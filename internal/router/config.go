/*
Package router — модуль роутера LLM-запросов: ротация API-ключей, алиасы
моделей и встроенный sing-box для трафика через VLESS.

Зачем. Бот должен уметь просить одного и того же «кодера» (алиас), не зная,
у какого провайдера сейчас жив ключ, и не умирая от 429 на одном ключе.
Роутер держит два TOML-конфига:

  - router.toml — провайдеры (url + список API-ключей) и алиасы моделей
    (алиас → упорядоченная цепочка «провайдер/модель»);
  - proxy.toml — секция [proxy] роутера плюс raw-секции sing-box
    (outbounds с ключами VLESS и т.п.), которые уходят в box без изменений.

Бот ходит в роутер как в обычный OpenAI-совместимый шлюз: base_url =
http://127.0.0.1:8791/v1, model = имя алиаса («coder»). Правки в Python-коде
не нужны — контракт ровно тот же, что у efi/llm/providers/openai_compatible.py.
*/
package router

import (
	"fmt"
	"net"
	"net/url"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/BurntSushi/toml"
)

// Дефолты, которые применяются, если конфиг молчит.
const (
	// DefaultAddr — адрес HTTP-сервера роутера. Только localhost: бот живёт
	// в том же хосте, отдавать перебор ключей наружу не нужно.
	DefaultAddr = "127.0.0.1:8791"
	// DefaultSocksAddr — socks5-инбаунд встроенного sing-box, через который
	// HTTP-клиент роутера ходит в провайдеры с proxy = true.
	DefaultSocksAddr = "127.0.0.1:20818"
	// DefaultProviderTimeout — базовый таймаут провайдера, если не задан
	// (нижняя граница бюджета запроса, см. requestTimeout).
	DefaultProviderTimeout = 60.0
)

// ProviderConfig — один LLM-провайдер в router.toml: [providers.<имя>].
type ProviderConfig struct {
	// URL — базовый адрес OpenAI-совместимого API (с /v1), например
	// "https://dashscope.aliyuncs.com/compatible-mode/v1".
	URL string `toml:"url"`
	// APIKeys — список ключей провайдера. Один ключ выжимается до отказа,
	// затем роутер переходит к следующему (см. rotation.go).
	APIKeys []string `toml:"api_keys"`
	// TimeoutSeconds — база бюджета запроса к провайдеру. Настоящий бюджет
	// шире: к нему прибавляется время на запрошенный объём вывода.
	TimeoutSeconds float64 `toml:"timeout_seconds"`
	// Proxy — гонять ли запросы к провайдеру через встроенный sing-box
	// (proxy.toml). false — соединение напрямую, из процесса роутера.
	Proxy bool `toml:"proxy"`
}

// Timeout возвращает нормализованный базовый таймаут провайдера.
func (p ProviderConfig) Timeout() time.Duration {
	seconds := p.TimeoutSeconds
	if seconds <= 0 {
		seconds = DefaultProviderTimeout
	}
	return time.Duration(seconds * float64(time.Second))
}

// AliasConfig — алиас модели в router.toml: [aliases.<имя>].
//
// Бот спрашивает роутер с model = "<алиас>"; цепочка — это порядок, в котором
// провайдеры пробуются: сначала все ключи первого, затем второго и т.д.
type AliasConfig struct {
	// Chain — упорядоченная цепочка кандидатов «провайдер + модель».
	Chain []ChainEntry `toml:"chain"`
	// Interactive — интерактивна ли задача для алиаса. Определяет потолок
	// бюджета запроса: 90с живому диалогу, 300с фону (паритет с
	// efi/llm/timeouts.py). По умолчанию true.
	Interactive *bool `toml:"interactive"`
}

// IsInteractive — интерактивен ли алиас (дефолт true, пока конфиг не сказал
// иначе).
func (a AliasConfig) IsInteractive() bool {
	return a.Interactive == nil || *a.Interactive
}

// ChainEntry — один шаг цепочки алиаса: провайдер и имя модели у него.
type ChainEntry struct {
	Provider string `toml:"provider"`
	Model    string `toml:"model"`
}

// ServerConfig — секция [server] router.toml.
type ServerConfig struct {
	// Addr — адрес HTTP-сервера роутера (DefaultAddr, если пусто).
	Addr string `toml:"addr"`
}

// Config — разобранный router.toml.
type Config struct {
	Server    ServerConfig              `toml:"server"`
	Providers map[string]ProviderConfig `toml:"providers"`
	Aliases   map[string]AliasConfig    `toml:"aliases"`
}

// AddrOrDefault — адрес сервера с подстановкой дефолта.
func (s ServerConfig) AddrOrDefault() string {
	if strings.TrimSpace(s.Addr) == "" {
		return DefaultAddr
	}
	return s.Addr
}

// LoadConfig читает и валидирует router.toml.
//
// Валидация строгая намеренно: опечатка в имени провайдера внутри алиаса —
// это тихий отказ в рантайме, когда запрос дойдёт до перебора, а не ошибка на
// старте. Дешевле не подняться, чем ловить «алиас coder молчит».
func LoadConfig(path string) (*Config, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("чтение %s: %w", path, err)
	}
	var config Config
	if err := toml.Unmarshal(raw, &config); err != nil {
		return nil, fmt.Errorf("разбор %s: %w", path, err)
	}
	if err := config.validate(path); err != nil {
		return nil, err
	}
	return &config, nil
}

// validate проверяет конфиг.
func (c *Config) validate(path string) error {
	if len(c.Providers) == 0 {
		return fmt.Errorf("%s: не задано ни одного провайдера ([providers.<имя>])", path)
	}
	if len(c.Aliases) == 0 {
		return fmt.Errorf("%s: не задано ни одного алиаса ([aliases.<имя>])", path)
	}

	for name, provider := range c.Providers {
		if err := validateProviderURL(provider.URL); err != nil {
			return fmt.Errorf("%s: [providers.%s].url: %w", path, name, err)
		}
		if len(provider.APIKeys) == 0 {
			return fmt.Errorf("%s: [providers.%s].api_keys пуст — роутеру нечем подписать запрос",
				path, name)
		}
		for i, key := range provider.APIKeys {
			if strings.TrimSpace(key) == "" {
				return fmt.Errorf("%s: [providers.%s].api_keys[%d] пуст", path, name, i)
			}
		}
	}

	for name, alias := range c.Aliases {
		if len(alias.Chain) == 0 {
			return fmt.Errorf("%s: [aliases.%s].chain пуст — алиас не на что развернуть", path, name)
		}
		for i, entry := range alias.Chain {
			if strings.TrimSpace(entry.Provider) == "" {
				return fmt.Errorf("%s: [aliases.%s].chain[%d].provider пуст", path, name, i)
			}
			if _, ok := c.Providers[entry.Provider]; !ok {
				return fmt.Errorf("%s: [aliases.%s].chain[%d] ссылается на неизвестного провайдера %q",
					path, name, i, entry.Provider)
			}
			if strings.TrimSpace(entry.Model) == "" {
				return fmt.Errorf("%s: [aliases.%s].chain[%d].model пуст", path, name, i)
			}
		}
	}

	if _, _, err := net.SplitHostPort(c.Server.AddrOrDefault()); err != nil {
		return fmt.Errorf("%s: [server].addr %q: %w", path, c.Server.AddrOrDefault(), err)
	}
	return nil
}

// validateProviderURL — адрес должен быть http(s) и с хостом.
func validateProviderURL(raw string) error {
	if strings.TrimSpace(raw) == "" {
		return fmt.Errorf("пусто")
	}
	parsed, err := url.Parse(raw)
	if err != nil {
		return err
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return fmt.Errorf("ожидался http/https, получено %q", parsed.Scheme)
	}
	if parsed.Host == "" {
		return fmt.Errorf("нет хоста")
	}
	return nil
}

// Alias возвращает алиас по имени.
func (c *Config) Alias(name string) (AliasConfig, bool) {
	alias, ok := c.Aliases[name]
	return alias, ok
}

// Provider возвращает провайдера по имени.
func (c *Config) Provider(name string) (ProviderConfig, bool) {
	provider, ok := c.Providers[name]
	return provider, ok
}

// AliasNames — отсортированные имена алиасов (для /v1/models).
func (c *Config) AliasNames() []string {
	names := make([]string, 0, len(c.Aliases))
	for name := range c.Aliases {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// --- proxy.toml -------------------------------------------------------------

// ProxyConfig — секция [proxy] proxy.toml: настройки самого роутера.
// Остальные секции файла raw — это конфиг sing-box, он уходит в box как есть.
type ProxyConfig struct {
	// Enabled — поднимать ли встроенный sing-box.
	Enabled bool `toml:"enabled"`
	// SocksAddr — адрес socks5-инбаунда, куда HTTP-клиент роутера шлёт
	// запросы к провайдерам с proxy = true.
	SocksAddr string `toml:"socks_addr"`
	// LogLevel — уровень логов sing-box (trace..fatal).
	LogLevel string `toml:"log_level"`
}

// normalize подставляет дефолты до валидации.
func (p *ProxyConfig) normalize() {
	if p.Enabled && p.SocksAddr == "" {
		p.SocksAddr = DefaultSocksAddr
	}
	if p.LogLevel == "" {
		p.LogLevel = "warn"
	}
}

// ProxyFile — разобранный proxy.toml: наша секция + raw-карта для sing-box.
type ProxyFile struct {
	// Proxy — настройки роутера ([proxy]).
	Proxy ProxyConfig
	// Raw — весь файл как есть, минус секция [proxy]: outbounds, route, dns…
	// Имена секций sing-box не конфликтуют с нашей секцией.
	Raw map[string]any
}

// LoadProxyConfig читает proxy.toml. Файл опционален: без него роутер
// работает напрямую; proxy = true у провайдера при выключенном sing-box —
// ошибка на старте (см. New).
func LoadProxyConfig(path string) (*ProxyFile, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("чтение %s: %w", path, err)
	}
	var file ProxyFile
	// Первый проход — в типизированную секцию [proxy]. Обёртка нужна:
	// напрямую в ProxyConfig парсер искал бы ключи в корне файла, а не
	// внутри секции, и enabled остался бы false.
	var wrapper struct {
		Proxy ProxyConfig `toml:"proxy"`
	}
	if err := toml.Unmarshal(raw, &wrapper); err != nil {
		return nil, fmt.Errorf("разбор %s: %w", path, err)
	}
	file.Proxy = wrapper.Proxy
	// Второй проход — во всю карту, чтобы отдать raw-секции sing-box.
	var whole map[string]any
	if err := toml.Unmarshal(raw, &whole); err != nil {
		return nil, fmt.Errorf("разбор %s: %w", path, err)
	}
	delete(whole, "proxy")
	file.Raw = whole
	file.Proxy.normalize()

	if file.Proxy.Enabled {
		if _, _, err := net.SplitHostPort(file.Proxy.SocksAddr); err != nil {
			return nil, fmt.Errorf("%s: [proxy].socks_addr %q: %w", path, file.Proxy.SocksAddr, err)
		}
	}
	return &file, nil
}
