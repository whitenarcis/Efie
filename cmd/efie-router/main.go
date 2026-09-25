/*
Command efie-router — локальный шлюз к LLM-провайдерам с ротацией ключей.

Бот (Python) общается с этим процессом как с обычным OpenAI-совместимым
провайдером: base_url = http://127.0.0.1:8791/v1, model = имя алиаса
(например, "coder"). Дальше роутер сам решает, у какого провайдера и каким
ключом отвечать (см. internal/router).

Зачем так. Три вещи, которых нет в наивном клиенте:

  - Ротация ключей. У провайдера обычно несколько ключей; на одном ключе
    бот умирает от 429. Роутер выжимает ключ до отказа и лишь потом берёт
    следующий — причём липко, а не по кругу: чередование ключей на каждом
    запросе рвёт кеш контекста у облачных агрегаторов (см. rotation.go).
  - Цепочка провайдеров. Алиас разворачивается в список кандидатов, и при
    отказе одного провайдера запрос уходит к следующему незаметно для бота.
  - Встроенный sing-box. Часть провайдеров (Qwen Cloud с зарубежного адреса,
    например) доступна только через VLESS. Вместо «прокси где-то в системе»
    роутер поднимает sing-box у себя в процессе, а трафик к конкретному
    провайдеру помечается в конфиге флагом proxy = true.

Запуск:

	efie-router -config ~/.efi_data/router.toml -proxy-config ~/.efi_data/proxy.toml

Файлы конфигурации:

	router.toml — провайдеры (url + api_keys) и алиасы моделей;
	proxy.toml  — секция [proxy] и outbounds sing-box с ключами VLESS.

Проверка: efie-router -check -config ~/.efi_data/router.toml
*/
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/whitenarcis/Efie/internal/router"
)

type options struct {
	configPath  string
	proxyPath   string
	logLevel    string
	check       bool
	showVersion bool
}

func main() {
	opts := resolveConfig(parseFlags())
	if err := run(opts); err != nil {
		fmt.Fprintln(os.Stderr, "efie-router:", err)
		os.Exit(1)
	}
}

func parseFlags() options {
	var opts options
	// Дефолты пустые: настоящий путь решает resolveConfig — он перебирает
	// каталоги в том же порядке, что и Efie (см. config/paths.py).
	flag.StringVar(&opts.configPath, "config", "", "путь к router.toml (по умолчанию — src/config/router.toml)")
	flag.StringVar(&opts.proxyPath, "proxy-config", "", "путь к proxy.toml (по умолчанию — src/config/proxy.toml)")
	flag.StringVar(&opts.logLevel, "log-level", "info", "уровень логов: debug|info|warn|error")
	flag.BoolVar(&opts.check, "check", false, "только проверить конфиги и выйти")
	flag.BoolVar(&opts.showVersion, "version", false, "напечатать версию и выйти")
	flag.Parse()
	return opts
}

// resolveConfig ищет конфиги там же, где их ищет сам Efie: src/config/ рядом
// с проектом, а на Termux (где каталог только для чтения) — ~/.efi_data/.
// Явный -config всегда побеждает.
//
// Возвращает пути (возможно, несуществующие — это разбирается в run: для
// router.toml отсутствие файла — ошибка, для proxy.toml — штатный режим).
func resolveConfig(opts options) options {
	dirs := configDirs()
	if opts.configPath == "" {
		opts.configPath = findConfig(dirs, "router.toml")
	}
	if opts.proxyPath == "" {
		opts.proxyPath = findConfig(dirs, "proxy.toml")
	}
	return opts
}

// configDirs — каталоги-кандидаты в порядке приоритета: тот же список, что
// в src/efi/config/paths.py (src/config, затем ~/.efi_data для Termux).
func configDirs() []string {
	dirs := make([]string, 0, 2)
	if cwd, err := os.Getwd(); err == nil {
		dirs = append(dirs, filepath.Join(cwd, "src", "config"))
	}
	if home, err := os.UserHomeDir(); err == nil {
		dirs = append(dirs, filepath.Join(home, ".efi_data"))
	}
	return dirs
}

// findConfig возвращает первый существующий <dir>/<name>, иначе — первый
// кандидат (чтобы ошибка называла ожидаемое место, а не пустую строку).
func findConfig(dirs []string, name string) string {
	fallback := filepath.Join(dirs[0], name)
	for _, dir := range dirs {
		candidate := filepath.Join(dir, name)
		if _, err := os.Stat(candidate); err == nil {
			return candidate
		}
	}
	return fallback
}

// version — проставь при релизе через -ldflags.
var version = "dev"

func run(opts options) error {
	if opts.showVersion {
		fmt.Println("efie-router", version)
		return nil
	}

	logger := newLogger(opts.logLevel)

	config, err := router.LoadConfig(opts.configPath)
	if err != nil {
		return err
	}

	// proxy.toml опционален: его отсутствие — не ошибка, а «работаем напрямую».
	// Если файл есть, но сломан — ошибка: молча потерять ключи VLESS нельзя.
	var proxyFile *router.ProxyFile
	if _, statErr := os.Stat(opts.proxyPath); statErr == nil {
		proxyFile, err = router.LoadProxyConfig(opts.proxyPath)
		if err != nil {
			return err
		}
	} else if !errors.Is(statErr, os.ErrNotExist) {
		return fmt.Errorf("проверка %s: %w", opts.proxyPath, statErr)
	} else {
		logger.Info("efie-router: proxy.toml не найден, трафик идёт напрямую",
			"path", opts.proxyPath)
	}

	if opts.check {
		return check(config, proxyFile, logger)
	}

	// Встроенный sing-box поднимаем до HTTP-сервера: если прокси не взлетел,
	// лучше не открывать порт, чем принимать запросы, которые не уйдут.
	proxy, err := router.StartProxy(proxyFile, logger)
	if err != nil {
		return err
	}
	defer func() { _ = proxy.Close() }()

	var proxied *http.Client
	if proxy != nil {
		proxied, err = proxy.HTTPClient()
		if err != nil {
			return err
		}
	}

	core, err := router.New(config, proxied, logger)
	if err != nil {
		return err
	}

	addr := config.Server.AddrOrDefault()
	server := &http.Server{
		Addr:              addr,
		Handler:           core,
		ReadHeaderTimeout: 10 * time.Second,
		// Без WriteTimeout: стрим может идти долго, а обрывать его по
		// часам — значит рвать живой диалог. Потолок на весь перебор
		// держит chainBudget внутри роутера.
	}

	listener, err := net.Listen("tcp", addr)
	if err != nil {
		return fmt.Errorf("слушаю %s: %w", addr, err)
	}
	logger.Info("efie-router: слушаю",
		"addr", addr,
		"aliases", strings.Join(config.AliasNames(), ","),
		"providers", len(config.Providers),
		"proxy", proxy != nil)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	errs := make(chan error, 1)
	go func() {
		if err := server.Serve(listener); err != nil && !errors.Is(err, http.ErrServerClosed) {
			errs <- err
		}
	}()

	select {
	case err := <-errs:
		return fmt.Errorf("HTTP-сервер: %w", err)
	case <-ctx.Done():
		logger.Info("efie-router: останавливаюсь")
	}

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	return server.Shutdown(shutdownCtx)
}

// check — режим -check: валидация конфигов без запуска сервера и прокси.
func check(config *router.Config, proxyFile *router.ProxyFile, logger *slog.Logger) error {
	names := config.AliasNames()
	logger.Info("efie-router: конфиг в порядке",
		"providers", len(config.Providers),
		"aliases", len(names))

	for _, name := range names {
		alias, _ := config.Alias(name)
		chain := make([]string, 0, len(alias.Chain))
		for _, entry := range alias.Chain {
			provider, _ := config.Provider(entry.Provider)
			chain = append(chain, fmt.Sprintf("%s/%s (ключей: %d, прокси: %v)",
				entry.Provider, entry.Model, len(provider.APIKeys), provider.Proxy))
		}
		logger.Info("алиас", "name", name, "chain", strings.Join(chain, " -> "))
	}

	if proxyFile == nil {
		logger.Info("прокси: выключен (прямые подключения)")
		return nil
	}
	if !proxyFile.Proxy.Enabled {
		logger.Info("прокси: выключен в proxy.toml")
		return nil
	}
	logger.Info("прокси: включён", "socks", proxyFile.Proxy.SocksAddr)
	return nil
}

// newLogger собирает slog с человекочитаемым выводом.
func newLogger(level string) *slog.Logger {
	var slogLevel slog.Level
	switch strings.ToLower(strings.TrimSpace(level)) {
	case "debug":
		slogLevel = slog.LevelDebug
	case "warn", "warning":
		slogLevel = slog.LevelWarn
	case "error":
		slogLevel = slog.LevelError
	default:
		slogLevel = slog.LevelInfo
	}
	return slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slogLevel}))
}
