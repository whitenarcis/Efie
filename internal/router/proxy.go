package router

/*
Встроенный sing-box: socks5-инбаунд в нашем процессе, трафик к провайдерам
с proxy = true уходит через outbounds из proxy.toml (VLESS и что ещё там
захочет владелец — sing-box принимает любые свои секции).

Почему встроенный, а не отдельный процесс. Роутер — уже отдельный процесс
рядом с Python-ботом (как efie-memory); плодить третий — значит иметь три
места, где могут разъехаться пути, версии и автозапуск. sing-box умеет
работать библиотекой: box.New + Start в том же процессе, конфиг — наши
[proxy]-настройки плюс raw-секции из proxy.toml, отдаваемые box без
изменений (то, что написано для sing-box, работает как есть).

Схема трафика:

    роутер --socks5--> [inbound: router-socks на 127.0.0.1:20818]
        -> route: private/loopback -> direct (свой же ollamacloud на
           localhost:20128 не должен вылезать наружу через VLESS)
        -> иначе -> outbounds из proxy.toml (vless и т.п.)
*/

import (
	"context"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"strconv"

	box "github.com/sagernet/sing-box"
	"github.com/sagernet/sing-box/include"
	"github.com/sagernet/sing-box/option"
	singjson "github.com/sagernet/sing/common/json"
)

// Теги, которые роутер гарантирует в конфиге sing-box.
const (
	// socksInboundTag — тег нашего инбаунда; по нему владелец может писать
	// свои правила «от инбаунда».
	socksInboundTag = "router-socks"
	// directOutboundTag — direct-outbound для loopback-правила и страховки.
	directOutboundTag = "direct"
)

// ProxyRuntime — запущенный sing-box + адрес его socks-инбаунда.
type ProxyRuntime struct {
	instance *box.Box
	cancel   context.CancelFunc
	// SocksAddr — host:port инбаунда для HTTP-клиента роутера.
	SocksAddr string
}

// StartProxy поднимает sing-box из proxy.toml. Возвращает (nil, nil), если
// proxy отключён — это не ошибка, а штатный режим «напрямую».
func StartProxy(file *ProxyFile, logger *slog.Logger) (*ProxyRuntime, error) {
	if file == nil || !file.Proxy.Enabled {
		return nil, nil
	}
	if logger == nil {
		logger = slog.Default()
	}

	options, err := buildBoxOptions(file)
	if err != nil {
		return nil, err
	}

	// include.Context кладёт в context реестры протоколов (vless, vmess, …);
	// без них box.New не сможет распаковать outbounds.
	runCtx, cancel := context.WithCancel(include.Context(context.Background()))

	instance, err := box.New(box.Options{
		Context: runCtx,
		Options: options,
	})
	if err != nil {
		cancel()
		return nil, fmt.Errorf("sing-box: создание: %w", err)
	}
	// Start сам вызывает PreStart (см. box.go), двойного вызова не нужно.
	if err := instance.Start(); err != nil {
		cancel()
		_ = instance.Close()
		return nil, fmt.Errorf("sing-box: запуск: %w", err)
	}
	logger.Info("efie-router: sing-box запущен", "socks", file.Proxy.SocksAddr)

	return &ProxyRuntime{
		instance:  instance,
		cancel:    cancel,
		SocksAddr: file.Proxy.SocksAddr,
	}, nil
}

// Close останавливает sing-box (безопасно вызывать на nil — proxy выключен).
func (p *ProxyRuntime) Close() error {
	if p == nil {
		return nil
	}
	p.cancel()
	return p.instance.Close()
}

// HTTPClient возвращает http.Client, идущий через socks-инбаунд sing-box.
// Схема socks5 в net/http = socks5h: имя хоста уходит в прокси, и DNS
// резолвится на стороне выхода — иначе DNS-блокировки обойти не выйдет.
func (p *ProxyRuntime) HTTPClient() (*http.Client, error) {
	if p == nil {
		return nil, fmt.Errorf("sing-box не запущен")
	}
	proxyURL, err := url.Parse("socks5://" + p.SocksAddr)
	if err != nil {
		return nil, fmt.Errorf("адрес socks-инбаунда %q: %w", p.SocksAddr, err)
	}
	return &http.Client{
		Transport: &http.Transport{Proxy: http.ProxyURL(proxyURL)},
	}, nil
}

// buildBoxOptions превращает proxy.toml в option.Options для sing-box.
//
// Что делаем с чужими секциями: почти ничего. Владелец пишет [outbounds],
// [[outbounds]], [route], [dns] как обычно — они уходят в box без изменений,
// и всё, что умеет sing-box, работает здесь так же. Мы только добавляем
// своё: socks-инбаунд, direct-outbound и правило loopback.
func buildBoxOptions(file *ProxyFile) (option.Options, error) {
	host, port, err := splitHostPort(file.Proxy.SocksAddr)
	if err != nil {
		return option.Options{}, fmt.Errorf("адрес socks-инбаунда %q: %w", file.Proxy.SocksAddr, err)
	}

	// Копия map: мутации должны остаться локальными, конфиг владельца не
	// трогаем (он может переиспользоваться для перезапуска).
	sections := make(map[string]any, len(file.Raw))
	for key, value := range file.Raw {
		sections[key] = value
	}

	// 1. Наш socks-инбаунд. Добавляем всегда; дубль тега sing-box сочтёт
	//    ошибкой конфига, и мы покажем её как есть.
	sections["inbounds"] = append(toAnySlice(sections["inbounds"]), map[string]any{
		"type":        "socks",
		"tag":         socksInboundTag,
		"listen":      host,
		"listen_port": port,
	})

	// 2. direct-outbound нужен для loopback-правила и как страховка, если
	//    владелец описал только vless.
	sections["outbounds"] = ensureDirectOutbound(sections["outbounds"])

	// 3. private/loopback всегда direct — локальные сервисы не вылезают в VLESS.
	sections["route"] = injectLoopbackRule(sections["route"])

	// 4. Логи: наш уровень, только если владелец не задал [log] сам.
	if _, ok := sections["log"]; !ok {
		sections["log"] = map[string]any{"level": file.Proxy.LogLevel}
	}

	// map -> JSON -> option.Options: sing-box парсит конфиг своим контекстным
	// unmarshaller'ом (реестры протоколов — из context).
	rawJSON, err := singjson.Marshal(sections)
	if err != nil {
		return option.Options{}, fmt.Errorf("sing-box: кодирование конфига: %w", err)
	}
	options, err := singjson.UnmarshalExtendedContext[option.Options](
		include.Context(context.Background()), rawJSON)
	if err != nil {
		return option.Options{}, fmt.Errorf("sing-box: разбор конфига: %w", err)
	}
	return options, nil
}

// splitHostPort разбирает "host:port" и возвращает host + числовой port.
func splitHostPort(addr string) (string, int, error) {
	host, portStr, err := net.SplitHostPort(addr)
	if err != nil {
		return "", 0, err
	}
	port, err := strconv.Atoi(portStr)
	if err != nil || port < 1 || port > 65535 {
		return "", 0, fmt.Errorf("некорректный порт %q", portStr)
	}
	return host, port, nil
}

// toAnySlice приводит значение секции к срезу значений: inbounds и
// outbounds — это массивы таблиц, но одиночная таблица тоже должна работать.
func toAnySlice(value any) []any {
	switch typed := value.(type) {
	case []any:
		return typed
	case nil:
		return nil
	default:
		return []any{typed}
	}
}

// ensureDirectOutbound гарантирует наличие direct-outbound в секции outbounds.
func ensureDirectOutbound(value any) []any {
	outbounds := toAnySlice(value)
	for _, raw := range outbounds {
		if entry, ok := raw.(map[string]any); ok {
			if tag, _ := entry["tag"].(string); tag == directOutboundTag {
				return outbounds
			}
		}
	}
	return append(outbounds, map[string]any{
		"type": "direct",
		"tag":  directOutboundTag,
	})
}

// injectLoopbackRule добавляет В НАЧАЛО route.rules правило «private/loopback
// -> direct». Начало важно: правила sing-box оцениваются сверху вниз, и
// локальный трафик должен перехватываться раньше, чем сработает final
// владельца (обычно vless).
func injectLoopbackRule(value any) map[string]any {
	route, _ := value.(map[string]any)
	if route == nil {
		route = map[string]any{}
	}
	rules := toAnySlice(route["rules"])
	loopback := map[string]any{
		"type":     "default",
		"action":   "route",
		"ip_cidr":  []any{"127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16", "::1/128", "fc00::/7", "fe80::/10"},
		"outbound": directOutboundTag,
	}
	route["rules"] = append([]any{loopback}, rules...)
	return route
}
