package router

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeTempConfig кладёт содержимое во временный файл и возвращает путь.
func writeTempConfig(t *testing.T, name, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

const validRouterTOML = `
[server]
addr = "127.0.0.1:8791"

[providers.qwencloud]
url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
api_keys = ["sk-1", "sk-2"]
timeout_seconds = 60.0
proxy = true

[providers.ollamacloud]
url = "https://ollama.com/v1"
api_keys = ["ollama-1"]

[aliases.coder]
chain = [
    { provider = "qwencloud", model = "qwen3-coder-plus" },
    { provider = "ollamacloud", model = "qwen3-coder:30b" },
]
`

func TestLoadConfig(t *testing.T) {
	config, err := LoadConfig(writeTempConfig(t, "router.toml", validRouterTOML))
	if err != nil {
		t.Fatalf("разбор не прошёл: %v", err)
	}
	if len(config.Providers) != 2 {
		t.Errorf("провайдеров %d, ожидалось 2", len(config.Providers))
	}
	alias, ok := config.Alias("coder")
	if !ok {
		t.Fatal("алиас coder не найден")
	}
	if len(alias.Chain) != 2 {
		t.Errorf("в цепочке %d шагов, ожидалось 2", len(alias.Chain))
	}
	if !alias.IsInteractive() {
		t.Error("interactive по умолчанию должен быть true")
	}
	// Дефолты применяются, а не остаются нулями.
	provider, _ := config.Provider("ollamacloud")
	if provider.Timeout() == 0 {
		t.Error("таймаут провайдера не должен быть нулевым")
	}
}

// Валидация ловит опечатки на старте, а не в рантайме: неизвестный провайдер
// в алиасе иначе «пропал» бы молча в момент запроса.
func TestLoadConfigRejectsBadInput(t *testing.T) {
	cases := []struct {
		name    string
		content string
		wantErr string
	}{
		{
			name: "провайдер не объявлен",
			content: `
[providers.p]
url = "https://example.com/v1"
api_keys = ["k"]

[aliases.a]
chain = [{ provider = "нет-такого", model = "m" }]
`,
			wantErr: "неизвестного провайдера",
		},
		{
			name: "пустые ключи",
			content: `
[providers.p]
url = "https://example.com/v1"
api_keys = []

[aliases.a]
chain = [{ provider = "p", model = "m" }]
`,
			wantErr: "api_keys пуст",
		},
		{
			name: "не http url",
			content: `
[providers.p]
url = "ftp://example.com"
api_keys = ["k"]

[aliases.a]
chain = [{ provider = "p", model = "m" }]
`,
			wantErr: "ожидался http/https",
		},
		{
			name: "пустая цепочка",
			content: `
[providers.p]
url = "https://example.com/v1"
api_keys = ["k"]

[aliases.a]
chain = []
`,
			wantErr: "chain пуст",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := LoadConfig(writeTempConfig(t, "router.toml", tc.content))
			if err == nil {
				t.Fatal("ожидалась ошибка валидации")
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("ошибка %q не содержит %q", err, tc.wantErr)
			}
		})
	}
}

// proxy.toml: секция [proxy] роутера идёт отдельно, остальное — raw для sing-box.
func TestLoadProxyConfig(t *testing.T) {
	path := writeTempConfig(t, "proxy.toml", `
[proxy]
enabled = true
log_level = "debug"

[[outbounds]]
type = "vless"
tag = "v"
server = "example.com"
server_port = 443
uuid = "00000000-0000-0000-0000-000000000000"
`)
	file, err := LoadProxyConfig(path)
	if err != nil {
		t.Fatalf("разбор не прошёл: %v", err)
	}
	if !file.Proxy.Enabled {
		t.Error("proxy.enabled должен быть true")
	}
	if file.Proxy.SocksAddr != DefaultSocksAddr {
		t.Errorf("socks_addr %q, ожидался дефолт %q", file.Proxy.SocksAddr, DefaultSocksAddr)
	}
	// Секция [proxy] не должна утечь в raw — она не часть конфига sing-box.
	if _, leaked := file.Raw["proxy"]; leaked {
		t.Error("секция [proxy] попала в raw и уйдёт в sing-box")
	}
	if _, ok := file.Raw["outbounds"]; !ok {
		t.Error("секция outbounds потерялась")
	}
}
