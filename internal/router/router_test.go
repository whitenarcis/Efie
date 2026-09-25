package router

import (
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// fakeProvider — тестовый провайдер: отвечает 401 на заведомо плохие ключи
// и 200 на хороший, запоминая, каким ключом и с какой моделью пришёл запрос.
type fakeProvider struct {
	server *httptest.Server
	// goodKey — единственный ключ, на который сервер отвечает успехом.
	goodKey string
	// seen — ключи в порядке обращений.
	seen []string
	// models — какие model реально приходили (проверяем подстановку).
	models []string
}

func newFakeProvider(t *testing.T, goodKey string) *fakeProvider {
	t.Helper()
	fp := &fakeProvider{goodKey: goodKey}
	fp.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		key := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")

		var payload map[string]any
		_ = json.NewDecoder(r.Body).Decode(&payload)
		model, _ := payload["model"].(string)

		fp.seen = append(fp.seen, key)
		fp.models = append(fp.models, model)

		if key != fp.goodKey {
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = w.Write([]byte(`{"error":"bad key"}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}]}`))
	}))
	t.Cleanup(fp.server.Close)
	return fp
}

// configFor собирает конфиг из описания провайдеров и алиаса "coder".
func configFor(providers map[string]ProviderConfig, chain []ChainEntry) *Config {
	return &Config{
		Providers: providers,
		Aliases:   map[string]AliasConfig{"coder": {Chain: chain}},
	}
}

// quietLogger убирает логи роутера из вывода тестов.
func quietLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

// chatRequest отправляет нестримовый запрос алиаса и возвращает запись.
func chatRequest(t *testing.T, core *Router, model string) *httptest.ResponseRecorder {
	t.Helper()
	body := `{"model":"` + model + `","stream":false,"max_tokens":100}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()
	core.ServeHTTP(rec, req)
	return rec
}

// Ключи одного провайдера перебираются по порядку, пока не найдётся живой.
func TestFailsOverAcrossKeysOfOneProvider(t *testing.T) {
	fp := newFakeProvider(t, "key-3")
	core, err := New(configFor(map[string]ProviderConfig{
		"qwencloud": {URL: fp.server.URL, APIKeys: []string{"key-1", "key-2", "key-3"}},
	}, []ChainEntry{{Provider: "qwencloud", Model: "qwen3-coder-plus"}}), nil, quietLogger())
	if err != nil {
		t.Fatal(err)
	}

	rec := chatRequest(t, core, "coder")
	if rec.Code != http.StatusOK {
		t.Fatalf("код %d, тело %s", rec.Code, rec.Body)
	}
	if len(fp.seen) != 3 {
		t.Fatalf("ожидались 3 попытки, было %d: %v", len(fp.seen), fp.seen)
	}
	// Порядок строго по конфигу: роутер не перемешивает ключи.
	want := []string{"key-1", "key-2", "key-3"}
	for i, key := range want {
		if fp.seen[i] != key {
			t.Fatalf("порядок ключей нарушен: %v, ожидалось %v", fp.seen, want)
		}
	}
}

// Провайдер целиком отказал — запрос уходит к следующему в цепочке.
func TestFailsOverAcrossProviders(t *testing.T) {
	first := newFakeProvider(t, "never-matches")
	second := newFakeProvider(t, "good")

	core, err := New(configFor(map[string]ProviderConfig{
		"qwencloud":   {URL: first.server.URL, APIKeys: []string{"k1", "k2"}},
		"ollamacloud": {URL: second.server.URL, APIKeys: []string{"good"}},
	}, []ChainEntry{
		{Provider: "qwencloud", Model: "qwen3-coder-plus"},
		{Provider: "ollamacloud", Model: "qwen3-coder:30b"},
	}), nil, quietLogger())
	if err != nil {
		t.Fatal(err)
	}

	rec := chatRequest(t, core, "coder")
	if rec.Code != http.StatusOK {
		t.Fatalf("код %d, тело %s", rec.Code, rec.Body)
	}
	// Сначала выбиты оба ключа первого провайдера, потом ключ второго.
	if len(first.seen) != 2 {
		t.Fatalf("первый провайдер должен был быть выбит целиком: %v", first.seen)
	}
	if len(second.seen) != 1 {
		t.Fatalf("второй провайдер должен был получить один запрос: %v", second.seen)
	}
	// Алиас разворачивается в реальное имя модели каждого провайдера.
	if first.models[0] != "qwen3-coder-plus" {
		t.Errorf("модель первого провайдера %q", first.models[0])
	}
	if second.models[0] != "qwen3-coder:30b" {
		t.Errorf("модель второго провайдера %q", second.models[0])
	}
}

// Липкость проверяется сквозь HTTP: успешный ключ не должен «прыгать»
// на каждом запросе — иначе провайдер теряет кеш контекста сессии.
func TestStickyKeyAcrossRequests(t *testing.T) {
	fp := newFakeProvider(t, "good")
	core, err := New(configFor(map[string]ProviderConfig{
		"p": {URL: fp.server.URL, APIKeys: []string{"good", "second"}},
	}, []ChainEntry{{Provider: "p", Model: "m"}}), nil, quietLogger())
	if err != nil {
		t.Fatal(err)
	}

	for i := 0; i < 3; i++ {
		if rec := chatRequest(t, core, "coder"); rec.Code != http.StatusOK {
			t.Fatalf("запрос %d: код %d, тело %s", i, rec.Code, rec.Body)
		}
	}
	for i, key := range fp.seen {
		if key != "good" {
			t.Fatalf("запрос %d ушёл на ключ %q — ротация должна быть липкой: %v", i, key, fp.seen)
		}
	}
}

// Неизвестный алиас — 404 с перечнем доступных, а не молчаливая подмена.
func TestUnknownAliasIsRejected(t *testing.T) {
	fp := newFakeProvider(t, "good")
	core, err := New(configFor(map[string]ProviderConfig{
		"p": {URL: fp.server.URL, APIKeys: []string{"good"}},
	}, []ChainEntry{{Provider: "p", Model: "m"}}), nil, quietLogger())
	if err != nil {
		t.Fatal(err)
	}

	rec := chatRequest(t, core, "nope")
	if rec.Code != http.StatusNotFound {
		t.Fatalf("код %d, ожидался 404; тело %s", rec.Code, rec.Body)
	}
	if len(fp.seen) != 0 {
		t.Fatalf("провайдер не должен вызываться при неизвестном алиасе: %v", fp.seen)
	}
}

// Стрим релеится как есть, включая первый чанк, прочитанный для фейловера.
func TestStreamIsRelayed(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		for _, chunk := range []string{"data: one\n\n", "data: two\n\n", "data: [DONE]\n\n"} {
			_, _ = io.WriteString(w, chunk)
			if flusher != nil {
				flusher.Flush()
			}
		}
	}))
	t.Cleanup(server.Close)

	core, err := New(configFor(map[string]ProviderConfig{
		"p": {URL: server.URL, APIKeys: []string{"k"}},
	}, []ChainEntry{{Provider: "p", Model: "m"}}), nil, quietLogger())
	if err != nil {
		t.Fatal(err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"coder","stream":true}`))
	rec := httptest.NewRecorder()
	core.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("код %d, тело %s", rec.Code, rec.Body)
	}
	out := rec.Body.String()
	for _, want := range []string{"data: one", "data: two", "data: [DONE]"} {
		if !strings.Contains(out, want) {
			t.Errorf("в ответе нет %q; получено: %q", want, out)
		}
	}
}

// /v1/models показывает алиасы: бот должен видеть "coder" как имя модели.
func TestModelsListsAliases(t *testing.T) {
	fp := newFakeProvider(t, "k")
	core, err := New(configFor(map[string]ProviderConfig{
		"p": {URL: fp.server.URL, APIKeys: []string{"k"}},
	}, []ChainEntry{{Provider: "p", Model: "m"}}), nil, quietLogger())
	if err != nil {
		t.Fatal(err)
	}

	rec := httptest.NewRecorder()
	core.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/models", nil))

	var payload struct {
		Data []struct{ ID string } `json:"data"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("разбор ответа: %v; тело %s", err, rec.Body)
	}
	if len(payload.Data) != 1 || payload.Data[0].ID != "coder" {
		t.Fatalf("ожидался алиас coder, получено %+v", payload.Data)
	}
}
