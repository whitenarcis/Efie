package embed

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// captureEngine запоминает, что именно ему передали, — так проверяются префиксы
// e5 без единой модели.
type captureEngine struct {
	texts []string
	err   error
}

func (c *captureEngine) EmbedQuery(_ context.Context, text string) ([]float64, error) {
	c.texts = append(c.texts, text)
	if c.err != nil {
		return nil, c.err
	}
	return []float64{1, 0}, nil
}

func (c *captureEngine) EmbedDocument(_ context.Context, text string) ([]float64, error) {
	c.texts = append(c.texts, text)
	if c.err != nil {
		return nil, c.err
	}
	return []float64{0, 1}, nil
}

func TestRemoteSpeaksOpenAICompatibleEmbeddings(t *testing.T) {
	var (
		gotPath    string
		gotAuth    string
		gotPayload map[string]any
	)
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		gotPath = request.URL.Path
		gotAuth = request.Header.Get("Authorization")
		raw, _ := io.ReadAll(request.Body)
		_ = json.Unmarshal(raw, &gotPayload)
		writer.Header().Set("Content-Type", "application/json")
		_, _ = writer.Write([]byte(`{"data":[{"embedding":[0.5,-0.25]}]}`))
	}))
	defer server.Close()

	remote := NewRemote(server.URL, "secret-key", "text-embedding-3-small", 2*time.Second)
	vector, err := remote.EmbedQuery(context.Background(), "привет")
	if err != nil {
		t.Fatalf("embed: %v", err)
	}
	// Форма запроса — как у OpenAICompatibleProvider.embedding: путь, поле input
	// и модель подставляются вызывающей стороной, а не зашиты в клиент.
	if gotPath != "/embeddings" {
		t.Errorf("путь = %q, ожидался /embeddings", gotPath)
	}
	if gotAuth != "Bearer secret-key" {
		t.Errorf("Authorization = %q", gotAuth)
	}
	if gotPayload["model"] != "text-embedding-3-small" || gotPayload["input"] != "привет" {
		t.Errorf("payload = %v", gotPayload)
	}
	// Без префиксов: конвенция e5 — свойство конкретной модели, а не транспорта.
	if len(vector) != 2 || vector[0] != 0.5 {
		t.Errorf("вектор = %v", vector)
	}
}

func TestRemoteReportsFailures(t *testing.T) {
	rateLimited := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.WriteHeader(http.StatusTooManyRequests)
		_, _ = writer.Write([]byte(`{"error":"slow down"}`))
	}))
	defer rateLimited.Close()

	remote := NewRemote(rateLimited.URL, "", "any", time.Second)
	if _, err := remote.EmbedQuery(context.Background(), "текст"); err == nil {
		t.Fatal("HTTP 429 не привёл к ошибке: лестница не сможет уйти на следующий уровень")
	}

	broken := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		_, _ = writer.Write([]byte(`{"unexpected":true}`))
	}))
	defer broken.Close()
	remote = NewRemote(broken.URL, "", "any", time.Second)
	if _, err := remote.EmbedQuery(context.Background(), "текст"); err == nil {
		t.Fatal("неожиданная форма ответа принята за вектор")
	}
}

func TestPrefixedAddsE5Prefixes(t *testing.T) {
	engine := &captureEngine{}
	prefixed := NewPrefixed("local", engine)
	if _, err := prefixed.EmbedQuery(context.Background(), "как дела"); err != nil {
		t.Fatalf("embed query: %v", err)
	}
	if _, err := prefixed.EmbedDocument(context.Background(), "прожитый вечер"); err != nil {
		t.Fatalf("embed document: %v", err)
	}
	want := []string{QueryPrefix + "как дела", DocumentPrefix + "прожитый вечер"}
	if strings.Join(engine.texts, "|") != strings.Join(want, "|") {
		t.Fatalf("тексты = %v, ожидались %v", engine.texts, want)
	}
}

func TestLadderFallsThroughAndReportsLevel(t *testing.T) {
	local := &captureEngine{err: errors.New("модель не установлена")}
	remote := &captureEngine{}
	var failures []string
	ladder := Ladder{
		Local:  NewPrefixed("local", local),
		Remote: remote,
		OnLevelFailure: func(level string, err error) {
			failures = append(failures, level+": "+err.Error())
		},
	}
	vector, source, err := ladder.EmbedDocument(context.Background(), "текст")
	if err != nil {
		t.Fatalf("ladder: %v", err)
	}
	if source != "cloud" {
		t.Fatalf("источник = %q, ожидался cloud", source)
	}
	if len(vector) != 2 {
		t.Fatalf("вектор = %v", vector)
	}
	// Причина отказа локального уровня доезжает до вызывающей стороны:
	// без этого «жалоба один раз на причину» в логе стала бы невозможной.
	if len(failures) != 1 || !strings.Contains(failures[0], "local") {
		t.Fatalf("отказы уровней = %v", failures)
	}
}

func TestLadderWithoutEngines(t *testing.T) {
	ladder := Ladder{}
	if ladder.Configured() {
		t.Fatal("пустая лестница считается настроенной")
	}
	if _, _, err := ladder.EmbedQuery(context.Background(), "текст"); !errors.Is(err, ErrNoEmbedder) {
		t.Fatalf("ошибка = %v, ожидался ErrNoEmbedder", err)
	}
}

func TestHashEmbedderIsDeterministicAndNormalized(t *testing.T) {
	engine := NewHash(64)
	first, err := engine.EmbedQuery(context.Background(), "Сегодня мы говорили про монтаж звука.")
	if err != nil {
		t.Fatalf("embed: %v", err)
	}
	second, err := engine.EmbedQuery(context.Background(), "Сегодня мы говорили про монтаж звука.")
	if err != nil {
		t.Fatalf("embed: %v", err)
	}
	if len(first) != 64 {
		t.Fatalf("размерность = %d, ожидалась 64", len(first))
	}
	for index := range first {
		if first[index] != second[index] {
			t.Fatalf("хеш-эмбеддер не детерминирован на позиции %d", index)
		}
	}
	norm := 0.0
	for _, component := range first {
		norm += component * component
	}
	if norm < 0.999 || norm > 1.001 {
		t.Fatalf("норма вектора = %v, ожидалась единичная", norm)
	}
}
