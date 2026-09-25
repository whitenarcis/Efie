package memory

import (
	"context"
	"errors"
	"log/slog"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/whitenarcis/Efie/internal/diary"
	"github.com/whitenarcis/Efie/internal/embed"
	"github.com/whitenarcis/Efie/internal/memdomain"
	"github.com/whitenarcis/Efie/internal/tfidf"
)

// stubEngine — источник эмбеддингов с заранее заданным вектором: тесты памяти
// не должны зависеть ни от модели, ни от сети (в Python-версии ровно поэтому
// LocalEmbeddingEngine и LLMRouter подменяются фейками).
type stubEngine struct {
	query    []float64
	document []float64
	fail     bool

	mu   sync.Mutex
	seen []string
}

func (s *stubEngine) EmbedQuery(_ context.Context, text string) ([]float64, error) {
	s.record(text)
	if s.fail {
		return nil, errors.New("stub engine failure")
	}
	return append([]float64{}, s.query...), nil
}

func (s *stubEngine) EmbedDocument(_ context.Context, text string) ([]float64, error) {
	s.record(text)
	if s.fail {
		return nil, errors.New("stub engine failure")
	}
	return append([]float64{}, s.document...), nil
}

func (s *stubEngine) record(text string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.seen = append(s.seen, text)
}

func (s *stubEngine) texts() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string{}, s.seen...)
}

// recordingHandler ловит записи лога: поведение «жалоба один раз на причину» —
// часть контракта (лог читают именно тогда, когда что-то сломалось), и
// проверять его надо тестом, а не глазами.
type recordingHandler struct {
	mu      sync.Mutex
	records []string
	levels  []slog.Level
}

func (h *recordingHandler) Enabled(context.Context, slog.Level) bool { return true }

func (h *recordingHandler) Handle(_ context.Context, record slog.Record) error {
	// Атрибуты попадают в запись так же, как в текст: причина отказа живёт
	// именно в них (message — общий), и без них тест «жалуется на локальный
	// движок» проверял бы только факт записи, а не её содержание.
	parts := []string{record.Message, record.Level.String()}
	record.Attrs(func(attribute slog.Attr) bool {
		parts = append(parts, attribute.Key+"="+attribute.Value.String())
		return true
	})
	h.mu.Lock()
	defer h.mu.Unlock()
	h.records = append(h.records, strings.Join(parts, " "))
	h.levels = append(h.levels, record.Level)
	return nil
}

func (h *recordingHandler) WithAttrs([]slog.Attr) slog.Handler { return h }
func (h *recordingHandler) WithGroup(string) slog.Handler      { return h }

func (h *recordingHandler) warnings() []string {
	h.mu.Lock()
	defer h.mu.Unlock()
	found := make([]string, 0, len(h.records))
	for index, level := range h.levels {
		if level == slog.LevelWarn {
			found = append(found, h.records[index])
		}
	}
	return found
}

func newTestMemory(t *testing.T, ladder embed.Ladder, logger *slog.Logger) *Memory {
	t.Helper()
	store := diary.NewStore(t.TempDir(), 0.97, logger)
	mem := New(store, tfidf.New(), ladder, logger)
	return mem
}

func seed(t *testing.T, mem *Memory, id, body string, domain string, embedding []float64) *diary.Entry {
	t.Helper()
	metadata := diary.NewMetadata(domain, 0.5)
	metadata.Embedding = embedding
	metadata.CreatedAt = time.Date(2026, 4, 5, 6, 7, 8, 0, time.UTC)
	entry := &diary.Entry{ID: id, Metadata: metadata, Body: body}
	if err := mem.Store().Save(entry); err != nil {
		t.Fatalf("cannot seed %s: %v", id, err)
	}
	mem.Index().Add(entry)
	return entry
}

func TestShortQueryGoesStraightToTfidf(t *testing.T) {
	logger := slog.New(&recordingHandler{})
	mem := newTestMemory(t, embed.Ladder{}, logger)
	seed(t, mem, "note", "Сегодня мы говорили про монтаж звука.", diary.DomainHistory, nil)

	outcome, err := mem.Search(context.Background(), "привет", diary.DefaultQueryOptions(), nil)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if outcome.Source != SourceTFIDF {
		t.Fatalf("короткая реплика ушла на уровень %q вместо TF-IDF", outcome.Source)
	}
}

func TestSearchWithoutEmbedderDegradesToTfidfAndComplainsOnce(t *testing.T) {
	handler := &recordingHandler{}
	mem := newTestMemory(t, embed.Ladder{}, slog.New(handler))
	seed(t, mem, "note", "Сегодня мы говорили про монтаж звука и дорожки.", diary.DomainHistory, nil)

	query := "расскажи, что мы обсуждали про монтаж звука"
	for attempt := 0; attempt < 3; attempt++ {
		outcome, err := mem.Search(context.Background(), query, diary.DefaultQueryOptions(), nil)
		if err != nil {
			t.Fatalf("search: %v", err)
		}
		// Поиск не имеет права падать из-за отсутствия эмбеддингов: без памяти
		// ответ возможен, без ответа память бесполезна.
		if outcome.Source != SourceTFIDF {
			t.Fatalf("без движка эмбеддингов поиск ушёл на %q", outcome.Source)
		}
		if len(outcome.Results) == 0 {
			t.Fatal("TF-IDF не нашёл запись, которую сам же проиндексировал")
		}
	}
	if warnings := handler.warnings(); len(warnings) != 1 {
		t.Fatalf("ожидалась ровно одна жалоба на причину, получено %d: %v", len(warnings), warnings)
	}
}

func TestSearchUsesLocalEmbedderAndReportsSource(t *testing.T) {
	engine := &stubEngine{query: []float64{1, 0, 0, 0}, document: []float64{1, 0, 0, 0}}
	mem := newTestMemory(t, embed.Ladder{Local: embed.NewPrefixed("local", engine)}, slog.New(&recordingHandler{}))
	seed(t, mem, "match", "Запись, которая совпадёт по вектору.", diary.DomainHistory, []float64{1, 0, 0, 0})

	outcome, err := mem.Search(
		context.Background(),
		"расскажи, что мы обсуждали про монтаж звука",
		diary.DefaultQueryOptions(),
		nil,
	)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if outcome.Source != "local" {
		t.Fatalf("источник = %q, ожидался local", outcome.Source)
	}
	if len(outcome.Results) != 1 || outcome.Results[0].Entry.ID != "match" {
		t.Fatalf("векторный поиск вернул %d результатов: %v", len(outcome.Results), outcome.Results)
	}
	// Запрос ушёл в e5-движок с префиксом query:, как в Python-версии.
	if texts := engine.texts(); len(texts) != 1 || !strings.HasPrefix(texts[0], embed.QueryPrefix) {
		t.Fatalf("движок получил %v, ожидался текст с префиксом %q", texts, embed.QueryPrefix)
	}
}

func TestLadderFallsBackFromLocalToCloud(t *testing.T) {
	local := &stubEngine{fail: true}
	remote := &stubEngine{query: []float64{0, 1, 0, 0}, document: []float64{0, 1, 0, 0}}
	handler := &recordingHandler{}
	mem := newTestMemory(
		t,
		embed.Ladder{Local: embed.NewPrefixed("local", local), Remote: remote},
		slog.New(handler),
	)
	seed(t, mem, "cloud-match", "Запись для облачного вектора.", diary.DomainHistory, []float64{0, 1, 0, 0})

	outcome, err := mem.Search(
		context.Background(),
		"расскажи, что мы обсуждали про монтаж звука",
		diary.DefaultQueryOptions(),
		nil,
	)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if outcome.Source != "cloud" {
		t.Fatalf("источник = %q, ожидался cloud (локальный движок сломан)", outcome.Source)
	}
	// Отказ локального уровня обязан быть виден в логе: молчаливая деградация
	// качества памяти — ровно тот сорт поломки, который ищут неделю.
	warnings := handler.warnings()
	if len(warnings) != 1 || !strings.Contains(warnings[0], "local embedding failed") {
		t.Fatalf("ожидалась одна жалоба на локальный движок, получено %v", warnings)
	}
}

func TestBothLevelsFailedStillAnswers(t *testing.T) {
	mem := newTestMemory(t, embed.Ladder{
		Local:  embed.NewPrefixed("local", &stubEngine{fail: true}),
		Remote: &stubEngine{fail: true},
	}, slog.New(&recordingHandler{}))
	seed(t, mem, "note", "Сегодня мы говорили про монтаж звука и дорожки.", diary.DomainHistory, nil)

	outcome, err := mem.Search(
		context.Background(),
		"расскажи, что мы обсуждали про монтаж звука",
		diary.DefaultQueryOptions(),
		nil,
	)
	if err != nil {
		t.Fatalf("search вернул ошибку вместо деградации: %v", err)
	}
	if outcome.Source != SourceTFIDF {
		t.Fatalf("источник = %q, ожидался TF-IDF", outcome.Source)
	}
}

func TestRememberWithoutEmbedderStoresTextAnyway(t *testing.T) {
	mem := newTestMemory(t, embed.Ladder{}, slog.New(&recordingHandler{}))
	outcome, err := mem.Remember(context.Background(), "Сегодня мы говорили про монтаж звука.", 0.7, diary.DomainHistory)
	if err != nil {
		t.Fatalf("remember: %v", err)
	}
	if !outcome.Saved || outcome.Entry == nil {
		t.Fatal("запись не сохранена, хотя текст важнее вектора")
	}
	if len(outcome.Entry.Metadata.Embedding) != 0 {
		t.Fatalf("вектор появился без источника эмбеддингов: %d измерений", len(outcome.Entry.Metadata.Embedding))
	}
	// Найти текстом: TF-IDF обязан быть синхронен с дневником.
	found, err := mem.Search(context.Background(), "расскажи про монтаж звука", diary.DefaultQueryOptions(), nil)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(found.Results) != 1 || found.Results[0].Entry.ID != outcome.Entry.ID {
		t.Fatalf("запись без вектора не находится через TF-IDF: %v", found.Results)
	}
}

func TestRememberRejectsDuplicateAndReportsIt(t *testing.T) {
	engine := &stubEngine{query: []float64{1, 0, 0, 0}, document: []float64{1, 0, 0, 0}}
	mem := newTestMemory(t, embed.Ladder{Local: embed.NewPrefixed("local", engine)}, slog.New(&recordingHandler{}))
	existing := seed(t, mem, "existing", "Сегодня мы говорили про монтаж звука.", diary.DomainHistory, []float64{1, 0, 0, 0})

	outcome, err := mem.Remember(context.Background(), "Мы говорили про монтаж звука сегодня.", 0.7, diary.DomainHistory)
	if err != nil {
		t.Fatalf("remember: %v", err)
	}
	if outcome.Saved {
		t.Fatal("дубль сохранён как новая запись")
	}
	if outcome.Duplicate == nil || outcome.Duplicate.ID != existing.ID {
		t.Fatalf("дубль не указан: %+v", outcome.Duplicate)
	}
	if entries := mem.Store().Len(); entries != 1 {
		t.Fatalf("в дневнике %d записей, ожидалась 1 (дубль не должен сохраняться)", entries)
	}
}

func TestRememberSavesEmbeddingAndIndexes(t *testing.T) {
	engine := &stubEngine{query: []float64{0, 0, 1, 0}, document: []float64{0, 0, 1, 0}}
	mem := newTestMemory(t, embed.Ladder{Local: embed.NewPrefixed("local", engine)}, slog.New(&recordingHandler{}))

	outcome, err := mem.Remember(context.Background(), "Новая запись про базы данных.", 0.5, diary.DomainCommon)
	if err != nil {
		t.Fatalf("remember: %v", err)
	}
	if !outcome.Saved || len(outcome.Entry.Metadata.Embedding) != 4 {
		t.Fatalf("вектор не сохранён: %+v", outcome.Entry.Metadata.Embedding)
	}
	if outcome.Entry.Metadata.Domain != diary.DomainCommon {
		t.Fatalf("домен = %q, ожидался C", outcome.Entry.Metadata.Domain)
	}
	// Документ ушёл в e5-движок с префиксом passage:.
	if texts := engine.texts(); len(texts) != 1 || !strings.HasPrefix(texts[0], embed.DocumentPrefix) {
		t.Fatalf("движок получил %v, ожидался текст с префиксом %q", texts, embed.DocumentPrefix)
	}
	found, err := mem.Search(context.Background(), "расскажи про базы данных", diary.DefaultQueryOptions(), nil)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(found.Results) != 1 || found.Results[0].Entry.ID != outcome.Entry.ID {
		t.Fatalf("сохранённая запись не находится векторным поиском: %v", found.Results)
	}
}

func TestDedupCollapsesNearDuplicatesInResults(t *testing.T) {
	engine := &stubEngine{query: []float64{1, 0, 0, 0}}
	mem := newTestMemory(t, embed.Ladder{Local: embed.NewPrefixed("local", engine)}, slog.New(&recordingHandler{}))
	first := seed(t, mem, "first", "Про монтаж звука, вариант один.", diary.DomainHistory, []float64{1, 0, 0, 0})
	seed(t, mem, "second", "Про монтаж звука, вариант два.", diary.DomainHistory, []float64{0.999, 0.01, 0, 0})

	outcome, err := mem.Search(
		context.Background(),
		"расскажи, что мы обсуждали про монтаж звука",
		diary.QueryOptions{ConfidenceFactor: 0.0, MaxEntryCount: 10},
		nil,
	)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(outcome.Results) != 1 {
		t.Fatalf("склейка дублей не сработала: %d результатов", len(outcome.Results))
	}
	if outcome.Results[0].Entry.ID != first.ID {
		t.Fatalf("оставлен %s, ожидался более релевантный %s", outcome.Results[0].Entry.ID, first.ID)
	}
	// Подтверждение дубля копится в счётчике использования: модель видит
	// «это подтверждалось дважды», даже если формулировка одна.
	if outcome.Results[0].Entry.Metadata.UsageCount != 1 {
		t.Fatalf("usage_count = %d, ожидался 1", outcome.Results[0].Entry.Metadata.UsageCount)
	}
}

func TestSearchAppliesDomainFilter(t *testing.T) {
	engine := &stubEngine{query: []float64{1, 0, 0, 0}}
	mem := newTestMemory(t, embed.Ladder{Local: embed.NewPrefixed("local", engine)}, slog.New(&recordingHandler{}))
	seed(t, mem, "world", "Про SQLite и WAL.", diary.DomainCommon, []float64{1, 0, 0, 0})
	seed(t, mem, "mine", "Про монтаж звука вечером.", diary.DomainHistory, []float64{1, 0, 0, 0})

	outcome, err := mem.Search(
		context.Background(),
		"расскажи, что мы обсуждали про монтаж звука",
		diary.DefaultQueryOptions(),
		[]memdomain.Domain{memdomain.History},
	)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(outcome.Results) != 1 || outcome.Results[0].Entry.ID != "mine" {
		t.Fatalf("доменный фильтр не применён: %v", outcome.Results)
	}
}
