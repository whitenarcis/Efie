/*
Package memory — Go-порт efi/memory/rag.py: единая точка входа для
семантического поиска и записи в долгосрочную память.

Критический путь — `Search`: вызывается прямо перед генерацией ответа, и ни на
одном из уровней сбой не должен ронять формирование ответа целиком. Отсюда
лестница отката, воспроизведённая по Python-версии буквально:

 1. короткое/фатическое сообщение («привет», «ок») — сразу TF-IDF: сетевой
    или модельный round-trip для него не имеет смысла;
 2. иначе — эмбеддинг запроса (локальный движок, затем облако);
 3. если эмбеддинга нет никаким путём — снова TF-IDF, вместо исключения.

Почему этот пакет первым в Go-переносе. В Python-версии он стоит на каждом
сообщении и держит в памяти numpy-матрицу корпуса, копию TF-IDF-индекса и
ONNX-сессию модели e5 (порядка гигабайта весов). Здесь та же логика работает
без numpy, без копий индекса и — после подключения ONNX-движка — в отдельном
процессе, который Python не тащит на себе.
*/
package memory

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/whitenarcis/Efie/internal/diary"
	"github.com/whitenarcis/Efie/internal/embed"
	"github.com/whitenarcis/Efie/internal/memdomain"
	"github.com/whitenarcis/Efie/internal/tfidf"
	"github.com/whitenarcis/Efie/internal/vec"
)

// DedupThreshold — порог склейки семантических дублей в выдаче (0.90 в
// Python-версии): ловит перефразы одного и того же, не склевая соседние по
// теме, но разные воспоминания.
const DedupThreshold = 0.90

// SourceTFIDF — имя уровня TF-IDF в SearchOutcome.Source.
const SourceTFIDF = "tfidf"

// SearchOutcome — результат поиска вместе с уровнем, который его обслужил.
// Уровень нужен вызывающей стороне, чтобы отличить «память промолчала» от
// «память отвечала по TF-IDF, потому что эмбеддингов нет».
type SearchOutcome struct {
	Results []diary.QueryResult
	Source  string // "tfidf" | "local" | "cloud"
}

// RememberOutcome — результат записи в память: сохранённая запись либо
// найденный дубль, из-за которого запись не сохранялась.
type RememberOutcome struct {
	Entry       *diary.Entry
	Saved       bool
	Duplicate   *diary.Entry
	Relatedness float64
}

/*
Memory — оркестратор памяти: дневник + TF-IDF-индекс + лестница эмбеддингов.

Все зависимости собираются снаружи (владение жизненным циклом — за
cmd/efie-memory), поэтому этот тип ничего не знает ни про HTTP, ни про файлы
конфигурации.
*/
type Memory struct {
	logger *slog.Logger
	store  *diary.Store
	index  *tfidf.Index
	ladder embed.Ladder

	// DedupThreshold — порог склейки дублей в выдаче (см. DedupThreshold).
	DedupThreshold float64

	mu       sync.Mutex
	reported map[string]struct{}

	now func() time.Time
}

// New собирает оркестратор. Уровни эмбеддингов необязательны: без них память
// продолжает работать на TF-IDF — это штатный режим деградации, а не поломка
// (ровно как в Python-версии, где облачный провайдер может не иметь
// embedding-моделей вовсе).
func New(store *diary.Store, index *tfidf.Index, ladder embed.Ladder, logger *slog.Logger) *Memory {
	if logger == nil {
		logger = slog.Default()
	}
	memory := &Memory{
		logger:         logger,
		store:          store,
		index:          index,
		DedupThreshold: DedupThreshold,
		reported:       make(map[string]struct{}),
		now:            time.Now,
	}
	ladder.OnLevelFailure = memory.onLevelFailure
	memory.ladder = ladder
	return memory
}

// Index возвращает TF-IDF-индекс (нужен консолидации: удалённая из дневника
// запись должна исчезнуть и из поиска).
func (m *Memory) Index() *tfidf.Index { return m.index }

// Store возвращает дневник.
func (m *Memory) Store() *diary.Store { return m.store }

// Sources — настроенные уровни эмбеддингов, для /v1/stats и логов.
func (m *Memory) Sources() []string { return m.ladder.Sources() }

// onLevelFailure — жалоба на отказ уровня эмбеддингов. Причины стабильны
// («модель не установлена» не изменится никогда, недоступный провайдер держится
// в cooldown минутами), а вызывается это на КАЖДОЕ сообщение, поэтому каждая
// причина попадает в лог один раз, дальше — в debug.
func (m *Memory) onLevelFailure(level string, err error) {
	switch level {
	case "local":
		m.complainOnce(fmt.Sprintf("local embedding failed (%v), falling back to cloud", err))
	default:
		m.complainOnce(fmt.Sprintf("cloud embedding request failed (%v)", err))
	}
}

func (m *Memory) complainOnce(reason string) {
	m.mu.Lock()
	_, seen := m.reported[reason]
	m.reported[reason] = struct{}{}
	m.mu.Unlock()
	if seen {
		m.logger.Debug("rag: embedding trouble", "reason", reason)
		return
	}
	m.logger.Warn("rag: embedding trouble", "reason", reason)
}

/*
Search — семантический поиск по долгосрочной памяти.

`domains` сужает выборку до нужных доменов (классификатор «какие домены
уместны под этот текст» остаётся в Python: см. efi/memory/router.py, словари и
правила читаются вместе с поведением Эфи). `nil` — без фильтрации.
*/
func (m *Memory) Search(
	ctx context.Context,
	queryText string,
	options diary.QueryOptions,
	domains []memdomain.Domain,
) (SearchOutcome, error) {
	filter := memdomain.Filter(domains)

	if tfidf.ShouldUseShortcut(queryText) {
		return SearchOutcome{Results: m.index.Search(queryText, options, filter), Source: SourceTFIDF}, nil
	}

	vector, source, err := m.ladder.EmbedQuery(ctx, queryText)
	if err != nil {
		if errors.Is(err, embed.ErrNoEmbedder) {
			m.complainOnce("no embedding engine configured, every search goes to TF-IDF")
		}
		// Деградация вместо исключения: без памяти ответ возможен, без ответа
		// память бесполезна.
		return SearchOutcome{Results: m.index.Search(queryText, options, filter), Source: SourceTFIDF}, nil
	}

	results, err := m.store.Query(vector, options, filter)
	if err != nil {
		return SearchOutcome{}, err
	}
	return SearchOutcome{Results: m.dedupByEmbedding(results), Source: source}, nil
}

// EmbedQuery — вектор поискового запроса тем же движком, что и сами поиски.
// Возвращает nil и пустой источник, если эмбеддинга нет ни с одного уровня.
func (m *Memory) EmbedQuery(ctx context.Context, text string) ([]float64, string) {
	vector, source, err := m.ladder.EmbedQuery(ctx, text)
	if err != nil {
		return nil, ""
	}
	return vector, source
}

// EmbedDocument — вектор сохраняемого текста (та же лестница уровней).
func (m *Memory) EmbedDocument(ctx context.Context, text string) ([]float64, string) {
	vector, source, err := m.ladder.EmbedDocument(ctx, text)
	if err != nil {
		return nil, ""
	}
	return vector, source
}

// Embed — публичный доступ к тому же источнику эмбеддингов, которым пользуется
// сам поиск. Нужен семантической дедупликации фактов: сравнивать факты
// вектором, полученным другим движком, чем дневник, — значит сравнивать
// несравнимое (в Python-версии это `RAGMemory.embed`).
func (m *Memory) Embed(ctx context.Context, text string) []float64 {
	vector, _ := m.EmbedDocument(ctx, text)
	return vector
}

/*
Remember добавляет новую запись в долгосрочную память: считает эмбеддинг,
проверяет на дубль/плагиат и при отсутствии дубля сохраняет запись (индексируя
её в TF-IDF, чтобы тот не отставал от дневника).

Если эмбеддинга нет ни с одного уровня — запись всё равно сохраняется, но без
вектора и без дубль-чека: лучше запомнить текстом и найти потом через TF-IDF,
чем потерять запись целиком.

`Saved == false` означает, что запись отбракована как дубль существующей —
и в этом случае `Duplicate` указывает, какого именно. Python-версия в этой
ситуации возвращает None, а инструмент `remember_diary_entry` отвечает
пользователю «похожее уже есть»; знать, что именно нашлось, полезнее, чем
догадываться по логу.
*/
func (m *Memory) Remember(
	ctx context.Context,
	body string,
	confidence float64,
	domain string,
) (RememberOutcome, error) {
	vector, _, err := m.ladder.EmbedDocument(ctx, body)
	if err != nil {
		m.logger.Warn("rag: no embedding available for new entry (local and cloud both failed), storing without one")
		entry := &diary.Entry{
			ID:       newEntryID(m.now()),
			Metadata: diary.NewMetadata(domain, confidence),
			Body:     body,
		}
		if err := m.store.Save(entry); err != nil {
			return RememberOutcome{}, err
		}
		m.index.Add(entry)
		return RememberOutcome{Entry: entry, Saved: true}, nil
	}

	duplicate, err := m.store.IsDuplicateOf(vector)
	if err != nil {
		return RememberOutcome{}, err
	}
	if duplicate != nil {
		m.logger.Info(
			"rag: skipping duplicate diary entry",
			"relatedness", duplicate.Relatedness,
			"existing_id", duplicate.Entry.ID,
		)
		return RememberOutcome{
			Saved:       false,
			Duplicate:   duplicate.Entry,
			Relatedness: duplicate.Relatedness,
		}, nil
	}

	metadata := diary.NewMetadata(domain, confidence)
	metadata.Embedding = vector
	entry := &diary.Entry{ID: newEntryID(m.now()), Metadata: metadata, Body: body}
	if err := m.store.Save(entry); err != nil {
		return RememberOutcome{}, err
	}
	m.index.Add(entry)
	return RememberOutcome{Entry: entry, Saved: true}, nil
}

/*
dedupByEmbedding склеивает семантические дубли в выдаче.

Без этого один и тот же эпизод — сохранённый в дневник несколько раз
(web_lookup + researcher + novelization одного гуглёжа) — всплывает в промпте
дважды-трижды под разными формулировками.

Склейка детерминистична: идём от самого релевантного и для каждой следующей
проверяем, не дубль ли она любой уже оставленной. При совпадении растёт
usage_count оставленной — модель видит «это подтверждалось дважды», даже если
формулировка одна. Правка счётчика идёт по живому объекту кэша: следующий
`Store.Save` её сохранит (в Python-версии — ровно так же).
*/
func (m *Memory) dedupByEmbedding(results []diary.QueryResult) []diary.QueryResult {
	if len(results) < 2 {
		return results
	}
	kept := make([]diary.QueryResult, 0, len(results))
	keptVectors := make([][]float64, 0, len(results))
	for _, result := range results {
		vector := result.Entry.Metadata.Embedding
		if vec.Norm(vector) <= 0 {
			// Вектора нет (или он нулевой) — сравнивать нечем, но и терять
			// запись нельзя: она уходит в выдачу как есть.
			kept = append(kept, result)
			continue
		}
		duplicateAt := -1
		for index, keptVector := range keptVectors {
			if vec.Cosine(vector, keptVector) >= m.DedupThreshold {
				duplicateAt = index
				break
			}
		}
		if duplicateAt >= 0 {
			kept[duplicateAt].Entry.Metadata.UsageCount++
			continue
		}
		kept = append(kept, result)
		keptVectors = append(keptVectors, vector)
	}
	return kept
}

// newEntryID — unix-таймстамп + короткий случайный суффикс: уникально даже при
// параллельных вызовах в одну секунду (аналог `_generate_entry_id`).
func newEntryID(now time.Time) string {
	suffix := make([]byte, 3)
	if _, err := rand.Read(suffix); err != nil {
		return fmt.Sprintf("%d_%06d", now.Unix(), now.Nanosecond()%1_000_000)
	}
	return fmt.Sprintf("%d_%s", now.Unix(), hex.EncodeToString(suffix))
}
