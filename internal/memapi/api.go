/*
Package memapi — HTTP-контракт Go-подсистемы памяти: то, через что Python-бот
в переходный период зовёт Go, а не наоборот.

Почему HTTP на localhost, а не cgo-библиотека или FFI. Подсистема памяти
должна переживать перезапуск бота и не утягивать его за собой: упавший поиск
не имеет права уронить диалог, а гигабайт весов модели — «подешеветь» от того,
что RAG стал Go-библиотекой в том же процессе. Отдельный процесс с
JSON-контрактом позволяет подменять реализацию по частям: сначала сюда
переезжает поиск и дневник, потом — эмбеддинги — и на каждом шаге Python-код
меняется на вызов метода, а не на переписывание подсистемы.

Наружу НЕ отдаются векторы: в выдаче поиска десять записей по 1024 измерения —
это сотни килобайт JSON на каждое сообщение. Вектор нужен ровно в двух местах
(сравнение фактов и проверка дубля), и для них есть отдельный эндпоинт
`/v1/embed/*`.
*/
package memapi

import (
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/whitenarcis/Efie/internal/diary"
	"github.com/whitenarcis/Efie/internal/memory"
	"github.com/whitenarcis/Efie/internal/memstats"
)

// maxBodyBytes — потолок на тело запроса. Записи дневника бывают длинными
// (новеллизация эпизода — до сотни килобайт), но не бесконечными.
const maxBodyBytes = 4 << 20

// Server — обработчики подсистемы памяти.
type Server struct {
	memory  *memory.Memory
	logger  *slog.Logger
	started time.Time

	searches     atomic.Int64
	tfidfAnswers atomic.Int64
	remembers    atomic.Int64
	duplicates   atomic.Int64
	unembedded   atomic.Int64
}

// New создаёт сервер поверх готового оркестратора памяти.
func New(mem *memory.Memory, logger *slog.Logger) *Server {
	if logger == nil {
		logger = slog.Default()
	}
	return &Server{memory: mem, logger: logger, started: time.Now()}
}

// Handler собирает маршруты.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", s.handleHealth)
	mux.HandleFunc("GET /v1/stats", s.handleStats)
	mux.HandleFunc("POST /v1/embed/query", s.handleEmbedQuery)
	mux.HandleFunc("POST /v1/embed/document", s.handleEmbedDocument)
	mux.HandleFunc("POST /v1/diary/search", s.handleSearch)
	mux.HandleFunc("POST /v1/diary/remember", s.handleRemember)
	mux.HandleFunc("GET /v1/diary/entries", s.handleEntries)
	mux.HandleFunc("GET /v1/diary/entry/{id}", s.handleEntry)
	mux.HandleFunc("DELETE /v1/diary/entry/{id}", s.handleDeleteEntry)
	mux.HandleFunc("POST /v1/diary/entry/{id}/touch", s.handleTouchEntry)
	mux.HandleFunc("POST /v1/tfidf/search", s.handleTfidfSearch)
	return mux
}

func (s *Server) handleHealth(writer http.ResponseWriter, _ *http.Request) {
	entries, err := s.memory.Store().Load()
	if err != nil {
		s.writeError(writer, http.StatusInternalServerError, err.Error())
		return
	}
	s.writeJSON(writer, http.StatusOK, map[string]any{
		"status":  "ok",
		"entries": entries,
		"sources": s.memory.Sources(),
	})
}

func (s *Server) handleStats(writer http.ResponseWriter, _ *http.Request) {
	entries, err := s.memory.Store().Load()
	if err != nil {
		s.writeError(writer, http.StatusInternalServerError, err.Error())
		return
	}
	s.writeJSON(writer, http.StatusOK, map[string]any{
		"entries":          entries,
		"indexed":          s.memory.Index().Len(),
		"embedder_sources": s.memory.Sources(),
		"searches":         s.searches.Load(),
		"tfidf_answers":    s.tfidfAnswers.Load(),
		"remembers":        s.remembers.Load(),
		"duplicates":       s.duplicates.Load(),
		"unembedded":       s.unembedded.Load(),
		"uptime_seconds":   time.Since(s.started).Seconds(),
		"rss_bytes":        memstats.RSSBytes(),
	})
}

func (s *Server) writeJSON(writer http.ResponseWriter, status int, payload any) {
	writer.Header().Set("Content-Type", "application/json; charset=utf-8")
	writer.WriteHeader(status)
	encoder := json.NewEncoder(writer)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(payload); err != nil {
		s.logger.Error("memapi: cannot encode response", "error", err)
	}
}

func (s *Server) writeError(writer http.ResponseWriter, status int, message string) {
	s.writeJSON(writer, status, map[string]string{"error": message})
}

// decodeBody читает и разбирает JSON-тело запроса, ограничивая его размер.
func decodeBody(request *http.Request, target any) error {
	if request.Body == nil {
		return errors.New("empty request body")
	}
	raw, err := io.ReadAll(io.LimitReader(request.Body, maxBodyBytes))
	if err != nil {
		return err
	}
	if len(strings.TrimSpace(string(raw))) == 0 {
		return errors.New("empty request body")
	}
	return json.Unmarshal(raw, target)
}

func parseQueryInt(request *http.Request, name string, fallback int) int {
	raw := request.URL.Query().Get(name)
	if raw == "" {
		return fallback
	}
	value, err := strconv.Atoi(raw)
	if err != nil {
		return fallback
	}
	return value
}

// queryOptions собирает опции поиска с дефолтами Python-версии
// (confidence_factor 0.01, max_entry_count 10, min_relatedness 0.0), но с
// возможностью переопределить любое из них запросом: настройки памяти
// (memory.min_relatedness = 0.80) задаёт бот, а не сервис.
func queryOptions(request searchRequest) diary.QueryOptions {
	options := diary.DefaultQueryOptions()
	if request.MaxEntryCount != nil {
		options.MaxEntryCount = *request.MaxEntryCount
	}
	if request.MinRelatedness != nil {
		options.MinRelatedness = *request.MinRelatedness
	}
	if request.ConfidenceFactor != nil {
		options.ConfidenceFactor = *request.ConfidenceFactor
	}
	return options
}

// --- DTO -------------------------------------------------------------------

// entryDTO — запись для внешнего мира: без вектора (см. докстринг пакета),
// но с указанием, есть ли он, — по этому признаку видно, попадёт ли запись в
// семантический поиск или только в TF-IDF.
type entryDTO struct {
	ID         string  `json:"id"`
	Body       string  `json:"body"`
	Confidence float64 `json:"confidence"`
	Domain     string  `json:"domain"`
	UsageCount int     `json:"usage_count"`
	CreatedAt  string  `json:"created_at"`
	LastUsed   *string `json:"last_used"`
	Dimensions int     `json:"dimensions"`
}

func newEntryDTO(entry *diary.Entry) entryDTO {
	dto := entryDTO{
		ID:         entry.ID,
		Body:       entry.Body,
		Confidence: entry.Metadata.Confidence,
		Domain:     entry.Metadata.Domain,
		UsageCount: entry.Metadata.UsageCount,
		CreatedAt:  diary.FormatTimestamp(entry.Metadata.CreatedAt),
		Dimensions: len(entry.Metadata.Embedding),
	}
	if entry.Metadata.LastUsed != nil {
		formatted := diary.FormatTimestamp(*entry.Metadata.LastUsed)
		dto.LastUsed = &formatted
	}
	return dto
}

// resultDTO — результат поиска: запись + релевантность (аналог DiaryQueryResult).
type resultDTO struct {
	Entry       entryDTO `json:"entry"`
	Relatedness float64  `json:"relatedness"`
}

func newResultDTO(result diary.QueryResult) resultDTO {
	return resultDTO{Entry: newEntryDTO(result.Entry), Relatedness: result.Relatedness}
}

type embedRequest struct {
	Text string `json:"text"`
}

type embedResponse struct {
	Source     string    `json:"source"`
	Dimensions int       `json:"dimensions"`
	Embedding  []float64 `json:"embedding"`
}

/*
searchRequest — тело поискового запроса.

`Domains` приходят от бота: классификатор «какие домены уместны под этот текст»
(efi/memory/router.py) остался в Python, и это осознанно — словари и правила
читаются вместе с поведением Эфи. Пустой список доменов означает «без
фильтрации», ровно как в Python-версии.
*/
type searchRequest struct {
	Text             string   `json:"text"`
	MaxEntryCount    *int     `json:"max_entry_count"`
	MinRelatedness   *float64 `json:"min_relatedness"`
	ConfidenceFactor *float64 `json:"confidence_factor"`
	Domains          []string `json:"domains"`
}

type searchResponse struct {
	Source  string      `json:"source"`
	Results []resultDTO `json:"results"`
}

type rememberRequest struct {
	Body       string   `json:"body"`
	Confidence *float64 `json:"confidence"`
	Domain     string   `json:"domain"`
}

type rememberResponse struct {
	Saved     bool      `json:"saved"`
	Entry     *entryDTO `json:"entry,omitempty"`
	Duplicate *entryDTO `json:"duplicate,omitempty"`
}
