package memapi

import (
	"errors"
	"net/http"
	"strings"

	"github.com/whitenarcis/Efie/internal/diary"
	"github.com/whitenarcis/Efie/internal/memdomain"
	"github.com/whitenarcis/Efie/internal/memory"
)

func (s *Server) handleEmbedQuery(writer http.ResponseWriter, request *http.Request) {
	s.handleEmbed(writer, request, true)
}

func (s *Server) handleEmbedDocument(writer http.ResponseWriter, request *http.Request) {
	s.handleEmbed(writer, request, false)
}

// handleEmbed отдаёт вектор текста. Отдельный эндпоинт, а не поле в выдаче
// поиска: векторы нужны ровно двум потребителям (дубль-чек фактов и проверка
// перед сохранением записи), а таскать их в каждом ответе поиска — это сотни
// килобайт JSON на каждое сообщение (см. докстринг пакета).
func (s *Server) handleEmbed(writer http.ResponseWriter, request *http.Request, asQuery bool) {
	var payload embedRequest
	if err := decodeBody(request, &payload); err != nil {
		s.writeError(writer, http.StatusBadRequest, err.Error())
		return
	}
	if strings.TrimSpace(payload.Text) == "" {
		s.writeError(writer, http.StatusBadRequest, "text is required")
		return
	}
	var (
		vector []float64
		source string
	)
	if asQuery {
		vector, source = s.memory.EmbedQuery(request.Context(), payload.Text)
	} else {
		vector, source = s.memory.EmbedDocument(request.Context(), payload.Text)
	}
	if vector == nil {
		// Это не ошибка сервиса: «эмбеддингов нет» — штатное состояние, при
		// котором поиск уходит на TF-IDF, а запись сохраняется без вектора.
		s.writeJSON(writer, http.StatusOK, embedResponse{Source: "", Dimensions: 0, Embedding: []float64{}})
		return
	}
	s.writeJSON(writer, http.StatusOK, embedResponse{Source: source, Dimensions: len(vector), Embedding: vector})
}

func (s *Server) handleSearch(writer http.ResponseWriter, request *http.Request) {
	var payload searchRequest
	if err := decodeBody(request, &payload); err != nil {
		s.writeError(writer, http.StatusBadRequest, err.Error())
		return
	}
	if strings.TrimSpace(payload.Text) == "" {
		s.writeError(writer, http.StatusBadRequest, "text is required")
		return
	}
	outcome, err := s.memory.Search(
		request.Context(),
		payload.Text,
		queryOptions(payload),
		domains(payload.Domains),
	)
	if err != nil {
		s.writeError(writer, http.StatusInternalServerError, err.Error())
		return
	}
	s.searches.Add(1)
	if outcome.Source == memory.SourceTFIDF {
		s.tfidfAnswers.Add(1)
	}
	s.writeJSON(writer, http.StatusOK, searchResponse{
		Source:  outcome.Source,
		Results: results(outcome.Results),
	})
}

/*
handleTfidfSearch — поиск строго по TF-IDF, минуя эмбеддинги.

Нужен там, где семантика неуместна или недоступна: ручная проверка памяти из
дашборда, отладка «почему нашлось именно это», а также сравнение двух выдач
(«что находит TF-IDF против того, что находит вектор») — без такого эндпоинта
такое сравнение возможно только правкой кода.
*/
func (s *Server) handleTfidfSearch(writer http.ResponseWriter, request *http.Request) {
	var payload searchRequest
	if err := decodeBody(request, &payload); err != nil {
		s.writeError(writer, http.StatusBadRequest, err.Error())
		return
	}
	if strings.TrimSpace(payload.Text) == "" {
		s.writeError(writer, http.StatusBadRequest, "text is required")
		return
	}
	found := s.memory.Index().Search(payload.Text, queryOptions(payload), memdomain.Filter(domains(payload.Domains)))
	s.writeJSON(writer, http.StatusOK, searchResponse{Source: memory.SourceTFIDF, Results: results(found)})
}

func (s *Server) handleRemember(writer http.ResponseWriter, request *http.Request) {
	var payload rememberRequest
	if err := decodeBody(request, &payload); err != nil {
		s.writeError(writer, http.StatusBadRequest, err.Error())
		return
	}
	if strings.TrimSpace(payload.Body) == "" {
		s.writeError(writer, http.StatusBadRequest, "body is required")
		return
	}
	confidence := 0.0
	if payload.Confidence != nil {
		confidence = *payload.Confidence
		if confidence < -1.0 || confidence > 1.0 {
			s.writeError(writer, http.StatusBadRequest, "confidence must be within [-1, 1]")
			return
		}
	}
	outcome, err := s.memory.Remember(request.Context(), payload.Body, confidence, payload.Domain)
	if err != nil {
		s.writeError(writer, http.StatusInternalServerError, err.Error())
		return
	}
	if !outcome.Saved {
		s.duplicates.Add(1)
		// Дубль отдан отдельным полем, а не строкой в логе: вызывающей стороне
		// (remember_diary_entry) нужен ответ «похожее уже есть» — и желательно
		// с указанием, что именно нашлось.
		response := rememberResponse{Saved: false}
		if outcome.Duplicate != nil {
			duplicate := newEntryDTO(outcome.Duplicate)
			response.Duplicate = &duplicate
		}
		s.writeJSON(writer, http.StatusOK, response)
		return
	}
	s.remembers.Add(1)
	if len(outcome.Entry.Metadata.Embedding) == 0 {
		s.unembedded.Add(1)
	}
	dto := newEntryDTO(outcome.Entry)
	s.writeJSON(writer, http.StatusOK, rememberResponse{Saved: true, Entry: &dto})
}

func results(found []diary.QueryResult) []resultDTO {
	converted := make([]resultDTO, 0, len(found))
	for _, result := range found {
		converted = append(converted, newResultDTO(result))
	}
	return converted
}

// handleEntries отдаёт корпус постранично. Нужен консолидации и дашборду:
// и то, и другое ходит по памяти целиком, а не поиском.
func (s *Server) handleEntries(writer http.ResponseWriter, request *http.Request) {
	all, err := s.memory.Store().All()
	if err != nil {
		s.writeError(writer, http.StatusInternalServerError, err.Error())
		return
	}
	limit := parseQueryInt(request, "limit", 200)
	if limit <= 0 {
		limit = 200
	}
	if limit > 5000 {
		limit = 5000
	}
	offset := parseQueryInt(request, "offset", 0)
	if offset < 0 {
		offset = 0
	}
	if offset > len(all) {
		offset = len(all)
	}
	end := offset + limit
	if end > len(all) {
		end = len(all)
	}
	page := make([]entryDTO, 0, end-offset)
	for _, entry := range all[offset:end] {
		page = append(page, newEntryDTO(entry))
	}
	s.writeJSON(writer, http.StatusOK, map[string]any{
		"total":   len(all),
		"offset":  offset,
		"limit":   limit,
		"entries": page,
	})
}

func (s *Server) handleEntry(writer http.ResponseWriter, request *http.Request) {
	entry, known := s.memory.Store().Get(request.PathValue("id"))
	if !known {
		s.writeError(writer, http.StatusNotFound, diary.ErrNotFound.Error())
		return
	}
	s.writeJSON(writer, http.StatusOK, newEntryDTO(entry))
}

func (s *Server) handleDeleteEntry(writer http.ResponseWriter, request *http.Request) {
	id := request.PathValue("id")
	if _, known := s.memory.Store().Get(id); !known {
		s.writeError(writer, http.StatusNotFound, diary.ErrNotFound.Error())
		return
	}
	if err := s.memory.Store().Delete(id); err != nil {
		s.writeError(writer, http.StatusInternalServerError, err.Error())
		return
	}
	// Индекс обязан следовать за дневником: удалённая запись не должна
	// продолжать находиться поиском (иначе она всплывёт в промпте, хотя её
	// уже нет в памяти).
	s.memory.Index().Remove(id)
	s.writeJSON(writer, http.StatusOK, map[string]any{"deleted": true, "id": id})
}

func (s *Server) handleTouchEntry(writer http.ResponseWriter, request *http.Request) {
	entry, err := s.memory.Store().Touch(request.PathValue("id"))
	if err != nil {
		if errors.Is(err, diary.ErrNotFound) {
			s.writeError(writer, http.StatusNotFound, err.Error())
			return
		}
		s.writeError(writer, http.StatusInternalServerError, err.Error())
		return
	}
	s.writeJSON(writer, http.StatusOK, newEntryDTO(entry))
}

func domains(raw []string) []memdomain.Domain {
	if len(raw) == 0 {
		// Пустой список — «без фильтрации»: так же понимает это
		// DiaryEntryFilter в Python-версии.
		return nil
	}
	parsed := make([]memdomain.Domain, 0, len(raw))
	for _, value := range raw {
		parsed = append(parsed, memdomain.Parse(value, memdomain.History))
	}
	return parsed
}
