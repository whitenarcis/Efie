/*
Package embed — источники эмбеддингов и лестница отката, порт
efi/memory/rag.py (`_compute_embedding`) вместе с efi/memory/local_embeddings.py.

Источник эмбеддингов трёхуровневый, от быстрого/бесплатного к самому
примитивному:

 1. локальный движок — без сети и без зависимости от того, какие модели держит
    текущий LLM-провайдер (основной путь: у Groq роли FAST нет
    embedding-моделей вообще);
 2. облачный фолбэк — тот же OpenAI-совместимый `/embeddings`;
 3. TF-IDF (в Go это пакет tfidf) — если оба пути выше недоступны.

Здесь живут первые два уровня. Третий подключается оркестратором (rag), потому
что решение «идти в TF-IDF или нет» — это поведение, а не транспорт.

Модели семейства e5 (дефолт: intfloat/multilingual-e5-large) асимметричны:
запрос и сохраняемый документ эмбеддятся с разными префиксами, отсюда
`EmbedQuery`/`EmbedDocument` вместо одного универсального `Embed`. Облачный
фолбэк префиксов НЕ добавляет: там модель задаёт провайдер, и навязывать ей
конвенцию e5 — значит портить чужие векторы.
*/
package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Префиксы асимметричных e5-моделей.
const (
	QueryPrefix    = "query: "
	DocumentPrefix = "passage: "
)

// ErrNoEmbedder — ни одного источника эмбеддингов не настроено. Не является
// ошибкой сама по себе: вызывающая сторона уходит на TF-IDF, как Python-версия
// при `embedding is None`.
var ErrNoEmbedder = errors.New("embed: no embedding engine configured")

// Engine — источник эмбеддингов.
type Engine interface {
	EmbedQuery(ctx context.Context, text string) ([]float64, error)
	EmbedDocument(ctx context.Context, text string) ([]float64, error)
}

// EngineError помечает отказ конкретного уровня лестницы: кто именно не смог
// (для лога) и почему.
type EngineError struct {
	Engine string
	Err    error
}

func (e *EngineError) Error() string {
	return fmt.Sprintf("%s: %v", e.Engine, e.Err)
}

func (e *EngineError) Unwrap() error { return e.Err }

// Prefixed добавляет префикс e5 к тексту перед отправкой в движок.
// Оборачивает локальные e5-движки; облачные — никогда.
type Prefixed struct {
	Engine Engine
	Name   string
}

// NewPrefixed оборачивает движок конвенцией префиксов e5
// (аналог embed_query/embed_document в LocalEmbeddingEngine).
func NewPrefixed(name string, engine Engine) *Prefixed {
	return &Prefixed{Engine: engine, Name: name}
}

// EmbedQuery — эмбеддинг ПОИСКОВОГО запроса: `query: <текст>`.
func (p *Prefixed) EmbedQuery(ctx context.Context, text string) ([]float64, error) {
	vector, err := p.Engine.EmbedQuery(ctx, QueryPrefix+text)
	if err != nil {
		return nil, &EngineError{Engine: p.Name, Err: err}
	}
	return vector, nil
}

// EmbedDocument — эмбеддинг СОХРАНЯЕМОГО текста: `passage: <текст>`.
func (p *Prefixed) EmbedDocument(ctx context.Context, text string) ([]float64, error) {
	vector, err := p.Engine.EmbedDocument(ctx, DocumentPrefix+text)
	if err != nil {
		return nil, &EngineError{Engine: p.Name, Err: err}
	}
	return vector, nil
}

/*
Ladder — порядок «локальный движок, затем облако».

Отказ локального уровня НЕ является отказом операции: он логируется один раз
на каждую причину (это делает rag) и лестница идёт дальше. Отказ обоих уровней
возвращает ошибку — вызывающая сторона превратит её в «эмбеддинга нет» и уйдёт
на TF-IDF.
*/
type Ladder struct {
	Local  Engine
	Remote Engine
	/*
		OnLevelFailure вызывается при отказе каждого уровня до перехода на
		следующий. Нужен, чтобы оркестратор мог пожаловаться в лог ОДИН РАЗ на
		каждую отдельную причину (поведение `_complain_once` из Python-версии):
		отказ локального движка и отказ облака — разные причины, и обе должны
		быть видны, а не затирать друг друга.
	*/
	OnLevelFailure func(level string, err error)
}

// Configured — есть ли хоть один уровень.
func (l Ladder) Configured() bool { return l.Local != nil || l.Remote != nil }

// Sources — имена настроенных уровней, для логов и `/v1/stats`.
func (l Ladder) Sources() []string {
	sources := make([]string, 0, 2)
	if l.Local != nil {
		sources = append(sources, "local")
	}
	if l.Remote != nil {
		sources = append(sources, "cloud")
	}
	return sources
}

// EmbedQuery проходит лестницу для поискового запроса.
func (l Ladder) EmbedQuery(ctx context.Context, text string) ([]float64, string, error) {
	return l.walk(ctx, text, true)
}

// EmbedDocument проходит лестницу для сохраняемого текста.
func (l Ladder) EmbedDocument(ctx context.Context, text string) ([]float64, string, error) {
	return l.walk(ctx, text, false)
}

func (l Ladder) walk(ctx context.Context, text string, asQuery bool) ([]float64, string, error) {
	if !l.Configured() {
		return nil, "", ErrNoEmbedder
	}
	var (
		lastErr error
		lastSrc string
	)
	for _, level := range []struct {
		name   string
		engine Engine
	}{{"local", l.Local}, {"cloud", l.Remote}} {
		if level.engine == nil {
			continue
		}
		var (
			vector []float64
			err    error
		)
		if asQuery {
			vector, err = level.engine.EmbedQuery(ctx, text)
		} else {
			vector, err = level.engine.EmbedDocument(ctx, text)
		}
		if err == nil {
			return vector, level.name, nil
		}
		if l.OnLevelFailure != nil {
			l.OnLevelFailure(level.name, err)
		}
		lastErr, lastSrc = err, level.name
	}
	return nil, lastSrc, lastErr
}

/*
Remote — облачный фолбэк: OpenAI-совместимый `POST {base}/embeddings`.

Форма запроса и разбор ответа совпадают с Python-провайдером
(`OpenAICompatibleProvider.embedding`): `{"model": ..., "input": <текст>}` и
`data[0].embedding`. Это позволяет подставить сюда тот же эндпоинт, что уже
настроен у Эфи, без отдельного конфига.
*/
type Remote struct {
	BaseURL string
	APIKey  string
	Model   string
	Client  *http.Client
}

// NewRemote создаёт облачный движок. `timeout` — таймаут одного запроса;
// на критическом пути он должен быть небольшим: эмбеддинг, которого ждут
// секунду, хуже, чем эмбеддинга нет (TF-IDF ответит сразу).
func NewRemote(baseURL, apiKey, model string, timeout time.Duration) *Remote {
	if timeout <= 0 {
		timeout = 5 * time.Second
	}
	return &Remote{
		BaseURL: strings.TrimRight(baseURL, "/"),
		APIKey:  apiKey,
		Model:   model,
		Client:  &http.Client{Timeout: timeout},
	}
}

// EmbedQuery — для облачного движка то же, что EmbedDocument: префиксы e5 —
// конвенция конкретной модели, а не транспорта.
func (r *Remote) EmbedQuery(ctx context.Context, text string) ([]float64, error) {
	return r.embed(ctx, text)
}

// EmbedDocument — см. EmbedQuery.
func (r *Remote) EmbedDocument(ctx context.Context, text string) ([]float64, error) {
	return r.embed(ctx, text)
}

func (r *Remote) embed(ctx context.Context, text string) ([]float64, error) {
	payload := struct {
		Model string `json:"model"`
		Input string `json:"input"`
	}{Model: r.Model, Input: text}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, &EngineError{Engine: "cloud", Err: err}
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, r.BaseURL+"/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, &EngineError{Engine: "cloud", Err: err}
	}
	request.Header.Set("Content-Type", "application/json")
	if r.APIKey != "" {
		request.Header.Set("Authorization", "Bearer "+r.APIKey)
	}
	response, err := r.Client.Do(request)
	if err != nil {
		return nil, &EngineError{Engine: "cloud", Err: err}
	}
	defer response.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(response.Body, 32<<20))
	if err != nil {
		return nil, &EngineError{Engine: "cloud", Err: err}
	}
	if response.StatusCode < 200 || response.StatusCode > 299 {
		preview := string(raw)
		if len(preview) > 500 {
			preview = preview[:500]
		}
		return nil, &EngineError{Engine: "cloud", Err: fmt.Errorf("HTTP %d: %s", response.StatusCode, preview)}
	}
	var parsed struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil || len(parsed.Data) == 0 {
		return nil, &EngineError{Engine: "cloud", Err: errors.New("unexpected /embeddings response shape")}
	}
	return parsed.Data[0].Embedding, nil
}
