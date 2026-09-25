package router

/*
Ядро роутера: алиас → цепочка кандидатов → перебор ключей одного провайдера,
затем следующих провайдеров, до первого успеха.

Путь запроса:

  1. бот шлёт POST /v1/chat/completions с model = алиас ("coder");
  2. алиас разворачивается в упорядоченную цепочку (провайдер, модель);
  3. для каждого кандидата роутер перебирает ключи ЕГО провайдера в порядке
     из rotation.order — липкий активный ключ первым (см. rotation.go);
  4. 401/403/429/5xx уводит ключ в cooldown, берётся следующий; когда ключи
     одного провайдера кончились — переходим к следующему провайдеру цепочки;
  5. первый 2xx выигрывает: ответ (стрим или нет) прозрачно релэится боту.

Классификация ошибок и cooldown-классы повторяют efi/llm/router.py: auth 600с,
rate Retry-After/90с, серверное 60с. Бюджет всего перебора — chainBudget
(сумма бюджетов кандидатов + буфер), без него цепочка копила бы таймауты.
*/

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
)

// maxUpstreamBody — потолок на тело НЕстримового ответа провайдера:
// ответ LLM — это текст, а не новелла; больше — значит что-то пошло не так.
const maxUpstreamBody = 32 << 20 // 32 МиБ: сюда попадают и vision-картинки в base64

// ChatRequest — тело запроса бота, разобранное роутером.
type ChatRequest struct {
	Model  string `json:"model"`
	Stream bool   `json:"stream"`
	// MaxOutputTokens — из max_tokens/max_completion_tokens; нужен для
	// расчёта бюджета запроса (см. timeouts.go).
	MaxOutputTokens int `json:"max_tokens"`
	// CompletionTokens — альтернативное имя того же поля у части провайдеров.
	CompletionTokens int `json:"max_completion_tokens"`
}

// rawMaxOutputTokens возвращает запрошенный объём вывода (оба имени поля).
func (r ChatRequest) rawMaxOutputTokens() int {
	if r.MaxOutputTokens > 0 {
		return r.MaxOutputTokens
	}
	return r.CompletionTokens
}

// Router — ядро модуля: конфиг + состояние ротации + HTTP-клиенты.
type Router struct {
	config  *Config
	rotate  *rotation
	logger  *slog.Logger
	direct  *http.Client
	proxied *http.Client // nil, если sing-box выключен

	// счётчики для /healthz.
	attempts  atomic.Int64
	successes atomic.Int64
}

// New создаёт ядро роутера. proxied может быть nil (sing-box выключен) —
// тогда провайдеры с proxy = true вызывают ошибку: молча ходить мимо прокси,
// который просили, нельзя.
func New(config *Config, proxied *http.Client, logger *slog.Logger) (*Router, error) {
	if logger == nil {
		logger = slog.Default()
	}
	if proxied == nil {
		for name, provider := range config.Providers {
			if provider.Proxy {
				return nil, fmt.Errorf(
					"провайдер %q настроен proxy = true, но sing-box не запущен (proxy.toml: [proxy].enabled = false или файл отсутствует)",
					name)
			}
		}
	}
	return &Router{
		config:  config,
		rotate:  newRotation(nil),
		logger:  logger,
		direct:  &http.Client{}, // таймауты — через context, не через Client.Timeout
		proxied: proxied,
	}, nil
}

// clientFor возвращает HTTP-клиент провайдера: через sing-box или напрямую.
func (r *Router) clientFor(provider ProviderConfig) (*http.Client, error) {
	if !provider.Proxy {
		return r.direct, nil
	}
	if r.proxied == nil {
		return nil, errors.New("провайдер требует proxy, но sing-box выключен")
	}
	return r.proxied, nil
}

// chatCompletionsURL склеивает url провайдера с путём эндпоинта.
func chatCompletionsURL(base string) string {
	return strings.TrimRight(base, "/") + "/chat/completions"
}

// ServeHTTP — OpenAI-совместимый шлюз: /v1/chat/completions, /v1/models,
// /healthz.
func (r *Router) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	switch req.URL.Path {
	case "/v1/chat/completions":
		if req.Method != http.MethodPost {
			writeJSON(w, http.StatusMethodNotAllowed, map[string]any{
				"error": map[string]any{"message": "нужен POST"},
			})
			return
		}
		r.handleChat(w, req)
	case "/v1/models":
		r.handleModels(w, req)
	case "/healthz":
		r.handleHealth(w, req)
	default:
		http.NotFound(w, req)
	}
}

// handleModels отдаёт список алиасов — бот должен видеть «coder» как модель.
func (r *Router) handleModels(w http.ResponseWriter, _ *http.Request) {
	names := r.config.AliasNames()
	models := make([]map[string]any, 0, len(names))
	for _, name := range names {
		models = append(models, map[string]any{
			"id":       name,
			"object":   "model",
			"owned_by": "efie-router",
		})
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"object": "list",
		"data":   models,
	})
}

// handleHealth — состояние роутера: счётчики и активные ключи по провайдерам.
func (r *Router) handleHealth(w http.ResponseWriter, _ *http.Request) {
	providers := make(map[string]any, len(r.config.Providers))
	for name, provider := range r.config.Providers {
		entry := map[string]any{
			"keys":    len(provider.APIKeys),
			"proxy":   provider.Proxy,
			"aliases": r.aliasesUsing(name),
		}
		if index, ok := r.rotate.activeKey(name); ok {
			entry["active_key"] = index
		} else {
			entry["active_key"] = nil
		}
		providers[name] = entry
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"status":    "ok",
		"attempts":  r.attempts.Load(),
		"successes": r.successes.Load(),
		"providers": providers,
	})
}

// aliasesUsing — имена алиасов, в цепочках которых встречается провайдер.
func (r *Router) aliasesUsing(provider string) []string {
	var names []string
	for name, alias := range r.config.Aliases {
		for _, entry := range alias.Chain {
			if entry.Provider == provider {
				names = append(names, name)
				break
			}
		}
	}
	sort.Strings(names)
	return names
}

// attemptFailure — что не сработало на одной попытке.
type attemptFailure struct {
	provider string
	keyIndex int
	status   int
	message  string
}

// failuresSummary — короткий перечень попыток для сообщения об ошибке.
func failuresSummary(failures []attemptFailure) string {
	parts := make([]string, 0, len(failures))
	for _, failure := range failures {
		label := fmt.Sprintf("%s[%d]", failure.provider, failure.keyIndex)
		if failure.status > 0 {
			parts = append(parts, fmt.Sprintf("%s:%d", label, failure.status))
		} else {
			parts = append(parts, label)
		}
	}
	return strings.Join(parts, ", ")
}

// handleChat — главный перебор: алиас → кандидаты → ключи → успех.
func (r *Router) handleChat(w http.ResponseWriter, req *http.Request) {
	raw, err := io.ReadAll(http.MaxBytesReader(w, req.Body, maxUpstreamBody))
	if err != nil {
		r.writeError(w, http.StatusBadRequest, "чтение тела запроса: %v", err)
		return
	}
	var chatReq ChatRequest
	if err := json.Unmarshal(raw, &chatReq); err != nil {
		r.writeError(w, http.StatusBadRequest, "разбор тела запроса: %v", err)
		return
	}
	if chatReq.Model == "" {
		r.writeError(w, http.StatusBadRequest, "не указана model (ожидается имя алиаса)")
		return
	}
	alias, ok := r.config.Alias(chatReq.Model)
	if !ok {
		// Неизвестный алиас — ошибка вызова, а не провайдера: молча
		// подставлять что-то другое нельзя, бот должен увидеть правду.
		r.writeError(w, http.StatusNotFound, "неизвестный алиас %q (доступны: %s)",
			chatReq.Model, strings.Join(r.config.AliasNames(), ", "))
		return
	}

	// Бюджет на весь перебор: сумма бюджетов кандидатов плюс буфер.
	var bases []time.Duration
	for _, entry := range alias.Chain {
		if provider, ok := r.config.Provider(entry.Provider); ok {
			bases = append(bases, provider.Timeout())
		}
	}
	budget := chainBudget(bases, chatReq.rawMaxOutputTokens(), alias.IsInteractive())

	ctx, cancel := context.WithTimeout(req.Context(), budget)
	defer cancel()

	failures := r.runChain(ctx, w, raw, alias, chatReq.Stream, chatReq.rawMaxOutputTokens())
	if failures == nil {
		return // успех: ответ уже отправлен
	}
	// Ничего не сработало — отдаём боту ошибку в OpenAI-формате.
	if len(failures) == 0 {
		// Цепочка пуста на практике быть не может (валидация требует не
		// пустой chain), но индекс последней попытки при пустом срезе
		// уронил бы обработчик. Лучше честная ошибка, чем паника.
		r.writeError(w, http.StatusBadGateway, "алиас %q не развернулся ни в одну попытку", chatReq.Model)
		return
	}
	last := failures[len(failures)-1]
	status := http.StatusBadGateway
	if last.status > 0 && last.status < 500 {
		status = last.status
	}
	message := last.message
	if len(failures) > 1 {
		message = fmt.Sprintf("%s (попыток: %d — %s)", message, len(failures), failuresSummary(failures))
	}
	r.writeError(w, status, "%s", message)
}

// runChain перебирает цепочку алиаса: у каждого кандидата сначала все ключи
// его провайдера, затем следующий кандидат. Возвращает nil, если ответ
// отправлен боту; иначе — список неудачных попыток.
func (r *Router) runChain(
	ctx context.Context,
	w http.ResponseWriter,
	body []byte,
	alias AliasConfig,
	stream bool,
	// maxOutputTokens — запрошенный объём вывода: бюджет одной попытки
	// считается от него (см. requestTimeout), а не от голой базы провайдера.
	maxOutputTokens int,
) []attemptFailure {
	var failures []attemptFailure

	for _, entry := range alias.Chain {
		provider, ok := r.config.Provider(entry.Provider)
		if !ok {
			// Провайдер исчез из конфига между стартом и запросом — не повод
			// отдавать ошибку, просто пропускаем шаг цепочки.
			continue
		}
		client, err := r.clientFor(provider)
		if err != nil {
			failures = append(failures, attemptFailure{
				provider: entry.Provider,
				message:  err.Error(),
			})
			continue
		}

		// Ключи провайдера в порядке ротации: липкий активный первым.
		for _, keyIndex := range r.rotate.order(entry.Provider, len(provider.APIKeys)) {
			if err := ctx.Err(); err != nil {
				failures = append(failures, attemptFailure{
					provider: entry.Provider,
					keyIndex: keyIndex,
					message:  "бюджет перебора исчерпан: " + err.Error(),
				})
				return failures
			}

			r.attempts.Add(1)
			result, failure := r.attempt(ctx, client, provider, entry, keyIndex,
				body, stream, requestTimeout(provider.Timeout(), maxOutputTokens, alias.IsInteractive()))
			if failure != nil {
				failures = append(failures, *failure)
				continue
			}

			// Успех: ключ закреплён, ответ отдаём боту. Ошибку релэя после
			// отправки заголовков исправить уже нечем — логируем и молчим.
			r.rotate.markSuccess(entry.Provider, keyIndex)
			r.successes.Add(1)
			r.logger.Info("router: ok",
				"provider", entry.Provider,
				"model", entry.Model,
				"key_index", keyIndex,
				"stream", stream)
			if err := r.relay(w, result, stream); err != nil {
				r.logger.Error("router: relay", "provider", entry.Provider, "err", err)
			}
			return nil
		}
		// Ключи провайдера исчерпаны — следующий кандидат цепочки.
	}

	// Ни один кандидат не ответил: отдаём список попыток вызывающему.
	return failures
}

// upstreamResult — принятый ответ провайдера.
type upstreamResult struct {
	body        []byte        // нестримовый ответ целиком
	stream      io.ReadCloser // стрим после первого чанка
	firstChunk  []byte        // первый чанк стрима, прочитанный для решения о фейловере
	contentType string
}

// attempt — одна попытка: подставляет ключ, шлёт запрос, читает ответ ровно
// настолько, чтобы понять, успех это или отказ.
//
// Почему так: при стриме фейловер возможен, пока боту не отправлено ни
// байта ответа. Поэтому первый чанк (или, для нестрима, всё тело) читается
// здесь — и только потом решается, релеить или пробовать следующий ключ.
// Уже начатый стрим докинуть нельзя: как только бот увидел первые токены,
// переключение даст ему два разных ответа вперемешку.
func (r *Router) attempt(
	ctx context.Context,
	client *http.Client,
	provider ProviderConfig,
	entry ChainEntry,
	keyIndex int,
	body []byte,
	stream bool,
	// budget — потолок одной попытки (requestTimeout). Сверху он ещё
	// ограничен контекстом всего перебора, так что минуты на цепочку не
	// превращаются в один долгий запрос.
	budget time.Duration,
) (*upstreamResult, *attemptFailure) {
	// Тело уходит провайдеру с заменой model на реальное имя и подставленной
	// подписью. Остальные поля бота не трогаем: temperature, tools, response_format
	// должны дойти как есть.
	payload, err := rewriteModel(body, entry.Model)
	if err != nil {
		return nil, &attemptFailure{
			provider: entry.Provider,
			keyIndex: keyIndex,
			message:  fmt.Sprintf("подстановка модели: %v", err),
		}
	}

	attemptCtx, cancel := context.WithTimeout(ctx, budget)
	defer cancel()

	req, err := http.NewRequestWithContext(attemptCtx, http.MethodPost,
		chatCompletionsURL(provider.URL), bytes.NewReader(payload))
	if err != nil {
		return nil, &attemptFailure{
			provider: entry.Provider,
			keyIndex: keyIndex,
			message:  err.Error(),
		}
	}
	req.Header.Set("Content-Type", "application/json")
	// Авторизация: у части провайдеров (Ollama Cloud) ключ пустой — пустую
	// подпись не шлём, иначе они получают 401 на валидном локальном запросе.
	if key := provider.APIKeys[keyIndex]; key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}

	resp, err := client.Do(req)
	if err != nil {

		// Транспорт/таймаут: ключ тут ни при чём, но перебор всё равно имеет
		// смысл — следующий провайдер может быть живым.
		r.logger.Warn("router: попытка не удалась",
			"provider", entry.Provider, "key_index", keyIndex, "err", err)
		return nil, &attemptFailure{
			provider: entry.Provider,
			keyIndex: keyIndex,
			message:  fmt.Sprintf("%s: транспорт: %v", entry.Provider, err),
		}
	}
	// Тело закрываем сами, но только пока оно наше. Как только стрим
	// передан в relayStream, закрывать его здесь нельзя: попытка уже
	// вернулась, и дочитывать остаток будет релэй.
	handedOff := false
	defer func() {
		if !handedOff && resp.Body != nil {
			_ = resp.Body.Close()
		}
	}()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		detail, _ := io.ReadAll(io.LimitReader(resp.Body, 8<<10))
		message := fmt.Sprintf("%s: HTTP %d: %s",
			entry.Provider, resp.StatusCode, truncate(detail, 400))
		class := classFromStatus(resp.StatusCode)
		// Ключ уводим в cooldown только если дело в ключе/лимите. 4xx-прочее
		// (плохой запрос, неизвестная модель) ключа не касается: виноват
		// запрос, и следующий ключ того же провайдера это не починит —
		// зато следующий провайдер с другой моделью починить может.
		if class == failureAuth || class == failureRate || class == failureServer {
			r.rotate.markFailure(entry.Provider, keyIndex, class,
				parseRetryAfter(resp.Header.Get("Retry-After")))
		}
		r.logger.Warn("router: отказ провайдера",
			"provider", entry.Provider,
			"key_index", keyIndex,
			"status", resp.StatusCode,
			"class", class,
			"detail", truncate(detail, 200))
		return nil, &attemptFailure{
			provider: entry.Provider,
			keyIndex: keyIndex,
			status:   resp.StatusCode,
			message:  message,
		}
	}

	result := &upstreamResult{contentType: resp.Header.Get("Content-Type")}
	if !stream {
		data, err := io.ReadAll(io.LimitReader(resp.Body, maxUpstreamBody))
		if err != nil {
			return nil, &attemptFailure{
				provider: entry.Provider,
				keyIndex: keyIndex,
				message:  fmt.Sprintf("%s: чтение ответа: %v", entry.Provider, err),
			}
		}
		result.body = data
		return result, nil
	}

	// Стрим: читаем первый чанк, чтобы понять, что провайдер реально пошёл.
	// Пустой поток — это валидный «ничего не ответил», а не ошибка.
	first := make([]byte, 32<<10)
	n, err := resp.Body.Read(first)
	if n > 0 {
		result.firstChunk = append([]byte(nil), first[:n]...)
	}
	if err != nil && !errors.Is(err, io.EOF) {
		return nil, &attemptFailure{
			provider: entry.Provider,
			keyIndex: keyIndex,
			message:  fmt.Sprintf("%s: чтение стрима: %v", entry.Provider, err),
		}
	}
	if n == 0 && errors.Is(err, io.EOF) {
		// Стрим закрылся сразу: отдаём пустой, ключ считаем отработавшим.
		result.stream = nil
		return result, nil
	}
	result.stream = resp.Body
	handedOff = true // дальше тело читает и закрывает relayStream
	return result, nil
}

// rewriteModel подменяет model в теле запроса, сохраняя остальное тело как
// есть (важно для нестандартных полей провайдеров).
func rewriteModel(body []byte, model string) ([]byte, error) {
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, err
	}
	payload["model"] = model
	return json.Marshal(payload)
}

// relay отдаёт успешный ответ боту: нестрим — целиком, стрим — с промежуточными
// flush, чтобы SSE доходил построчно, как того ждёт openai_compatible.py.
func (r *Router) relay(w http.ResponseWriter, result *upstreamResult, stream bool) error {
	if stream {
		return relayStream(w, result)
	}
	contentType := result.contentType
	if contentType == "" {
		contentType = "application/json"
	}
	w.Header().Set("Content-Type", contentType)
	w.WriteHeader(http.StatusOK)
	_, err := w.Write(result.body)
	return err
}

// relayStream прокидывает SSE-поток провайдера боту построчно.
//
// Первый чанк уже прочитан в attempt (там же решается, возможен ли ещё
// фейловер) — он пишется первым, чтобы данные не ждали в буфере.
func relayStream(w http.ResponseWriter, result *upstreamResult) error {
	if result.stream != nil {
		defer result.stream.Close()
	}

	contentType := result.contentType
	if contentType == "" {
		contentType = "text/event-stream"
	}
	header := w.Header()
	header.Set("Content-Type", contentType)
	header.Set("Cache-Control", "no-cache")
	header.Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	flusher, _ := w.(http.Flusher)
	write := func(chunk []byte) error {
		if len(chunk) == 0 {
			return nil
		}
		if _, err := w.Write(chunk); err != nil {
			return err
		}
		if flusher != nil {
			flusher.Flush()
		}
		return nil
	}
	if err := write(result.firstChunk); err != nil {
		return err
	}
	if result.stream == nil {
		return nil // пустой стрим: заголовки ушли, тела нет
	}

	buf := make([]byte, 32<<10)
	for {
		n, err := result.stream.Read(buf)
		if n > 0 {
			if writeErr := write(buf[:n]); writeErr != nil {
				return writeErr
			}
		}
		if err != nil {
			if errors.Is(err, io.EOF) {
				return nil
			}
			return err
		}
	}
}

// writeError отдаёт ошибку в формате OpenAI, как ждёт клиент бота.
func (r *Router) writeError(w http.ResponseWriter, status int, format string, args ...any) {
	message := fmt.Sprintf(format, args...)
	r.logger.Error("router: ошибка", "status", status, "message", message)
	writeJSON(w, status, map[string]any{
		"error": map[string]any{
			"message": message,
			"type":    "efie_router_error",
			"code":    status,
		},
	})
}

// writeJSON — единая точка записи JSON-ответов.
func writeJSON(w http.ResponseWriter, status int, payload any) {
	body, err := json.Marshal(payload)
	if err != nil {
		body = []byte(`{"error":{"message":"не удалось собрать ответ"}}`)
		status = http.StatusInternalServerError
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write(body)
}

// parseRetryAfter понимает и «120» (секунды), и HTTP-дату — как и
// _parse_retry_after в openai_compatible.py.
func parseRetryAfter(value string) time.Duration {
	value = strings.TrimSpace(value)
	if value == "" {
		return 0
	}
	if seconds, err := strconv.Atoi(value); err == nil && seconds > 0 {
		return time.Duration(seconds) * time.Second
	}
	if at, err := http.ParseTime(value); err == nil {
		if delta := time.Until(at); delta > 0 {
			return delta
		}
	}
	return 0
}

// truncate обрезает текст для сообщений об ошибках — превью, а не роман.
func truncate(data []byte, limit int) string {
	text := strings.TrimSpace(string(data))
	if len(text) <= limit {
		return text
	}
	return text[:limit] + "…"
}
