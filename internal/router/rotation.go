package router

/*
Ротация API-ключей с липким выбором: один ключ выжимается до отказа.

Идея (важно): round-robin тут вреден. Если чередовать ключи на каждом
запросе, то у многих провайдеров — особенно у облачных агрегаторов с
кешированием контекста сессии на стороне ключа/аккаунта — страдает именно
контекст: запросы одного диалога разъезжаются по разным ключам, и провайдер
видит обрывы сессий. Поэтому поведение такое:

  - запрос начинается с активного ключа провайдера и использует ЕГО, пока
    тот не получит отказ;
  - отказ (401/403/429/5xx/таймаут) отправляет ключ в cooldown, и только
    тогда активным становится следующий — ближайший живой;
  - успешный ответ никогда не меняет активный ключ;
  - cooldown-семантика совпадает с efi/llm/router.py: auth 600с,
    rate-limit Retry-After или 90с, прочее 60с.

Порядок попыток внутри одного запроса: сначала активный, затем остальные
ключи в порядке конфига, оставшиеся в cooldown — в конец (они хуже, но лучше
них — ничего). Это даёт и «выжимать до отказа», и живой фолбэк, когда
активный в cooldown.
*/

import (
	"sync"
	"time"
)

// Классы отказа ключа — таймауты cooldown по ним.
const (
	authCooldownSeconds   = 600.0 // 401/403: мёртвый ключ сам себя не починит
	rateCooldownSeconds   = 90.0  // 429 без Retry-After
	serverCooldownSeconds = 60.0  // 5xx, транспорт, таймаут
)

// keyState — состояние одного ключа.
type keyState struct {
	// active — выжимаем ли мы этот ключ сейчас (липкий выбор).
	active bool
	// cooldownUntil — до какого момента ключ нельзя трогать.
	cooldownUntil time.Time
}

// providerState — состояние всех ключей одного провайдера.
type providerState struct {
	keys []keyState
}

// rotation — потокобезопасное состояние ротации всех провайдеров.
type rotation struct {
	mu        sync.Mutex
	providers map[string]*providerState
	now       func() time.Time
}

// newRotation создаёт пустое состояние. clock — для тестов (подмена времени).
func newRotation(clock func() time.Time) *rotation {
	if clock == nil {
		clock = time.Now
	}
	return &rotation{
		providers: make(map[string]*providerState),
		now:       clock,
	}
}

// ensure регистрирует провайдера, если ещё не встречался. Возвращает
// состояние (под мьютексом не хранится — вызывать под lock).
func (r *rotation) ensure(provider string, keyCount int) *providerState {
	state, ok := r.providers[provider]
	if ok {
		return state
	}
	state = &providerState{keys: make([]keyState, keyCount)}
	if keyCount > 0 {
		// Липкость: первый ключ активен с самого начала.
		state.keys[0].active = true
	}
	r.providers[provider] = state
	return state
}

// order возвращает индексы ключей в порядке попыток для одного запроса:
// активный ключ первым, затем остальные живые в порядке конфига, затем
// охлаждённые (лучше попробовать подохлаждённый, чем не пытаться вовсе —
// та же философия, что в efi/llm/router.py: available + cooling_down).
//
// Активный в cooldown получает приоритет среди охлаждённых: как только его
// окно истечёт, он снова будет первым.
func (r *rotation) order(provider string, keyCount int) []int {
	r.mu.Lock()
	defer r.mu.Unlock()

	state := r.ensure(provider, keyCount)
	now := r.now()

	active := -1
	var live, cooling []int
	for i := range state.keys {
		if state.keys[i].cooldownUntil.After(now) {
			if state.keys[i].active {
				// активный, но остывает — идёт первым среди cooling
				cooling = append([]int{i}, cooling...)
			} else {
				cooling = append(cooling, i)
			}
			continue
		}
		if state.keys[i].active {
			active = i
			continue
		}
		live = append(live, i)
	}

	order := make([]int, 0, len(state.keys))
	if active >= 0 {
		order = append(order, active)
	}
	order = append(order, live...)
	order = append(order, cooling...)
	return order
}

// markSuccess фиксирует успех ключа и оставляет его активным.
//
// Повторный вызов для уже активного ключа — no-op: успех ничего не меняет,
// именно это и отличает липкий выбор от round-robin.
func (r *rotation) markSuccess(provider string, keyIndex int) {
	r.mu.Lock()
	defer r.mu.Unlock()

	state, ok := r.providers[provider]
	if !ok || keyIndex < 0 || keyIndex >= len(state.keys) {
		return
	}
	// Ключ, переживший отказ и выдержавший успешный запрос, снова становится
	// активным: окно прошло, а копить его «на будущее» незачем.
	if !state.keys[keyIndex].active {
		for i := range state.keys {
			state.keys[i].active = false
		}
		state.keys[keyIndex].active = true
	}
}

// markFailure отправляет ключ в cooldown по классу отказа и переключает
// активного на следующего живого. retryAfter — из заголовка Retry-After
// (только для rate-класса, 0 = не был указан).
//
// ВАЖНО: ключ, получивший отказ, остаётся неактивным до ближайшего успеха
// другого — роутер не возвращается к нему «на всякий случай», а выжимает
// следующий (см. докстринг пакета).
func (r *rotation) markFailure(provider string, keyIndex int, class failureClass, retryAfter time.Duration) {
	r.mu.Lock()
	defer r.mu.Unlock()

	state, ok := r.providers[provider]
	if !ok || keyIndex < 0 || keyIndex >= len(state.keys) {
		return
	}

	var cooldown time.Duration
	switch class {
	case failureAuth:
		cooldown = time.Duration(authCooldownSeconds * float64(time.Second))
	case failureRate:
		if retryAfter > 0 {
			cooldown = retryAfter
		} else {
			cooldown = time.Duration(rateCooldownSeconds * float64(time.Second))
		}
	default:
		cooldown = time.Duration(serverCooldownSeconds * float64(time.Second))
	}

	now := r.now()
	state.keys[keyIndex].cooldownUntil = now.Add(cooldown)
	state.keys[keyIndex].active = false

	// Переключаем активного на ближайший живой после отказавшего.
	// Если живых нет — пусть активным будет следующий по порядку: его окно
	// истечёт раньше, и он подхватит первым.
	n := len(state.keys)
	if n == 0 {
		return
	}
	for offset := 1; offset <= n; offset++ {
		candidate := (keyIndex + offset) % n
		if !state.keys[candidate].cooldownUntil.After(now) {
			for i := range state.keys {
				state.keys[i].active = false
			}
			state.keys[candidate].active = true
			return
		}
	}
	next := (keyIndex + 1) % n
	for i := range state.keys {
		state.keys[i].active = false
	}
	state.keys[next].active = true
}

// activeKey отдаёт (индекс, true) активного ключа, если он сейчас жив —
// для /healthz и логов.
func (r *rotation) activeKey(provider string) (int, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()

	state, ok := r.providers[provider]
	if !ok {
		return -1, false
	}
	now := r.now()
	for i := range state.keys {
		if state.keys[i].active && !state.keys[i].cooldownUntil.After(now) {
			return i, true
		}
	}
	return -1, false
}

// failureClass — класс ошибки провайдера, определяющий cooldown.
type failureClass int

const (
	// failureAuth — 401/403: ключ неверен/отозван.
	failureAuth failureClass = iota
	// failureRate — 429: лимит запросов.
	failureRate
	// failureServer — 5xx/таймаут/транспорт: временно у провайдера.
	failureServer
	// failureClient — 4xx прочее: запрос не удался по нашей вине; короткий
	// cooldown, чтобы не долбить заведомо плохой кандидат.
	failureClient
)

// classFromStatus маппит HTTP-статус на класс отказа.
func classFromStatus(status int) failureClass {
	switch {
	case status == 401 || status == 403:
		return failureAuth
	case status == 429:
		return failureRate
	case status >= 500:
		return failureServer
	default:
		return failureClient
	}
}
