package router

/*
Бюджет одного запроса — порт efi/llm/timeouts.py (см. его докстринг там).

Правило то же: один таймаут на эндпоинт — ошибка, которая долго выглядит
как настройка. Бюджет складывается из базы провайдера (нижняя граница,
заданная владельцем) плюс время на запрошенный объём вывода, и ограничен
потолком своей роли: 90с живому диалогу (иначе собеседник уже ушёл), 300с
фону (иначе зависшая фоновая задача живёт вечно).
*/

import "time"

const (
	// secondsPer1KToken — сколько секунд закладывать на тысячу токенов
	// вывода. Ориентир — медленная самохостируемая модель (десятки токенов
	// в секунду); на быстром облаке запас просто не понадобится.
	secondsPer1KToken = 40.0
	// interactiveCeilingSeconds — потолок для диалога.
	interactiveCeilingSeconds = 90.0
	// backgroundCeilingSeconds — потолок для фоновых задач.
	backgroundCeilingSeconds = 300.0
)

// requestTimeout считает бюджет одной попытки: база провайдера плюс время
// на maxOutputTokens, ограниченный потолком роли. Никогда не меньше базы —
// настройка владельца это нижняя граница, а не единственное значение.
func requestTimeout(base time.Duration, maxOutputTokens int, interactive bool) time.Duration {
	ceiling := backgroundCeilingSeconds
	if interactive {
		ceiling = interactiveCeilingSeconds
	}
	baseSeconds := base.Seconds()
	if baseSeconds <= 0 {
		baseSeconds = DefaultProviderTimeout
	}
	generation := float64(max(0, maxOutputTokens)) / 1000.0 * secondsPer1KToken
	budget := baseSeconds + generation
	ceilingFloor := ceiling
	if baseSeconds > ceilingFloor {
		// база важнее потолка: владелец явно попросил ждать дольше.
		ceilingFloor = baseSeconds
	}
	if budget > ceilingFloor {
		budget = ceilingFloor
	}
	return time.Duration(budget * float64(time.Second))
}

// chainBudget — общий потолок на весь перебор кандидатов одной цепочки:
// сумма бюджетов всех кандидатов плюс небольшой буфер. Без него цепочка из
// трёх кандидатов копила бы таймауты и превращала один ответ в минуты
// ожидания (та же история, что описана в efi/llm/router.py).
func chainBudget(bases []time.Duration, maxOutputTokens int, interactive bool) time.Duration {
	var total time.Duration
	for _, base := range bases {
		total += requestTimeout(base, maxOutputTokens, interactive)
	}
	return total + 5*time.Second
}
