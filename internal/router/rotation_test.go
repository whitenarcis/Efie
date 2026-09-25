package router

import (
	"testing"
	"time"
)

// clockStub — подставные часы: тесты ротации не должны ждать реальных минут.
type clockStub struct{ now time.Time }

func (c *clockStub) Now() time.Time { return c.now }

func (c *clockStub) advance(d time.Duration) { c.now = c.now.Add(d) }

// Проверяем главное свойство ротации: успех НЕ двигает активный ключ.
// Это ровно то, чем ротация отличается от round-robin, и из-за чего
// последний вреден для кеша контекста.
func TestMarkSuccessKeepsActiveKey(t *testing.T) {
	clock := &clockStub{now: time.Unix(1_700_000_000, 0)}
	rot := newRotation(clock.Now)

	if order := rot.order("p", 3); order[0] != 0 {
		t.Fatalf("первый ключ должен быть активным, порядок %v", order)
	}

	// Три успеха подряд — активным остаётся тот же ключ.
	for i := 0; i < 3; i++ {
		rot.markSuccess("p", 0)
		if order := rot.order("p", 3); order[0] != 0 {
			t.Fatalf("после успеха %d активным стал %v, ожидался 0", i, order)
		}
	}
}

// Отказ активного ключа переводит его в cooldown и активным становится
// следующий живой.
func TestMarkFailureSwitchesActiveKey(t *testing.T) {
	clock := &clockStub{now: time.Unix(1_700_000_000, 0)}
	rot := newRotation(clock.Now)
	rot.order("p", 2) // регистрация

	rot.markFailure("p", 0, failureRate, 0)

	order := rot.order("p", 2)
	if order[0] != 1 {
		t.Fatalf("после отказа ключа 0 активным должен стать 1, порядок %v", order)
	}
	// Охлаждённый ключ уходит в конец, но не исчезает: лучше попробовать
	// подохлаждённый, чем не пытаться вовсе.
	if order[len(order)-1] != 0 {
		t.Fatalf("охлаждённый ключ должен быть последним в порядке, %v", order)
	}
}

// Retry-After из 429 уважается буквально.
func TestMarkFailureHonoursRetryAfter(t *testing.T) {
	clock := &clockStub{now: time.Unix(1_700_000_000, 0)}
	rot := newRotation(clock.Now)
	rot.order("p", 2)

	rot.markFailure("p", 0, failureRate, 30*time.Second)

	// Сразу после отказа ключ 0 ещё в хвосте порядка.
	if order := rot.order("p", 2); order[0] != 1 {
		t.Fatalf("ожидался переход на ключ 1, порядок %v", order)
	}
	// Окно истекло — ключ снова доступен, но активным остаётся второй.
	clock.advance(31 * time.Second)
	order := rot.order("p", 2)
	if order[0] != 1 {
		t.Fatalf("после истечения окна активным остаётся 1, порядок %v", order)
	}
	if order[1] != 0 {
		t.Fatalf("истёкший ключ должен вернуться в живые, порядок %v", order)
	}
}

// Auth-ошибка держит ключ в cooldown дольше всех прочих (ключ мёртвый).
func TestAuthCooldownIsLongest(t *testing.T) {
	clock := &clockStub{now: time.Unix(1_700_000_000, 0)}
	rot := newRotation(clock.Now)
	rot.order("p", 2)

	rot.markFailure("p", 0, failureAuth, 0)
	clock.advance(5 * time.Minute) // меньше authCooldown (600с)

	order := rot.order("p", 2)
	if order[0] != 1 {
		t.Fatalf("auth-ключ должен молчать 600с, порядок %v", order)
	}
}

// classFromStatus повторяет разбор ошибок провайдера.
func TestClassFromStatus(t *testing.T) {
	cases := []struct {
		status int
		want   failureClass
	}{
		{401, failureAuth},
		{403, failureAuth},
		{429, failureRate},
		{500, failureServer},
		{503, failureServer},
		{400, failureClient},
		{404, failureClient},
	}
	for _, tc := range cases {
		if got := classFromStatus(tc.status); got != tc.want {
			t.Errorf("статус %d: класс %v, ожидался %v", tc.status, got, tc.want)
		}
	}
}
