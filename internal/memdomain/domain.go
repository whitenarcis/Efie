/*
Package memdomain — Go-порт доменной части efi/memory/router.py: три
непересекающихся вида знания и фильтр по ним.

Перенесён сюда именно ФИЛЬТР, а не классификатор. Классификатор
(`domains_for_message`, «какие домены уместны под этот текст») остаётся в
Python: он состоит из словарей на русском и правил, которые читаются и правятся
вместе с поведением Эфи, и переносить его раньше остальной памяти смысла нет.
А вот фильтр обязан совпадать побайтово — иначе Go-версия поиска начнёт
возвращать записи чужих доменов, и поведение памяти станет зависеть от того,
через какую реализацию идёт запрос.
*/
package memdomain

import (
	"strings"

	"github.com/whitenarcis/Efie/internal/diary"
)

// Domain — домен памяти. Значение — то, что физически лежит в поле `domain`.
type Domain string

const (
	// Common — интерсубъективное знание о мире: технологии, концепции, факты.
	Common Domain = "C"
	// Personal — модель владельца и конкретных людей.
	Personal Domain = "P"
	// History — личный эпизодический и эмоциональный опыт самой Эфи.
	History Domain = "H"
)

// aliases — то же, что `_DOMAIN_ALIASES` в Python-версии: восприятие модели
// пишет домен то буквой, то словом, то по-русски, и требовать одного формата
// под угрозой отбраковки — способ терять валидные записи на форматной мелочи.
var aliases = map[string]Domain{
	"c": Common, "common": Common, "world": Common, "общее": Common,
	"p": Personal, "personal": Personal, "person": Personal, "личное": Personal,
	"h": History, "history": History, "episodic": History, "опыт": History,
}

// Parse разбирает домен из внешнего представления. Пустая строка и неизвестное
// значение дают `fallback` — если он не задан (пустой), возвращается History:
// так же, как `MemoryDomain.parse(raw, default=H)` в Python-версии.
func Parse(raw string, fallback Domain) Domain {
	text := strings.ToLower(strings.TrimSpace(raw))
	if text == "" {
		if fallback != "" {
			return fallback
		}
		return History
	}
	if parsed, known := aliases[text]; known {
		return parsed
	}
	if fallback != "" {
		return fallback
	}
	return History
}

// Of — домен записи дневника. Записи, созданные до разделения на домены,
// читаются как H (личный опыт): именно ими дневник был целиком, и молча
// выкинуть их из выдачи значило бы стереть всю память об общении.
func Of(entry *diary.Entry) Domain {
	return Parse(entry.Metadata.Domain, History)
}

// Filter строит предикат для diary.Store.Query / tfidf.Index.Search.
//
// Пустой набор доменов означает «без фильтрации» — как `DiaryEntryFilter`,
// у которого `if not self.allowed: return True`. Именно так Python-версия
// понимает `diary_filter([])`, и повторять это в Go нужно буквально.
func Filter(allowed []Domain) diary.Filter {
	if len(allowed) == 0 {
		return nil
	}
	set := make(map[Domain]struct{}, len(allowed))
	for _, domain := range allowed {
		set[Parse(string(domain), History)] = struct{}{}
	}
	return func(entry *diary.Entry) bool {
		_, ok := set[Of(entry)]
		return ok
	}
}
