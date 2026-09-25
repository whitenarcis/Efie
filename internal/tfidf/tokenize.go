/*
Package tfidf — Go-порт efi/memory/tfidf_fallback.py: локальный TF-IDF поиск
по памяти без сети и без эмбеддингов.

Зачем это переносится в Go. В Python-версии это «дешёвый» путь критического
маршрута — и он действительно дешёвый по смыслу (никаких моделей и сети), но
дорогой по исполнению: `search()` копирует ВЕСЬ индекс (`dict(self._documents)`
плюс `Counter(...)`) под локом, затем токенизирует весь корпус заново в
`asyncio.to_thread`. То есть на каждое короткое «привет» Python пересобирает
и копирует индекс целиком — ровно то, что Go делает без копий и потоков.

Токенизация воспроизводит `[^\W\d_]+` из Python дословно, включая неочевидное:
`²`, `½`, `Ⅷ` — «буквы» для этого шаблона (они проходят `\w` и не являются
десятичными цифрами), а `٣` (арабиндская цифра) и `_` — нет.
*/
package tfidf

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

// ShortTextLengthThreshold — тексты короче или равные этому порогу (в рунах,
// как Python `len()`) обслуживаются TF-IDF без семантического поиска.
const ShortTextLengthThreshold = 12

// maxPhaticTokenCount — сколько фатических слов ещё считается «болтовнёй».
const maxPhaticTokenCount = 3

// phaticWords — частотные фатические реплики (RU): сообщения из них одних не
// заслуживают семантического поиска. Список такой же, как в Python-версии.
var phaticWords = map[string]struct{}{
	"привет": {}, "прив": {}, "хай": {}, "ку": {}, "здравствуй": {}, "здравствуйте": {},
	"йо": {}, "ало": {}, "ок": {}, "окей": {}, "ладно": {}, "понял": {}, "поняла": {},
	"ясно": {}, "спасибо": {}, "благодарю": {}, "спс": {}, "пока": {}, "бай": {},
	"чмок": {}, "ага": {}, "угу": {}, "да": {}, "нет": {}, "неа": {}, "лол": {},
	"хах": {}, "ахах": {}, "кек": {},
}

/*
Tokenize — слова любого алфавита в нижнем регистре, без цифр и знаков.

Правило символа-части-слова повторяет Python `[^\W\d_]`:
`\w` без десятичных цифр и подчёркивания. Практическая разница с
«интуитивным» `unicode.IsLetter` — числительные символы вроде `²` и `½`:
Python считает их частями слова, и Go-версия обязана делать то же, иначе
индекс и выдача разойдутся с Python-версией на одном и том же корпусе.
*/
func Tokenize(text string) []string {
	tokens := make([]string, 0, 16)
	var current []rune
	flush := func() {
		if len(current) == 0 {
			return
		}
		tokens = append(tokens, strings.ToLower(string(current)))
		current = current[:0]
	}
	for _, symbol := range text {
		if isWordRune(symbol) {
			current = append(current, symbol)
			continue
		}
		flush()
	}
	flush()
	return tokens
}

// isWordRune — символ входит в токен: буква любого алфавита или числительный
// символ (¼, ½, Ⅷ), но не десятичная цифра и не подчёркивание.
func isWordRune(symbol rune) bool {
	if unicode.IsLetter(symbol) {
		return true
	}
	return unicode.IsNumber(symbol) && !unicode.IsDigit(symbol)
}

/*
ShouldUseShortcut — эвристика короткого замыкания критического пути: True,
если текст либо короче порога, либо состоит целиком из нескольких фатических
слов. Для такого сообщения семантический поиск избыточен.

Ложноотрицательный результат здесь не страшен (уйдём в полный RAG), поэтому
эвристика сознательно грубая — как и в Python-версии.
*/
func ShouldUseShortcut(text string) bool {
	stripped := strings.TrimSpace(text)
	if utf8.RuneCountInString(stripped) <= ShortTextLengthThreshold {
		return true
	}
	tokens := Tokenize(stripped)
	if len(tokens) == 0 {
		return true
	}
	if len(tokens) > maxPhaticTokenCount {
		return false
	}
	for _, token := range tokens {
		if _, ok := phaticWords[token]; !ok {
			return false
		}
	}
	return true
}
