/*
Package diary — Go-порт efi/memory/diary.py: дневник долговременной памяти Эфи.

Формат хранения ровно тот же, что у Python-версии, и это не косметика: пока
идёт постепенный перенос, обе реализации читают и пишут ОДИН И ТОТ ЖЕ каталог,
и запись, сделанная Go-подсистемой, обязана быть понятна Python-версии (и
наоборот). Отсюда — воспроизведение деталей, которые легко «улучшить» и тем
самым разойтись с Python:

  - файл записи — `<id>.md` с JSON front-matter БЕЗ эмбеддинга и свободным
    текстом после второго `---` (вектор раздувал файл до ~25 КБ при теле в
    ~1 КБ и делал `git diff` нечитаемым);
  - вектор — в sidecar `<id>.vec.json` вида `{"embedding": [...]}`;
  - datetime сериализуются как RFC3339: `Z` для UTC, микросекунды шестью
    цифрами и только когда они не нулевые (так это делает pydantic);
  - вещественные числа — в repr-подобном виде (`0.0`, а не `0`), потому что
    именно так их пишет `model_dump_json`;
  - если front-matter не найден, телом считается ФАЙЛ ЦЕЛИКОМ, вместе с
    ведущими пробелами (Python присваивает `body = raw` до разбора).

Запись на диск — атомарная (запись дневника это прожитый вечер, см.
efi/utils/atomic.py): временный файл рядом с целевым, fsync, `rename`.
*/
package diary

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
	"time"
	"unicode"
)

// FrontMatterDelimiter — разделитель front-matter, как в Python-версии.
const FrontMatterDelimiter = "---"

// Домены памяти (efi/memory/router.py). Значение — то, что физически лежит
// в поле `domain` front-matter.
const (
	DomainCommon   = "C" // знание о мире
	DomainPersonal = "P" // модель человека
	DomainHistory  = "H" // личный опыт (дефолт: до разделения доменами дневник был целиком H)
)

// ErrNotFound — записи по указанному id нет.
var ErrNotFound = errors.New("diary: entry not found")

/*
Metadata — Go-аналог DiaryEntryMetadata (efi/llm/schemas.py).

Порядок полей в JSON — тот же, что порядок объявления полей pydantic-модели:
`score, confidence, last_used, usage_count, embedding, domain, created_at`.
На него опирается побайтовое совпадение с Python-сериализацией, поэтому
переставлять поля в этой структуре нельзя.
*/
type Metadata struct {
	Score      float64    `json:"score"`
	Confidence float64    `json:"confidence"`
	LastUsed   *time.Time `json:"last_used"`
	UsageCount int        `json:"usage_count"`
	Embedding  []float64  `json:"embedding"`
	Domain     string     `json:"domain"`
	CreatedAt  time.Time  `json:"created_at"`
}

// NewMetadata — метаданные новой записи. Embedding всегда не-nil: в JSON это
// `[]`, а не `null` (Python-сериализатор пишет `[]`, и парсер обеих сторон
// не должен спотыкаться о null).
func NewMetadata(domain string, confidence float64) Metadata {
	metadata := defaultMetadata()
	if domain != "" {
		metadata.Domain = domain
	}
	metadata.Confidence = confidence
	return metadata
}

/*
defaultMetadata — ровно те дефолты, которые в Python-версии проставляет
pydantic: домен H, пустой (не nil) вектор и created_at = момент создания.

created_at = «сейчас» здесь не формальность. У записей, сохранённых до
появления поля, дата в Python-версии проставляется моментом ЧТЕНИЯ — и
Go-версия обязана читать их так же, иначе одна и та же запись получит две
разные даты в зависимости от того, кто её прочитал.
*/
func defaultMetadata() Metadata {
	return Metadata{Embedding: []float64{}, Domain: DomainHistory, CreatedAt: time.Now().UTC()}
}

// Touch отмечает использование записи: инкремент счётчика + last_used.
// Аналог DiaryEntryMetadata.touch().
func (m *Metadata) Touch() {
	m.UsageCount++
	now := time.Now().UTC()
	m.LastUsed = &now
}

// IsGroundTruth — confidence >= 1.0 (подтверждённый факт).
func (m Metadata) IsGroundTruth() bool { return m.Confidence >= 1.0 }

// IsMarkedFalse — confidence <= -1.0 (заведомая ложь/шутка).
func (m Metadata) IsMarkedFalse() bool { return m.Confidence <= -1.0 }

// metadataWire — приёмник «сырого» front-matter. Поля-указатели отделяют
// «поле отсутствует» (дефолт pydantic) от «поле равно нулю».
type metadataWire struct {
	Score      *float64        `json:"score"`
	Confidence *float64        `json:"confidence"`
	LastUsed   json.RawMessage `json:"last_used"`
	UsageCount json.RawMessage `json:"usage_count"`
	Embedding  json.RawMessage `json:"embedding"`
	Domain     *string         `json:"domain"`
	CreatedAt  json.RawMessage `json:"created_at"`
}

// UnmarshalJSON разбирает метаданные с той же строгостью, что pydantic:
// confidence вне [-1, 1] и отрицательный usage_count — ошибка валидации
// (в Python она означает «файл пропущен с предупреждением», см. `_read_entry`).
// Неизвестные ключи игнорируются — это поведение pydantic по умолчанию.
func (m *Metadata) UnmarshalJSON(data []byte) error {
	var wire metadataWire
	if err := json.Unmarshal(data, &wire); err != nil {
		return fmt.Errorf("diary: front matter is not valid JSON: %w", err)
	}
	out := defaultMetadata()
	if wire.Score != nil {
		out.Score = *wire.Score
	}
	if wire.Confidence != nil {
		confidence := *wire.Confidence
		if math.IsNaN(confidence) || confidence < -1.0 || confidence > 1.0 {
			return fmt.Errorf("diary: confidence %v out of range [-1, 1]", confidence)
		}
		out.Confidence = confidence
	}
	if len(wire.UsageCount) > 0 {
		count, err := parseUsageCount(wire.UsageCount)
		if err != nil {
			return err
		}
		out.UsageCount = count
	}
	if len(wire.Embedding) > 0 {
		if err := json.Unmarshal(wire.Embedding, &out.Embedding); err != nil {
			return fmt.Errorf("diary: embedding is not a list of numbers: %w", err)
		}
		if out.Embedding == nil {
			out.Embedding = []float64{}
		}
	}
	if wire.Domain != nil {
		out.Domain = *wire.Domain
	}
	if len(wire.CreatedAt) > 0 {
		created, err := parseTimestamp(wire.CreatedAt)
		if err != nil {
			return fmt.Errorf("diary: created_at: %w", err)
		}
		out.CreatedAt = created
	}
	if len(wire.LastUsed) > 0 {
		if !bytes.Equal(bytes.TrimSpace(wire.LastUsed), []byte("null")) {
			used, err := parseTimestamp(wire.LastUsed)
			if err != nil {
				return fmt.Errorf("diary: last_used: %w", err)
			}
			out.LastUsed = &used
		}
	}
	*m = out
	return nil
}

func parseUsageCount(raw json.RawMessage) (int, error) {
	text := strings.TrimSpace(string(raw))
	if number, err := strconv.ParseInt(text, 10, 64); err == nil {
		if number < 0 {
			return 0, fmt.Errorf("diary: usage_count %d is negative", number)
		}
		return int(number), nil
	}
	// pydantic lax-режим принимает и `3.0` — но только если дробной части нет.
	value, err := strconv.ParseFloat(text, 64)
	if err != nil || value != math.Trunc(value) {
		return 0, fmt.Errorf("diary: usage_count is not an integer: %s", text)
	}
	if value < 0 {
		return 0, fmt.Errorf("diary: usage_count %v is negative", value)
	}
	return int(value), nil
}

// parseTimestamp принимает ISO-8601-строку (как pydantic) или число секунд
// эпохи — pydantic в JSON-режиме понимает и то, и другое.
func parseTimestamp(raw json.RawMessage) (time.Time, error) {
	text := strings.TrimSpace(string(raw))
	if strings.HasPrefix(text, `"`) {
		var value string
		if err := json.Unmarshal(raw, &value); err != nil {
			return time.Time{}, err
		}
		if value == "" {
			return time.Time{}, errors.New("empty datetime")
		}
		for _, layout := range timestampLayouts {
			if parsed, err := time.Parse(layout, value); err == nil {
				return parsed.UTC(), nil
			}
		}
		return time.Time{}, fmt.Errorf("unrecognised datetime %q", value)
	}
	seconds, err := strconv.ParseFloat(text, 64)
	if err != nil {
		return time.Time{}, fmt.Errorf("unrecognised datetime %q", text)
	}
	whole, fraction := math.Modf(seconds)
	return time.Unix(int64(whole), int64(fraction*float64(time.Second))).UTC(), nil
}

var timestampLayouts = []string{
	time.RFC3339Nano,
	time.RFC3339,
	"2006-01-02T15:04:05.999999999",
	"2006-01-02T15:04:05",
	"2006-01-02 15:04:05.999999999",
	"2006-01-02 15:04:05",
	"2006-01-02",
}

// FormatTimestamp воспроизводит сериализацию datetime из pydantic v2:
// UTC — суффикс `Z`, микросекунды — ровно шесть цифр и только если они есть,
// прочие смещения — как есть.
func FormatTimestamp(t time.Time) string {
	if t.Nanosecond() == 0 {
		return t.Format("2006-01-02T15:04:05Z07:00")
	}
	return t.Format("2006-01-02T15:04:05.000000Z07:00")
}

// FormatNumber пишет float так, как его пишет Python (`repr`): целые значения
// получают `.0`, а не превращаются в `0`, — иначе побайтовый паритет
// front-matter с Python-версией теряется и `git diff` дневника пухнет.
func FormatNumber(value float64) string {
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return "null"
	}
	if value == math.Trunc(value) && math.Abs(value) < 1e16 {
		return strconv.FormatFloat(value, 'f', 1, 64)
	}
	return strconv.FormatFloat(value, 'g', -1, 64)
}

/*
Entry — запись дневника: идентификатор, метаданные и свободный текст.
Аналог DiaryEntry (efi/llm/schemas.py).

ID одновременно является именем файла, поэтому он проверяется на файловую
безопасность (`_id_must_be_filesystem_safe`): id с `../` не должен иметь
возможности выйти за каталог дневника.
*/
type Entry struct {
	ID       string   `json:"id"`
	Metadata Metadata `json:"metadata"`
	Body     string   `json:"body"`
}

// Filename — имя markdown-файла записи: `<id>.md`.
func (e *Entry) Filename() string { return e.ID + ".md" }

// SidecarFilename — имя файла с вектором: `<id>.vec.json`.
func (e *Entry) SidecarFilename() string { return e.ID + ".vec.json" }

// forbiddenIDChars — символы, запрещённые в id (как в Python-валидаторе).
const forbiddenIDChars = `/\:*?"<>|`

// ValidateID проверяет, что id пригоден как имя файла.
func ValidateID(id string) error {
	if id == "" {
		return errors.New("diary: entry id is empty")
	}
	if strings.ContainsAny(id, forbiddenIDChars) {
		return fmt.Errorf("diary: entry id is not filesystem-safe: %q", id)
	}
	return nil
}

/*
ParseEntry разбирает markdown-файл записи.

`id` берётся из имени файла (как `path.stem` в Python-версии) и в файле не
хранится. Разделители front-matter ищутся после отбрасывания ведущих пробелов;
если закрывающего `---` нет, метаданные остаются дефолтными, а тело — сырым
файлом целиком.
*/
func ParseEntry(id, raw string) (*Entry, error) {
	if err := ValidateID(id); err != nil {
		return nil, err
	}
	entry := &Entry{
		ID:       id,
		Metadata: defaultMetadata(),
		Body:     raw,
	}
	stripped := strings.TrimLeftFunc(raw, unicode.IsSpace)
	if !strings.HasPrefix(stripped, FrontMatterDelimiter) {
		return entry, nil
	}
	remainder := stripped[len(FrontMatterDelimiter):]
	closing := strings.Index(remainder, FrontMatterDelimiter)
	if closing == -1 {
		return entry, nil
	}
	entry.Body = strings.TrimLeft(remainder[closing+len(FrontMatterDelimiter):], "\n")
	front := strings.TrimSpace(remainder[:closing])
	if front == "" {
		return entry, nil
	}
	var metadata Metadata
	if err := json.Unmarshal([]byte(front), &metadata); err != nil {
		return nil, err
	}
	entry.Metadata = metadata
	return entry, nil
}

// Serialize пишет запись так, как её пишет Python: компактный читаемый
// markdown с front-matter БЕЗ эмбеддинга (вектор живёт в sidecar-файле).
func (e *Entry) Serialize() string {
	var builder strings.Builder
	builder.WriteString(FrontMatterDelimiter)
	builder.WriteString("\n")
	builder.WriteString(e.frontMatterJSON())
	builder.WriteString("\n")
	builder.WriteString(FrontMatterDelimiter)
	builder.WriteString("\n")
	builder.WriteString(e.Body)
	return builder.String()
}

// frontMatterJSON — рукописная сериализация метаданных: семь известных полей
// в фиксированном порядке, отступ в два пробела и `0.0` вместо `0`.
// `json.Marshal` так не умеет, а расходиться с Python в формате файла нельзя:
// файлы дневника читают и правят глазами, а разница в оформлении — это
// постоянный шум в `git diff`.
func (e *Entry) frontMatterJSON() string {
	m := e.Metadata
	var builder strings.Builder
	builder.WriteString("{\n")
	writeJSONField(&builder, "score", FormatNumber(m.Score))
	writeJSONField(&builder, "confidence", FormatNumber(m.Confidence))
	if m.LastUsed == nil {
		writeJSONField(&builder, "last_used", "null")
	} else {
		writeJSONField(&builder, "last_used", strconv.Quote(FormatTimestamp(*m.LastUsed)))
	}
	writeJSONField(&builder, "usage_count", strconv.Itoa(m.UsageCount))
	writeJSONField(&builder, "embedding", "[]")
	writeJSONField(&builder, "domain", strconv.Quote(m.Domain))
	writeJSONField(&builder, "created_at", strconv.Quote(FormatTimestamp(m.CreatedAt)))
	// Закрывающая скобка — на своей строке: так делает `json.dumps(indent=2)`
	// в Python-версии, и на этом побайтовый паритет файла держится тоже.
	builder.WriteString("\n}")
	return builder.String()
}

func writeJSONField(builder *strings.Builder, name, value string) {
	if builder.Len() > 2 {
		builder.WriteString(",\n")
	}
	builder.WriteString("  ")
	builder.WriteString(strconv.Quote(name))
	builder.WriteString(": ")
	builder.WriteString(value)
}

// SidecarJSON — содержимое sidecar-файла с вектором: `{"embedding": [...]}`.
// Формат совпадает с `json.dumps({"embedding": [...]})` Python-версии
// (пробелы после `:` и `,` — часть этого формата, а не вкусовщина: sidecar
// тоже попадает в `git diff`).
func (e *Entry) SidecarJSON() string {
	var builder strings.Builder
	builder.WriteString(`{"embedding": [`)
	for index, component := range e.Metadata.Embedding {
		if index > 0 {
			builder.WriteString(", ")
		}
		builder.WriteString(FormatNumber(component))
	}
	builder.WriteString("]}")
	return builder.String()
}

// ParseSidecar разбирает sidecar-файл с вектором. Отсутствие ключа или
// не-числа в массиве означает «вектора нет» — как в Python-версии, где
// такой payload просто игнорируется.
func ParseSidecar(raw string) []float64 {
	var payload struct {
		Embedding *[]float64 `json:"embedding"`
	}
	if err := json.Unmarshal([]byte(raw), &payload); err != nil || payload.Embedding == nil {
		return nil
	}
	return *payload.Embedding
}
