package embed

import (
	"context"
	"hash/fnv"
	"math"
)

/*
Hash — детерминированный «эмбеддер» на хешах символьных n-грамм.

ОН НЕ ЗАМЕНЯЕТ e5 И НЕ ПРЕТЕНДУЕТ НА ЕЁ КАЧЕСТВО. Он существует ровно для
двух вещей:

 1. Go-сервис памяти должен запускаться и проверяться БЕЗ гигабайта весов
    ONNX — в тестах, в CI и на машине разработчика, где модели ещё нет;
 2. лестнице нужен работающий «локальный» уровень, чтобы проверить сам
    механизм отката (локальный → облачный → TF-IDF) без настоящей модели.

В продакшене включается только осознанно (`-embedder=hash-dev`) и с громким
предупреждением в логе: качество семантического поиска на нём будет заметно
хуже, и это изменение поведения Эфи, а не оптимизация.

Свойства, важные для тестов: результат детерминирован (FNV-1a стабилен между
запусками и платформами), вектор нормирован к единичной длине, одинаковый
текст даёт одинаковый вектор.
*/
type Hash struct {
	Dims int
}

// NewHash создаёт хеш-эмбеддер заданной размерности.
func NewHash(dims int) *Hash {
	if dims <= 0 {
		dims = 256
	}
	return &Hash{Dims: dims}
}

// EmbedQuery — эмбеддинг запроса (префиксы добавляет embed.Prefixed).
func (h *Hash) EmbedQuery(_ context.Context, text string) ([]float64, error) {
	return h.embed(text), nil
}

// EmbedDocument — эмбеддинг сохраняемого текста.
func (h *Hash) EmbedDocument(_ context.Context, text string) ([]float64, error) {
	return h.embed(text), nil
}

func (h *Hash) embed(text string) []float64 {
	vector := make([]float64, h.Dims)
	runes := []rune(text)
	if len(runes) == 0 {
		return vector
	}
	// Символьные биграммы и триграммы: для русского текста этого достаточно,
	// чтобы «похожие фразы» не разъезжались, а «разные» не слипались.
	for size := 2; size <= 3; size++ {
		for start := 0; start+size <= len(runes); start++ {
			gram := string(runes[start : start+size])
			hasher := fnv.New32a()
			if _, err := hasher.Write([]byte(gram)); err != nil {
				continue
			}
			sum := hasher.Sum32()
			index := int(sum % uint32(h.Dims))
			if sum&0x80000000 != 0 {
				vector[index] -= 1
			} else {
				vector[index] += 1
			}
		}
	}
	length := 0.0
	for _, component := range vector {
		length += component * component
	}
	length = math.Sqrt(length)
	if length == 0 {
		return vector
	}
	for index := range vector {
		vector[index] /= length
	}
	return vector
}
