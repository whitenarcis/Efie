package tfidf

import (
	"math"
	"sort"
	"sync"

	"github.com/whitenarcis/Efie/internal/diary"
)

// Document — проиндексированная запись: счётчики термов и длина в токенах.
type Document struct {
	Entry  *diary.Entry
	Terms  []string // порядок первого появления в тексте
	Counts map[string]int
	Length int
}

/*
Index — инкрементальный TF-IDF индекс поверх текстов записей, живущий в
памяти процесса (сам себя не персистит).

`Build`/`Add` — операции записи, `Search` — критический путь: без сети, без
копирования корпуса и без потоков (в отличие от Python-версии, где под локом
снимается полная копия индекса).

Порядок документов значим: сортировка результатов стабильная, поэтому при
равных релевантностях выдача идёт в порядке индексации — ровно как порядок
ключей dict в Python.
*/
type Index struct {
	mu                sync.RWMutex
	documents         map[string]*Document
	order             []string
	documentFrequency map[string]int
}

// New создаёт пустой индекс.
func New() *Index {
	return &Index{
		documents:         make(map[string]*Document),
		documentFrequency: make(map[string]int),
	}
}

// Build полностью перестраивает индекс по переданному набору записей.
func (i *Index) Build(entries []*diary.Entry) {
	documents := make(map[string]*Document, len(entries))
	documentFrequency := make(map[string]int, len(entries))
	order := make([]string, 0, len(entries))
	for _, entry := range entries {
		document := indexDocument(entry)
		documents[entry.ID] = document
		order = append(order, entry.ID)
		for _, term := range document.Terms {
			documentFrequency[term]++
		}
	}
	i.mu.Lock()
	defer i.mu.Unlock()
	i.documents = documents
	i.documentFrequency = documentFrequency
	i.order = order
}

// Add инкрементально добавляет одну запись в уже построенный индекс.
// Повторное добавление той же записи обновляет её документ, но НЕ двигает
// позицию в порядке индексации (поведение присваивания ключу dict).
func (i *Index) Add(entry *diary.Entry) {
	document := indexDocument(entry)
	i.mu.Lock()
	defer i.mu.Unlock()
	if _, known := i.documents[entry.ID]; !known {
		i.order = append(i.order, entry.ID)
	}
	i.documents[entry.ID] = document
	for _, term := range document.Terms {
		i.documentFrequency[term]++
	}
}

// Remove убирает запись из индекса (нужно консолидации: удалённая из дневника
// запись не должна продолжать находиться поиском).
func (i *Index) Remove(id string) {
	i.mu.Lock()
	defer i.mu.Unlock()
	document, known := i.documents[id]
	if !known {
		return
	}
	delete(i.documents, id)
	for _, term := range document.Terms {
		i.documentFrequency[term]--
		if i.documentFrequency[term] <= 0 {
			delete(i.documentFrequency, term)
		}
	}
	for position, candidate := range i.order {
		if candidate == id {
			i.order = append(i.order[:position], i.order[position+1:]...)
			break
		}
	}
}

// Len — число проиндексированных записей.
func (i *Index) Len() int {
	i.mu.RLock()
	defer i.mu.RUnlock()
	return len(i.documents)
}

/*
Search — поиск по TF-IDF без единого сетевого вызова.

`filter` — тот же контракт, что у diary.Store.Query: фолбэк обязан уважать
доменную фильтрацию (иначе поведение памяти зависело бы от того, работает ли
сегодня локальный embedding-движок). Фильтр применяется ДО ранжирования.

min_relatedness здесь не применяется — так же, как в Python-версии: у TF-IDF
шкала сходства не сопоставима с косинусом эмбеддингов, и общий порог
`memory.min_relatedness` (0.80) выкосил бы всю выдачу.
*/
func (i *Index) Search(query string, options diary.QueryOptions, filter diary.Filter) []diary.QueryResult {
	i.mu.RLock()
	defer i.mu.RUnlock()
	if len(i.documents) == 0 {
		return nil
	}
	documents := make([]*Document, 0, len(i.order))
	for _, id := range i.order {
		document, known := i.documents[id]
		if !known {
			continue
		}
		if filter != nil && !filter(document.Entry) {
			continue
		}
		documents = append(documents, document)
	}
	if len(documents) == 0 {
		return nil
	}
	return scoreQuery(query, documents, i.documentFrequency, options.MaxEntryCount)
}

func indexDocument(entry *diary.Entry) *Document {
	tokens := Tokenize(entry.Body)
	counts := make(map[string]int, len(tokens))
	terms := make([]string, 0, len(tokens))
	for _, token := range tokens {
		if _, known := counts[token]; !known {
			terms = append(terms, token)
		}
		counts[token]++
	}
	return &Document{Entry: entry, Terms: terms, Counts: counts, Length: len(tokens)}
}

/*
scoreQuery — косинусное сходство TF-IDF векторов запроса и документов.

Формула — порт `_score_query` дословно, вместе со сглаженным IDF
(`log((N+1)/(df+1)) + 1`, всегда > 0) и нормировкой частоты на длину
документа. Менять её нельзя: выдача TF-IDF уходит в промпт, то есть является
поведением, а не внутренней деталью.

Итерация по термам документа идёт в порядке первого появления (`Terms`), а не
по map: порядок суммирования float64 — единственное, что отличает результат
от Python-версии, и он должен быть хотя бы детерминированным от запуска
к запуску.
*/
func scoreQuery(
	queryText string,
	documents []*Document,
	documentFrequency map[string]int,
	maxEntryCount int,
) []diary.QueryResult {
	queryTokens := Tokenize(queryText)
	if len(queryTokens) == 0 {
		return nil
	}

	queryCounts := make(map[string]int, len(queryTokens))
	for _, token := range queryTokens {
		queryCounts[token]++
	}
	corpusSize := len(documents)
	if corpusSize < 1 {
		corpusSize = 1
	}
	idf := func(term string) float64 {
		return math.Log(float64(corpusSize+1)/float64(documentFrequency[term]+1)) + 1.0
	}

	queryVector := make(map[string]float64, len(queryCounts))
	queryNormSquared := 0.0
	for term, count := range queryCounts {
		weight := (float64(count) / float64(len(queryTokens))) * idf(term)
		queryVector[term] = weight
		queryNormSquared += weight * weight
	}
	queryNorm := math.Sqrt(queryNormSquared)
	if queryNorm == 0 {
		queryNorm = 1.0
	}

	scored := make([]diary.QueryResult, 0, len(documents))
	for _, document := range documents {
		if document.Length == 0 {
			continue
		}
		dotProduct := 0.0
		documentNormSquared := 0.0
		for _, term := range document.Terms {
			weight := (float64(document.Counts[term]) / float64(document.Length)) * idf(term)
			documentNormSquared += weight * weight
			if queryWeight, ok := queryVector[term]; ok {
				dotProduct += weight * queryWeight
			}
		}
		if dotProduct <= 0.0 {
			continue
		}
		documentNorm := math.Sqrt(documentNormSquared)
		if documentNorm == 0 {
			documentNorm = 1.0
		}
		similarity := dotProduct / (queryNorm * documentNorm)
		scored = append(scored, diary.QueryResult{
			Entry:       document.Entry,
			Relatedness: math.Min(math.Max(similarity, 0.0), 1.0),
		})
	}

	sort.SliceStable(scored, func(left, right int) bool {
		return scored[left].Relatedness > scored[right].Relatedness
	})
	if maxEntryCount > 0 && len(scored) > maxEntryCount {
		scored = scored[:maxEntryCount]
	}
	return scored
}
