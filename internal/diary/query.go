package diary

import (
	"math"
	"sort"

	"github.com/whitenarcis/Efie/internal/vec"
)

// Filter — предикат по записи: чем Python-версия передаёт `filter_fn`
// (например, доменный фильтр из memory/router.py).
type Filter func(*Entry) bool

/*
QueryOptions — аналог DiaryQueryOptions (efi/llm/schemas.py).

Значения по умолчанию повторяют pydantic: confidence_factor = 0.01,
max_entry_count = 10, min_relatedness = 0.0. Расходиться с ними нельзя:
это те же числа, которыми Python-версия ранжирует память, а ранжирование
видно в промпте — то есть является поведением, а не настройкой.
*/
type QueryOptions struct {
	ConfidenceFactor float64
	MaxEntryCount    int
	MinRelatedness   float64
}

// DefaultQueryOptions — значения по умолчанию, как у DiaryQueryOptions().
func DefaultQueryOptions() QueryOptions {
	return QueryOptions{ConfidenceFactor: 0.01, MaxEntryCount: 10}
}

// QueryResult — запись + нормализованная релевантность (Diry::EntryExAndRelatedness).
type QueryResult struct {
	Entry       *Entry
	Relatedness float64
}

/*
CosineSimilarity — косинусное сходство двух векторов; 0, если любой из них
нулевой.

Тонкая обёртка над vec.Cosine: формула живёт в одном месте (internal/vec),
потому что ею же пользуется дедупликация выдачи в RAG. Считается в float64 и
по одному вектору за раз — так результат совпадает с numpy до 1e-15, чего
достаточно: разница в порядке суммирования не меняет ни порядок выдачи, ни её
состав.
*/
func CosineSimilarity(left, right []float64) float64 { return vec.Cosine(left, right) }

/*
Query ищет ближайшие записи к эмбеддингу запроса: косинусное сходство,
слегка скорректированное на confidence (аналог Diary::query у референса).

Порядок операций важен и повторяет Python-версию:
 1. кандидаты — только записи С эмбеддингом;
 2. фильтр домена применяется ДО ранжирования (иначе десять релевантных
    записей чужого домена вытеснили бы единственную нужную);
 3. записи другой размерности отбрасываются со счётчиком в лог;
 4. сортировка — стабильная, по убыванию relatedness.

Последнее — не деталь: сортировка в Python стабильна, поэтому при равных
релевантностях порядок выдачи задаёт порядок кэша (`order`), а не случайность
сравнения.
*/
func (s *Store) Query(embedding []float64, options QueryOptions, filter Filter) ([]QueryResult, error) {
	if err := s.ensureLoaded(); err != nil {
		return nil, err
	}
	s.mu.RLock()
	candidates := make([]*Entry, 0, len(s.order))
	for _, id := range s.order {
		entry, known := s.entries[id]
		if !known || len(entry.Metadata.Embedding) == 0 {
			continue
		}
		candidates = append(candidates, entry)
	}
	s.mu.RUnlock()

	if filter != nil {
		kept := candidates[:0]
		for _, entry := range candidates {
			if filter(entry) {
				kept = append(kept, entry)
			}
		}
		candidates = kept
	}
	if len(candidates) == 0 {
		return nil, nil
	}

	scored, skipped := scoreEntries(embedding, candidates, options.ConfidenceFactor)
	if skipped > 0 {
		s.logger.Warn(
			"diary: skipping entries with embedding dimensionality mismatch",
			"skipped", skipped,
			"expected", len(embedding),
		)
	}
	sort.SliceStable(scored, func(i, j int) bool { return scored[i].Relatedness > scored[j].Relatedness })
	if options.MinRelatedness > 0.0 {
		kept := scored[:0]
		for _, result := range scored {
			if result.Relatedness >= options.MinRelatedness {
				kept = append(kept, result)
			}
		}
		scored = kept
	}
	if options.MaxEntryCount > 0 && len(scored) > options.MaxEntryCount {
		scored = scored[:options.MaxEntryCount]
	}
	return scored, nil
}

// IsDuplicateOf — проверка на дубль/плагиат перед сохранением новой записи
// (аналог diaryPlagiarismThreshold): возвращает лучший найденный дубль, если
// его relatedness выше порога, иначе nil.
func (s *Store) IsDuplicateOf(candidate []float64) (*QueryResult, error) {
	best, err := s.Query(candidate, QueryOptions{ConfidenceFactor: 0.0, MaxEntryCount: 1}, nil)
	if err != nil {
		return nil, err
	}
	if len(best) > 0 && best[0].Relatedness > s.plagiarismThreshold {
		return &best[0], nil
	}
	return nil, nil
}

// PlagiarismThreshold — порог, выше которого новая запись считается дублем.
func (s *Store) PlagiarismThreshold() float64 { return s.plagiarismThreshold }

/*
scoreEntries — CPU-bound часть поиска, порт `_score_entries` (numpy).

Формула воспроизведена буквально, включая нормализацию косинуса в [0, 1]
и обрезку итоговой релевантности: `(cos + 1) / 2` — это то, что видит модель
в промпте, и «улучшать» её здесь нельзя.
*/
func scoreEntries(query []float64, entries []*Entry, confidenceFactor float64) ([]QueryResult, int) {
	dimension := len(query)
	same := make([]*Entry, 0, len(entries))
	skipped := 0
	for _, entry := range entries {
		if len(entry.Metadata.Embedding) == dimension {
			same = append(same, entry)
		} else {
			skipped++
		}
	}
	if len(same) == 0 {
		return nil, skipped
	}

	queryNorm := vec.Norm(query)
	results := make([]QueryResult, 0, len(same))
	for _, entry := range same {
		vector := entry.Metadata.Embedding
		cosine := 0.0
		if denominator := vec.Norm(vector) * queryNorm; denominator > 0 {
			cosine = vec.Dot(vector, query) / denominator
		}
		normalized := (cosine + 1.0) / 2.0
		adjusted := normalized + confidenceFactor*entry.Metadata.Confidence
		results = append(results, QueryResult{
			Entry:       entry,
			Relatedness: math.Min(math.Max(adjusted, 0.0), 1.0),
		})
	}
	return results, skipped
}
