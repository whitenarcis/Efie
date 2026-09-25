package main

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"

	"github.com/whitenarcis/Efie/internal/diary"
	"github.com/whitenarcis/Efie/internal/embed"
	"github.com/whitenarcis/Efie/internal/memory"
	"github.com/whitenarcis/Efie/internal/memstats"
	"github.com/whitenarcis/Efie/internal/tfidf"
)

// runBench — замер подсистемы памяти на синтетическом корпусе.
//
// Зачем это в бинарнике, а не отдельной тулзой. Перенос подсистемы в Go имеет
// смысл ровно настолько, насколько он уменьшает RAM и CPU; значит, цифры
// должны быть воспроизводимы одной командой на любой машине, включая ту, где
// ещё нет ни бота, ни модели. Замер пишет корпус во временный каталог, читает
// его холодным стартом, строит индекс и гоняет поиски — то есть повторяет
// ровно те операции, которые бот делает на старте и на каждом сообщении.
func runBench(opts options) error {
	dir, err := os.MkdirTemp("", "efie-memory-bench")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)

	logger := newLogger("error")
	baseline := memstats.RSSBytes()

	generated := time.Now()
	for index := 0; index < opts.bench; index++ {
		entry := syntheticEntry(index, opts.benchDims)
		if err := os.WriteFile(filepath.Join(dir, entry.Filename()), []byte(entry.Serialize()), 0o644); err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(dir, entry.SidecarFilename()), []byte(entry.SidecarJSON()), 0o644); err != nil {
			return err
		}
	}
	generationDuration := time.Since(generated)

	// Холодный старт: бот перечитывает каталог целиком после каждого
	// рестарта, и это его самая долгая операция с памятью.
	store := diary.NewStore(dir, 0.97, logger)
	loading := time.Now()
	entries, err := store.Load()
	if err != nil {
		return err
	}
	loadDuration := time.Since(loading)
	all, err := store.All()
	if err != nil {
		return err
	}
	afterLoad := memstats.RSSBytes()

	building := time.Now()
	index := tfidf.New()
	index.Build(all)
	indexDuration := time.Since(building)
	afterIndex := memstats.RSSBytes()

	mem := memory.New(store, index, embed.Ladder{}, logger)

	// Поиски: ровно то, что происходит на каждое сообщение.
	ctx := context.Background()
	searchDuration := time.Duration(0)
	vectorDuration := time.Duration(0)
	for index := 0; index < opts.benchQueries; index++ {
		query := syntheticQuery(index)
		started := time.Now()
		if _, err := mem.Search(ctx, query, diary.QueryOptions{MaxEntryCount: 10, MinRelatedness: 0.80}, nil); err != nil {
			return err
		}
		searchDuration += time.Since(started)

		started = time.Now()
		if _, err := store.Query(syntheticVector(index, opts.benchDims), diary.QueryOptions{
			ConfidenceFactor: 0.01,
			MaxEntryCount:    10,
			MinRelatedness:   0.80,
		}, nil); err != nil {
			return err
		}
		vectorDuration += time.Since(started)
	}

	printBenchReport(benchReport{
		entries:          entries,
		dims:             opts.benchDims,
		queries:          opts.benchQueries,
		generation:       generationDuration,
		load:             loadDuration,
		index:            indexDuration,
		search:           searchDuration,
		vector:           vectorDuration,
		baselineRSS:      baseline,
		afterLoad:        afterLoad,
		afterIndex:       afterIndex,
		finalRSS:         memstats.RSSBytes(),
		indexedDocuments: index.Len(),
	})
	return nil
}

type benchReport struct {
	entries          int
	dims             int
	queries          int
	generation       time.Duration
	load             time.Duration
	index            time.Duration
	search           time.Duration
	vector           time.Duration
	baselineRSS      int64
	afterLoad        int64
	afterIndex       int64
	finalRSS         int64
	indexedDocuments int
}

func printBenchReport(report benchReport) {
	perSearch := report.search / time.Duration(report.queries)
	perVector := report.vector / time.Duration(report.queries)
	fmt.Println("efie-memory benchmark")
	fmt.Printf("  corpus:            %d записей, размерность %d, проиндексировано %d\n",
		report.entries, report.dims, report.indexedDocuments)
	fmt.Printf("  генерация корпуса: %.2f с (%.2f мс на запись, включая запись на диск)\n",
		report.generation.Seconds(), float64(report.generation.Milliseconds())/float64(report.entries))
	fmt.Printf("  холодная загрузка: %.2f с (%.3f мс на запись: чтение .md + sidecar + разбор)\n",
		report.load.Seconds(), float64(report.load.Microseconds())/float64(report.entries)/1000)
	fmt.Printf("  сборка TF-IDF:     %.2f с\n", report.index.Seconds())
	fmt.Printf("  поиск (полный путь памяти, %d запросов): %.3f мс на запрос\n",
		report.queries, float64(perSearch.Microseconds())/1000)
	fmt.Printf("  поиск (векторный, без TF-IDF и дедупликации): %.3f мс на запрос\n",
		float64(perVector.Microseconds())/1000)
	fmt.Printf("  RSS: старт %.1f МБ -> после загрузки %.1f МБ -> после индекса %.1f МБ (итого %.1f МБ)\n",
		megabytes(report.baselineRSS), megabytes(report.afterLoad), megabytes(report.afterIndex),
		megabytes(report.finalRSS))
	fmt.Printf("  прирост RSS на корпус: %.1f МБ (%.1f КБ на запись, включая вектора в памяти)\n",
		megabytes(report.finalRSS-report.baselineRSS),
		float64(report.finalRSS-report.baselineRSS)/float64(report.entries)/1024)
}

func megabytes(bytes int64) float64 { return float64(bytes) / (1 << 20) }

// syntheticEntry — запись для замера: русскоязычное тело из словаря и
// детерминированный вектор заданной размерности.
func syntheticEntry(seed, dims int) *diary.Entry {
	metadata := diary.NewMetadata(diary.DomainHistory, 0.7)
	metadata.Embedding = syntheticVector(seed, dims)
	metadata.CreatedAt = time.Unix(1_700_000_000+int64(seed), 0).UTC()
	return &diary.Entry{
		ID:       fmt.Sprintf("%d_bench%06d", 1_700_000_000+seed, seed),
		Metadata: metadata,
		Body:     syntheticBody(seed),
	}
}

var bodyWords = []string{
	"сегодня", "он", "сказал", "что", "монтаж", "звука", "оказался", "сложнее",
	"чем", "я", "думала", "и", "весь", "вечер", "перебирала", "дорожки",
	"в", "редакторе", "смешно", "но", "мне", "это", "понравилось", "утром",
	"мы", "говорили", "про", "базы", "данных", "sqlite", "её",
	"промпты", "стали", "короче", "после", "правки", "памяти",
}

func syntheticBody(seed int) string {
	length := 20 + seed%40
	builder := make([]byte, 0, length*12)
	for index := 0; index < length; index++ {
		builder = append(builder, bodyWords[(seed*7+index*13)%len(bodyWords)]...)
		builder = append(builder, ' ')
	}
	return string(builder)
}

func syntheticQuery(seed int) string {
	return fmt.Sprintf("%s %s %s %s",
		bodyWords[seed%len(bodyWords)],
		bodyWords[(seed+5)%len(bodyWords)],
		bodyWords[(seed+11)%len(bodyWords)],
		"памяти",
	)
}

// syntheticVector — детерминированный псевдослучайный вектор единичной длины:
// воспроизводимость важнее «настоящей» случайности, иначе замеры между
// запусками не сравнить.
func syntheticVector(seed, dims int) []float64 {
	vector := make([]float64, dims)
	state := uint64(seed)*6364136223846793005 + 1442695040888963407
	norm := 0.0
	for index := range vector {
		state = state*6364136223846793005 + 1442695040888963407
		value := float64(int64(state>>11))/float64(uint64(1)<<53)*2 - 1
		vector[index] = value
		norm += value * value
	}
	norm = math.Sqrt(norm)
	if norm == 0 {
		return vector
	}
	for index := range vector {
		vector[index] /= norm
	}
	return vector
}
