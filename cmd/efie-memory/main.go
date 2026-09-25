/*
Command efie-memory — процесс памяти Эфи: дневник, векторный поиск, TF-IDF и
эмбеддинги, вынесенные из Python-бота в отдельный Go-процесс.

Это первый шаг переноса, и выбран он не наугад. Замеры по коду и конфигу
Python-версии (см. docs/go-port.md) дают такую картину по стоимости:

  - RAM: локальный embedding-движок (fastembed, `intfloat/multilingual-e5-large`,
    включён по умолчанию `memory.use_local_embeddings = true`) — порядка
    гигабайта весов ONNX плюс numpy/onnxruntime/Python в том же процессе,
    в котором живёт бот;
  - CPU: он же — эмбеддинг считается НА КАЖДУЮ сборку промпта, то есть на
    каждое сообщение, под одним `threading.Lock` (запросы выстраиваются в
    очередь);
  - CPU сверх того: на каждый поиск Python копирует весь TF-IDF-индекс под
    локом и строит numpy-матрицу корпуса заново.

Отсюда этот сервис: он берёт на себя корпус, индекс, поиск и лестницу
эмбеддингов, отдаёт их по HTTP на localhost и может быть остановлен или
перезапущен без остановки бота. Python в переходный период остаётся
оркестратором и вызывает сюда; когда перенос завершится, вызывающим станет
уже Go-бот.

Запуск:

	efie-memory -diary-dir ~/.efi_data/diary -addr 127.0.0.1:8790

Проверка цифр (без бота и без модели — на синтетическом корпусе):

	efie-memory -bench 5000 -bench-dims 1024 -bench-queries 200
*/
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/whitenarcis/Efie/internal/diary"
	"github.com/whitenarcis/Efie/internal/embed"
	"github.com/whitenarcis/Efie/internal/memapi"
	"github.com/whitenarcis/Efie/internal/memory"
	"github.com/whitenarcis/Efie/internal/tfidf"
)

type options struct {
	addr                string
	diaryDir            string
	plagiarismThreshold float64
	dedupThreshold      float64
	embedder            string
	embedBaseURL        string
	embedAPIKey         string
	embedModel          string
	embedTimeout        time.Duration
	logLevel            string

	bench        int
	benchDims    int
	benchQueries int
}

func main() {
	opts := parseFlags()
	if err := run(opts); err != nil {
		fmt.Fprintln(os.Stderr, "efie-memory:", err)
		os.Exit(1)
	}
}

func parseFlags() options {
	var opts options
	flag.StringVar(&opts.addr, "addr", envString("EFIE_MEMORY_ADDR", "127.0.0.1:8790"),
		"адрес HTTP-сервера памяти")
	flag.StringVar(&opts.diaryDir, "diary-dir", envString("EFIE_MEMORY_DIARY_DIR", ""),
		"каталог дневника (те же .md + .vec.json, что у Python-версии)")
	flag.Float64Var(&opts.plagiarismThreshold, "plagiarism-threshold", 0.97,
		"порог, выше которого новая запись считается дублем (memory.plagiarism_threshold)")
	flag.Float64Var(&opts.dedupThreshold, "dedup-threshold", memory.DedupThreshold,
		"порог склейки дублей в выдаче поиска")
	flag.StringVar(&opts.embedder, "embedder", envString("EFIE_MEMORY_EMBEDDER", "none"),
		"источник эмбеддингов: none | local-onnx | hash-dev | remote")
	flag.StringVar(&opts.embedBaseURL, "embed-base-url", envString("EFIE_EMBED_BASE_URL", ""),
		"base_url OpenAI-совместимого /embeddings (тот же, что у Эфи)")
	flag.StringVar(&opts.embedAPIKey, "embed-api-key", envString("EFIE_EMBED_API_KEY", ""),
		"ключ для /embeddings")
	flag.StringVar(&opts.embedModel, "embed-model", envString("EFIE_EMBED_MODEL", ""),
		"имя embedding-модели у провайдера")
	flag.DurationVar(&opts.embedTimeout, "embed-timeout", 5*time.Second,
		"таймаут одного запроса эмбеддинга")
	flag.StringVar(&opts.logLevel, "log-level", envString("EFIE_MEMORY_LOG_LEVEL", "info"),
		"уровень логов: debug | info | warn | error")
	flag.IntVar(&opts.bench, "bench", 0,
		"не поднимать сервер, а прогнать замер на синтетическом корпусе из N записей")
	flag.IntVar(&opts.benchDims, "bench-dims", 1024, "размерность синтетических векторов в замере")
	flag.IntVar(&opts.benchQueries, "bench-queries", 200, "сколько поисковых запросов прогнать в замере")
	flag.Parse()
	return opts
}

func envString(name, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(name)); value != "" {
		return value
	}
	return fallback
}

func run(opts options) error {
	if opts.bench > 0 {
		return runBench(opts)
	}
	if opts.diaryDir == "" {
		return errors.New("-diary-dir is required (or EFIE_MEMORY_DIARY_DIR)")
	}
	logger := newLogger(opts.logLevel)

	ladder, err := buildLadder(opts, logger)
	if err != nil {
		return err
	}
	mem := buildMemory(opts, ladder, logger)
	entries, err := mem.Store().Load()
	if err != nil {
		return err
	}
	all, err := mem.Store().All()
	if err != nil {
		return err
	}
	mem.Index().Build(all)
	logger.Info(
		"efie-memory: corpus loaded",
		"entries", entries,
		"diary_dir", opts.diaryDir,
		"embedding_sources", mem.Sources(),
	)

	server := &http.Server{
		Addr:              opts.addr,
		Handler:           memapi.New(mem, logger).Handler(),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       30 * time.Second,
		WriteTimeout:      60 * time.Second,
		IdleTimeout:       2 * time.Minute,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	failed := make(chan error, 1)
	go func() {
		logger.Info("efie-memory: listening", "addr", opts.addr)
		if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			failed <- err
		}
	}()

	select {
	case err := <-failed:
		return err
	case <-ctx.Done():
		logger.Info("efie-memory: shutting down")
	}
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	return server.Shutdown(shutdownCtx)
}

func newLogger(level string) *slog.Logger {
	parsed := slog.LevelInfo
	switch strings.ToLower(level) {
	case "debug":
		parsed = slog.LevelDebug
	case "warn":
		parsed = slog.LevelWarn
	case "error":
		parsed = slog.LevelError
	}
	return slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: parsed}))
}

/*
buildLadder собирает лестницу эмбеддингов по флагу.

`none` — не поломка, а штатный режим: память работает на TF-IDF, ровно как
Python-версия при недоступных эмбеддингах. `hash-dev` включается только
осознанно и с предупреждением: это хеш-эмбеддер, а не e5, и качество
семантического поиска на нём будет другим (см. internal/embed/hash.go).

`local-onnx` — следующий шаг переноса: тот же e5, но инференс в Go
(onnxruntime + tokenizer), без numpy и без Python в процессе. Место под него
уже готово: достаточно вернуть здесь движок, и лестница, дневник, поиск и API
заработают с ним без единой правки в остальных пакетах.
*/
func buildLadder(opts options, logger *slog.Logger) (embed.Ladder, error) {
	var ladder embed.Ladder
	switch strings.ToLower(opts.embedder) {
	case "", "none":
		logger.Info("efie-memory: embeddings disabled, поиск работает по TF-IDF")
	case "hash-dev":
		logger.Warn(
			"efie-memory: hash-dev embedder is NOT multilingual-e5 — качество памяти будет другим, это режим для тестов",
		)
		ladder.Local = embed.NewPrefixed("hash-dev", embed.NewHash(256))
	case "local-onnx":
		return embed.Ladder{}, errors.New("local-onnx ещё не реализован: это следующий шаг переноса (см. docs/go-port.md)")
	case "remote":
		if opts.embedBaseURL == "" || opts.embedModel == "" {
			return embed.Ladder{}, errors.New("-embed-base-url и -embed-model обязательны для -embedder=remote")
		}
		ladder.Remote = embed.NewRemote(opts.embedBaseURL, opts.embedAPIKey, opts.embedModel, opts.embedTimeout)
	default:
		return embed.Ladder{}, fmt.Errorf("unknown -embedder %q", opts.embedder)
	}
	return ladder, nil
}

func buildMemory(opts options, ladder embed.Ladder, logger *slog.Logger) *memory.Memory {
	store := diary.NewStore(opts.diaryDir, opts.plagiarismThreshold, logger)
	mem := memory.New(store, tfidf.New(), ladder, logger)
	mem.DedupThreshold = opts.dedupThreshold
	return mem
}
