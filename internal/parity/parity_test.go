package parity

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/whitenarcis/Efie/internal/diary"
	"github.com/whitenarcis/Efie/internal/tfidf"
)

/*
Тесты паритета с Python-версией.

goldens.json сгенерирован настоящим Python-кодом (internal/parity/gen_goldens.py):
токенизация, эвристика короткого замыкания, ранжирование TF-IDF, векторный
поиск Diary, разбор и сериализация файлов дневника. Здесь то же самое
прогоняется через Go-реализацию и сравнивается.

Смысл не в «покрытии», а в конкретном риске перехода: пока обе реализации
работают на одном каталоге дневника, любое расхождение в формуле или формате
означает, что память Эфи меняется в зависимости от того, какая реализация
ответила — то есть поведение начинает зависеть от деталей миграции.
*/

const floatTolerance = 1e-9

type goldens struct {
	Tokenize []struct {
		Text   string   `json:"text"`
		Tokens []string `json:"tokens"`
	} `json:"tokenize"`
	Shortcut []struct {
		Text     string `json:"text"`
		Shortcut bool   `json:"shortcut"`
	} `json:"shortcut"`
	TFIDF struct {
		Corpus []struct {
			ID   string `json:"id"`
			Body string `json:"body"`
		} `json:"corpus"`
		Queries []struct {
			Text          string `json:"text"`
			MaxEntryCount int    `json:"max_entry_count"`
			Results       []struct {
				ID          string  `json:"id"`
				Relatedness float64 `json:"relatedness"`
			} `json:"results"`
		} `json:"queries"`
	} `json:"tfidf"`
	DiaryQuery struct {
		Entries []struct {
			ID         string    `json:"id"`
			Confidence float64   `json:"confidence"`
			Embedding  []float64 `json:"embedding"`
			Body       string    `json:"body"`
		} `json:"entries"`
		Query   []float64 `json:"query"`
		Options struct {
			ConfidenceFactor float64 `json:"confidence_factor"`
			MaxEntryCount    int     `json:"max_entry_count"`
			MinRelatedness   float64 `json:"min_relatedness"`
		} `json:"options"`
		Results []struct {
			ID          string  `json:"id"`
			Relatedness float64 `json:"relatedness"`
		} `json:"results"`
		Files []string `json:"files"`
	} `json:"diary_query"`
	Serialize []struct {
		Case struct {
			ID         string  `json:"id"`
			Score      float64 `json:"score"`
			Confidence float64 `json:"confidence"`
			UsageCount int     `json:"usage_count"`
			Domain     string  `json:"domain"`
			CreatedAt  string  `json:"created_at"`
			LastUsed   *string `json:"last_used"`
			Body       string  `json:"body"`
		} `json:"case"`
		Expected string `json:"expected"`
	} `json:"serialize"`
	Parse []struct {
		Case struct {
			ID  string `json:"id"`
			Raw string `json:"raw"`
		} `json:"case"`
		Expected struct {
			Body       string    `json:"body"`
			Confidence float64   `json:"confidence"`
			Domain     string    `json:"domain"`
			UsageCount int       `json:"usage_count"`
			Score      float64   `json:"score"`
			CreatedAt  *string   `json:"created_at"`
			Defaulted  bool      `json:"created_at_defaulted"`
			Embedding  []float64 `json:"embedding"`
		} `json:"expected"`
	} `json:"parse"`
	Sidecar struct {
		Embedding []float64 `json:"embedding"`
		Expected  string    `json:"expected"`
	} `json:"sidecar"`
}

func loadGoldens(t *testing.T) goldens {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "goldens.json"))
	if err != nil {
		t.Fatalf("cannot read goldens: %v (сгенерируйте: PYTHONPATH=src python internal/parity/gen_goldens.py)", err)
	}
	var parsed goldens
	if err := json.Unmarshal(raw, &parsed); err != nil {
		t.Fatalf("cannot parse goldens: %v", err)
	}
	return parsed
}

func closeEnough(left, right float64) bool {
	return math.Abs(left-right) <= floatTolerance
}

func parseGoldenTime(t *testing.T, value string) time.Time {
	t.Helper()
	parsed, err := time.Parse(time.RFC3339Nano, value)
	if err != nil {
		t.Fatalf("cannot parse golden timestamp %q: %v", value, err)
	}
	return parsed.UTC()
}

func TestTokenizeMatchesPython(t *testing.T) {
	golden := loadGoldens(t)
	for _, item := range golden.Tokenize {
		t.Run(item.Text, func(t *testing.T) {
			got := tfidf.Tokenize(item.Text)
			if len(got) != len(item.Tokens) {
				t.Fatalf("tokenize(%q) = %q, Python даёт %q", item.Text, got, item.Tokens)
			}
			for index := range got {
				if got[index] != item.Tokens[index] {
					t.Fatalf("tokenize(%q) = %q, Python даёт %q", item.Text, got, item.Tokens)
				}
			}
		})
	}
}

func TestShortcutMatchesPython(t *testing.T) {
	golden := loadGoldens(t)
	for _, item := range golden.Shortcut {
		got := tfidf.ShouldUseShortcut(item.Text)
		if got != item.Shortcut {
			t.Errorf("ShouldUseShortcut(%q) = %v, Python даёт %v", item.Text, got, item.Shortcut)
		}
	}
}

func TestTFIDFRankingMatchesPython(t *testing.T) {
	golden := loadGoldens(t)
	entries := make([]*diary.Entry, 0, len(golden.TFIDF.Corpus))
	for _, item := range golden.TFIDF.Corpus {
		entries = append(entries, &diary.Entry{
			ID:       item.ID,
			Metadata: diary.NewMetadata(diary.DomainHistory, 0.5),
			Body:     item.Body,
		})
	}
	index := tfidf.New()
	index.Build(entries)

	for _, query := range golden.TFIDF.Queries {
		t.Run(query.Text, func(t *testing.T) {
			got := index.Search(query.Text, diary.QueryOptions{MaxEntryCount: query.MaxEntryCount}, nil)
			if len(got) != len(query.Results) {
				t.Fatalf("получено %d результатов, Python даёт %d", len(got), len(query.Results))
			}
			for position, expected := range query.Results {
				if got[position].Entry.ID != expected.ID {
					t.Fatalf("результат %d: получено %s, Python даёт %s", position, got[position].Entry.ID, expected.ID)
				}
				if !closeEnough(got[position].Relatedness, expected.Relatedness) {
					t.Fatalf("relatedness %s: получено %.12f, Python даёт %.12f",
						expected.ID, got[position].Relatedness, expected.Relatedness)
				}
			}
		})
	}
}

func TestDiaryVectorQueryMatchesPython(t *testing.T) {
	golden := loadGoldens(t)
	dir := t.TempDir()

	store := diary.NewStore(filepath.Join(dir, "diary"), 0.97, nil)
	for _, item := range golden.DiaryQuery.Entries {
		metadata := diary.NewMetadata(diary.DomainHistory, item.Confidence)
		metadata.Embedding = item.Embedding
		metadata.CreatedAt = parseGoldenTime(t, "2026-03-01T00:00:00Z")
		if err := store.Save(&diary.Entry{ID: item.ID, Metadata: metadata, Body: item.Body}); err != nil {
			t.Fatalf("cannot save %s: %v", item.ID, err)
		}
	}

	got, err := store.Query(golden.DiaryQuery.Query, diary.QueryOptions{
		ConfidenceFactor: golden.DiaryQuery.Options.ConfidenceFactor,
		MaxEntryCount:    golden.DiaryQuery.Options.MaxEntryCount,
		MinRelatedness:   golden.DiaryQuery.Options.MinRelatedness,
	}, nil)
	if err != nil {
		t.Fatalf("query failed: %v", err)
	}
	if len(got) != len(golden.DiaryQuery.Results) {
		t.Fatalf("получено %d результатов, Python даёт %d (порог и обрезка должны совпадать)",
			len(got), len(golden.DiaryQuery.Results))
	}
	for position, expected := range golden.DiaryQuery.Results {
		if got[position].Entry.ID != expected.ID {
			t.Fatalf("результат %d: получено %s, Python даёт %s", position, got[position].Entry.ID, expected.ID)
		}
		if !closeEnough(got[position].Relatedness, expected.Relatedness) {
			t.Fatalf("relatedness %s: получено %.12f, Python даёт %.12f",
				expected.ID, got[position].Relatedness, expected.Relatedness)
		}
	}

	// Имена файлов на диске: Go-версия обязана писать ровно тот же набор, что
	// Python, иначе два процесса перестанут видеть записи друг друга.
	names := listFiles(t, filepath.Join(dir, "diary"))
	if strings.Join(names, ",") != strings.Join(golden.DiaryQuery.Files, ",") {
		t.Fatalf("файлы на диске %v, Python пишет %v", names, golden.DiaryQuery.Files)
	}
}

func TestSerializeMatchesPythonByteForByte(t *testing.T) {
	golden := loadGoldens(t)
	for _, item := range golden.Serialize {
		t.Run(item.Case.ID, func(t *testing.T) {
			metadata := diary.NewMetadata(item.Case.Domain, item.Case.Confidence)
			metadata.Score = item.Case.Score
			metadata.UsageCount = item.Case.UsageCount
			metadata.CreatedAt = parseGoldenTime(t, item.Case.CreatedAt)
			if item.Case.LastUsed != nil {
				used := parseGoldenTime(t, *item.Case.LastUsed)
				metadata.LastUsed = &used
			}
			entry := &diary.Entry{ID: item.Case.ID, Metadata: metadata, Body: item.Case.Body}
			if got := entry.Serialize(); got != item.Expected {
				t.Fatalf("сериализация разошлась с Python.\nGo:\n%s\nPython:\n%s", got, item.Expected)
			}
		})
	}
}

func TestParseMatchesPython(t *testing.T) {
	golden := loadGoldens(t)
	for _, item := range golden.Parse {
		t.Run(item.Case.ID, func(t *testing.T) {
			entry, err := diary.ParseEntry(item.Case.ID, item.Case.Raw)
			if err != nil {
				t.Fatalf("ParseEntry failed: %v", err)
			}
			if entry.Body != item.Expected.Body {
				t.Fatalf("body = %q, Python даёт %q", entry.Body, item.Expected.Body)
			}
			if !closeEnough(entry.Metadata.Confidence, item.Expected.Confidence) {
				t.Errorf("confidence = %v, Python даёт %v", entry.Metadata.Confidence, item.Expected.Confidence)
			}
			if !closeEnough(entry.Metadata.Score, item.Expected.Score) {
				t.Errorf("score = %v, Python даёт %v", entry.Metadata.Score, item.Expected.Score)
			}
			if entry.Metadata.Domain != item.Expected.Domain {
				t.Errorf("domain = %q, Python даёт %q", entry.Metadata.Domain, item.Expected.Domain)
			}
			if entry.Metadata.UsageCount != item.Expected.UsageCount {
				t.Errorf("usage_count = %d, Python даёт %d", entry.Metadata.UsageCount, item.Expected.UsageCount)
			}
			if len(entry.Metadata.Embedding) != len(item.Expected.Embedding) {
				t.Errorf("embedding = %v, Python даёт %v", entry.Metadata.Embedding, item.Expected.Embedding)
			}
			// created_at: у файла без этого поля Python проставляет «сейчас»
			// (default_factory pydantic) — сравнивать значение нельзя, но можно
			// и нужно проверить, что дата действительно свежая, а не нулевая.
			if item.Expected.Defaulted {
				if age := time.Since(entry.Metadata.CreatedAt); age < 0 || age > 5*time.Minute {
					t.Errorf("created_at = %s: ожидался момент чтения (сейчас), Python проставляет default_factory",
						entry.Metadata.CreatedAt)
				}
			} else if item.Expected.CreatedAt != nil {
				// Python отдаёт isoformat (+00:00 для UTC), Go пишет 'Z' —
				// сравниваем момент времени, а не строку.
				expectedCreated := parseGoldenTime(t, strings.Replace(*item.Expected.CreatedAt, "+00:00", "Z", 1))
				if !entry.Metadata.CreatedAt.Equal(expectedCreated) {
					t.Errorf("created_at = %s, Python даёт %s", entry.Metadata.CreatedAt, expectedCreated)
				}
			}
		})
	}
}

func TestSidecarMatchesPython(t *testing.T) {
	golden := loadGoldens(t)
	metadata := diary.NewMetadata(diary.DomainHistory, 0.5)
	metadata.Embedding = golden.Sidecar.Embedding
	entry := &diary.Entry{ID: "sidecar", Metadata: metadata, Body: "тело"}
	if got := entry.SidecarJSON(); got != golden.Sidecar.Expected {
		t.Fatalf("sidecar = %s, Python пишет %s", got, golden.Sidecar.Expected)
	}
	if parsed := diary.ParseSidecar(golden.Sidecar.Expected); len(parsed) != len(golden.Sidecar.Embedding) {
		t.Fatalf("ParseSidecar вернул %v, ожидался вектор длины %d", parsed, len(golden.Sidecar.Embedding))
	}
}

func listFiles(t *testing.T, dir string) []string {
	t.Helper()
	items, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("cannot read %s: %v", dir, err)
	}
	names := make([]string, 0, len(items))
	for _, item := range items {
		names = append(names, item.Name())
	}
	sort.Strings(names)
	return names
}
