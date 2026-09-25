package diary

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func newTestStore(t *testing.T) *Store {
	t.Helper()
	return NewStore(filepath.Join(t.TempDir(), "diary"), 0.97, nil)
}

func TestSaveAndLoadRoundTrip(t *testing.T) {
	store := newTestStore(t)
	metadata := NewMetadata(DomainPersonal, 0.5)
	metadata.Embedding = []float64{0.1, -0.2, 0.3}
	metadata.CreatedAt = time.Date(2026, 3, 4, 5, 6, 7, 0, time.UTC)
	entry := &Entry{ID: "2026-03-04_aaaaaa", Metadata: metadata, Body: "Первая запись.\n\nВторая строка."}
	if err := store.Save(entry); err != nil {
		t.Fatalf("save: %v", err)
	}

	// Новый Store поверх того же каталога — как после перезапуска процесса.
	fresh := NewStore(store.Dir(), 0.97, nil)
	if _, err := fresh.Load(); err != nil {
		t.Fatalf("load: %v", err)
	}
	loaded, known := fresh.Get(entry.ID)
	if !known {
		t.Fatalf("запись %s не найдена после перезагрузки", entry.ID)
	}
	if loaded.Body != entry.Body {
		t.Errorf("body = %q, ожидалось %q", loaded.Body, entry.Body)
	}
	if loaded.Metadata.Domain != DomainPersonal || loaded.Metadata.Confidence != 0.5 {
		t.Errorf("метаданные разошлись: %+v", loaded.Metadata)
	}
	if len(loaded.Metadata.Embedding) != 3 {
		t.Fatalf("вектор не восстановлен из sidecar: %v", loaded.Metadata.Embedding)
	}
	if !loaded.Metadata.CreatedAt.Equal(metadata.CreatedAt) {
		t.Errorf("created_at = %s, ожидалось %s", loaded.Metadata.CreatedAt, metadata.CreatedAt)
	}
}

func TestSaveWithoutEmbeddingRemovesStaleSidecar(t *testing.T) {
	store := newTestStore(t)
	metadata := NewMetadata(DomainHistory, 0.0)
	metadata.Embedding = []float64{1, 2, 3}
	entry := &Entry{ID: "stale", Metadata: metadata, Body: "с вектором"}
	if err := store.Save(entry); err != nil {
		t.Fatalf("save: %v", err)
	}
	sidecar := filepath.Join(store.Dir(), entry.SidecarFilename())
	if _, err := os.Stat(sidecar); err != nil {
		t.Fatalf("sidecar не создан: %v", err)
	}

	entry.Metadata.Embedding = []float64{}
	if err := store.Save(entry); err != nil {
		t.Fatalf("save: %v", err)
	}
	// Sidecar от прошлой версии записи не должен врать, что вектор есть.
	if _, err := os.Stat(sidecar); !os.IsNotExist(err) {
		t.Fatalf("устаревший sidecar не удалён (err=%v)", err)
	}
}

func TestSaveIsAtomicAndLeavesNoTemp(t *testing.T) {
	store := newTestStore(t)
	if err := store.Save(&Entry{ID: "atomic", Metadata: NewMetadata(DomainHistory, 0), Body: "тело"}); err != nil {
		t.Fatalf("save: %v", err)
	}
	items, err := os.ReadDir(store.Dir())
	if err != nil {
		t.Fatalf("read dir: %v", err)
	}
	for _, item := range items {
		if strings.HasSuffix(item.Name(), ".tmp") {
			t.Fatalf("после записи остался временный файл: %s", item.Name())
		}
	}
}

func TestDeleteUnknownEntryKeepsFiles(t *testing.T) {
	// Случай, который здесь важен: файл лежит в каталоге, но записи в кэше нет —
	// потому что разбор не удался (например, испорчен front-matter). Python-версия
	// такую запись не удаляет (`cache.pop(id, None)` → `return`), и Go-версия
	// обязана вести себя так же: консолидация не имеет права стирать с диска
	// то, чего она не читала, — иначе одна опечатка в файле уносит запись
	// навсегда и без следа.
	store := newTestStore(t)
	if err := os.MkdirAll(store.Dir(), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	broken := filepath.Join(store.Dir(), "broken.md")
	if err := os.WriteFile(broken, []byte("---\n{\"confidence\": 9.9}\n---\nбитая запись"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	if err := store.Delete("broken"); err != nil {
		t.Fatalf("delete: %v", err)
	}
	if _, err := os.Stat(broken); err != nil {
		t.Fatalf("битый файл удалён, хотя записи не было в кэше: %v", err)
	}

	// А нормально прочитанная запись удаляется вместе с обоими своими файлами.
	metadata := NewMetadata(DomainHistory, 0.0)
	metadata.Embedding = []float64{0.1, 0.2}
	entry := &Entry{ID: "healthy", Metadata: metadata, Body: "тело"}
	if err := store.Save(entry); err != nil {
		t.Fatalf("save: %v", err)
	}
	if err := store.Delete("healthy"); err != nil {
		t.Fatalf("delete: %v", err)
	}
	for _, name := range []string{entry.Filename(), entry.SidecarFilename()} {
		if _, err := os.Stat(filepath.Join(store.Dir(), name)); !os.IsNotExist(err) {
			t.Fatalf("файл %s не удалён вместе с записью (err=%v)", name, err)
		}
	}
}

func TestLoadSkipsBrokenFilesButKeepsRest(t *testing.T) {
	store := newTestStore(t)
	if err := store.Save(&Entry{ID: "good", Metadata: NewMetadata(DomainHistory, 0.1), Body: "нормальная"}); err != nil {
		t.Fatalf("save: %v", err)
	}
	// Битый front-matter: confidence вне диапазона — валидация падает, и запись
	// пропускается, а не роняет загрузку всего корпуса.
	broken := "---\n{\"confidence\": 7.5}\n---\nбитая запись"
	if err := os.WriteFile(filepath.Join(store.Dir(), "broken.md"), []byte(broken), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	count, err := store.Load()
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if count != 1 {
		t.Fatalf("загружено %d записей, ожидалась 1 (битая должна быть пропущена)", count)
	}
	if _, known := store.Get("good"); !known {
		t.Fatal("целая запись потеряна из-за битой")
	}
}

func TestReloadPicksUpExternalChanges(t *testing.T) {
	store := newTestStore(t)
	if err := store.Save(&Entry{ID: "one", Metadata: NewMetadata(DomainHistory, 0), Body: "первая"}); err != nil {
		t.Fatalf("save: %v", err)
	}
	if _, err := store.Load(); err != nil {
		t.Fatalf("load: %v", err)
	}
	if err := os.WriteFile(filepath.Join(store.Dir(), "three.md"), []byte("третья"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	// Запись, появившаяся на диске в обход этого объекта (например, её написал
	// Python-процесс), видна только после Reload — как и в Python-версии.
	if _, known := store.Get("three"); known {
		t.Fatal("кэш подхватил запись без Reload")
	}
	store.Reload()
	if _, err := store.Load(); err != nil {
		t.Fatalf("load: %v", err)
	}
	if _, known := store.Get("three"); !known {
		t.Fatal("после Reload запись с диска не подхвачена")
	}
}

func TestTouchPersistsUsageCounter(t *testing.T) {
	store := newTestStore(t)
	if err := store.Save(&Entry{ID: "touched", Metadata: NewMetadata(DomainHistory, 0), Body: "тело"}); err != nil {
		t.Fatalf("save: %v", err)
	}
	touched, err := store.Touch("touched")
	if err != nil {
		t.Fatalf("touch: %v", err)
	}
	if touched.Metadata.UsageCount != 1 || touched.Metadata.LastUsed == nil {
		t.Fatalf("touch не отметил использование: %+v", touched.Metadata)
	}
	fresh := NewStore(store.Dir(), 0.97, nil)
	if _, err := fresh.Load(); err != nil {
		t.Fatalf("load: %v", err)
	}
	loaded, _ := fresh.Get("touched")
	if loaded.Metadata.UsageCount != 1 || loaded.Metadata.LastUsed == nil {
		t.Fatalf("счётчик использования не сохранился на диск: %+v", loaded.Metadata)
	}
}

func TestQuerySkipsDimensionMismatch(t *testing.T) {
	store := newTestStore(t)
	good := NewMetadata(DomainHistory, 0)
	good.Embedding = []float64{1, 0, 0, 0}
	other := NewMetadata(DomainHistory, 0)
	other.Embedding = []float64{1, 0}
	if err := store.Save(&Entry{ID: "good", Metadata: good, Body: "совпадает"}); err != nil {
		t.Fatalf("save: %v", err)
	}
	if err := store.Save(&Entry{ID: "other", Metadata: other, Body: "другая размерность"}); err != nil {
		t.Fatalf("save: %v", err)
	}
	results, err := store.Query([]float64{1, 0, 0, 0}, DefaultQueryOptions(), nil)
	if err != nil {
		t.Fatalf("query: %v", err)
	}
	if len(results) != 1 || results[0].Entry.ID != "good" {
		t.Fatalf("ожидался один результат (good), получено %d", len(results))
	}
}

func TestQueryAppliesFilterBeforeRanking(t *testing.T) {
	store := newTestStore(t)
	for _, item := range []struct {
		id        string
		domain    string
		embedding []float64
	}{
		{"common", DomainCommon, []float64{1, 0, 0, 0}},
		{"history", DomainHistory, []float64{0.9, 0.1, 0, 0}},
	} {
		metadata := NewMetadata(item.domain, 0.0)
		metadata.Embedding = item.embedding
		if err := store.Save(&Entry{ID: item.id, Metadata: metadata, Body: item.id}); err != nil {
			t.Fatalf("save: %v", err)
		}
	}
	results, err := store.Query([]float64{1, 0, 0, 0}, DefaultQueryOptions(), func(entry *Entry) bool {
		return entry.Metadata.Domain == DomainHistory
	})
	if err != nil {
		t.Fatalf("query: %v", err)
	}
	if len(results) != 1 || results[0].Entry.ID != "history" {
		t.Fatalf("фильтр применён после ранжирования: %v", results)
	}
}
