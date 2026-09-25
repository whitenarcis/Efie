package diary

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
)

/*
Store — Go-аналог класса Diary: дневник как каталог markdown-файлов с ленивым
кэшем в памяти процесса.

Кэш ленивый: каталог не читается, пока не понадобятся данные (`Get`, `All`,
`Query`). `Reload` инвалидирует кэш, следующий доступ перечитывает каталог —
полезно, если файлы дневника были изменены в обход этого объекта (например,
вручную или Python-версией).

Почему это вообще переносится в Go первым. Python-версия держит корпус как
`dict[str, DiaryEntry]` из pydantic-моделей и копирует его на каждый поиск
(`TfidfFallbackIndex.search`) либо строит numpy-матрицу из списков float на
каждый запрос к дневнику. Здесь корпус читается один раз и лежит в плоских
структурах; вместо GIL-потоков и `asyncio.to_thread` — обычные горутины.
*/
type Store struct {
	dir                 string
	plagiarismThreshold float64
	logger              *slog.Logger

	mu      sync.RWMutex
	entries map[string]*Entry
	// order — порядок записей, в котором их увидел Python-код: `sorted(glob)`
	// при загрузке и порядок появления при `save`. Он важен не для красоты:
	// сортировка результатов в Python стабильная, поэтому ничьи разрешаются в
	// этом порядке, и от него зависит выдача RAG при одинаковых релевантностях.
	order []string
	// loaded отделяет «кэш пуст» от «кэш ещё не читали»: пустой дневник —
	// это тоже загруженный дневник.
	loaded bool
}

// NewStore создаёт хранилище поверх каталога `dir`. Каталог создаётся при
// первой загрузке, а не в конструкторе (как в Python-версии).
func NewStore(dir string, plagiarismThreshold float64, logger *slog.Logger) *Store {
	if logger == nil {
		logger = slog.Default()
	}
	return &Store{
		dir:                 dir,
		plagiarismThreshold: plagiarismThreshold,
		logger:              logger,
		entries:             make(map[string]*Entry),
	}
}

// Dir возвращает каталог дневника.
func (s *Store) Dir() string { return s.dir }

// Reload инвалидирует кэш: следующий доступ перечитает каталог с диска.
func (s *Store) Reload() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.loaded = false
	s.entries = make(map[string]*Entry)
	s.order = nil
}

// Load читает каталог, если кэш ещё пуст, и отдаёт число записей.
func (s *Store) Load() (int, error) {
	if err := s.ensureLoaded(); err != nil {
		return 0, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.entries), nil
}

func (s *Store) ensureLoaded() error {
	s.mu.RLock()
	if s.loaded {
		s.mu.RUnlock()
		return nil
	}
	s.mu.RUnlock()

	s.mu.Lock()
	defer s.mu.Unlock()
	if s.loaded {
		return nil
	}
	if err := os.MkdirAll(s.dir, 0o755); err != nil {
		return fmt.Errorf("diary: cannot create %s: %w", s.dir, err)
	}
	paths, err := s.listEntryFiles()
	if err != nil {
		return err
	}
	entries := make(map[string]*Entry, len(paths))
	order := make([]string, 0, len(paths))
	for _, path := range paths {
		entry := s.readEntry(path)
		if entry == nil {
			continue
		}
		entries[entry.ID] = entry
		order = append(order, entry.ID)
	}
	s.entries = entries
	s.order = order
	s.loaded = true
	return nil
}

// listEntryFiles возвращает пути к записям в лексикографическом порядке —
// ровно как `sorted(self._diary_dir.glob("*.md"))`. Sidecar-файлы
// (`*.vec.json`) в этот список не попадают по построению.
func (s *Store) listEntryFiles() ([]string, error) {
	items, err := os.ReadDir(s.dir)
	if err != nil {
		return nil, fmt.Errorf("diary: cannot read %s: %w", s.dir, err)
	}
	paths := make([]string, 0, len(items))
	for _, item := range items {
		if item.IsDir() || !strings.HasSuffix(item.Name(), ".md") {
			continue
		}
		paths = append(paths, filepath.Join(s.dir, item.Name()))
	}
	sort.Strings(paths)
	return paths, nil
}

/*
readEntry читает одну запись: markdown-файл, а если эмбеддинга в нём нет —
добирает вектор из sidecar-файла рядом.

Ошибки чтения и разбора не прерывают загрузку корпуса: одна испорченная
запись не должна лишать Эфи всей памяти. Нечитаемый файл пропускается с
предупреждением (в Python-версии — так же).
*/
func (s *Store) readEntry(path string) *Entry {
	raw, err := os.ReadFile(path)
	if err != nil {
		s.logger.Warn("diary: failed to read entry", "path", path, "error", err)
		return nil
	}
	id := strings.TrimSuffix(filepath.Base(path), ".md")
	entry, err := ParseEntry(id, string(raw))
	if err != nil {
		s.logger.Warn("diary: failed to parse entry", "path", path, "error", err)
		return nil
	}
	if len(entry.Metadata.Embedding) == 0 {
		sidecar := filepath.Join(s.dir, entry.SidecarFilename())
		payload, err := os.ReadFile(sidecar)
		switch {
		case err == nil:
			if vector := ParseSidecar(string(payload)); vector != nil {
				entry.Metadata.Embedding = vector
			}
		case os.IsNotExist(err):
			// Вектора нет — запись живёт текстом и находится через TF-IDF.
		default:
			s.logger.Warn("diary: failed to read sidecar", "path", sidecar, "error", err)
		}
	}
	return entry
}

// Get возвращает запись по id. Указатель на живую запись кэша, а не копию:
// Python-версия отдаёт тот же объект, и вызывающая сторона (например,
// дедупликация выдачи в RAG) рассчитывает, что правка счетчика использования
// в нём видна до следующего `Save`.
func (s *Store) Get(id string) (*Entry, bool) {
	if err := s.ensureLoaded(); err != nil {
		return nil, false
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	entry, ok := s.entries[id]
	return entry, ok
}

// All отдаёт все записи в порядке кэша (см. комментарий к полю `order`).
func (s *Store) All() ([]*Entry, error) {
	if err := s.ensureLoaded(); err != nil {
		return nil, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	entries := make([]*Entry, 0, len(s.order))
	for _, id := range s.order {
		if entry, ok := s.entries[id]; ok {
			entries = append(entries, entry)
		}
	}
	return entries, nil
}

// Len — число записей в кэше (не трогает диск, если корпус уже загружен).
func (s *Store) Len() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.entries)
}

// Save персистит запись на диск и обновляет кэш.
//
// Дубль-чек здесь сознательно отсутствует (как в Python-версии): он требует
// эмбеддинга запроса и делается уровнем выше — memory/rag.go, ДО вызова Save.
func (s *Store) Save(entry *Entry) error {
	if err := ValidateID(entry.ID); err != nil {
		return err
	}
	if err := s.ensureLoaded(); err != nil {
		return err
	}
	if err := writeTextAtomic(filepath.Join(s.dir, entry.Filename()), entry.Serialize()); err != nil {
		return err
	}
	sidecar := filepath.Join(s.dir, entry.SidecarFilename())
	if len(entry.Metadata.Embedding) > 0 {
		if err := writeTextAtomic(sidecar, entry.SidecarJSON()); err != nil {
			return err
		}
	} else if err := os.Remove(sidecar); err != nil && !os.IsNotExist(err) {
		// Залежавшийся sidecar от прошлой версии записи не должен врать,
		// что вектор есть, — но его отсутствие ошибкой не является.
		s.logger.Warn("diary: cannot remove stale sidecar", "path", sidecar, "error", err)
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if _, known := s.entries[entry.ID]; !known {
		s.order = append(s.order, entry.ID)
	}
	s.entries[entry.ID] = entry
	return nil
}

// Delete удаляет запись с диска и из кэша. Используется программной
// консолидацией; обычный путь записи в дневник удаление не вызывает никогда.
// Запись, которой нет в кэше, не удаляется с диска — как в Python-версии
// (`cache.pop(id, None)` → `return`), которая, возможно, и не знает о ней,
// потому что кэш ещё не читали.
func (s *Store) Delete(id string) error {
	if err := s.ensureLoaded(); err != nil {
		return err
	}
	s.mu.Lock()
	entry, known := s.entries[id]
	delete(s.entries, id)
	if known {
		remaining := s.order[:0]
		for _, candidate := range s.order {
			if candidate != id {
				remaining = append(remaining, candidate)
			}
		}
		s.order = remaining
	}
	s.mu.Unlock()
	if !known {
		return nil
	}
	if err := os.Remove(filepath.Join(s.dir, entry.Filename())); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("diary: cannot delete %s: %w", entry.Filename(), err)
	}
	if err := os.Remove(filepath.Join(s.dir, entry.SidecarFilename())); err != nil && !os.IsNotExist(err) {
		s.logger.Warn("diary: cannot delete sidecar", "id", id, "error", err)
	}
	return nil
}

// Touch отмечает запись использованной и сохраняет её (аналог Diary.touch).
func (s *Store) Touch(id string) (*Entry, error) {
	entry, ok := s.Get(id)
	if !ok {
		return nil, ErrNotFound
	}
	entry.Metadata.Touch()
	if err := s.Save(entry); err != nil {
		return nil, err
	}
	return entry, nil
}

/*
writeTextAtomic — порт efi/utils/atomic.py: временный файл РЯДОМ с целевым
(обязательно в том же каталоге, иначе `rename` перестанет быть атомарным),
fsync до подмены, затем `rename`.

Почему это не «перестраховка»: на телефоне Android убивает фон по своему
усмотрению, и с обычной записью («обрезать в ноль, налить заново») в моменте
обрыва на диске остаётся пустой файл. В файле дневниковой записи — прожитый
вечер.
*/
func writeTextAtomic(path, content string) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("diary: cannot create %s: %w", dir, err)
	}
	temporary := filepath.Join(dir, fmt.Sprintf(".%s.%d.tmp", filepath.Base(path), os.Getpid()))
	handle, err := os.OpenFile(temporary, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o644)
	if err != nil {
		return fmt.Errorf("diary: cannot open %s: %w", temporary, err)
	}
	if _, err := handle.WriteString(content); err != nil {
		handle.Close()
		os.Remove(temporary)
		return fmt.Errorf("diary: cannot write %s: %w", temporary, err)
	}
	// Данные должны дойти до диска ДО подмены: иначе после внезапного
	// выключения на месте старого файла окажется новый, но пустой.
	if err := handle.Sync(); err != nil {
		handle.Close()
		os.Remove(temporary)
		return fmt.Errorf("diary: cannot sync %s: %w", temporary, err)
	}
	if err := handle.Close(); err != nil {
		os.Remove(temporary)
		return fmt.Errorf("diary: cannot close %s: %w", temporary, err)
	}
	if err := os.Rename(temporary, path); err != nil {
		os.Remove(temporary)
		return fmt.Errorf("diary: cannot replace %s: %w", path, err)
	}
	return nil
}
