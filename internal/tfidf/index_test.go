package tfidf

import (
	"testing"

	"github.com/whitenarcis/Efie/internal/diary"
)

func entry(id, body string, domain string) *diary.Entry {
	metadata := diary.NewMetadata(domain, 0.5)
	return &diary.Entry{ID: id, Metadata: metadata, Body: body}
}

func TestIncrementalAddMatchesRebuild(t *testing.T) {
	entries := []*diary.Entry{
		entry("one", "Сегодня мы говорили про монтаж звука и дорожки.", diary.DomainHistory),
		entry("two", "Утром обсуждали SQLite и WAL в базе данных.", diary.DomainCommon),
		entry("three", "Вечером перебирала дорожки в редакторе, звук стал ровнее.", diary.DomainHistory),
	}

	rebuilt := New()
	rebuilt.Build(entries)

	incremental := New()
	for _, item := range entries {
		incremental.Add(item)
	}

	for _, query := range []string{"монтаж звука", "sqlite база", "дорожки редактор", "совсем другое слово"} {
		expected := rebuilt.Search(query, diary.QueryOptions{MaxEntryCount: 5}, nil)
		got := incremental.Search(query, diary.QueryOptions{MaxEntryCount: 5}, nil)
		if len(got) != len(expected) {
			t.Fatalf("запрос %q: инкрементальный индекс дал %d результатов, пересобранный %d",
				query, len(got), len(expected))
		}
		for position := range got {
			if got[position].Entry.ID != expected[position].Entry.ID {
				t.Fatalf("запрос %q: порядок разошёлся на позиции %d (%s против %s)",
					query, position, got[position].Entry.ID, expected[position].Entry.ID)
			}
		}
	}
}

func TestRemoveDropsDocumentFromSearch(t *testing.T) {
	index := New()
	index.Build([]*diary.Entry{
		entry("keep", "монета упала на пол и покатилась под диван", diary.DomainHistory),
		entry("drop", "монета упала на пол и покатилась под стол", diary.DomainHistory),
	})
	if len(index.Search("монета упала", diary.QueryOptions{MaxEntryCount: 5}, nil)) != 2 {
		t.Fatalf("до удаления ожидались обе записи, индекс знает %d", index.Len())
	}
	index.Remove("drop")
	found := index.Search("монета упала", diary.QueryOptions{MaxEntryCount: 5}, nil)
	if len(found) != 1 || found[0].Entry.ID != "keep" {
		t.Fatalf("после удаления найдено %d записей: %v", len(found), found)
	}
	if index.Len() != 1 {
		t.Fatalf("в индексе осталось %d записей, ожидалась 1", index.Len())
	}
	// Повторное удаление не должно ломать частоты термов.
	index.Remove("drop")
	if len(index.Search("монета упала", diary.QueryOptions{MaxEntryCount: 5}, nil)) != 1 {
		t.Fatal("повторное удаление изменило выдачу")
	}
}

func TestSearchRespectsDomainFilter(t *testing.T) {
	index := New()
	index.Build([]*diary.Entry{
		entry("world", "дорожки в редакторе и монтаж звука", diary.DomainCommon),
		entry("mine", "дорожки в редакторе и монтаж звука", diary.DomainHistory),
	})
	found := index.Search("монтаж звука", diary.QueryOptions{MaxEntryCount: 5}, func(candidate *diary.Entry) bool {
		return candidate.Metadata.Domain == diary.DomainCommon
	})
	if len(found) != 1 || found[0].Entry.ID != "world" {
		t.Fatalf("фильтр домена не применён: %v", found)
	}
}

func TestSearchOnEmptyIndexIsEmpty(t *testing.T) {
	index := New()
	if found := index.Search("что угодно", diary.QueryOptions{MaxEntryCount: 5}, nil); len(found) != 0 {
		t.Fatalf("пустой индекс вернул %d результатов", len(found))
	}
}

func TestShortcutNeedsNoIndex(t *testing.T) {
	// Фатическая реплика распознаётся эвристикой ДО обращения к индексу —
	// на этом стоит весь смысл короткого замыкания критического пути.
	if !ShouldUseShortcut("ок") {
		t.Fatal("«ок» должно уходить в короткое замыкание")
	}
	if ShouldUseShortcut("расскажи, что мы обсуждали про монтаж звука вчера вечером") {
		t.Fatal("длинная содержательная фраза не должна попадать в короткое замыкание")
	}
}
