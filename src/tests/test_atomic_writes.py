"""
Тесты записи, которая переживает выключение телефона.

На сервере обрыв посреди записи — событие раз в год. На телефоне это норма:
Android убивает фоновые процессы, батарея садится ночью, Termux закрывается
вместе с приложением. Цена обрыва здесь не абстрактная — в рабочей памяти
лежат её состояние и НЕЗАКРЫТЫЕ ОБЕЩАНИЯ, то есть то, что человек ей сказал
и чего ждёт.

Поэтому проверяется главное свойство: по целевому пути в любой момент лежит
либо старое содержимое целиком, либо новое целиком.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from efi.memory.diary import Diary
from efi.memory.working_memory import WorkingMemory
from efi.utils.atomic import write_text_atomic


async def test_the_file_is_replaced_whole(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    await write_text_atomic(target, '{"было": 1}')

    await write_text_atomic(target, '{"стало": 2}')

    assert target.read_text(encoding="utf-8") == '{"стало": 2}'


async def test_no_debris_is_left_behind(tmp_path: Path) -> None:
    """Временный файл — деталь реализации, и оставаться на диске он не должен."""
    target = tmp_path / "state.json"

    await write_text_atomic(target, "содержимое")

    assert os.listdir(tmp_path) == ["state.json"]


async def test_a_failed_write_keeps_the_old_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Не записалось — значит, на диске осталось прежнее. Это и есть то, ради
    чего всё затевалось: половины файла не бывает.

    Беда имитируется отказом самой подмены: права каталога тут не годятся,
    потому что в Termux (и в CI) процесс нередко идёт от root, а ему запрет
    на запись не писан.
    """
    target = tmp_path / "state.json"
    await write_text_atomic(target, "старое")

    async def disk_is_full(*_args: object, **_kwargs: object) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("efi.utils.atomic.aiofiles.os.replace", disk_is_full)

    with pytest.raises(OSError):
        await write_text_atomic(target, "новое")

    assert target.read_text(encoding="utf-8") == "старое"
    # И временный файл не остался лежать мусором рядом.
    assert os.listdir(tmp_path) == ["state.json"]


async def test_working_memory_survives_being_rewritten(tmp_path: Path) -> None:
    """Открытые обещания обязаны пережить перезапуск — они уже даны человеку."""
    memory = WorkingMemory(tmp_path / "wm.json")
    await memory.add_item("скинуть ссылку на трек", chat_id=42)

    reread = await WorkingMemory(tmp_path / "wm.json").load()

    assert [item.text for item in reread.items] == ["скинуть ссылку на трек"]
    assert not any(name.startswith(".") for name in os.listdir(tmp_path))


async def test_a_diary_entry_lands_whole(tmp_path: Path) -> None:
    from efi.llm.schemas import DiaryEntry, DiaryEntryMetadata

    diary = Diary(tmp_path / "diary")
    await diary.save(
        DiaryEntry(id="e1", metadata=DiaryEntryMetadata(confidence=0.5), body="Прожитый вечер.")
    )

    entries = await Diary(tmp_path / "diary").all_entries()

    assert [entry.body for entry in entries] == ["Прожитый вечер."]
