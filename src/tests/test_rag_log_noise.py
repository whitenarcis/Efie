"""
Тест на шум в логе от подсчёта эмбеддингов.

Эмбеддинг считается на КАЖДУЮ сборку промпта, то есть на каждое сообщение, а
обе причины отказа стабильны: «fastembed не установлен» не изменится никогда,
а недоступный провайдер держится в cooldown минутами. Пока каждая попытка
писала WARNING, лог за вечер состоял из одной и той же пары строк — и
настоящая ошибка в нём просто не была видна.

Лог здесь читают ровно тогда, когда что-то сломалось, так что цена шума не
эстетическая: он прячет то, ради чего в лог и полезли.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from efi.config.schema import TaskRole
from efi.llm.errors import LLMError
from efi.llm.schemas import EmbeddingVector
from efi.memory.diary import Diary
from efi.memory.rag import RAGMemory
from efi.memory.tfidf_fallback import TfidfFallbackIndex

#: Достаточно длинный текст, чтобы не сработала TF-IDF-срезка для коротышей.
_LONG_ENOUGH = "расскажи, что нового по тому проекту с разбором логов, который мы обсуждали"


class _DeadRouter:
    """Провайдер, до которого не достучаться, — обычное дело на бесплатном тире."""

    def __init__(self) -> None:
        self.calls = 0

    async def embedding(self, role: TaskRole, text: str) -> EmbeddingVector:
        self.calls += 1
        raise LLMError("transport error: 403 Forbidden")


class _MissingLocalEngine:
    """Локальный движок, который не установлен. Причина неустранимая и вечная."""

    async def embed_query(self, text: str) -> EmbeddingVector:
        raise RuntimeError("fastembed не установлен")

    async def embed_document(self, text: str) -> EmbeddingVector:
        raise RuntimeError("fastembed не установлен")


def _rag(tmp_path: Path, *, local: object | None = None) -> tuple[RAGMemory, _DeadRouter]:
    router = _DeadRouter()
    memory = RAGMemory(
        Diary(tmp_path / "diary"),
        router,  # type: ignore[arg-type]
        TfidfFallbackIndex(),
        local_embeddings=local,  # type: ignore[arg-type]
    )
    return memory, router


def _rag_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING and record.name.startswith("efi.memory.rag")
    ]


async def test_the_same_trouble_is_reported_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Главное: одна причина — одна строчка, сколько бы сообщений ни пришло."""
    memory, router = _rag(tmp_path)

    with caplog.at_level(logging.DEBUG):
        for _ in range(5):
            await memory._compute_embedding(_LONG_ENOUGH, is_query=True)

    assert router.calls == 5, "жаловаться реже — не значит перестать пытаться"
    assert len(_rag_warnings(caplog)) == 1


async def test_a_new_trouble_is_still_reported(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """
    Обратная сторона: приглушать нужно повтор, а не сам факт. Сменилась
    причина — значит, изменилось состояние мира, и это стоит сказать.
    """
    memory, _router = _rag(tmp_path, local=_MissingLocalEngine())

    with caplog.at_level(logging.DEBUG):
        await memory._compute_embedding(_LONG_ENOUGH, is_query=True)

    warnings = _rag_warnings(caplog)
    assert len(warnings) == 2, "локальный движок и облако — две разные беды"
    assert any("fastembed" in line for line in warnings)
    assert any("403" in line for line in warnings)


async def test_muting_the_log_does_not_mute_the_failure(tmp_path: Path) -> None:
    """
    Отказ по-прежнему возвращается вызывающей стороне как None — иначе она
    решила бы, что эмбеддинг посчитан, и записала бы в память пустоту.
    """
    memory, _router = _rag(tmp_path)

    assert await memory._compute_embedding(_LONG_ENOUGH, is_query=True) is None
    assert await memory._compute_embedding(_LONG_ENOUGH, is_query=True) is None
