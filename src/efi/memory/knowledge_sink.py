"""
efi/memory/knowledge_sink.py

Переходник между «эпизод прожит» и конвейером приёма знаний.

Консолидация (efi/memory/consolidation.py) знает только то, что эпизод
закончился, и умеет отдать его текстом. Конвейер (efi/memory/ingest.py) знает,
как провести кандидатов через границу доверия, но не знает, какие люди вообще
существуют — каталог сущностей ему обязана дать вызывающая сторона.

Склеить их напрямую в `EfiApp` было бы можно, но тогда сборка каталога
(«кого Эфи может иметь в виду в ЭТОМ чате») переехала бы в точку сборки
приложения — то есть туда, где её никто не протестирует, а сам вызов
превратился бы в лямбду с захваченными зависимостями. Здесь же это обычный
класс с двумя зависимостями и одним методом.

Почему каталог собирается на каждый эпизод, а не один раз при старте: люди
появляются по ходу дела, и кэш «кто существует» через сутки работы означал бы,
что новых собеседников разрешение упоминаний не видит вовсе. Эпизоды случаются
раз в десять минут в лучшем случае — один локальный SQL-запрос на эпизод здесь
не стоит ничего.
"""

from __future__ import annotations

import logging

from efi.memory.catalog import apply_confirmed_answers, build_people_catalog
from efi.memory.ingest import IngestResult, MemoryIngestor
from efi.memory.people import PeopleStore

logger = logging.getLogger(__name__)

#: Сколько известных людей попадает в каталог разрешения упоминаний.
#: Не «все»: каталог нужен, чтобы отличить Рому от Ромы, а не чтобы держать
#: в памяти всю адресную книгу. Полсотни свежих собеседников покрывают
#: практически любой разговор, а те, о ком речь не заходила месяцами, скорее
#: внесут шум в уточняющие вопросы, чем помогут.
_CATALOG_PEOPLE_LIMIT = 50


class EpisodeKnowledgeSink:
    """
    Реализация `consolidation.KnowledgeSink`: прожитый эпизод -> строгая память.

    Ничего не решает сама — только собирает каталог сущностей для этого чата
    и передаёт эпизод конвейеру. Вся логика доверия остаётся там, где ей и
    место.
    """

    def __init__(self, ingestor: MemoryIngestor, people: PeopleStore) -> None:
        self._ingestor = ingestor
        self._people = people

    async def ingest_episode(self, episode_text: str, *, chat_id: int | None = None) -> IngestResult:
        catalog = build_people_catalog(
            await self._people.recent(limit=_CATALOG_PEOPLE_LIMIT), chat_id=chat_id
        )
        if chat_id is not None:
            catalog = apply_confirmed_answers(
                catalog, self._ingestor.pending_clarifications.confirmed_entities(chat_id)
            )
        result = await self._ingestor.ingest_conversation(
            episode_text, source=_source_label(chat_id), chat_id=chat_id, catalog=catalog
        )
        if result.needs_clarification:
            # Не ошибка: конвейер сознательно не записал факт и попросил
            # уточнение. Вопрос дойдёт до собеседника через системный промпт
            # (см. efi/prompts/builder.py) — здесь его только видно в логе.
            logger.info(
                "knowledge: по chat_id=%s нужно уточнение — %s",
                chat_id, "; ".join(result.clarifications),
            )
        return result


def _source_label(chat_id: int | None) -> str:
    """Метка источника в knowledge_facts/knowledge_rejections — по ней потом видно, откуда факт."""
    return f"episode:chat:{chat_id}" if chat_id is not None else "episode"


__all__ = ["EpisodeKnowledgeSink"]
