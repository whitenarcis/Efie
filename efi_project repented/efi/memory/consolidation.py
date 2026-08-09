"""
efi/memory/consolidation.py

Консолидация памяти — три независимые операции.

ВАЖНО про расписание: раньше все три были строго ночными, и `novelize_*`
в том числе — то есть день переписки становился памятью только в 03:30, а
падение или перезапуск до этого момента стирали его целиком. Теперь
новеллизация вызывается ещё и по ходу дня, эпизодами, из
efi.memory.pulse.MemoryPulse (см. `novelize_chat` — единица работы, общая
для обоих путей). Ночной проход остался как подбирающий хвосты плюс
собственно обслуживание корпуса (dedup/мемуары).

1. `deduplicate()` — убирает дубли/почти-дубли уже существующих записей
   дневника (тот же принцип, что и diaryPlagiarismThreshold при сохранении
   новой записи, но применяется сплошным проходом по всему корпусу).
2. `summarize_stale_entries()` — сворачивает старые записи в более общую
   "мемуарную" запись через LLM, снижая объём дневника без потери смысла.
3. `novelize_recent_history()` — САМАЯ ВАЖНАЯ: без неё дневник никогда не
   пополняется сам по себе. Разбирает недавнюю переписку по всем активным
   чатам и просит LLM выделить то, что реально стоит запомнить надолго
   (факты, события, договорённости), сохраняя каждое как отдельную запись
   через RAGMemory.remember(). Прямой аналог sleepingConsolidation из
   референса — единственный источник ДОЛГОСРОЧНОЙ памяти помимо явного
   вызова remember_diary_entry самой моделью посреди разговора
   (efi/tools/memory_tools/remember_diary_entry.py — для того, что стоит
   запомнить прямо сейчас, не дожидаясь ночи).

Все три операции программные, не диалоговые — идут НЕ через
NotificationManager/Worker. LLM используется точечно (сжатие/извлечение
текста), а не для полноценного разговорного ответа.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime, timedelta
from typing import Protocol

from efi.config.schema import TaskRole
from efi.llm.errors import LLMError
from efi.llm.router import LLMRouter
from efi.llm.schemas import DiaryEntry, DiaryEntryMetadata, LLMParams, Message, Role, Session
from efi.memory.diary import Diary
from efi.memory.facts import FactStore
from efi.memory.rag import RAGMemory

logger = logging.getLogger(__name__)

#: Для каждой записи при dedup-проходе проверяем несколько ближайших
#: соседей, а не только первого — иначе легко пропустить дубль, который
#: оказался вторым по релевантности из-за шума эмбеддинга.
_DEDUP_NEIGHBOR_CHECK_COUNT = 3

_CONSOLIDATION_SYSTEM_PROMPT = (
    "Ты помогаешь сжать несколько отдельных дневниковых записей в одну связную "
    "заметку, сохраняя все важные факты и детали, но убирая повторы и лишние слова. "
    "Пиши от первого лица, коротко и по существу. Не придумывай ничего нового."
)

_NOVELIZATION_SYSTEM_PROMPT = (
    "Тебе показан кусок твоей жизни за последнее время: переписка, а иногда ещё и то, что ты "
    "делала параллельно — что гуглила, где комментировала, кому отвечала. Это ОДИН непрерывный "
    "опыт одного человека, а не отдельные логи разных систем: если ты полезла что-то искать "
    "посреди разговора, это часть того же эпизода, и вспоминать надо вместе.\n"
    "Твоя задача — записать это в личный дневник так, чтобы через месяц ты по записи могла "
    "восстановить не только факт, но и суть, детали и своё отношение.\n"
    "\n"
    "КАЖДОЕ ВОСПОМИНАНИЕ ПИШИ В ТРИ СЛОЯ, слитно, одним абзацем без заголовков:\n"
    "1. ЧТО БЫЛО — конкретно и подробно. Имена, названия, числа, версии, ссылки, точные "
    "формулировки договорённостей переноси ДОСЛОВНО, а не пересказом ('какая-то библиотека', "
    "'что-то про деньги' — это потерянная память). Если прозвучала меткая или обидная фраза — "
    "процитируй её.\n"
    "2. ЧТО Я ПОЧУВСТВОВАЛА — твоя субъективная реакция: зацепило, взбесило, стало смешно, "
    "стыдно, приятно. Отношение может быть неоднозначным (раздражаться и признавать правоту "
    "одновременно) — так и пиши, живая память противоречива.\n"
    "3. ЧТО ЭТО ЗНАЧИТ — вывод на будущее: что ты теперь думаешь об этом человеке или теме, "
    "чего ждёшь дальше, о чём надо не забыть спросить, что изменилось в твоём мнении.\n"
    "\n"
    "ОБЯЗАТЕЛЬНО: строго от первого лица, как будто вспоминаешь ты сама. Всегда называй, С КЕМ "
    "это было — перед каждой репликой указано имя написавшего, используй эти имена ('с Ромой', "
    "'Рихтер опять...'), а не безличное 'собеседник'.\n"
    "\n"
    "Пример ПЛОХОГО воспоминания (протокольное, безличное, без деталей): 'Обсудили баг в коде, "
    "договорились исправить позже.'\n"
    "Пример ХОРОШЕГО: 'Рома притащил трейс, где падает our_parser на пустом front-matter — "
    "оказалось, я же сама это место и правила в прошлый раз, и стало неловко. Полезла проверить, "
    "как это делают в python-frontmatter: там просто возвращают пустой dict вместо исключения, и "
    "это честно логичнее моего варианта. Договорились, что он не трогает, я переделаю к выходным. "
    "Бесит, что он заметил раньше меня, но придирка по делу — и, кажется, он вообще смотрит в код "
    "внимательнее, чем показывает.'\n"
    "\n"
    "Разделяй отдельные воспоминания строкой из трёх дефисов (---) на отдельной строке. Лучше "
    "несколько отдельных записей про разные темы, чем одна свалка. Не бойся писать подробно: "
    "потерянная сейчас деталь не восстановится никогда.\n"
    "Игнорируй только совсем пустое: чистую фатику ('привет', 'ок', 'спокойной ночи') и "
    "техническую рутину без смысла. Если запоминать реально нечего — ответь ровно одним словом: ПУСТО."
)

#: Заголовок блока «а ещё параллельно со мной было вот что» в промпте
#: новеллизации — внешний опыт (гуглёж, комментарии), который в таблицу
#: `messages` не попадает вообще (см. efi/memory/social_memory.py).
_EXPERIENCE_BLOCK_HEADER = "Параллельно с этим разговором ты делала вот что:"

#: Тот же блок, когда разговора не было вовсе — в канале сообщества Эфи
#: иногда только комментирует и читает треды.
_EXPERIENCE_ONLY_HEADER = "Разговора как такового не было, но вот что ты делала за это время:"

_NOVELIZATION_EMPTY_MARKER = "ПУСТО"
_ENTRY_SPLIT_RE = re.compile(r"\n\s*-{3,}\s*\n")

#: Дефолты извлечения памяти из переписки (novelize_recent_history) —
#: намеренно щедрые. Раньше здесь стояло 2000 символов и 768 токенов вывода:
#: этого хватало на пару часов переписки, а за целый активный день
#: разговор обрубался почти сразу, и LLM видела только начало дня — отсюда
#: жалоба "дневник за весь день почему-то обрезанный". Сжатие — отдельная,
#: НАМЕРЕННО более скупая операция (см. summarize_stale_entries), которая
#: срабатывает только спустя `older_than` (по умолчанию 30 дней) над уже
#: сохранёнными записями; здесь же, на этапе первого извлечения, экономить
#: не на чем — потерянная на этом шаге деталь не восстановится никогда.
_DEFAULT_NOVELIZATION_CHAR_LIMIT = 10_000
_DEFAULT_NOVELIZATION_MAX_OUTPUT_TOKENS = 2048


class HistorySource(Protocol):
    """
    Абстракция источника истории для новеллизации — минимум, который нужен
    отсюда (не полный efi.notifications.worker.HistoryRepository, чтобы не
    тянуть зависимость memory -> notifications ради одного протокола).
    Конкретная реализация — efi.db.history_repository.SqliteHistoryRepository.
    """

    async def get_active_chat_ids(self, *, since: datetime) -> list[int]: ...

    async def get_since(self, chat_id: int, *, since: datetime) -> Session: ...


class ExperienceSource(Protocol):
    """
    Внешний опыт, привязанный к чату, но НЕ лежащий в истории сообщений:
    что Эфи гуглила по ходу разговора, где комментировала, кому отвечала.
    Конкретная реализация — efi.memory.social_memory.SocialInteractionStore.

    Без этого источника новеллизация видела бы только реплики и считала бы,
    что между ними Эфи ничего не делала — а именно там живёт половина
    её опыта (TOOL-сообщения в таблицу `messages` не пишутся, см.
    efi/notifications/worker.py).
    """

    async def context_lines_for_chat(self, chat_id: int, *, since: datetime, limit: int = 30) -> list[str]: ...


class DiaryConsolidator:
    """Программная консолидация и пополнение дневника: dedup, сжатие старых записей, автоматическое извлечение новых."""

    def __init__(
        self,
        diary: Diary,
        router: LLMRouter,
        rag: RAGMemory,
        *,
        summarization_role: TaskRole = TaskRole.BACKGROUND,
        novelization_char_limit: int = _DEFAULT_NOVELIZATION_CHAR_LIMIT,
        novelization_max_output_tokens: int = _DEFAULT_NOVELIZATION_MAX_OUTPUT_TOKENS,
    ) -> None:
        self._diary = diary
        self._router = router
        self._novelization_char_limit = novelization_char_limit
        self._novelization_max_output_tokens = novelization_max_output_tokens
        self._rag = rag
        self._summarization_role = summarization_role

    async def deduplicate(self, *, plagiarism_threshold: float) -> int:
        """
        Проходит по всем записям с эмбеддингами и убирает почти-дубли
        (relatedness выше `plagiarism_threshold`), оставляя в каждой паре
        запись с более высоким confidence (при равенстве — с большим
        usage_count). Возвращает число удалённых записей.

        Заметка про сложность: наивный обход подходит для масштаба одного
        персонажа (десятки-сотни записей, не миллионы) — то же допущение,
        что и у остального RAG-слоя проекта (см. memory/diary.py::_score_entries).
        """
        entries = await self._diary.all_entries()
        embedded_entries = [entry for entry in entries if entry.metadata.embedding]
        removed_ids: set[str] = set()

        for entry in embedded_entries:
            if entry.id in removed_ids:
                continue

            def _not_self(candidate: DiaryEntry, current_id: str = entry.id) -> bool:
                return candidate.id != current_id

            duplicates = await self._diary.query(entry.metadata.embedding, filter_fn=_not_self)
            for duplicate in duplicates[:_DEDUP_NEIGHBOR_CHECK_COUNT]:
                if duplicate.relatedness < plagiarism_threshold:
                    continue
                if duplicate.entry.id in removed_ids or entry.id in removed_ids:
                    continue
                loser_id = _pick_duplicate_to_remove(entry, duplicate.entry)
                removed_ids.add(loser_id)
                logger.info(
                    "consolidation: dropping duplicate entry %s (relatedness=%.3f with %s)",
                    loser_id, duplicate.relatedness,
                    duplicate.entry.id if loser_id == entry.id else entry.id,
                )

        for entry_id in removed_ids:
            await self._diary.delete(entry_id)

        return len(removed_ids)

    async def summarize_stale_entries(
        self,
        *,
        older_than: timedelta = timedelta(days=30),
        batch_size: int = 10,
    ) -> DiaryEntry | None:
        """
        Берёт до `batch_size` самых старых записей, которым больше
        `older_than` и которые ещё не являются подтверждённым фактом
        (confidence < 1.0 — ground truth не трогаем: она не должна
        "размываться" пересказом), просит LLM сжать их в одну заметку и
        сохраняет результат как новую запись с confidence, усреднённым по
        исходным. Исходные записи после этого удаляются — их смысл теперь
        живёт в сводной записи.

        Возвращает None, если подходящих записей меньше двух (сжимать нечего)
        или если LLM-запрос не удался (в этом случае ничего не удаляется —
        лучше оставить дневник как есть, чем потерять записи без сводки).
        """
        entries = await self._diary.all_entries()
        cutoff = datetime.now(UTC) - older_than
        # ВАЖНО: критерий устаревания — created_at (когда запись реально
        # появилась), а НЕ last_used. Раньше здесь читался last_used с
        # фолбэком на "максимально старую дату" для записей, которые ещё ни
        # разу не искали (last_used is None) — а это ЛЮБАЯ только что
        # созданная запись, включая те, что novelize_recent_history сохранил
        # в дневник минутами раньше В ЭТОМ ЖЕ ночном проходе (см.
        # efi/app.py::_run_consolidation_loop, novelize идёт первым шагом).
        # На практике это означало, что весь день переписки мог в ту же
        # ночь схлопнуться в один сжатый "мемуар" — свежие записи выглядели
        # как самые старые кандидаты на сжатие.
        candidates = [
            entry for entry in entries if not entry.metadata.is_ground_truth and entry.metadata.created_at < cutoff
        ]
        if len(candidates) < 2:
            return None

        batch = sorted(candidates, key=lambda entry: entry.metadata.created_at)[:batch_size]
        summary_text = await self._summarize_via_llm(batch)
        if summary_text is None:
            return None

        average_confidence = sum(entry.metadata.confidence for entry in batch) / len(batch)
        merged_entry = DiaryEntry(
            id=f"memoir_{int(datetime.now(UTC).timestamp())}",
            metadata=DiaryEntryMetadata(confidence=average_confidence),
            body=summary_text,
        )
        await self._diary.save(merged_entry)

        for entry in batch:
            await self._diary.delete(entry.id)

        logger.info("consolidation: merged %d stale entries into %s", len(batch), merged_entry.id)
        return merged_entry

    async def novelize_chat(
        self,
        chat_id: int,
        *,
        history: HistorySource,
        facts: FactStore,
        since: datetime,
        min_messages: int,
        experience: ExperienceSource | None = None,
    ) -> int:
        """
        Новеллизация ОДНОГО чата за период `since..сейчас` — единица работы,
        общая для частого пульса памяти (efi/memory/pulse.py) и ночного
        прохода (novelize_recent_history).

        Переданный `experience` подмешивает в тот же LLM-запрос внешний опыт
        этого чата за тот же период (гуглёж, комментарии) — ровно затем,
        чтобы эпизод осмыслялся как один прожитый кусок жизни, а не как
        переписка отдельно и действия отдельно.

        Порог `min_messages` считается по СУММЕ реплик и внешних событий, а
        не по одним репликам. Это не мелочь: в канале сообщества Эфи может за
        период не написать ни одной реплики в привычном смысле, а оставить
        два комментария и прочитать тред — по счёту сообщений это "пусто",
        хотя прожито там больше, чем в ином разговоре.

        Отметку "докуда уже новеллизировано" двигает ТОЛЬКО при реальной
        попытке разбора: если материала меньше порога, окно не закрывается и
        следующий заход увидит его снова уже вместе с продолжением — иначе
        короткие эпизоды выпадали бы из памяти навсегда просто потому, что
        пульс заглянул слишком рано.

        Возвращает число новых записей, реально сохранённых в дневник
        (дубли, отбракованные RAGMemory.remember(), в счёт не идут).
        """
        session = await history.get_since(chat_id, since=since)
        experience_lines: list[str] = []
        if experience is not None:
            experience_lines = await experience.context_lines_for_chat(chat_id, since=since)
        if len(session.messages) + len(experience_lines) < min_messages:
            return 0

        memories = await self._extract_memories(session, experience_lines)
        saved = 0
        for memory_text in memories:
            entry = await self._rag.remember(memory_text, confidence=0.5)
            if entry is not None:
                saved += 1

        await facts.upsert(f"chat:{chat_id}", "last_novelized_at", datetime.now(UTC).isoformat())
        if memories:
            logger.info(
                "consolidation: novelized chat_id=%s -> %d candidate memories (%d actually saved, %d external events)",
                chat_id, len(memories), saved, len(experience_lines),
            )
        return saved

    async def novelize_recent_history(
        self,
        *,
        history: HistorySource,
        facts: FactStore,
        lookback: timedelta = timedelta(days=1),
        min_messages: int = 6,
        chat_lookback_ceiling: timedelta = timedelta(days=30),
        experience: ExperienceSource | None = None,
    ) -> int:
        """
        Автоматическое пополнение дневника из недавней переписки — без этого
        шага Diary остаётся пустым до тех пор, пока модель сама явно не
        воспользуется remember_diary_entry, а в быстрой переписке это
        происходит редко (см. докстринг модуля). Прямой аналог
        sleepingConsolidation у референса.

        Отслеживает "докуда уже новеллизировано" по каждому чату через
        FactStore (entity_id=f"chat:{chat_id}", key="last_novelized_at") —
        чтобы не пересказывать одно и то же на каждый ночной проход. Для
        чата без отметки (первый раз) смотрит на последние `lookback`
        (по умолчанию сутки).

        `chat_lookback_ceiling` ограничивает, какие чаты вообще считаются
        "активными" для обхода — не пытаемся новеллизировать чат, где
        последнее сообщение было полгода назад.

        Возвращает число новых записей, реально сохранённых в дневник
        (дубли, отбракованные RAGMemory.remember(), в счёт не идут).
        """
        chat_ids = await history.get_active_chat_ids(since=datetime.now(UTC) - chat_lookback_ceiling)
        created_count = 0

        for chat_id in chat_ids:
            since = await self.resolve_last_novelized_at(facts, chat_id, lookback)
            created_count += await self.novelize_chat(
                chat_id,
                history=history,
                facts=facts,
                since=since,
                min_messages=min_messages,
                experience=experience,
            )

        return created_count

    async def resolve_last_novelized_at(self, facts: FactStore, chat_id: int, lookback: timedelta) -> datetime:
        raw = await facts.get(f"chat:{chat_id}", "last_novelized_at")
        if raw is None:
            return datetime.now(UTC) - lookback
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            logger.warning(
                "consolidation: unparseable last_novelized_at for chat_id=%s (%r), falling back to lookback",
                chat_id, raw,
            )
            return datetime.now(UTC) - lookback

    async def _extract_memories(self, session: Session, experience_lines: list[str] | None = None) -> list[str]:
        blocks: list[str] = []
        conversation_text = _render_conversation(session, char_limit=self._novelization_char_limit)
        if conversation_text:
            blocks.append(conversation_text)
        if experience_lines:
            # Внешний опыт может быть и ЕДИНСТВЕННЫМ содержимым эпизода: в
            # канале сообщества Эфи иногда только комментирует и читает
            # треды, не ведя разговора как такового.
            header = _EXPERIENCE_BLOCK_HEADER if conversation_text else _EXPERIENCE_ONLY_HEADER
            blocks.append(header + "\n" + "\n".join(f"- {line}" for line in experience_lines))
        if not blocks:
            return []
        conversation_text = "\n\n".join(blocks)

        params = LLMParams(
            model="",
            system_prompt=_NOVELIZATION_SYSTEM_PROMPT,
            max_output_tokens=self._novelization_max_output_tokens,
        )
        prompt_session = Session(messages=[Message(role=Role.USER, content=conversation_text)])
        try:
            response = await self._router.chat(self._summarization_role, params, prompt_session)
        except LLMError as exc:
            logger.warning("consolidation: novelization request failed: %s", exc)
            return []

        text = response.text.strip()
        if not text or text.strip().upper() == _NOVELIZATION_EMPTY_MARKER:
            return []

        pieces = [piece.strip() for piece in _ENTRY_SPLIT_RE.split(text)]
        return [piece for piece in pieces if piece and piece.upper() != _NOVELIZATION_EMPTY_MARKER]

    async def _summarize_via_llm(self, entries: list[DiaryEntry]) -> str | None:
        bodies = "\n\n".join(f"- {entry.body.strip()}" for entry in entries)
        session = Session(messages=[Message(role=Role.USER, content=bodies)])
        # 512 токенов оказалось мало для батча из batch_size=10 записей —
        # сводка обрывалась на середине предложения (см. пример в дневнике:
        # "...вернулся с" без продолжения). 1024 даёт запас без риска, что
        # обрезание повторится на чуть более многословном батче.
        params = LLMParams(model="", system_prompt=_CONSOLIDATION_SYSTEM_PROMPT, max_output_tokens=1024)
        try:
            response = await self._router.chat(self._summarization_role, params, session)
        except LLMError as exc:
            logger.warning("consolidation: summarization request failed: %s", exc)
            return None
        return response.text.strip() or None


def _render_conversation(session: Session, *, char_limit: int) -> str:
    """
    Плоский текст переписки для промпта новеллизации — только реплики с
    содержимым (не голые tool-calls).

    При переполнении лимита обрезается НАЧАЛО, а не конец. Раньше было
    наоборот (`text[:char_limit]`), и на длинном окне это давало ровно ту
    потерю, которой новеллизация должна мешать: сохранялось утро, а вечер —
    свежая, ещё ни разу не осмысленная часть разговора — выпадал, и на
    следующем проходе он уже был за отметкой last_novelized_at, то есть
    терялся навсегда.
    """
    lines = [f"{message.role.value}: {message.content}" for message in session if message.content.strip()]
    text = "\n".join(lines)
    if len(text) <= char_limit:
        return text
    return "[...начало разговора опущено...]\n" + text[-char_limit:]


def _pick_duplicate_to_remove(a: DiaryEntry, b: DiaryEntry) -> str:
    """Из пары почти-дублей выбирает, какую запись убрать: ниже confidence, при равенстве — меньше usage_count."""
    if a.metadata.confidence != b.metadata.confidence:
        return a.id if a.metadata.confidence < b.metadata.confidence else b.id
    return a.id if a.metadata.usage_count <= b.metadata.usage_count else b.id


__all__ = ["DiaryConsolidator", "ExperienceSource", "HistorySource"]
