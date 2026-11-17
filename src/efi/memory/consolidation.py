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
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from efi.config.schema import TaskRole
from efi.llm.errors import LLMError, LLMTimeoutError
from efi.llm.router import LLMRouter
from efi.llm.schemas import DiaryEntry, DiaryEntryMetadata, LLMParams, Message, Role, Session
from efi.memory.diary import Diary
from efi.memory.facts import FactStore
from efi.memory.rag import RAGMemory
from efi.memory.transcript import SELF_MARKER as _SELF_MARKER
from efi.memory.transcript import render_transcript
from efi.utils.text import salvage_truncated

logger = logging.getLogger(__name__)

#: Для каждой записи при dedup-проходе проверяем несколько ближайших
#: соседей, а не только первого — иначе легко пропустить дубль, который
#: оказался вторым по релевантности из-за шума эмбеддинга.
_DEDUP_NEIGHBOR_CHECK_COUNT = 3

_CONSOLIDATION_SYSTEM_PROMPT = (
    "Ты сжимаешь несколько дневниковых записей в одну связную заметку. "
    "Главная цель — сохранить всё ВАЖНОЕ, убрав повторы и лишние слова.\n"
    "\n"
    "ВАЖНОЕ — это:\n"
    "• Факты о людях: имена, занятия, вкусы, обстоятельства\n"
    "• Договорённости: сроки, обещания, планы\n"
    "• Эмоциональные моменты: что зацепило, обидело, обрадовало\n"
    "• Конфликты и их разрешение\n"
    "\n"
    "НЕ ВАЖНОЕ (можно опустить):\n"
    "• Рутина без нового смысла\n"
    "• Повторы одного и того же в разных формулировках\n"
    "• Технические детали без отношения к людям\n"
    "\n"
    "Пиши от первого лица, коротко и по существу. Не придумывай ничего нового. "
    "Если в записях есть конкретные имена, числа, даты — сохрани их дословно."
)

_NOVELIZATION_SYSTEM_PROMPT = (
    "Тебе показан кусок твоей жизни за последнее время: переписка, а иногда ещё и то, что ты "
    "делала параллельно — что гуглила, где комментировала, кому отвечала. Это ОДИН непрерывный "
    "опыт одного человека, а не отдельные логи разных систем: если ты полезла что-то искать "
    "посреди разговора, это часть того же эпизода, и вспоминать надо вместе.\n"
    "Твоя задача — записать в дневник ТОЛЬКО то, что действительно стоит помнить. Не каждое "
    "сообщение, не каждую мысль — память конечна, и забивать её мусором значит терять важное.\n"
    "\n"
    "ЧТО СТОИТ ЗАПОМНИТЬ:\n"
    "• Факты о человеке: имя, чем занимается, вкусы, привычки, важные обстоятельства жизни.\n"
    "• Договорённости: сроки, обещания, планы, к чему пришли в споре.\n"
    "• Сильные эмоции: что зацепило, обидело, обрадовало, взбесило — твоё или собеседника.\n"
    "• Неожиданное: человек сказал что-то, что изменило твоё мнение о нём или теме.\n"
    "• Конфликты и их разрешение: о чём спорили, кто прав, к чему пришли.\n"
    "\n"
    "ЧТО НЕ СТОИТ ЗАПОМИНАТЬ:\n"
    "• Рутина без нового смысла: 'опять работал до трёх', 'как обычно пообедал'.\n"
    "• Чистая фатика: 'привет', 'ок', 'спасибо', 'спокойной ночи' — без контекста.\n"
    "• Повторы: если ты уже записывала про это раньше, не пиши снова — просто отметь, что было.\n"
    "• Случайные мелочи без связи с человеком: 'интересная статья' — не записывай, если она не "
    "меняет отношения или не содержит важного факта.\n"
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
    "КТО ЧТО СКАЗАЛ — проверь это отдельно, ПЕРЕД тем как писать. Реплики с пометкой "
    f"«({_SELF_MARKER})» — твои собственные слова: то, что сказала ТЫ. Все остальные строки "
    "написали другие люди, и их имя стоит в начале строки. Не приписывай себе чужие "
    "мысли, работу и настроение и не отдавай собеседнику свои: перепутанное направление "
    "превращает воспоминание в ложное — через месяц ты будешь уверена, что это ты чинила "
    "тот баг, хотя чинил его он.\n"
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
    "несколько отдельных записей про разные темы, чем одна свалка. Не бойся писать подробно, "
    "НО будь избирательной: меньше записей — острее память. "
    "Если запоминать реально нечего — ответь ровно одним словом: ПУСТО."
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

#: Бюджет вывода на один проход новеллизации. 2048 токенов, стоявшие здесь
#: раньше, — это примерно ОДНА подробная запись по-русски: у токенизаторов
#: бесплатных моделей кириллица стоит в 2-3 раза дороже английского, а промпт
#: просит несколько записей и требует подробностей. Отсюда и брались
#: постоянные обрывы в дневнике: лимит выбирался по английским меркам, а
#: писала модель по-русски.
_DEFAULT_NOVELIZATION_MAX_OUTPUT_TOKENS = 4096

#: Сколько раз просить дописать оборванный текст. Один раз — это ещё столько
#: же токенов сверху; два прохода закрывают любой реальный эпизод, а дальше
#: дело не в лимите, а в том, что модель не умеет останавливаться.
_MAX_CONTINUATION_ROUNDS = 2

#: Ниже этого урезать бюджет бессмысленно: получится не короткая запись, а
#: обрывок фразы, из которого потом нечего вспомнить.
_MIN_NOVELIZATION_TOKENS = 512

#: Сколько последних символов уже написанного показывать при просьбе
#: дописать. Нужен ровно хвост: по нему модель находит место обрыва, а весь
#: текст целиком занял бы контекст, который нужен под продолжение.
_CONTINUATION_TAIL_CHARS = 600


@dataclass(slots=True, frozen=True)
class _Novelization:
    """Ответ новеллизации: текст и признак того, что он оборван по лимиту."""

    body: str
    truncated: bool


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


class KnowledgeSink(Protocol):
    """
    Строгое хранилище знаний с точки зрения консолидации — ровно один вызов.

    Протокол, а не прямой импорт `MemoryIngestor`: консолидации незачем знать
    ни про границу доверия, ни про домены, ни про разрешение сущностей. Она
    умеет одно — сказать «вот прожитый эпизод», и это всё, что между этими
    двумя подсистемами должно быть общего.

    Конкретная реализация — efi.memory.knowledge_sink.EpisodeKnowledgeSink.
    """

    async def ingest_episode(self, episode_text: str, *, chat_id: int | None = None) -> object: ...


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
        character_name: str = "Эфи",
        knowledge: KnowledgeSink | None = None,
    ) -> None:
        self._diary = diary
        #: Своим именем Эфи подписана в плоском тексте переписки — иначе её
        #: собственные реплики неотличимы от чужих (см. efi/memory/transcript.py).
        self._character_name = character_name
        self._router = router
        self._novelization_char_limit = novelization_char_limit
        self._novelization_max_output_tokens = novelization_max_output_tokens
        self._rag = rag
        self._summarization_role = summarization_role
        #: Необязателен: без него консолидация ведёт себя ровно как раньше и
        #: пополняет только дневник. Это не «фича под флагом», а честная
        #: граница — строгое хранилище требует и БД, и эмбеддингов, а тесты
        #: дневника не должны тащить за собой ни то, ни другое.
        self._knowledge = knowledge

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
        importance_threshold: float = 0.6,
    ) -> DiaryEntry | None:
        """
        Берёт до `batch_size` самых старых записей, которым больше
        `older_than` и которые ещё не являются подтверждённым фактом
        (confidence < 1.0 — ground truth не трогаем: она не должна
        "размываться" пересказом), просит LLM сжать их в одну заметку и
        сохраняет результат как новую запись с confidence, усреднённым по
        исходным. Исходные записи после этого удаляются — их смысл теперь
        живёт в сводной записи.

        ВАЖНОЕ ИЗМЕНЕНИЕ: записи с importance >= importance_threshold
        НЕ сжимаются — они сохраняются как есть, даже если старые.
        Важные воспоминания (факты о людях, договорённости, сильные эмоции)
        живут дольше рутины.

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
        #
        # НОВОЕ: исключаем важные записи из сжатия — они заслуживают
        # долгой жизни без пересказа.
        candidates = [
            entry for entry in entries
            if not entry.metadata.is_ground_truth
            and entry.metadata.created_at < cutoff
            and not _is_important(entry, threshold=importance_threshold)
        ]
        if len(candidates) < 2:
            return None

        batch = sorted(candidates, key=lambda entry: entry.metadata.created_at)[:batch_size]
        summary_text = await self._summarize_via_llm(batch)
        if summary_text is None:
            return None

        average_confidence = sum(entry.metadata.confidence for entry in batch) / len(batch)
        # Важность мемуара — среднее по исходным, но не ниже 0.5
        average_importance = max(0.5, sum(_score_importance(e) for e in batch) / len(batch))
        merged_entry = DiaryEntry(
            id=f"memoir_{int(datetime.now(UTC).timestamp())}",
            metadata=DiaryEntryMetadata(confidence=average_confidence, importance=average_importance),
            body=summary_text,
        )
        await self._diary.save(merged_entry)

        for entry in batch:
            await self._diary.delete(entry.id)

        logger.info("consolidation: merged %d stale entries into %s (avg importance=%.2f)", len(batch), merged_entry.id, average_importance)
        return merged_entry

    async def prune_low_importance(
        self,
        *,
        older_than: timedelta = timedelta(days=60),
        importance_threshold: float = 0.35,
        max_removals: int = 20,
    ) -> int:
        """
        Удаляет записи с очень низкой важностью, которые старше `older_than`.

        Это последняя линия обороны от мусора: если запись прожила 2 месяца,
        ни разу не использовалась, не содержит эмоций, фактов или договорённостей —
        она не нужна в памяти.

        НЕ удаляет:
        - Ground truth (confidence=1.0)
        - Записи с importance >= importance_threshold
        - Записи, которые использовались недавно (usage_count > 0 за последние 30 дней)

        Возвращает число удалённых записей.
        """
        entries = await self._diary.all_entries()
        cutoff = datetime.now(UTC) - older_than

        candidates = [
            entry for entry in entries
            if not entry.metadata.is_ground_truth
            and entry.metadata.created_at < cutoff
            and _score_importance(entry) < importance_threshold
            and entry.metadata.usage_count == 0  # никто ни разу не искал
        ]

        # Удаляем самые неважные и старые
        candidates.sort(key=lambda e: (_score_importance(e), e.metadata.created_at))
        to_remove = candidates[:max_removals]

        removed = 0
        for entry in to_remove:
            try:
                await self._diary.delete(entry.id)
                removed += 1
                logger.debug(
                    "consolidation: pruned low-importance entry %s (importance=%.2f, age=%d days)",
                    entry.id, _score_importance(entry),
                    (datetime.now(UTC) - entry.metadata.created_at).days
                )
            except Exception as e:
                logger.warning("consolidation: failed to delete entry %s: %s", entry.id, e)

        if removed:
            logger.info("consolidation: pruned %d low-importance entries (threshold=%.2f)", removed, importance_threshold)
        return removed

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

        Тот же текст эпизода уходит в строгое хранилище знаний, если оно
        передано (`knowledge`): дневник и knowledge_facts — два разных среза
        одного прожитого куска, и разъезжаться им нельзя. Здесь, а не в
        пульсе памяти: `novelize_chat` — единственная точка, общая для
        частого пульса и ночного прохода, и повесив разбор на одну из них,
        мы получили бы память, которая зависит от того, каким путём эпизод
        дошёл до осмысления.

        Возвращает число новых записей, реально сохранённых в дневник
        (дубли, отбракованные RAGMemory.remember(), в счёт не идут).
        Результат разбора знаний в это число не входит: это отдельный срез
        памяти со своим счётом, см. IngestResult.
        """
        session = await history.get_since(chat_id, since=since)
        experience_lines: list[str] = []
        if experience is not None:
            experience_lines = await experience.context_lines_for_chat(chat_id, since=since)
        if len(session.messages) + len(experience_lines) < min_messages:
            return 0

        episode_text = self._compose_episode(session, experience_lines)
        memories = await self._extract_memories(episode_text)
        await self._ingest_knowledge(episode_text, chat_id=chat_id)
        saved = 0
        for memory_text in memories:
            entry = await self._rag.remember(memory_text, confidence=0.5, chat_id=chat_id)
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

    def _compose_episode(self, session: Session, experience_lines: list[str] | None = None) -> str:
        """
        Прожитый кусок одним текстом: переписка плюс то, что Эфи делала
        параллельно. Пустая строка — эпизода не было вовсе.

        Собирается один раз и уходит СРАЗУ В ДВА разбора — дневниковый и
        знаниевый. Собирать дважды значило бы допустить, что дневник и
        knowledge_facts осмысляют слегка разный текст, а расхождение между
        двумя срезами одной памяти потом не отследить ничем.
        """
        blocks: list[str] = []
        conversation_text = render_transcript(
            session, self_name=self._character_name, char_limit=self._novelization_char_limit
        )
        if conversation_text:
            blocks.append(conversation_text)
        if experience_lines:
            # Внешний опыт может быть и ЕДИНСТВЕННЫМ содержимым эпизода: в
            # канале сообщества Эфи иногда только комментирует и читает
            # треды, не ведя разговора как такового.
            header = _EXPERIENCE_BLOCK_HEADER if conversation_text else _EXPERIENCE_ONLY_HEADER
            blocks.append(header + "\n" + "\n".join(f"- {line}" for line in experience_lines))
        return "\n\n".join(blocks)

    async def _ingest_knowledge(self, episode_text: str, *, chat_id: int | None) -> None:
        """
        Второй разбор того же эпизода — в строгое хранилище знаний.

        Сбой здесь не должен отменять дневник: это два независимых среза
        памяти, и потерять оба из-за проблем в одном — хуже, чем потерять
        один. Поэтому исключение только логируется.
        """
        if self._knowledge is None or not episode_text:
            return
        try:
            await self._knowledge.ingest_episode(episode_text, chat_id=chat_id)
        except Exception:
            logger.warning(
                "consolidation: разбор знаний для chat_id=%s не удался, дневник это не отменяет",
                chat_id, exc_info=True,
            )

    async def _extract_memories(self, conversation_text: str) -> list[str]:
        """
        Воспоминания за эпизод. Если ответ упёрся в лимит — просит ДОПИСАТЬ,
        а не обрезает.

        Почему дописать. Промпт требует подробностей («потерянная деталь не
        восстановится никогда») и нескольких записей за проход, поэтому
        обрыв по лимиту — не исключительная ситуация, а норма на активном
        дне. Раньше единственным лечением было отрезание хвоста по последней
        законченной фразе: дневник наполнялся записями, обрывающимися на
        полумысли, — ровно то, чего этот код должен был не допустить.
        Продолжение стоит одного фонового запроса и возвращает потерянное
        целиком.
        """
        if not conversation_text:
            return []

        text = await self._novelize(conversation_text)
        if text is None:
            return []

        rounds = 0
        while text.truncated and rounds < _MAX_CONTINUATION_ROUNDS:
            rounds += 1
            logger.info(
                "consolidation: новеллизация упёрлась в лимит (%s токенов), прошу дописать (%d/%d)",
                self._novelization_max_output_tokens, rounds, _MAX_CONTINUATION_ROUNDS,
            )
            continuation = await self._novelize(conversation_text, written_so_far=text.body)
            if continuation is None or not continuation.body:
                break
            text = _Novelization(body=_join_continuation(text.body, continuation.body),
                                 truncated=continuation.truncated)

        body = text.body.strip()
        if not body or body.upper() == _NOVELIZATION_EMPTY_MARKER:
            return []

        pieces = [piece.strip() for piece in _ENTRY_SPLIT_RE.split(body)]
        pieces = [piece for piece in pieces if piece and piece.upper() != _NOVELIZATION_EMPTY_MARKER]

        # Дописать не удалось (лимит держится или продолжение не пришло).
        # Тогда — как раньше: обрыв бьёт только по ПОСЛЕДНЕЙ записи, всё
        # перед разделителем модель успела закончить, и выбрасывать весь
        # проход из-за хвоста нельзя.
        if pieces and text.truncated:
            tail = salvage_truncated(pieces[-1], truncated=True)
            logger.warning(
                "consolidation: дописать не вышло даже за %d подход(а); последняя запись %s",
                _MAX_CONTINUATION_ROUNDS,
                "обрезана по последней законченной фразе" if tail else "выброшена: спасать нечего",
            )
            if tail:
                pieces[-1] = tail
            else:
                pieces.pop()

        return pieces

    async def _novelize(self, conversation_text: str, *, written_so_far: str = "") -> _Novelization | None:
        """
        Один запрос новеллизации. `written_so_far` непуст — значит, это
        просьба продолжить прерванный текст с того места, где он оборвался.
        """
        params = LLMParams(
            model="",
            system_prompt=_NOVELIZATION_SYSTEM_PROMPT,
            max_output_tokens=self._novelization_max_output_tokens,
        )
        content = conversation_text
        if written_so_far:
            content = (
                f"{conversation_text}\n\n"
                "--- Ты уже начала записывать этот эпизод, но текст оборвался на полуслове. "
                "Вот его конец:\n"
                f"{written_so_far[-_CONTINUATION_TAIL_CHARS:]}\n\n"
                "Продолжи РОВНО с этого места и допиши до конца: не начинай заново, не повторяй "
                "уже написанное и не здоровайся. Первый же твой символ — продолжение оборванной фразы."
            )
        session = Session(messages=[Message(role=Role.USER, content=content)])
        try:
            response = await self._router.chat(self._summarization_role, params, session)
        except LLMTimeoutError as exc:
            # Медленная модель не успевает написать столько, сколько мы
            # попросили, — и тогда пропадает ВСЯ запись, а не её часть. Проще
            # попросить меньше: короткий эпизод в дневнике лучше, чем ещё
            # одна дыра в памяти за этот вечер.
            logger.warning("consolidation: novelization timed out (%s), retrying with a smaller budget", exc)
            return await self._novelize_briefly(params, session)
        except LLMError as exc:
            logger.warning("consolidation: novelization request failed: %s", exc)
            return None
        return _Novelization(body=response.text.strip(), truncated=response.was_truncated)

    async def _novelize_briefly(self, params: LLMParams, session: Session) -> _Novelization | None:
        """
        Повтор после таймаута — с половинным бюджетом вывода.

        Урезается именно объём, а не время: если модель пишет медленно, ждать
        её дольше на том же объёме значит упереться в тот же таймаут. Просьба
        написать короче — единственное, что здесь действительно помогает.
        """
        smaller = max(_MIN_NOVELIZATION_TOKENS, params.max_output_tokens // 2)
        if smaller >= params.max_output_tokens:
            return None
        try:
            response = await self._router.chat(
                self._summarization_role,
                params.model_copy(update={"max_output_tokens": smaller}),
                session,
            )
        except LLMError as exc:
            logger.warning("consolidation: shortened novelization failed too: %s", exc)
            return None
        logger.info("consolidation: novelized within a smaller budget of %d tokens", smaller)
        return _Novelization(body=response.text.strip(), truncated=response.was_truncated)

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

        summary = salvage_truncated(response.text, truncated=response.was_truncated)
        if response.was_truncated:
            logger.warning(
                "consolidation: summarization hit the output limit (1024 tokens); %s",
                "trimmed to the last complete sentence" if summary else "nothing salvageable, keeping originals",
            )
        return summary or None




def _join_continuation(head: str, tail: str) -> str:
    """
    Склейка оборванного текста с его продолжением.

    Пробел между ними ставится, только если модель не начала продолжение с
    разделителя или знака препинания: «…он замет» + «ил раньше меня» должно
    склеиться в слово, а не в «замет ил».
    """
    if not head:
        return tail
    if not tail:
        return head
    if head[-1].isspace() or tail[0].isspace() or tail[0] in ".,!?;:)»":
        return f"{head}{tail}"
    return f"{head} {tail}"


def _pick_duplicate_to_remove(a: DiaryEntry, b: DiaryEntry) -> str:
    """Из пары почти-дублей выбирает, какую запись убрать: ниже confidence, при равенстве — меньше usage_count."""
    if a.metadata.confidence != b.metadata.confidence:
        return a.id if a.metadata.confidence < b.metadata.confidence else b.id
    return a.id if a.metadata.usage_count <= b.metadata.usage_count else b.id


# ---------------------------------------------------------------------------
# Оценка важности записи
# ---------------------------------------------------------------------------

#: Маркеры эмоциональной окраски в тексте записи
_EMOTIONAL_MARKERS = frozenset({
    # Позитивные
    "рад", "рада", "приятно", "классно", "здорово", "супер", "обожаю", "обожала",
    "нравится", "понравилось", "счастлив", "счастлива", "повезло", "удача",
    # Негативные
    "бесит", "бесила", "раздражает", "раздражало", "злит", "злило",
    "обидно", "обидела", "обижен", "обижена", "больно", "ранит",
    "стыдно", "неловко", "ужас", "кошмар", "ненавижу", "противно",
    # Сильные
    "взбесило", "взорвало", "зацепило", "пробило", "проняло",
    "удивил", "удивила", "шокировал", "шокировала", "ошарашил",
    # Отношения
    "полюбил", "полюбила", "разлюбил", "разлюбила", "скучаю", "скучала",
    "люблю", "любила", "ненавижу", "ненавидела",
})

#: Маркеры договорённостей и планов
_AGREEMENT_MARKERS = frozenset({
    "договорились", "договорилась", "договорился", "обещал", "обещала",
    "пообещал", "пообещала", "условились", "условилась",
    "надо", "нужно", "не забыть", "напомни", "напомню",
    "встретимся", "встретимся", "позвоню", "позвонит", "скину", "пришлю",
    "до пятницы", "к понедельнику", "завтра", "послезавтра", "на следующей",
    "в следующий", "через неделю", "через месяц",
})

#: Маркеры фактов о человеке
_PERSON_FACT_MARKERS = frozenset({
    "работает", "работал", "учится", "учился", "живёт", "жил", "жила",
    "родом", "из москвы", "из питера", "из санкт", "из киев",
    "любим", "нравится", "обожает", "ненавидит",
    "женат", "замужем", "холост", "не замужем", "встречается",
    "кошка", "кот", "собака", "пёс", "питомец",
    "аллергия", "болеет", "болел", "заболел",
})


def estimate_importance(entry: DiaryEntry) -> float:
    """
    Оценивает важность записи для личности и отношений.

    Критерии (накапливаются):
    - Эмоциональная окраска: +0.2 за каждый маркер (до 0.4)
    - Договорённости и планы: +0.25
    - Факты о человеке: +0.15 за каждый маркер (до 0.3)
    - Использование (usage_count): +0.1 за каждые 3 использования (до 0.3)
    - Ground truth (confidence=1.0): +0.3

    Возвращает значение от 0.0 до 1.0.
    """
    text = entry.body.lower()
    base = 0.3  # Базовая важность — нейтральная запись

    # Эмоциональные маркеры
    emotional_hits = sum(1 for marker in _EMOTIONAL_MARKERS if marker in text)
    base += min(0.4, emotional_hits * 0.2)

    # Договорённости
    if any(marker in text for marker in _AGREEMENT_MARKERS):
        base += 0.25

    # Факты о человеке
    person_hits = sum(1 for marker in _PERSON_FACT_MARKERS if marker in text)
    base += min(0.3, person_hits * 0.15)

    # Использование — часто вспоминаемое = важное
    usage_bonus = min(0.3, (entry.metadata.usage_count // 3) * 0.1)
    base += usage_bonus

    # Ground truth — проверенные факты важнее
    if entry.metadata.is_ground_truth:
        base += 0.3

    return min(1.0, base)


def _is_important(entry: DiaryEntry, threshold: float = 0.6) -> bool:
    """Быстрая проверка: запись достаточно важна, чтобы сохранять надолго."""
    return entry.metadata.importance >= threshold or estimate_importance(entry) >= threshold


def _score_importance(entry: DiaryEntry) -> float:
    """Возвращает важность из метаданных или оценивает заново."""
    if entry.metadata.importance > 0.5:
        return entry.metadata.importance
    return estimate_importance(entry)


__all__ = ["DiaryConsolidator", "ExperienceSource", "HistorySource", "estimate_importance"]
