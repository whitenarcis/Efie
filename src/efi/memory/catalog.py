"""
efi/memory/catalog.py

Каталог сущностей для разрешения упоминаний — «кого именно Эфи может иметь
в виду, когда в разговоре звучит имя».

Зачем это отдельно от конвейера приёма знаний (efi/memory/ingest.py). Сам
конвейер сущностей не ищет и не должен: список того, кто вообще существует, —
это знание приложения, а не механики валидации. Конвейер только спрашивает
каталог, получает варианты и решает, разрешено упоминание или неоднозначно
(см. efi/behavior/ambiguity.py).

Про оценки. Никакой «уверенности», выведенной из статистики общения, здесь
нет намеренно. У этого слоя есть ровно два вида свидетельств, и оба честные:

  1. имя совпало — иначе кандидата бы не было вовсе;
  2. этот человек участвует ИМЕННО В ЭТОМ чате — а не просто известен Эфи.

Соблазн добавить третье («с этим Ромой я переписывалась 500 раз, а с тем —
трижды, значит речь про первого») выглядит разумным ровно до первой ошибки:
именно так упоминание тихо приписывается не тому человеку, и отличить такой
факт от настоящего потом уже нельзя. Поэтому два одинаково названных
участника одного чата дают равные оценки — то есть уточняющий вопрос, а не
догадку.
"""

from __future__ import annotations

from collections import defaultdict

from efi.behavior.ambiguity import EntityCandidate
from efi.memory.people import PersonProfile
from efi.memory.validator import normalize_mention

#: Оценка кандидата, который участвует в текущем чате. Разрыв с
#: `_SCORE_ELSEWHERE` заведомо больше DEFAULT_MARGIN (0.08) — значит, свой
#: перевешивает чужого без вопросов.
_SCORE_IN_CHAT = 1.0

#: Кандидат с тем же именем, но из другого чата. Выше DEFAULT_MIN_SCORE
#: (0.35): он остаётся жизнеспособным вариантом, и если своих в чате нет —
#: упоминание разрешится в него.
_SCORE_ELSEWHERE = 0.6


def build_people_catalog(
    people: list[PersonProfile], *, chat_id: int | None = None
) -> dict[str, list[EntityCandidate]]:
    """
    Каталог «упоминание -> кто это может быть» по известным людям.

    Ключ нормализуется той же функцией, что и сущность факта в валидаторе
    (`normalize_mention`), иначе каталог и конвейер разошлись бы на регистре
    и пробелах: искали бы «рома», а в каталоге лежало бы «Рома».

    Люди без имени пропускаются: по пустому упоминанию разрешать нечего.
    """
    catalog: dict[str, list[EntityCandidate]] = defaultdict(list)
    for profile in people:
        name = profile.display_name.strip()
        if not name:
            continue
        catalog[normalize_mention(name)].append(_to_candidate(profile, chat_id=chat_id))
    return dict(catalog)


def apply_confirmed_answers(
    catalog: dict[str, list[EntityCandidate]], confirmed: dict[str, str]
) -> dict[str, list[EntityCandidate]]:
    """
    Убирает из каталога всех, кроме того, на кого человек уже указал сам.

    Без этого шага уточнение работало бы вхолостую: вопрос задан, ответ
    получен и закрыт — а следующий эпизод снова упирается в те же два
    одинаковых имени и снова спрашивает. Вопрос, ответ на который не
    запомнили, раздражает сильнее, чем молчание.

    Именно вычёркивание, а не «повысить оценку»: человек не ранжировал
    варианты, он назвал один. Оставлять рядом остальных — значит оставлять
    возможность снова счесть упоминание спорным.
    """
    if not confirmed:
        return catalog

    updated = dict(catalog)
    for mention, entity_id in confirmed.items():
        candidates = updated.get(mention)
        if not candidates:
            continue
        chosen = [candidate for candidate in candidates if candidate.entity_id == entity_id]
        if chosen:
            updated[mention] = chosen
    return updated


def _to_candidate(profile: PersonProfile, *, chat_id: int | None) -> EntityCandidate:
    in_this_chat = chat_id is not None and profile.last_chat_id == chat_id
    return EntityCandidate(
        entity_id=f"user:{profile.user_id}",
        label=profile.display_name,
        score=_SCORE_IN_CHAT if in_this_chat else _SCORE_ELSEWHERE,
        hint=_hint(profile, in_this_chat=in_this_chat),
    )


def _hint(profile: PersonProfile, *, in_this_chat: bool) -> str:
    """
    Короткое пояснение для уточняющего вопроса. Без него вопрос выродится в
    «ты про Рому или про Рому?» — то есть в бессмыслицу.
    """
    if in_this_chat:
        return "из этого чата"
    if profile.last_chat_title:
        return f"из «{profile.last_chat_title}»"
    return "из другого чата"


__all__ = ["apply_confirmed_answers", "build_people_catalog"]
