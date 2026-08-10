-- efi/db/schema.sql
--
-- DDL строгого хранилища знаний. Вынесен в .sql, а не оставлен строкой в
-- Python, потому что это единственная часть схемы, которую приходится читать
-- глазами целиком: домены, ограничения и дедуплицирующий уникальный индекс
-- имеют смысл только вместе, и разглядывать их удобнее как схему, а не как
-- набор строковых констант. Применяется миграцией из efi/db/models.py.
--
-- Всё здесь идемпотентно (IF NOT EXISTS) — миграции прогоняются на каждом
-- старте без таблицы версий, см. докстринг efi/db/models.py.
--
-- ДОМЕНЫ ПАМЯТИ (обязательный атрибут `domain` у каждой записи):
--   'C' (Common)   — интерсубъективные знания о мире: технологии, концепции,
--                    факты, не привязанные к конкретному человеку.
--   'P' (Personal) — модель владельца и конкретных людей: предпочтения,
--                    характеристики, личные данные.
--   'H' (History)  — личный эпизодический и эмоциональный опыт самой Эфи:
--                    с кем общалась, как узнала факт, что при этом чувствовала.
--
-- Смысл разделения — не бухгалтерия, а качество выборки: без него RAG на
-- вопрос «как работает WAL в SQLite» одинаково охотно подмешивал и статью про
-- WAL, и воспоминание о том, как в позапрошлый вторник кто-то грустил.

CREATE TABLE IF NOT EXISTS knowledge_facts (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Домен обязателен и ограничен на уровне БД: запись без домена или с
    -- выдуманным доменом не должна существовать физически, а не «не должна
    -- появляться по договорённости».
    domain           TEXT NOT NULL CHECK (domain IN ('C', 'P', 'H')),

    entity_id        TEXT NOT NULL,   -- нормализованный субъект: 'user:625207005', 'topic:sqlite'
    attribute        TEXT NOT NULL,   -- нормализованный slug атрибута: 'режим_сна'
    value            TEXT NOT NULL,   -- очищенное значение

    -- sha256 от (domain|entity_id|attribute|нормализованное значение).
    -- Первая ступень дедупликации: точный повтор ловится сравнением строк,
    -- без единого обращения к эмбеддингам.
    normalized_hash  TEXT NOT NULL,

    -- Вектор значения для ВТОРОЙ ступени дедупликации (косинусное сходство).
    -- BLOB из float32, а не JSON: тысяча фактов по 1024 измерения — это
    -- 4 МБ бинарно против ~20 МБ текстом, и парсить его не нужно.
    embedding        BLOB,
    embedding_dim    INTEGER NOT NULL DEFAULT 0,

    confidence       REAL NOT NULL DEFAULT 0.5 CHECK (confidence >= 0.0 AND confidence <= 1.0),

    -- Сколько раз этот факт подтверждался. Именно это число превращает
    -- «однажды сказал» в «систематически делает» и уезжает в промпт как
    -- «(упомянуто 12 раз)».
    occurrence_count INTEGER NOT NULL DEFAULT 1 CHECK (occurrence_count >= 1),

    source           TEXT NOT NULL DEFAULT '',  -- откуда узнала: 'chat:-100123', 'web', 'self'
    first_seen_at    TEXT NOT NULL,
    last_seen_at     TEXT NOT NULL,

    -- Дедупликация точным совпадением обеспечена самой БД: даже если два
    -- воркера одновременно решат записать один и тот же факт, вторая вставка
    -- не пройдёт, и код честно перейдёт к инкременту счётчика.
    UNIQUE (domain, normalized_hash)
);

CREATE INDEX IF NOT EXISTS idx_knowledge_facts_domain_entity
    ON knowledge_facts (domain, entity_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_facts_last_seen
    ON knowledge_facts (last_seen_at DESC);
CREATE INDEX IF NOT EXISTS idx_knowledge_facts_occurrence
    ON knowledge_facts (occurrence_count DESC);

-- Журнал отклонённых кандидатов: то, что модель предложила запомнить, а
-- валидатор не пропустил. Нужен не для памяти, а для наблюдаемости границы
-- доверия — без него «Эфи не запомнила» и «Эфи запомнила чушь» выглядят
-- одинаково: тишиной.
CREATE TABLE IF NOT EXISTS knowledge_rejections (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    domain        TEXT NOT NULL DEFAULT '',
    entity_id     TEXT NOT NULL DEFAULT '',
    attribute     TEXT NOT NULL DEFAULT '',
    value         TEXT NOT NULL DEFAULT '',
    reason        TEXT NOT NULL,
    source        TEXT NOT NULL DEFAULT '',
    created_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_knowledge_rejections_created
    ON knowledge_rejections (created_at DESC);
