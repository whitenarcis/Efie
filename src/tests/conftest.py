"""
Общие фикстуры тестов.

Единственная задача этого файла — закрывать соединения с БД после каждого
теста. С тех пор как `Database` держит ОДНО долгоживущее соединение вместо
короткоживущих (см. efi/db/core.py), незакрытое соединение оставляет за собой
НЕ-daemon-поток aiosqlite. В приложении закрывать его есть кому (EfiApp.stop),
а в тестах владельца нет — и сотня тестов оставляла бы сотню живых потоков,
которые под конец сессии начинают ломиться в уже закрытый event loop.

Ровно этот класс ошибок однажды уже давал плавающее зависание всего прогона на
выходе из интерпретатора, поэтому он закрыт здесь разом, а не по одному
`await database.close()` в каждом тесте.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from efi.db.core import close_all_databases


@pytest.fixture(autouse=True)
async def _close_databases() -> AsyncIterator[None]:
    # asyncio_mode = "auto" (см. pyproject.toml) делает async-фикстуры рабочими
    # и через обычный pytest.fixture.
    yield
    await close_all_databases()
