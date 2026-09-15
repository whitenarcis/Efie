#!/usr/bin/env python3
"""
scripts/diary.py — обслуживание корпуса дневника.

Что умеет:
    prune --dry-run   показать записи-обрубки (тело кончается на полуслове),
                      ничего не трогая;
    prune --apply     удалить их (полный текст веб-поиска при этом НЕ теряется:
                      он живёт в SQLite-журнале social_interactions, в дневник
                      шла только его изуродованная копия);
    migrate           вынести эмбеддинги из старых `.md` в sidecar-файлы
                      `<id>.vec.json`, после чего `.md` снова читается глазами
                      и `git diff` перестаёт шуметь.

Примеры:
    PYTHONPATH=src .venv/bin/python scripts/diary.py prune --dry-run
    PYTHONPATH=src .venv/bin/python scripts/diary.py prune --apply
    PYTHONPATH=src .venv/bin/python scripts/diary.py migrate
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from efi.config.schema import get_settings  # noqa: E402
from efi.memory.diary import Diary, dedup_diary_corpus, migrate_diary_corpus, prune_unfinished_entries  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обслуживание корпуса дневника Эфи")
    sub = parser.add_subparsers(dest="command", required=True)

    prune = sub.add_parser("prune", help="найти/удалить записи-обрубки")
    group = prune.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="только показать кандидатов")
    group.add_argument("--apply", action="store_true", help="удалить их")

    sub.add_parser("migrate", help="вынести эмбеддинги из .md в sidecar-файлы")
    dedup = sub.add_parser("dedup", help="найти/удалить семантические дубли записей")
    dedup.add_argument("--threshold", type=float, default=0.95, help="порог косинуса (по умолчанию 0.95)")
    dedup_dry = dedup.add_mutually_exclusive_group(required=True)
    dedup_dry.add_argument("--dry-run", action="store_true", help="только показать пары")
    dedup_dry.add_argument("--apply", action="store_true", help="удалить дубли")
    return parser.parse_args(argv)


async def _main_async(args: argparse.Namespace) -> int:
    settings = get_settings()
    diary = Diary(settings.memory.resolve_diary_dir(settings.paths))
    if args.command == "migrate":
        migrated = await migrate_diary_corpus(diary)
        print(f"migrated {migrated} entries")
        return 0
    if args.command == "dedup":
        pairs = await dedup_diary_corpus(diary, dry_run=not args.apply, threshold=args.threshold)
        if not pairs:
            print("duplicate pairs: none")
            return 0
        for remove_id, keep_id in pairs:
            print(f"- {remove_id} -> {keep_id}")
        if args.apply:
            print(f"removed {len(pairs)} entries")
        else:
            print(f"candidates {len(pairs)} pairs (run with --apply to delete)")
        return 0
    ids = await prune_unfinished_entries(diary, dry_run=not args.apply)
    if not ids:
        print("unfinished entries: none")
        return 0
    entries = {entry.id: entry for entry in await diary.all_entries()}
    for entry_id in ids:
        body = entries.get(entry_id).body if entries.get(entry_id) else "<already deleted>"
        preview = " ".join(body.split())[:160]
        print(f"- {entry_id}: {preview}")
    if args.apply:
        print(f"deleted {len(ids)} entries")
    else:
        print(f"candidates {len(ids)} (run with --apply to delete)")
    return 0


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(asyncio.run(_main_async(_parse_args(argv))))


if __name__ == "__main__":
    main()
