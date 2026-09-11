from __future__ import annotations

import getpass
import asyncio
import os
import shutil
import sys
from pathlib import Path

# Корректно поднимаемся на 3 уровня вверх: setup.py -> scripts/ -> src/ -> Efie/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "src" / "config"
DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_URL = "http://localhost:20128/v1"
DEFAULT_MODELS = {
    "main": "ollamacloud/gemma4:31b",
    "main_fallback": "ollamacloud/nemotron-3-super",
    "fast": "ollamacloud/gpt-oss:20b",
    "vision": "qwen/qwen3.6-27b",
    "background": "ollamacloud/gpt-oss:20b",
    "coder": "siliconflow/zai-org/GLM-4.7",
}

# Роли для прохода в цикле настройки провайдеров
ROLES = ("main", "fast", "vision", "background", "coder")


def clear() -> None:
    os.system("clear" if os.name != "nt" else "cls")


def prompt(label: str, default: str = "", is_password: bool = False, required: bool = False) -> str:
    suffix = f" [{default}]" if default else ""
    prompt_str = f"\033[1;36m>\033[0m {label}{suffix}: "

    while True:
        if is_password:
            val = getpass.getpass(prompt_str).strip()
        else:
            val = input(prompt_str).strip()

        final_val = val if val else default

        if required and not final_val:
            print("\033[1;31mОшибка: это поле обязательно для заполнения!\033[0m")
            continue
        return final_val


def confirm(label: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    val = input(f"\033[1;33m?\033[0m {label} [{hint}]: ").strip().lower()
    if not val:
        return default
    return val in ("y", "yes", "д", "да")


def write_toml(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(content.lstrip())


async def authorize_telegram(api_id: int, api_hash: str) -> bool:
    try:
        from pyrogram import Client
    except ImportError:
        venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
        if venv_python.exists():
            import subprocess
            print("\033[1;34mПереключение контекста на .venv для авторизации Telegram...\033[0m")

            cmd = [str(venv_python), __file__, "--auth-only", str(api_id), api_hash]
            result = subprocess.run(cmd)
            return result.returncode == 0

        print("\033[1;31mОшибка: pyrofork не установлен, и .venv не найден. Пропуск авторизации.\033[0m")
        return False

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session_name = str(DATA_DIR / "efi_session")

    print("\n\033[1;32mИнициализация сессии Pyrofork в окружении venv...\033[0m")
    client = Client(
        name=session_name,
        api_id=api_id,
        api_hash=api_hash,
        workdir=str(PROJECT_ROOT),
    )

    async with client:
        me = await client.get_me()
        print(
            f"\033[1;32mУспешный вход в аккаунт бота: {me.first_name} (@{me.username or 'без юзернейма'}, ID: {me.id})\033[0m")
        return True


def setup_path() -> None:
    bin_dir = Path.home() / ".local" / "bin"
    target_script = PROJECT_ROOT / "src" / "scripts" / "run.py"

    if not target_script.exists():
        print(f"\n\033[1;31mОшибка: Файл не найден по пути {target_script}!\033[0m")
        return

    wrapper_path = bin_dir / "efie"
    if not wrapper_path.parent.exists():
        bin_dir.mkdir(parents=True, exist_ok=True)

    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    interpreter = str(venv_python) if venv_python.exists() else sys.executable

    wrapper_content = f"""#!/bin/sh
exec "{interpreter}" "{target_script}" "$@"
"""
    with wrapper_path.open("w", encoding="utf-8") as f:
        f.write(wrapper_content)
    wrapper_path.chmod(0o755)

    shell = os.environ.get("SHELL", "")
    rc_file = Path.home() / ".zshrc" if "zsh" in shell else (Path.home() / ".bashrc" if "bash" in shell else None)

    path_env = os.environ.get("PATH", "")
    if str(bin_dir) not in path_env and rc_file and rc_file.is_file():
        export_line = f'\nexport PATH="{bin_dir}:$PATH"\n'
        with rc_file.open("a", encoding="utf-8") as f:
            f.write(export_line)
        print(f"\033[1;32mСтрока экспорта добавлена в {rc_file}.\033[0m")

    print(f"\033[1;32mКоманда 'efie' успешно создана для: {target_script}\033[0m")


def main() -> None:
    clear()
    print("\033[1;35m========================================")
    print("        Мастер настройки Efie")
    print("========================================\033[0m\n")

    single_provider = confirm("Хотите использовать одного провайдера?", default=True)
    providers: dict[str, dict[str, str]] = {}

    if single_provider:
        url = prompt("URL API", DEFAULT_URL, required=True)
        key = prompt("API Key", is_password=True, required=True)
        for role in ROLES:
            providers[role] = {"base_url": url, "api_key": key}
    else:
        for role in ROLES:
            print(f"\n\033[1;34m--- Настройка роли: {role.upper()} ---\033[0m")
            url = prompt(f"URL для {role}", DEFAULT_URL, required=True)
            key = prompt(f"API Key для {role}", is_password=True, required=True)
            providers[role] = {"base_url": url, "api_key": key}

    print("")
    custom_models = confirm("Хотите сами выбрать модели?", default=False)
    models: dict[str, str] = {}

    if custom_models:
        for role in ROLES:
            models[role] = prompt(f"Модель для {role}", DEFAULT_MODELS[role], required=True)
        models["main_fallback"] = prompt("Модель для MAIN (fallback)", DEFAULT_MODELS["main_fallback"], required=True)
    else:
        models = DEFAULT_MODELS.copy()

    llm_content = f"""[llm_roles.main]
degrade_to = "fast"

[llm_roles.main.primary]
base_url = "{providers['main']['base_url']}"
api_key = "{providers['main']['api_key']}"
model = "{models['main']}"
timeout_seconds = 15.0

[llm_roles.main.fallback]
base_url = "{providers['main']['base_url']}"
api_key = "{providers['main']['api_key']}"
model = "{models['main_fallback']}"
timeout_seconds = 15.0

[llm_roles.fast.primary]
base_url = "{providers['fast']['base_url']}"
api_key = "{providers['fast']['api_key']}"
model = "{models['fast']}"
timeout_seconds = 15.0

[llm_roles.background.primary]
base_url = "{providers['background']['base_url']}"
api_key = "{providers['background']['api_key']}"
model = "{models['background']}"
timeout_seconds = 45.0

[llm_roles.vision.primary]
base_url = "{providers['vision']['base_url']}"
api_key = "{providers['vision']['api_key']}"
model = "{models['vision']}"
timeout_seconds = 60.0

[llm_roles.coder]
degrade_to = "fast"

[llm_roles.coder.primary]
base_url = "{providers['coder']['base_url']}"
api_key = "{providers['coder']['api_key']}"
model = "{models['coder']}"
timeout_seconds = 90.0
"""
    write_toml(CONFIG_DIR / "llm.toml", llm_content)

    print("\n\033[1;34m--- Настройка Telegram ---\033[0m")

    while True:
        api_id_str = prompt("Telegram API ID", required=True)
        if api_id_str.isdigit():
            api_id = int(api_id_str)
            break
        print("\033[1;31mОшибка: API ID должен состоять только из цифр.\033[0m")

    api_hash = prompt("Telegram API Hash (скрытый ввод)", is_password=True, required=True)

    while True:
        owner_id_str = prompt("Telegram ID Владельца бота", required=True)
        if owner_id_str.isdigit():
            owner_id = int(owner_id_str)
            break
        print("\033[1;31mОшибка: ID Владельца должен состоять только из цифр.\033[0m")

    asyncio.run(authorize_telegram(api_id, api_hash))

    telegram_content = f"""[telegram]
api_id = {api_id}
api_hash = "{api_hash}"
owner_id = {owner_id}
allowed_chats = []
community_chats = []
lockdown_mode = "owner_only"
check_chats_on_startup = true
can_join_chats = false
can_leave_chats = true

[community]
enabled = true
comment_probability = 0.35
min_delay_seconds = 300.0
max_delay_seconds = 1800.0
thread_scan_interval_seconds = 1800.0
max_replies_per_thread = 1
topic_match_min_score = 0.34
"""
    # Записываем сгенерированный telegram.toml
    write_toml(CONFIG_DIR / "telegram.toml", telegram_content)

    # --- НОВЫЙ БЛОК: Копирование остальных конфигурационных файлов из templates ---
    templates_dir = PROJECT_ROOT / "templates"
    print("\n\033[1;34m--- Проверка дополнительных конфигурационных файлов ---\033[0m")

    if templates_dir.exists():
        # Список файлов, которые нужно перенести «как есть», если их еще нет в config
        extra_configs = ["behaviour.toml", "dashboard.toml", "experemental.toml", "web_search.toml"]

        for config_file in extra_configs:
            src_file = templates_dir / config_file
            dst_file = CONFIG_DIR / config_file

            if src_file.exists():
                if not dst_file.exists():
                    shutil.copy2(src_file, dst_file)
                    print(f"  • {config_file} успешно скопирован из шаблонов.")
                else:
                    print(f"  • {config_file} уже существует в config (пропущен).")
            else:
                print(f"\033[1;33mПредупреждение: Шаблон {config_file} не найден в папе templates.\033[0m")
    else:
        print("\033[1;31mПредупреждение: Папка templates не найдена в корне проекта.\033[0m")
    # -----------------------------------------------------------------------------

    print("")
    if confirm("Добавить команду 'efie' в PATH (~/.local/bin)?", default=True):
        setup_path()

    print("\n\033[1;32m========================================")
    print("          Настройка завершена")
    print("========================================\033[0m")
    print("\nКонфигурационные файлы:")
    print(f"  • LLM:          {CONFIG_DIR / 'llm.toml'}")
    print(f"  • Telegram:     {CONFIG_DIR / 'telegram.toml'}")
    print(f"  • Поведение:    {CONFIG_DIR / 'behaviour.toml'}")
    print(f"  • Панель:       {CONFIG_DIR / 'dashboard.toml'}")
    print(f"  • Экспер-ты:    {CONFIG_DIR / 'experemental.toml'}")
    print(f"  • База:         {DATA_DIR / 'efi.db'}")
    print(f"  • Сессия:       {DATA_DIR / 'efi_session.session'}")
    print("")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--auth-only":
        try:
            api_id = int(sys.argv[2])
            api_hash = sys.argv[3]
            asyncio.run(authorize_telegram(api_id, api_hash))
            sys.exit(0)
        except Exception as e:
            print(f"Ошибка подпроцесса авторизации: {e}", file=sys.stderr)
            sys.exit(1)

    try:
        main()
    except KeyboardInterrupt:
        print("\n\033[1;33mНастройка отменена пользователем.\033[0m")
        sys.exit(130)
