/*
 * efi/dashboard/static/app.js
 *
 * Дашборд без сборки, фреймворков и внешних библиотек: маршрутизация по
 * хэшу, fetch к /api/* и отрисовка через innerHTML. Причина та же, что и у
 * собственного HTTP-сервера, — Эфи должна запускаться в Termux и открываться
 * с телефона в локальной сети, где ни npm, ни CDN нет.
 *
 * Всё, что приходит с сервера, проходит через esc() перед вставкой в
 * разметку: в логах, дневнике и переписке лежит текст от посторонних людей,
 * и он не должен уметь выполниться в странице.
 */

(() => {
  'use strict';

  const view = document.getElementById('view');
  const nav = document.getElementById('nav');
  const burger = document.getElementById('burger');
  const presence = document.getElementById('presence');
  const presenceText = document.getElementById('presence-text');
  const footerNote = document.getElementById('footer-note');
  const footerRefresh = document.getElementById('footer-refresh');

  /** Как часто перезапрашиваются «живые» разделы. */
  const REFRESH_MS = 5000;

  const state = {
    route: 'overview',
    arg: null,
    timer: null,
    stream: null,
    logs: { level: 'INFO', query: '', paused: false, follow: true, lastId: 0 },
  };

  // ------------------------------------------------------------- утилиты

  const ESCAPES = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
  const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (char) => ESCAPES[char]);

  const numberFormat = new Intl.NumberFormat('ru-RU');
  const num = (value) => (value === null || value === undefined ? '—' : numberFormat.format(value));

  const fixed = (value, digits = 2) =>
    value === null || value === undefined || Number.isNaN(Number(value))
      ? '—'
      : Number(value).toFixed(digits);

  const pct = (value) => (value === null || value === undefined ? '—' : `${Math.round(Number(value) * 100)}%`);

  function parseDate(value) {
    if (!value) return null;
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? null : parsed;
  }

  function fmtDateTime(value) {
    const date = parseDate(value);
    if (!date) return '—';
    return date.toLocaleString('ru-RU', {
      day: '2-digit', month: '2-digit', year: '2-digit', hour: '2-digit', minute: '2-digit',
    });
  }

  function fmtTime(value) {
    const date = parseDate(value);
    if (!date) return '—';
    return date.toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  }

  function fmtAgo(value) {
    const date = parseDate(value);
    if (!date) return '—';
    const seconds = Math.max(0, (Date.now() - date.getTime()) / 1000);
    if (seconds < 60) return 'только что';
    if (seconds < 3600) return `${Math.floor(seconds / 60)} мин назад`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)} ч назад`;
    return `${Math.floor(seconds / 86400)} дн назад`;
  }

  function fmtDuration(seconds) {
    if (seconds === null || seconds === undefined) return '—';
    const total = Math.floor(Number(seconds));
    const days = Math.floor(total / 86400);
    const hours = Math.floor((total % 86400) / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    if (days > 0) return `${days} д ${hours} ч`;
    if (hours > 0) return `${hours} ч ${minutes} мин`;
    if (minutes > 0) return `${minutes} мин`;
    return `${total} с`;
  }

  async function api(path, params = {}) {
    const url = new URL(path, window.location.origin);
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== '') url.searchParams.set(key, value);
    });
    const response = await fetch(url, { headers: { Accept: 'application/json' } });
    if (response.status === 401) {
      window.location.reload();
      throw new Error('нужна авторизация');
    }
    if (!response.ok) throw new Error(`${response.status}`);
    return response.json();
  }

  function setPresence(online, note) {
    presence.dataset.state = online ? 'online' : 'offline';
    presenceText.textContent = online ? 'НА СВЯЗИ' : 'НЕТ СВЯЗИ';
    if (note) footerNote.textContent = note;
  }

  // -------------------------------------------------------- строительные

  const hero = (kicker, title, lede) => `
    <section class="hero">
      <p class="kicker">${esc(kicker)}</p>
      <h1>${esc(title)}</h1>
      ${lede ? `<p class="lede">${esc(lede)}</p>` : ''}
    </section>`;

  const cell = (label, value, note, extra = '') => `
    <div class="cell">
      <span class="label">${esc(label)}</span>
      <div class="value ${extra}">${value}</div>
      ${note ? `<p class="note">${note}</p>` : ''}
    </div>`;

  const meter = (fraction, soft = false) => {
    const width = Math.max(0, Math.min(1, Number(fraction) || 0)) * 100;
    return `<div class="meter${soft ? ' soft' : ''}"><i style="width:${width.toFixed(1)}%"></i></div>`;
  };

  const row = (title, sub, aside, attrs = '') => `
    <div class="row ${attrs ? 'clickable' : ''}" ${attrs}>
      <div class="row-main">
        <div class="row-title">${title}</div>
        ${sub ? `<div class="row-sub">${sub}</div>` : ''}
      </div>
      ${aside ? `<div class="row-aside">${aside}</div>` : ''}
    </div>`;

  const rows = (items, emptyText = 'Пусто') =>
    items.length ? `<div class="rows">${items.join('')}</div>` : `<div class="empty">${esc(emptyText)}</div>`;

  const title = (text) => `<h2 class="section-title">${esc(text)}</h2>`;

  const STATE_TAGS = {
    running: ['ok', 'работает'],
    stopped: ['warn', 'остановлен'],
    failed: ['err', 'упал'],
    finished: ['warn', 'завершился'],
    disabled: ['mute', 'выключен'],
    not_started: ['mute', 'не запущен'],
  };

  const stateTag = (name) => {
    const [kind, label] = STATE_TAGS[name] || ['mute', name];
    return `<span class="tag ${kind}">${esc(label)}</span>`;
  };

  function sparkline(points) {
    // По одной точке линию не построить — на её месте получался бы
    // бессмысленный клин во всю ширину карточки.
    if (points.length < 2) return '';
    const values = points.map((point) => Number(point.total) || 0);
    const max = Math.max(...values, 1);
    const width = 100;
    const height = 40;
    const step = values.length > 1 ? width / (values.length - 1) : width;
    const coords = values.map((value, index) => [index * step, height - (value / max) * (height - 4) - 2]);
    const line = coords.map(([x, y], index) => `${index ? 'L' : 'M'}${x.toFixed(2)},${y.toFixed(2)}`).join(' ');
    const area = `${line} L${width},${height} L0,${height} Z`;
    return `
      <svg class="spark" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" aria-hidden="true">
        <path class="area" d="${area}"></path>
        <path d="${line}"></path>
      </svg>`;
  }

  // -------------------------------------------------------------- разделы

  const PAGES = {
    overview: { kicker: 'Эфи · сейчас', title: 'Живой снимок', live: true, load: renderOverview },
    self: { kicker: 'Эфи · внутри', title: 'Внутреннее состояние', live: true, load: renderSelf },
    functions: { kicker: 'Эфи · механика', title: 'Функции и службы', live: true, load: renderFunctions },
    logs: { kicker: 'Эфи · поток', title: 'Подробные логи', live: false, load: renderLogs },
    diary: { kicker: 'Эфи · память', title: 'Дневник', live: false, load: renderDiary },
    memory: { kicker: 'Эфи · память', title: 'Накопленное', live: false, load: renderMemory },
    people: { kicker: 'Эфи · окружение', title: 'Знакомые люди', live: true, load: renderPeople },
    projects: { kicker: 'Эфи · ремесло', title: 'Её проекты', live: true, load: renderProjects },
    chats: { kicker: 'Эфи · разговоры', title: 'Чаты и переписка', live: true, load: renderChats },
    llm: { kicker: 'Эфи · инференс', title: 'Модели и вызовы', live: true, load: renderLLM },
    prompts: { kicker: 'Эфи · основа', title: 'Основа личности', live: false, load: renderPrompts },
  };

  async function renderOverview() {
    const data = await api('/api/overview');
    const counts = data.counts || {};
    const telegram = data.telegram || {};
    const memory = data.working_memory || {};
    const quiet = data.quiet_hours || {};
    const llm = data.llm || {};
    const queue = data.queue || {};
    const generations = data.generations || {};

    const connected = telegram.connected;
    const connectionValue = connected === null || connected === undefined
      ? 'неизвестно'
      : connected ? 'на связи' : 'отключена';

    const grid = [
      cell('Telegram', esc(connectionValue), `режим доступа: ${esc(telegram.lockdown_mode || '—')}`, 'small'),
      cell('Аптайм', esc(fmtDuration(data.uptime_seconds)), `запущена ${esc(fmtDateTime(data.started_at))}`, 'small'),
      cell('Очередь событий', num(queue.total), `${num(queue.worker_count)} воркер(ов)`),
      cell(
        'Пишет прямо сейчас',
        num(generations.active),
        (generations.chats || []).map((chat) => esc(chat.label)).join(', ') || 'ничего не генерирует',
      ),
      cell(
        'Энергия',
        memory.energy === undefined || memory.energy === null ? '—' : pct(memory.energy),
        `${esc(memory.energy_label || memory.emotional_state || '—')}${meter(memory.energy)}`,
        'small',
      ),
      cell(
        'Тихие часы',
        quiet.active_now ? 'идут' : 'нет',
        quiet.enabled ? `${quiet.start_hour}:00 — ${quiet.end_hour}:00, у неё ${esc(fmtTime(quiet.local_time))} ${esc(quiet.timezone || '')} (${esc(quiet.timezone_source || 'система')})` : 'выключены',
        'small',
      ),
      cell('Вызовы моделей', num(llm.calls), `ошибок ${num(llm.errors)} · среднее ${fixed(llm.avg_seconds)} c`),
      cell('Записи дневника', num(counts.diary), `сообщений в истории ${num(counts.messages)}`),
      cell('Люди и факты', `${num(counts.people)} / ${num(counts.facts)}`, 'знакомых / фактов о них', 'small'),
      cell('Убеждения', num(counts.beliefs), `семян любопытства ${num(counts.curiosity_seeds)}`),
      cell('Внешний опыт', num(counts.social_interactions), `тредов на карандаше ${num(counts.thread_state)}`),
      cell(
        'Записей в логе',
        num((data.logs || {}).total),
        `в буфере ${num((data.logs || {}).buffered)} из ${num((data.logs || {}).capacity)}`,
      ),
    ].join('');

    const services = (data.services || []).map((service) =>
      row(
        esc(service.title),
        esc(service.description || service.name),
        `${stateTag(service.state)}${service.detail ? `<div class="row-sub">${esc(service.detail)}</div>` : ''}`,
      ),
    );

    const activity = data.activity || [];
    const activityBlock = activity.length > 1
      ? `<div class="rows"><div class="row"><div class="row-main">
           <div class="row-title">Сообщений за последние ${activity.length} дн.</div>
           <div class="row-sub">${esc(activity[0].day)} — ${esc(activity[activity.length - 1].day)}</div>
           ${sparkline(activity)}
         </div></div></div>`
      : '';

    setPresence(true, `${data.character_name} · ${data.environment}`);

    return `
      ${hero(PAGES.overview.kicker, PAGES.overview.title, `Всё, что Эфи делает и помнит, на одной странице. Обновлено ${fmtTime(data.now)}.`)}
      <div class="grid wide">${grid}</div>
      ${activityBlock ? title('Активность переписки') + activityBlock : ''}
      ${title('Фоновые службы')}
      ${rows(services, 'Службы не запущены')}`;
  }

  async function renderSelf() {
    const data = await api('/api/self');
    const memory = data.working_memory || {};
    const busy = data.busy || {};

    const grid = [
      cell('Эмоциональное состояние', esc(memory.emotional_state || '—'), memory.is_derived ? 'выведено из энергии и часа' : 'её собственные слова', 'text'),
      cell('Физическое состояние', esc(memory.physical_state || '—'), memory.is_derived ? 'выведено из энергии и часа' : 'её собственные слова', 'text'),
      cell('Энергия', pct(memory.energy), `${esc(memory.energy_label || '')}${memory.is_sleepy ? ' · клонит в сон' : ''}${meter(memory.energy)}`, 'small'),
      cell(
        'Занятость',
        busy.delay_seconds === undefined ? '—' : `${fixed(busy.delay_seconds, 1)} c`,
        'оценка паузы перед ответом прямо сейчас',
        'small',
      ),
      cell(
        'Фоновое исследование',
        data.researching === null || data.researching === undefined ? '—' : data.researching ? 'идёт' : 'нет',
        'движок жизни копает тему из любопытства',
        'small',
      ),
      cell(
        'Тихие часы',
        (data.quiet_hours || {}).active_now ? 'идут' : 'нет',
        'в тихие часы она не пишет первой',
        'small',
      ),
    ].join('');

    const items = (memory.items || []).map((item) =>
      row(
        esc(item.text),
        `заведено ${esc(fmtDateTime(item.created_at))} · обновлено ${esc(fmtAgo(item.last_updated))}`,
        item.done ? '<span class="tag mute">закрыто</span>' : '<span class="tag">открыто</span>',
      ),
    );

    const beliefs = (data.beliefs || []).map((belief) =>
      row(
        esc(belief.topic),
        esc(belief.stance),
        `${fixed(belief.confidence_score)}${meter(belief.confidence_score, true)}`,
      ),
    );

    const affinity = (data.affinity || []).map((chat) =>
      row(
        esc(chat.label),
        `сообщений ${num(chat.message_count)} · последнее ${esc(fmtAgo(chat.last_at))}`,
        `близость ${fixed(chat.affinity)} · уважение ${fixed(chat.respect_level)}${meter(chat.affinity, true)}`,
      ),
    );

    return `
      ${hero(PAGES.self.kicker, PAGES.self.title, 'Рабочая память, убеждения и отношение к собеседникам — то, из чего складывается её текущий характер.')}
      <div class="grid">${grid}</div>
      ${title('Обещания и открытые пункты')}
      ${rows(items, 'Ничего не обещала')}
      ${title('Убеждения')}
      ${rows(beliefs, 'Убеждений пока нет')}
      ${title('Близость к чатам')}
      ${rows(affinity, 'Близость ни к кому не накоплена')}`;
  }

  async function renderFunctions() {
    const data = await api('/api/functions');
    const behavior = data.behavior || {};

    const services = (data.services || []).map((service) =>
      row(
        esc(service.title),
        esc(service.description || ''),
        `${stateTag(service.state)}${service.detail ? `<div class="row-sub">${esc(service.detail)}</div>` : ''}`,
      ),
    );

    const workers = (data.queue.workers || []).map((worker) =>
      row(`Воркер ${worker.index}`, `в очереди ${num(worker.queued)}`, stateTag(worker.state)),
    );

    const tools = (data.tools || []).map((tool) =>
      row(
        `<span class="mono">${esc(tool.name)}</span>`,
        esc(tool.description || ''),
        tool.available ? '<span class="tag ok">доступен</span>' : '<span class="tag mute">выключен</span>',
      ),
    );

    const settingsGrid = [
      cell(
        'Скорость набора',
        `${num((behavior.humanizer || {}).typing_wpm?.[0])}–${num((behavior.humanizer || {}).typing_wpm?.[1])}`,
        'слов в минуту',
        'small',
      ),
      cell('Опечатки', pct((behavior.humanizer || {}).typo_probability), `сама исправляет ${pct((behavior.humanizer || {}).typo_self_correct_probability)}`, 'small'),
      cell('Глубина истории', num((behavior.memory || {}).history_limit), 'сообщений в каждом запросе к модели', 'small'),
      cell('Порог релевантности', fixed((behavior.memory || {}).min_relatedness), 'ниже — запись дневника не считается подходящей', 'small'),
      cell('Пульс памяти', (behavior.memory_pulse || {}).enabled ? 'включён' : 'выключен', `эпизод закрывается после ${num((behavior.memory_pulse || {}).episode_idle_seconds)} c тишины`, 'small'),
      cell('Сообщество', (behavior.community || {}).enabled ? 'участвует' : 'выключено', `каналов ${num((behavior.community || {}).chats)} · шанс ${pct((behavior.community || {}).comment_probability)}`, 'small'),
      cell('Локальные эмбеддинги', (behavior.memory || {}).use_local_embeddings ? 'да' : 'нет', esc((behavior.memory || {}).local_embedding_model || ''), 'small'),
      cell('Движок жизни', `${num((behavior.life_engine || {}).check_interval_seconds)} c`, `порог важности ${fixed((behavior.life_engine || {}).ping_importance_threshold)}`, 'small'),
    ].join('');

    return `
      ${hero(PAGES.functions.kicker, PAGES.functions.title, 'Что у неё запущено, чем она умеет пользоваться и по каким правилам работает.')}
      ${title('Фоновые службы')}
      ${rows(services, 'Служб нет')}
      ${title('Воркеры очереди')}
      ${rows(workers, 'Воркеры не запущены')}
      ${title('Инструменты модели')}
      ${rows(tools, 'Инструменты не зарегистрированы')}
      ${title('Настройки поведения')}
      <div class="grid wide">${settingsGrid}</div>`;
  }

  // ------------------------------------------------------------------ логи

  function logLine(entry) {
    return `
      <div class="log-line" data-id="${entry.id}">
        <div class="log-meta">
          <span>${esc(fmtTime(entry.timestamp))}</span>
          <span class="log-level ${esc(entry.level)}">${esc(entry.level)}</span>
          <span title="${esc(entry.logger)}">${esc(entry.logger)}</span>
          ${entry.task ? `<span>${esc(entry.task)}</span>` : ''}
        </div>
        <div>
          <div class="log-message">${esc(entry.message)}</div>
          ${entry.exception ? `<pre class="log-exception">${esc(entry.exception)}</pre>` : ''}
        </div>
      </div>`;
  }

  async function renderLogs() {
    const data = await api('/api/logs', { level: state.logs.level, q: state.logs.query, limit: 300 });
    state.logs.lastId = data.last_id || 0;
    const counts = data.counts || {};

    const levels = ['ALL', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
      .map((level) => `<option value="${level}" ${state.logs.level === level ? 'selected' : ''}>${level}</option>`)
      .join('');

    return `
      ${hero(PAGES.logs.kicker, PAGES.logs.title, `В буфере ${num(data.buffered)} из ${num(data.capacity)} записей · всего с запуска ${num(data.total)}.`)}
      <div class="grid wide">
        ${cell('Ошибки', num((counts.ERROR || 0) + (counts.CRITICAL || 0)), 'за всё время работы процесса')}
        ${cell('Предупреждения', num(counts.WARNING || 0), 'WARNING в ленте')}
        ${cell('Информационные', num(counts.INFO || 0), 'INFO в ленте')}
        ${cell('Отладочные', num(counts.DEBUG || 0), 'видны, если dashboard.log_level = DEBUG')}
      </div>
      ${title('Живая лента')}
      <div class="controls">
        <div class="field">
          <label for="log-level">Уровень</label>
          <select id="log-level">${levels}</select>
        </div>
        <div class="field">
          <label for="log-query">Поиск</label>
          <input id="log-query" type="search" placeholder="подстрока в тексте или имени логгера" value="${esc(state.logs.query)}">
        </div>
        <button class="ghost" id="log-pause" type="button" aria-pressed="${state.logs.paused}">${state.logs.paused ? 'Продолжить' : 'Пауза'}</button>
        <button class="ghost" id="log-follow" type="button" aria-pressed="${state.logs.follow}">Автопрокрутка</button>
      </div>
      <div class="log" id="log-container">${(data.entries || []).map(logLine).join('') || '<div class="empty">Записей нет</div>'}</div>`;
  }

  function attachLogs() {
    const container = document.getElementById('log-container');
    const levelSelect = document.getElementById('log-level');
    const querySelect = document.getElementById('log-query');
    const pauseButton = document.getElementById('log-pause');
    const followButton = document.getElementById('log-follow');
    if (!container) return;

    const scrollToEnd = () => { if (state.logs.follow) container.scrollTop = container.scrollHeight; };
    scrollToEnd();

    levelSelect.addEventListener('change', () => { state.logs.level = levelSelect.value; navigate(); });
    let debounce = null;
    querySelect.addEventListener('input', () => {
      window.clearTimeout(debounce);
      debounce = window.setTimeout(() => { state.logs.query = querySelect.value.trim(); navigate(); }, 350);
    });
    pauseButton.addEventListener('click', () => {
      state.logs.paused = !state.logs.paused;
      pauseButton.textContent = state.logs.paused ? 'Продолжить' : 'Пауза';
      pauseButton.setAttribute('aria-pressed', String(state.logs.paused));
      if (state.logs.paused) closeStream(); else openLogStream(container);
    });
    followButton.addEventListener('click', () => {
      state.logs.follow = !state.logs.follow;
      followButton.setAttribute('aria-pressed', String(state.logs.follow));
      scrollToEnd();
    });

    if (!state.logs.paused) openLogStream(container);
  }

  function openLogStream(container) {
    closeStream();
    const url = new URL('/api/logs/stream', window.location.origin);
    if (state.logs.level && state.logs.level !== 'ALL') url.searchParams.set('level', state.logs.level);
    if (state.logs.query) url.searchParams.set('q', state.logs.query);

    const source = new EventSource(url);
    state.stream = source;

    source.addEventListener('message', (event) => {
      let entry;
      try { entry = JSON.parse(event.data); } catch { return; }
      if (entry.id <= state.logs.lastId) return;
      state.logs.lastId = entry.id;

      const empty = container.querySelector('.empty');
      if (empty) empty.remove();
      container.insertAdjacentHTML('beforeend', logLine(entry));
      // Лента не должна расти бесконечно во вкладке, оставленной на сутки.
      while (container.children.length > 600) container.firstElementChild.remove();
      if (state.logs.follow) container.scrollTop = container.scrollHeight;
      setPresence(true);
    });
    source.addEventListener('error', () => setPresence(false));
  }

  function closeStream() {
    if (state.stream) {
      state.stream.close();
      state.stream = null;
    }
  }

  // --------------------------------------------------------------- дневник

  async function renderDiary() {
    if (state.arg) return renderDiaryEntry(state.arg);

    const query = state.diaryQuery || '';
    const data = await api('/api/diary', { q: query, limit: 100 });
    const entries = (data.entries || []).map((entry) =>
      row(
        esc(entry.preview || '(пустая запись)'),
        `${esc(fmtDateTime(entry.created_at))} · ${num(entry.length)} символов · использована ${num(entry.usage_count)} раз(а)`,
        `${entry.confidence >= 1 ? '<span class="tag ok">факт</span>' : ''}${entry.confidence <= -1 ? '<span class="tag err">ложь</span>' : ''}${entry.embedding_dim ? '' : '<span class="tag warn">без эмбеддинга</span>'}${entry.unfinished ? '<span class="tag warn">обрыв</span>' : ''}`,
        `data-diary="${esc(entry.id)}"`,
      ),
    );

    return `
      ${hero(PAGES.diary.kicker, PAGES.diary.title, `${num(data.total)} записей от первого лица — то, что она запомнила о прожитом.`)}
      <div class="controls">
        <div class="field">
          <label for="diary-query">Поиск</label>
          <input id="diary-query" type="search" placeholder="слово или фраза в записях" value="${esc(query)}">
        </div>
        <button class="ghost" id="diary-refresh" type="button">Обновить</button>
      </div>
      ${rows(entries, query ? 'Ничего не найдено' : 'Дневник пока пуст')}`;
  }

  async function renderDiaryEntry(entryId) {
    const entry = await api('/api/diary/entry', { id: entryId });
    return `
      ${hero(PAGES.diary.kicker, 'Запись дневника', fmtDateTime(entry.created_at))}
      <div class="controls">
        <button class="ghost" id="diary-back" type="button">← К списку</button>
      </div>
      <article class="reader">
        <p class="reader-title">${esc(entry.id)}${entry.unfinished ? ' <span class="tag warn">обрыв</span>' : ''}</p>
        <div class="reader-body">${esc(entry.body)}</div>
        ${entry.unfinished ? '<p class="reader-note">Запись обрывается на полуслове — её сохранили до починки бюджетов вывода. Новые записи так не сохраняются.</p>' : ''}
      </article>
      <div class="grid" style="margin-top:1px">
        ${cell('Использована', num(entry.usage_count), `последний раз ${esc(fmtAgo(entry.last_used))}`, 'small')}
        ${cell('Уверенность', fixed(entry.confidence), entry.is_ground_truth ? 'подтверждённый факт' : entry.is_marked_false ? 'помечена как ложь' : 'обычная запись', 'small')}
        ${cell('Эмбеддинг', entry.embedding_dim ? `${num(entry.embedding_dim)} измерений` : 'нет', 'нужен для семантического поиска', 'small')}
      </div>`;
  }

  // ---------------------------------------------------------------- память

  async function renderMemory() {
    const query = state.memoryQuery || '';
    const data = await api('/api/memory', { q: query, limit: 200 });

    const domainTitles = { C: 'о мире', P: 'о человеке', H: 'её опыт' };
    const knowledge = (data.knowledge || []).map((fact) =>
      row(
        `<span class="mono">${esc(fact.entity_id)}</span> · ${esc(String(fact.attribute).replace(/_/g, ' '))}`,
        `${esc(fact.value)}<br>домен ${esc(domainTitles[fact.domain] || fact.domain)} · впервые ${esc(fmtDateTime(fact.first_seen_at))}`,
        `${fact.occurrence_count > 1 ? `<span class="tag ok">подтверждено ${num(fact.occurrence_count)}×</span>` : '<span class="tag mute">однажды</span>'} ${fixed(fact.confidence)}`,
      ),
    );

    const rejections = (data.rejections || []).map((item) =>
      row(
        `${esc(item.entity_id || '—')} · ${esc(item.attribute || '—')}`,
        `${esc(item.value || '')}<br>${esc(fmtDateTime(item.created_at))}`,
        `<span class="tag warn">${esc(item.reason)}</span>`,
      ),
    );

    const facts = (data.facts || []).map((fact) =>
      row(
        `<span class="mono">${esc(fact.entity_id)}</span> · ${esc(fact.fact_key)}`,
        esc(fact.fact_value),
        `${fixed(fact.confidence)} · ${esc(fmtAgo(fact.updated_at))}`,
      ),
    );

    const seeds = (data.seeds || []).map((seed) =>
      row(
        esc(seed.topic),
        `${esc(fmtDateTime(seed.created_at))}${seed.source_chat_id ? ` · из чата ${esc(seed.source_chat_id)}` : ''}`,
        `${seed.status === 'pending' ? '<span class="tag">ждёт</span>' : '<span class="tag ok">изучено</span>'} ${fixed(seed.weight)}`,
      ),
    );

    const social = (data.social || []).map((item) =>
      row(
        esc(item.text),
        `${esc(item.kind)}${item.peer_name ? ` · ${esc(item.peer_name)}` : ''} · ${esc(fmtDateTime(item.created_at))}`,
        esc(item.tags || ''),
      ),
    );

    const conversations = (data.conversations || []).map((item) =>
      row(
        `${esc(item.chat_label)} · собеседник ${esc(item.peer_user_id)}`,
        item.closed_reason ? esc(item.closed_reason) : 'причин закрытия не было',
        `${item.status === 'closed' ? '<span class="tag warn">закрыт</span>' : '<span class="tag ok">активен</span>'} раздражение ${fixed(item.annoyance_score)}`,
      ),
    );

    const threads = (data.threads || []).map((item) =>
      row(
        `Тред ${esc(item.thread_id)} в ${esc(item.chat_id)}`,
        `замечен ${esc(fmtAgo(item.last_seen_at))}`,
        item.commented_at ? `<span class="tag ok">отписалась</span>` : '<span class="tag mute">молчит</span>',
      ),
    );

    const tasks = (data.tasks || []).map((item) =>
      row(
        `${esc(item.task_type)} · ${esc(item.chat_label)}`,
        `на ${esc(fmtDateTime(item.scheduled_at))}`,
        `<span class="tag ${item.status === 'pending' ? '' : 'mute'}">${esc(item.status)}</span>`,
      ),
    );

    return `
      ${hero(PAGES.memory.kicker, PAGES.memory.title, 'Факты, любопытство, внешний опыт и состояние диалогов — всё, что лежит в базе.')}
      <div class="controls">
        <div class="field">
          <label for="memory-query">Поиск по фактам</label>
          <input id="memory-query" type="search" placeholder="сущность, ключ или значение" value="${esc(query)}">
        </div>
        <button class="ghost" id="memory-refresh" type="button">Обновить</button>
      </div>
      ${title('Проверенные знания')}
      ${rows(knowledge, 'Проверенных фактов пока нет')}
      ${title('Отклонено на входе')}
      ${rows(rejections, 'Отклонённых кандидатов нет')}
      ${title('Служебные факты')}
      ${rows(facts, 'Фактов нет')}
      ${title('Семена любопытства')}
      ${rows(seeds, 'Любопытство пока ни за что не зацепилось')}
      ${title('Внешний опыт')}
      ${rows(social, 'Внешнего опыта нет')}
      ${title('Состояние диалогов')}
      ${rows(conversations, 'Диалогов с посторонними не было')}
      ${title('Треды сообщества')}
      ${rows(threads, 'Треды не отслеживались')}
      ${title('Проактивные задачи')}
      ${rows(tasks, 'Задач не запланировано')}`;
  }

  async function renderPeople() {
    const data = await api('/api/people', { limit: 200 });
    const people = (data.people || []).map((person) =>
      row(
        `${esc(person.display_name || 'без имени')} ${person.is_owner ? '<span class="tag ok">владелец</span>' : ''}`,
        `${esc(person.impression || 'впечатление ещё не сложилось')}<br>сообщений ${num(person.message_count)} · последний раз ${esc(fmtAgo(person.last_seen_at))}${person.last_chat_title ? ` · ${esc(person.last_chat_title)}` : ''}`,
        `близость ${fixed(person.affinity)} · уважение ${fixed(person.respect_level)}${meter(person.affinity, true)}`,
      ),
    );

    return `
      ${hero(PAGES.people.kicker, PAGES.people.title, 'Эфи помнит людей отдельно от чатов: у каждого своя история и своё впечатление.')}
      ${rows(people, 'Она пока ни с кем не знакома')}`;
  }

  /*
   * Раздел «Проекты» отвечает на единственный вопрос, который иначе
   * проверяется только руками через GitHub: она правда что-то делает или
   * только говорит, что делает. Отсюда и состав: статус, ссылка и число
   * правок после релиза — именно правки отличают «сгенерировала репозиторий»
   * от «возвращается к своему коду».
   */
  const PROJECT_STATUS_TAGS = {
    pending: ['mute', 'в очереди'],
    speccing: ['warn', 'продумывает'],
    coding: ['warn', 'пишет код'],
    publishing: ['warn', 'выкладывает'],
    done: ['ok', 'готово'],
    failed: ['err', 'не вышло'],
  };

  const projectStatusTag = (status) => {
    const [kind, label] = PROJECT_STATUS_TAGS[status] || ['mute', status];
    return `<span class="tag ${kind}">${esc(label)}</span>`;
  };

  async function renderProjects() {
    const data = await api('/api/projects', { limit: 100 });
    const stats = data.stats || {};
    const projects = data.projects || [];

    if (!data.enabled && !projects.length) {
      return `
        ${hero(PAGES.projects.kicker, PAGES.projects.title, 'Своё ремесло выключено: dev.enabled = false в конфигурации.')}
        <div class="empty">Эфи не пишет проекты. Включите раздел [dev] в behavior.toml, чтобы она начала.</div>`;
    }

    const grid = [
      cell('В работе', num(stats.in_work), 'проектов пишется прямо сейчас'),
      cell('Выложено', num(stats.released), 'доведено до репозитория'),
      cell('Правок после релиза', num(stats.revisions), 'возвращалась и меняла'),
      cell('Не вышло', num(stats.failed), 'брошено на полпути'),
      cell('Правок в чужом коде', num(stats.code_work), 'веток сдано по чужим репозиториям'),
    ].join('');

    const items = projects.map((project) => {
      const tags = [
        projectStatusTag(project.status),
        project.kind === 'swe' ? '<span class="tag">чужой код</span>' : '',
        project.is_collab ? '<span class="tag">вместе</span>' : '<span class="tag">своя затея</span>',
        project.revisions ? `<span class="tag">правок ${esc(project.revisions)}</span>` : '',
      ].join('');
      const link = project.repo_url
        ? `<a href="${esc(project.repo_url)}" target="_blank" rel="noreferrer noopener">${esc(project.repo_url)}</a>`
        : '';
      const stack = (project.stack || []).join(' · ');
      const meta = [
        stack ? esc(stack) : '',
        project.source ? esc(project.source) : '',
        project.branch ? `ветка ${esc(project.branch)}` : '',
        project.reviewed_at ? `перечитывала ${esc(fmtAgo(project.reviewed_at))}` : 'ещё не перечитывала',
        `обновлён ${esc(fmtAgo(project.updated_at))}`,
      ].filter(Boolean).join(' · ');
      const files = (project.files || [])
        .map((file) => `<li><code>${esc(file.path)}</code>${file.purpose ? ` — ${esc(file.purpose)}` : ''}</li>`)
        .join('');

      return `
        <article class="project">
          <header class="project-head">
            <h3>${esc(project.title)}</h3>
            <div class="project-tags">${tags}</div>
          </header>
          ${project.problem && project.problem !== project.title ? `<p class="project-problem">${esc(project.problem)}</p>` : ''}
          ${link ? `<p class="project-link">${link}</p>` : ''}
          ${project.error ? `<p class="project-error">${esc(project.error)}</p>` : ''}
          ${files ? `<ul class="project-files">${files}</ul>` : ''}
          <p class="project-meta">${meta}</p>
        </article>`;
    });

    return `
      ${hero(PAGES.projects.kicker, PAGES.projects.title, 'Что она написала сама, что пишет сейчас и к чему возвращалась после релиза.')}
      <div class="grid wide">${grid}</div>
      ${items.length ? `<div class="projects">${items.join('')}</div>` : '<div class="empty">Проектов пока нет</div>'}`;
  }

  async function renderChats() {
    if (state.arg) return renderChatMessages(state.arg);

    const data = await api('/api/chats', { limit: 200 });
    const chats = (data.chats || []).map((chat) => {
      const tags = [
        chat.is_owner ? '<span class="tag ok">владелец</span>' : '',
        chat.is_allowed ? '<span class="tag">разрешён</span>' : '',
        chat.is_community ? '<span class="tag">сообщество</span>' : '',
        chat.is_generating ? '<span class="tag warn">печатает</span>' : '',
      ].join('');
      return row(
        esc(chat.label),
        `${num(chat.message_count)} сообщений · её ${num(chat.assistant_count)} · последнее ${esc(fmtAgo(chat.last_at))}`,
        `${tags}${chat.affinity !== null && chat.affinity !== undefined ? `<div class="row-sub">близость ${fixed(chat.affinity)}</div>` : ''}`,
        `data-chat="${esc(chat.chat_id)}"`,
      );
    });

    return `
      ${hero(PAGES.chats.kicker, PAGES.chats.title, 'Все чаты, о которых у Эфи есть история. Нажмите, чтобы прочитать переписку.')}
      ${rows(chats, 'Переписки ещё не было')}`;
  }

  async function renderChatMessages(chatId) {
    const data = await api('/api/chats/messages', { chat_id: chatId, limit: 200 });
    const messages = (data.messages || []).map((message) => {
      const role = String(message.role || 'user');
      const label = { assistant: 'Эфи', user: 'собеседник', tool: 'инструмент', system: 'система' }[role] || role;
      const body = message.content || (message.tool_calls ? `вызов инструмента: ${message.tool_calls}` : '');
      return `<div class="bubble ${esc(role)}">${esc(body)}<span class="bubble-meta">${esc(label)} · ${esc(fmtDateTime(message.created_at))}</span></div>`;
    });

    return `
      ${hero(PAGES.chats.kicker, 'Переписка', data.label)}
      <div class="controls">
        <button class="ghost" id="chats-back" type="button">← К списку чатов</button>
      </div>
      ${messages.length ? `<div class="messages">${messages.join('')}</div>` : '<div class="empty">Сообщений нет</div>'}`;
  }

  async function renderLLM() {
    const [metrics, functions] = await Promise.all([api('/api/metrics', { limit: 100 }), api('/api/functions')]);
    const totals = metrics.totals || {};

    const grid = [
      cell('Вызовов', num(totals.calls), `ошибок ${num(totals.errors)} (${pct(totals.error_rate)})`),
      cell('Среднее время', `${fixed(totals.avg_seconds)}<span class="unit">c</span>`, 'на один вызов'),
      cell('Токены запроса', num(totals.prompt_tokens), `ответа ${num(totals.completion_tokens)}`),
      cell('Стоимость', fixed(totals.cost, 4), 'если провайдер её сообщает'),
    ].join('');

    const roles = (functions.llm_roles || []).map((role) => {
      const candidates = role.candidates
        .map(
          (candidate) =>
            `<div class="row-sub"><span class="mono">${esc(candidate.model)}</span> · ${esc(candidate.slot)} · таймаут ${fixed(candidate.timeout_seconds, 0)} c${
              candidate.cooldown_seconds ? ` · <span class="tag warn">пауза ${fixed(candidate.cooldown_seconds, 0)} c</span>` : ''
            }</div>`,
        )
        .join('');
      return row(
        esc(role.role.toUpperCase()),
        candidates || 'кандидатов нет',
        role.degrade_to ? `деградация в ${esc(role.degrade_to)}` : '',
      );
    });

    const endpoints = (metrics.endpoints || []).map((endpoint) =>
      row(
        `<span class="mono">${esc(endpoint.model || endpoint.provider)}</span>`,
        `${esc(endpoint.operation)} · ${esc(endpoint.base_url)}`,
        `${num(endpoint.calls)} вызов(ов) · среднее ${fixed(endpoint.avg_seconds)} c${
          endpoint.errors ? ` · <span class="tag err">ошибок ${num(endpoint.errors)}</span>` : ''
        }`,
      ),
    );

    const recent = (metrics.recent || []).map((event) =>
      row(
        `<span class="mono">${esc(event.model || event.provider)}</span> · ${esc(event.operation)}`,
        `${esc(fmtDateTime(event.timestamp))}${event.error ? ` · <span style="color:var(--err)">${esc(event.error)}</span>` : ''}`,
        `${fixed(event.duration_seconds)} c · ${num(event.prompt_tokens)}/${num(event.completion_tokens)} т.`,
      ),
    );

    return `
      ${hero(PAGES.llm.kicker, PAGES.llm.title, 'Метрики собираются только пока процесс жив — это снимок текущего запуска.')}
      <div class="grid wide">${grid}</div>
      ${title('Роли и маршруты')}
      ${rows(roles, 'Роли не настроены')}
      ${title('Эндпоинты')}
      ${rows(endpoints, 'Вызовов ещё не было')}
      ${title('Последние вызовы')}
      ${rows(recent, 'Лента пуста')}`;
  }

  async function renderPrompts() {
    const data = await api('/api/prompts');
    const template = data.personality_template || data.personality_prompt || '';
    return `
      ${hero(PAGES.prompts.kicker, PAGES.prompts.title, `${esc(data.character_name)} — то, из чего собирается системный промпт.`)}
      <div class="grid duo">
        ${cell('Имя', esc(data.character_name), 'подставляется в шаблон личности', 'small')}
        ${cell('Обращение к владельцу', esc(data.owner_display_name || 'из Telegram-профиля'), 'подстановка {user_name}', 'small')}
      </div>
      ${title('Шаблон личности')}
      <article class="reader">
        <p class="reader-title">${data.personality_template ? 'personality.md' : 'personality_prompt из конфигурации'}</p>
        <div class="reader-body">${esc(template || 'Шаблон не задан')}</div>
      </article>
      ${title('Защита от угодливости')}
      <article class="reader">
        <div class="reader-body">${esc(data.sycophancy_protection)}</div>
      </article>`;
  }

  // ------------------------------------------------------------ маршрутизация

  function parseHash() {
    const raw = window.location.hash.replace(/^#\/?/, '');
    const [route, ...rest] = raw.split('/');
    return { route: PAGES[route] ? route : 'overview', arg: rest.length ? decodeURIComponent(rest.join('/')) : null };
  }

  function markNav(route) {
    nav.querySelectorAll('a').forEach((link) => {
      if (link.dataset.route === route) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
    });
  }

  function attachHandlers() {
    const back = (target) => () => { window.location.hash = target; };

    document.getElementById('diary-back')?.addEventListener('click', back('#/diary'));
    document.getElementById('chats-back')?.addEventListener('click', back('#/chats'));
    document.getElementById('diary-refresh')?.addEventListener('click', () => navigate());
    document.getElementById('memory-refresh')?.addEventListener('click', () => navigate());

    const diaryQuery = document.getElementById('diary-query');
    diaryQuery?.addEventListener('change', () => { state.diaryQuery = diaryQuery.value.trim(); navigate(); });
    const memoryQuery = document.getElementById('memory-query');
    memoryQuery?.addEventListener('change', () => { state.memoryQuery = memoryQuery.value.trim(); navigate(); });

    view.querySelectorAll('[data-diary]').forEach((element) => {
      element.addEventListener('click', () => { window.location.hash = `#/diary/${encodeURIComponent(element.dataset.diary)}`; });
    });
    view.querySelectorAll('[data-chat]').forEach((element) => {
      element.addEventListener('click', () => { window.location.hash = `#/chats/${encodeURIComponent(element.dataset.chat)}`; });
    });

    if (state.route === 'logs') attachLogs();
  }

  async function navigate() {
    const page = PAGES[state.route];
    try {
      view.innerHTML = await page.load();
      setPresence(true);
    } catch (error) {
      setPresence(false);
      view.innerHTML = `
        ${hero('Эфи', 'Нет связи', 'Процесс не отвечает или дашборд остановлен. Страница попробует ещё раз автоматически.')}
        <div class="empty">${esc(error.message || error)}</div>`;
    }
    attachHandlers();
  }

  function scheduleRefresh() {
    window.clearInterval(state.timer);
    const page = PAGES[state.route];
    if (!page.live) {
      footerRefresh.textContent = state.route === 'logs' ? 'живой поток событий' : 'обновление по кнопке';
      return;
    }
    footerRefresh.textContent = `обновление каждые ${REFRESH_MS / 1000} с`;
    state.timer = window.setInterval(() => {
      // Не перерисовываем страницу под руками: пока пользователь набирает в
      // поле или вкладка скрыта, автообновление ждёт.
      if (document.hidden) return;
      if (document.activeElement && ['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
      navigate();
    }, REFRESH_MS);
  }

  async function route() {
    closeStream();
    const { route: name, arg } = parseHash();
    state.route = name;
    state.arg = arg;
    markNav(name);
    nav.classList.remove('open');
    burger.setAttribute('aria-expanded', 'false');
    window.scrollTo({ top: 0, behavior: 'instant' in window ? 'instant' : 'auto' });
    await navigate();
    scheduleRefresh();
  }

  // ------------------------------------------------------------------ тема

  function applyTheme(theme) {
    if (theme) document.documentElement.setAttribute('data-theme', theme);
    else document.documentElement.removeAttribute('data-theme');
  }

  function initTheme() {
    const saved = window.localStorage.getItem('efi-theme');
    applyTheme(saved);
    document.getElementById('theme-toggle').addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-theme');
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      const next = current ? (current === 'dark' ? 'light' : 'dark') : prefersDark ? 'light' : 'dark';
      window.localStorage.setItem('efi-theme', next);
      applyTheme(next);
    });
  }

  // ------------------------------------------------------------------ старт

  burger.addEventListener('click', () => {
    const open = nav.classList.toggle('open');
    burger.setAttribute('aria-expanded', String(open));
  });

  window.addEventListener('hashchange', route);
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden && PAGES[state.route].live) navigate();
  });

  initTheme();
  if (!window.location.hash) window.location.hash = '#/overview';
  route();
})();
