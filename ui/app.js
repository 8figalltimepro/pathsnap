/* PathSnap evaluator UI — vanilla JS + bundled Leaflet. */
(function () {
  'use strict';

  var DEFAULT_API = 'http://localhost:8000';
  var STORAGE = { api: 'pathsnap.apiBase', model: 'pathsnap.model' };
  var TIMEOUT_CONNECT = 20000;
  var TIMEOUT_EVAL = 180000;
  var MODELS_RETRY_MS = 45000;

  var MAP = {
    network: '#8F8F8F',
    gps: '#0072F5',
    clf: '#E5484D',
    hmm: '#45A557',
    casing: '#ffffff'
  };

  var state = {
    apiBase: '',
    health: null,
    trips: null,
    models: null,
    tripId: null,
    modelId: null,
    current: null,
    cache: new Map(),
    token: 0,
    active: 0,
    map: null,
    groups: null,
    initial: null,
    modelsTimer: null
  };

  var els = {};

  /* ---------------------------------------------------------------- helpers */
  function $(id) { return document.getElementById(id); }

  function bindIds(map) {
    Object.keys(map).forEach(function (key) { els[key] = $(map[key]); });
  }

  function isNum(v) { return typeof v === 'number' && isFinite(v); }

  function ratio(v, digits) {
    if (!isNum(v)) return '\u2014';
    var x = v > 1.5 ? v / 100 : v;
    return x.toFixed(digits == null ? 3 : digits);
  }

  function pct(v, digits) {
    if (!isNum(v)) return '\u2014';
    var x = v > 1.5 ? v : v * 100;
    return x.toFixed(digits == null ? 1 : digits) + '%';
  }

  function pctPlain(v) {
    if (!isNum(v)) return '\u2014';
    return (v > 1.5 ? v : v * 100).toFixed(1);
  }

  function count(v) {
    if (!isNum(v)) return '\u2014';
    return Math.round(v).toLocaleString('en-US');
  }

  function msCompact(v) {
    if (!isNum(v)) return '\u2014';
    return v >= 10 ? String(Math.round(v)) : v.toFixed(1);
  }

  function msTotal(v) {
    if (!isNum(v)) return '\u2014';
    return v.toFixed(1) + ' ms';
  }

  function setText(el, text) { if (el) el.textContent = text; }

  function normalizeApiBase(raw) {
    var v = String(raw == null ? '' : raw).trim();
    if (!v) throw new Error('empty URL');
    if (/\s/.test(v)) throw new Error('invalid URL');
    if (!/^[a-z][a-z0-9+.-]*:\/\//i.test(v)) v = 'http://' + v;
    v = v.replace(/\/+$/, '');
    var url = new URL(v);
    if (url.protocol !== 'http:' && url.protocol !== 'https:') throw new Error('unsupported protocol');
    return v;
  }

  function hostOf(base) {
    try { return new URL(base).host; } catch (err) { return base || '\u2014'; }
  }

  function readStorage(key) {
    try { return window.localStorage.getItem(key); } catch (err) { return null; }
  }

  function writeStorage(key, value) {
    try { window.localStorage.setItem(key, value); } catch (err) { /* private mode */ }
  }

  /* -------------------------------------------------------------- API client */
  async function apiGet(path, timeoutMs) {
    var controller = new AbortController();
    var timer = setTimeout(function () { controller.abort(); }, timeoutMs || TIMEOUT_CONNECT);
    try {
      var res = await fetch(state.apiBase + path, {
        method: 'GET',
        headers: { Accept: 'application/json' },
        signal: controller.signal,
        cache: 'no-store'
      });
      if (!res.ok) {
        var detail = '';
        try {
          var body = await res.json();
          if (body && body.detail) detail = String(body.detail);
        } catch (ignored) { /* not JSON */ }
        throw new Error('HTTP ' + res.status + (detail ? ' \u00b7 ' + detail : ''));
      }
      return await res.json();
    } catch (err) {
      if (err && err.name === 'AbortError') throw new Error('No response from API');
      if (err instanceof TypeError) throw new Error('Could not reach API');
      throw err;
    } finally {
      clearTimeout(timer);
    }
  }

  /* ------------------------------------------------------------- page chrome */
  function setConn(kind, text) {
    els.conn.className = 'conn' + (kind ? ' ' + kind : '');
    setText(els.connText, text);
  }

  function begin() {
    state.active += 1;
    if (els.progress) els.progress.hidden = false;
  }

  function end() {
    state.active = Math.max(0, state.active - 1);
    if (state.active === 0 && els.progress) els.progress.hidden = true;
  }

  function setResultState(text) { setText(els.resultState, text); }

  function updateStatusBar() {
    var health = state.health;
    var trips = state.trips;
    var connected = !!(health && trips);
    var trained = connected && health.mode === 'trained';

    els.modeDot.className = 'mode-dot' + (connected ? (trained ? ' ok' : ' err') : '');
    setText(els.modeText, connected ? (trips.mode === 'evaluation' ? 'Evaluation' : 'Demo') : 'Offline');
    setText(els.partitionText, connected ? (health.eval_partition || trips.partition || '\u2014') : '\u2014');
    setText(els.trainText, connected ? (trained ? 'Trained' : 'Baseline') : '\u2014');

    var n = state.models && state.models.models ? state.models.models.length : null;
    if (!n && health && isNum(health.n_models)) n = health.n_models;
    setText(els.modelsStatus, n ? n + ' models' : (connected ? 'model \u2014' : '\u2014'));
    setText(els.endpointText, state.apiBase ? hostOf(state.apiBase) : '\u2014');
  }

  function clearMetricValues() {
    [els.top1clf, els.top1hmm, els.contClf, els.contHmm, els.seqClf, els.seqHmm,
      els.f1, els.logloss, els.miss, els.scored, els.totalMs, els.latencyLine].forEach(function (el) {
      setText(el, '\u2014');
    });
  }

  function clearResult(message) {
    clearMetricValues();
    els.metricsSection.hidden = true;
    els.scoringSection.hidden = true;
    els.latencySection.hidden = true;
    els.resultEmpty.hidden = false;
    setText(els.resultEmpty, message || 'Select a trip');
    els.resultEmpty.title = '';
    setText(els.resultTrip, '\u2014');
    setText(els.resultObs, '');
    setText(els.resultModel, '\u2014');
    els.resultModel.title = '';
    setResultState('Idle');
  }

  function showResultError(message) {
    els.metricsSection.hidden = true;
    els.scoringSection.hidden = true;
    els.latencySection.hidden = true;
    els.resultEmpty.hidden = false;
    setText(els.resultEmpty, 'Evaluation failed');
    els.resultEmpty.title = message || '';
    setResultState('Error');
  }

  /* ----------------------------------------------------------------- models */
  function modelName(id) {
    if (!state.models || !state.models.models) return id || '';
    var found = state.models.models.filter(function (m) { return m.id === id; })[0];
    return found && found.name ? found.name : (id || '');
  }

  function renderModelsTable(list) {
    els.modelsBody.innerHTML = '';
    list.forEach(function (m) {
      var tr = document.createElement('tr');
      tr.dataset.model = m.id;
      tr.tabIndex = 0;
      var name = document.createElement('td');
      name.textContent = m.name || m.id;
      name.title = (m.name || m.id) + (m.family ? ' \u00b7 ' + m.family : '');
      var cv = document.createElement('td');
      cv.textContent = ratio(m.cv_macro_f1, 3);
      cv.title = 'Recorded 5-fold CV macro F1';
      var test = document.createElement('td');
      test.textContent = pctPlain(m.test_top1);
      test.title = 'Recorded student-test top-1 accuracy';
      var hmm = document.createElement('td');
      hmm.textContent = pctPlain(m.hmm_top1 != null ? m.hmm_top1 : m.test_top1_hmm);
      hmm.title = 'Recorded student-test top-1 after HMM decoding';
      tr.appendChild(name);
      tr.appendChild(cv);
      tr.appendChild(test);
      tr.appendChild(hmm);
      tr.addEventListener('click', function () { setModel(m.id); });
      tr.addEventListener('keydown', function (event) {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          setModel(m.id);
        }
      });
      els.modelsBody.appendChild(tr);
    });
    highlightModelRow();
  }

  function highlightModelRow() {
    var rows = els.modelsBody.querySelectorAll('tr');
    for (var i = 0; i < rows.length; i += 1) {
      rows[i].classList.toggle('selected', rows[i].dataset.model === state.modelId);
    }
  }

  function applyModels(payload) {
    var list = payload && Array.isArray(payload.models) ? payload.models : [];
    state.models = list.length ? payload : null;
    els.modelSelect.innerHTML = '';

    if (!list.length) {
      var empty = document.createElement('option');
      empty.value = '';
      empty.textContent = '\u2014';
      els.modelSelect.appendChild(empty);
      els.modelSelect.disabled = true;
      els.modelsTable.hidden = true;
      els.modelsEmpty.hidden = false;
      setText(els.modelsEmpty, 'Model list unavailable');
      setText(els.modelCount, '\u2014');
      state.modelId = null;
      return;
    }

    var candidates = [state.modelId, state.initial && state.initial.get('model'),
      readStorage(STORAGE.model), payload.default];
    var chosen = null;
    for (var i = 0; i < candidates.length; i += 1) {
      var candidate = candidates[i];
      if (!candidate) continue;
      if (list.some(function (m) { return m.id === candidate; })) { chosen = candidate; break; }
    }
    if (!chosen) chosen = list[0].id;

    list.forEach(function (m) {
      var option = document.createElement('option');
      option.value = m.id;
      option.textContent = m.name || m.id;
      els.modelSelect.appendChild(option);
    });
    els.modelSelect.disabled = false;
    els.modelSelect.value = chosen;
    state.modelId = chosen;
    writeStorage(STORAGE.model, chosen);

    els.modelsEmpty.hidden = true;
    els.modelsTable.hidden = false;
    setText(els.modelCount, String(list.length));
    renderModelsTable(list);
  }

  function setModel(id) {
    if (!id || id === state.modelId) return;
    if (!state.models || !state.models.models.some(function (m) { return m.id === id; })) return;
    state.modelId = id;
    els.modelSelect.value = id;
    writeStorage(STORAGE.model, id);
    highlightModelRow();
    if (state.tripId) loadTrip(state.tripId);
    else setText(els.resultModel, id);
  }

  function scheduleModelsRetry() {
    clearTimeout(state.modelsTimer);
    state.modelsTimer = setTimeout(refreshModels, MODELS_RETRY_MS);
  }

  async function refreshModels() {
    if (!state.apiBase) return;
    try {
      var payload = await apiGet('/models', 15000);
      if (!payload || !Array.isArray(payload.models) || !payload.models.length) throw new Error('empty model list');
      applyModels(payload);
      updateStatusBar();
      if (state.tripId) loadTrip(state.tripId);
    } catch (err) {
      scheduleModelsRetry();
    }
  }

  /* ------------------------------------------------------------------ trips */
  function renderTrips(list) {
    els.tripList.innerHTML = '';
    els.tripEmpty.hidden = !!list.length;
    setText(els.tripCount, list.length ? String(list.length) : '0');
    list.forEach(function (trip) {
      var btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'trip';
      btn.dataset.tripId = trip.trip_id;
      var id = document.createElement('span');
      id.className = 'tid';
      id.textContent = trip.trip_id;
      var obs = document.createElement('span');
      obs.className = 'tobs';
      obs.textContent = count(trip.n_obs);
      btn.appendChild(id);
      btn.appendChild(obs);
      btn.addEventListener('click', function () { selectTrip(trip.trip_id); });
      els.tripList.appendChild(btn);
    });
    highlightTripRow();
  }

  function highlightTripRow() {
    var rows = els.tripList.querySelectorAll('.trip');
    for (var i = 0; i < rows.length; i += 1) {
      rows[i].classList.toggle('active', rows[i].dataset.tripId === state.tripId);
    }
  }

  function selectTrip(id) {
    if (!id) return;
    state.tripId = id;
    highlightTripRow();
    loadTrip(id);
  }

  async function loadTrip(tripId) {
    var seq = ++state.token;
    var model = state.modelId;
    var key = state.apiBase + '|' + tripId + '|' + (model || '-');

    var cached = state.cache.get(key);
    if (cached) {
      state.current = cached;
      drawPayload(cached);
      renderPayload(cached);
      return;
    }

    setResultState('Loading');
    setText(els.resultTrip, tripId);
    var trip = findTrip(tripId);
    setText(els.resultObs, trip ? count(trip.n_obs) + ' fixes' : '');
    setText(els.resultModel, modelName(model) || model || 'default');
    els.resultModel.title = model || '';
    clearMetricValues();

    begin();
    try {
      var path = '/evaluate/' + encodeURIComponent(tripId) + (model ? '?model=' + encodeURIComponent(model) : '');
      var payload = await apiGet(path, TIMEOUT_EVAL);
      if (seq !== state.token) return;
      state.cache.set(key, payload);
      state.current = payload;
      drawPayload(payload);
      renderPayload(payload);
    } catch (err) {
      if (seq !== state.token) return;
      if (err.message === 'Could not reach API' || err.message === 'No response from API') setConn('err', 'Offline');
      showResultError(err.message);
    } finally {
      end();
    }
  }

  function findTrip(id) {
    var list = (state.trips && state.trips.trips) || [];
    return list.filter(function (t) { return t.trip_id === id; })[0] || null;
  }

  function renderPayload(payload) {
    var m = payload.metrics || {};
    var hasReference = m.reference_available !== false;

    setText(els.resultTrip, payload.trip_id || state.tripId || '\u2014');
    setText(els.resultObs, count(payload.n_obs) + ' fixes');
    var modelId = payload.model || state.modelId;
    setText(els.resultModel, modelName(modelId) || payload.model_name || modelId || '\u2014');
    els.resultModel.title = modelId || '';

    if (modelId && state.models && modelId !== state.modelId &&
        state.models.models.some(function (x) { return x.id === modelId; })) {
      state.modelId = modelId;
      els.modelSelect.value = modelId;
      highlightModelRow();
    }

    if (!hasReference) {
      els.metricsSection.hidden = true;
      els.scoringSection.hidden = true;
      els.latencySection.hidden = true;
      els.resultEmpty.hidden = false;
      setText(els.resultEmpty, 'No reference route for this trip');
      els.resultEmpty.title = '';
      setResultState('Live');
      return;
    }

    els.resultEmpty.hidden = true;
    els.metricsSection.hidden = false;
    els.scoringSection.hidden = false;
    els.latencySection.hidden = false;

    setText(els.top1clf, pct(m.top1_accuracy));
    setText(els.top1hmm, pct(m.top1_accuracy_hmm));
    setText(els.contClf, pct(m.route_continuity));
    setText(els.contHmm, pct(m.route_continuity_hmm));
    setText(els.seqClf, ratio(m.seq_f1_clf, 3));
    setText(els.seqHmm, ratio(m.seq_f1_hmm, 3));
    setText(els.f1, ratio(m.macro_f1, 3));
    setText(els.logloss, ratio(m.log_loss, 3));
    setText(els.miss, pct(m.candidate_miss_rate, 2));
    setText(els.scored, pct(m.scored_fraction));

    var t = payload.timings_ms || {};
    setText(els.totalMs, msTotal(t.total_ms));
    setText(els.latencyLine, 'NET ' + msCompact(t.load_network_ms) +
      ' \u00b7 CAND ' + msCompact(t.candidates_ms) +
      ' \u00b7 FEAT ' + msCompact(t.features_model_ms) +
      ' \u00b7 HMM ' + msCompact(t.hmm_ms));
    setResultState('Live');
  }

  /* -------------------------------------------------------------------- map */
  function toLatLng(pair) { return [pair[1], pair[0]]; }

  function initMap() {
    var map = L.map('map', { preferCanvas: true, minZoom: 3, attributionControl: false });
    map.setView([56.15, 54.6], 9);
    L.control.attribution({ prefix: false, position: 'bottomright' }).addTo(map);

    var tiles = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a>'
    });
    tiles.addTo(map);

    var tileStats = { loaded: false, errors: 0 };
    tiles.on('tileload', function () {
      tileStats.loaded = true;
      els.tileNote.hidden = true;
    });
    tiles.on('tileerror', function () {
      tileStats.errors += 1;
      if (!tileStats.loaded && tileStats.errors >= 6) els.tileNote.hidden = false;
    });

    var groups = {
      network: L.layerGroup().addTo(map),
      gps: L.layerGroup().addTo(map),
      clf: L.layerGroup().addTo(map),
      hmm: L.layerGroup().addTo(map)
    };

    addLayerControl(map, groups);
    state.map = map;
    state.groups = groups;
    map.invalidateSize();
    setTimeout(function () { map.invalidateSize(); }, 0);
    window.addEventListener('resize', function () { map.invalidateSize(); });
  }

  function addLayerControl(map, groups) {
    var control = L.control({ position: 'topright' });
    control.onAdd = function () {
      var card = L.DomUtil.create('div', 'layer-card');
      L.DomEvent.disableClickPropagation(card);
      L.DomEvent.disableScrollPropagation(card);
      var head = document.createElement('div');
      head.className = 'layer-head';
      head.textContent = 'Layers';
      card.appendChild(head);
      [
        ['gps', 'GPS', 'swatch-gps'],
        ['hmm', 'HMM', 'swatch-hmm'],
        ['clf', 'CLF', 'swatch-clf'],
        ['network', 'NET', 'swatch-net']
      ].forEach(function (row) {
        var label = document.createElement('label');
        label.className = 'layer-row';
        var input = document.createElement('input');
        input.type = 'checkbox';
        input.checked = true;
        input.setAttribute('aria-label', row[1] + ' layer');
        var swatch = document.createElement('span');
        swatch.className = 'swatch ' + row[2];
        var text = document.createElement('span');
        text.textContent = row[1];
        label.appendChild(input);
        label.appendChild(swatch);
        label.appendChild(text);
        input.addEventListener('change', function () {
          if (input.checked) map.addLayer(groups[row[0]]);
          else map.removeLayer(groups[row[0]]);
        });
        card.appendChild(label);
      });
      return card;
    };
    control.addTo(map);
  }

  function clearLayers() {
    if (!state.groups) return;
    Object.keys(state.groups).forEach(function (name) { state.groups[name].clearLayers(); });
  }

  function drawPayload(payload) {
    clearLayers();
    drawNetwork(payload.network);
    drawGps(payload.gps);
    drawStrokes(payload.predicted_strokes_hmm, { color: MAP.casing, weight: 5.5, opacity: 0.9 }, state.groups.hmm);
    drawStrokes(payload.predicted_strokes_hmm, {
      color: MAP.hmm, weight: 2.8, opacity: 0.92, lineCap: 'round', lineJoin: 'round'
    }, state.groups.hmm);
    drawStrokes(payload.predicted_strokes_clf, {
      color: MAP.clf, weight: 2, opacity: 0.95, dashArray: '6 4'
    }, state.groups.clf);
    fitToPayload(payload);
  }

  function drawNetwork(network) {
    if (!network || !network.nodes || !network.nodes.length) return;
    var latlngs = network.nodes.map(toLatLng);
    var segments = [];
    (network.arcs || []).forEach(function (arc) {
      var a = arc[0];
      var b = arc[1];
      if (a < 0 || b < 0 || a >= latlngs.length || b >= latlngs.length) return;
      segments.push([latlngs[a], latlngs[b]]);
    });
    if (segments.length) {
      L.polyline(segments, { color: MAP.network, weight: 1, opacity: 0.9, interactive: false })
        .addTo(state.groups.network);
    }
  }

  function drawGps(points) {
    if (!points || !points.length) return;
    var markers = points.map(function (point) {
      // Draw the fixes as a single polyline for weight; circle markers for validity.
      return L.circleMarker(toLatLng(point), {
        radius: 2.2, stroke: false, fillColor: MAP.gps, fillOpacity: 0.8
      });
    });
    L.layerGroup(markers).addTo(state.groups.gps);
  }

  function drawStrokes(strokes, style, group) {
    (strokes || []).forEach(function (stroke) {
      if (!stroke || stroke.length < 2) return;
      L.polyline(stroke.map(toLatLng), Object.assign({}, style)).addTo(group);
    });
  }

  function fitToPayload(payload) {
    var bounds = L.latLngBounds([]);
    function addPairs(list) {
      if (!list) return;
      list.forEach(function (point) {
        if (!point || point.length < 2) return;
        var ll = toLatLng(point);
        if (isFinite(ll[0]) && isFinite(ll[1])) bounds.extend(ll);
      });
    }
    addPairs(payload.gps);
    if (payload.network) addPairs(payload.network.nodes);
    (payload.predicted_strokes_hmm || []).forEach(addPairs);
    (payload.predicted_strokes_clf || []).forEach(addPairs);
    if (!bounds.isValid()) return;
    state.map.fitBounds(bounds, { padding: [26, 26], maxZoom: 18 });
  }

  /* --------------------------------------------------------------- settings */
  function setSettingsOpen(open) {
    els.settings.hidden = !open;
    els.settingsToggle.setAttribute('aria-expanded', String(open));
    if (open) {
      els.apiInput.value = state.apiBase || els.apiInput.value || DEFAULT_API;
      els.apiInput.focus();
      els.apiInput.select();
    }
  }

  function applyApiInput() {
    connect();
  }

  /* -------------------------------------------------------------- connection */
  async function connect() {
    var base;
    try {
      base = normalizeApiBase(els.apiInput.value || DEFAULT_API);
    } catch (err) {
      setConn('err', 'Invalid URL');
      els.apiInput.focus();
      return;
    }

    state.apiBase = base;
    els.apiInput.value = base;
    writeStorage(STORAGE.api, base);
    state.cache.clear();
    state.current = null;
    state.models = null;
    state.modelId = null;
    state.tripId = null;
    clearLayers();
    highlightTripRow();
    clearResult('Select a trip');
    setSettingsOpen(false);
    setConn('busy', 'Connecting');
    updateStatusBar();

    begin();
    var results = await Promise.allSettled([
      apiGet('/health', TIMEOUT_CONNECT),
      apiGet('/trips', TIMEOUT_CONNECT),
      apiGet('/models', TIMEOUT_CONNECT)
    ]);
    end();

    var health = results[0].status === 'fulfilled' ? results[0].value : null;
    var trips = results[1].status === 'fulfilled' ? results[1].value : null;

    if (!health || !trips) {
      state.health = null;
      state.trips = null;
      setConn('err', 'Offline');
      els.tripList.innerHTML = '';
      els.tripEmpty.hidden = false;
      setText(els.tripEmpty, 'API unreachable');
      setText(els.tripCount, '\u2014');
      els.modelsTable.hidden = true;
      els.modelsEmpty.hidden = false;
      setText(els.modelsEmpty, 'Model list unavailable');
      setText(els.modelCount, '\u2014');
      showResultError('Could not reach ' + hostOf(base));
      updateStatusBar();
      return;
    }

    state.health = health;
    state.trips = trips;
    setConn('ok', 'Connected');

    if (results[2].status === 'fulfilled' && results[2].value) {
      clearTimeout(state.modelsTimer);
      applyModels(results[2].value);
    } else {
      applyModels(null);
      scheduleModelsRetry();
    }

    renderTrips(trips.trips || []);
    updateStatusBar();

    if (!(trips.trips || []).length) {
      showResultError('No trips in this partition');
      return;
    }
    var wanted = state.initial && state.initial.get('trip');
    var list = trips.trips || [];
    if (wanted && list.some(function (t) { return t.trip_id === wanted; })) selectTrip(wanted);
    else clearResult('Select a trip');
  }

  /* ------------------------------------------------------------------- boot */
  function bindEvents() {
    els.modelSelect.addEventListener('change', function () { setModel(els.modelSelect.value); });
    els.settingsToggle.addEventListener('click', function () { setSettingsOpen(els.settings.hidden); });
    els.settingsClose.addEventListener('click', function () { setSettingsOpen(false); });
    els.apiApply.addEventListener('click', applyApiInput);
    els.apiReset.addEventListener('click', function () {
      els.apiInput.value = DEFAULT_API;
      els.apiInput.focus();
      els.apiInput.select();
    });
    els.apiInput.addEventListener('keydown', function (event) {
      if (event.key === 'Enter') { event.preventDefault(); applyApiInput(); }
    });
    document.addEventListener('click', function (event) {
      if (els.settings.hidden) return;
      if (els.settings.contains(event.target) || els.settingsToggle.contains(event.target)) return;
      setSettingsOpen(false);
    });
    document.addEventListener('keydown', function (event) {
      if (event.key === 'Escape' && !els.settings.hidden) {
        setSettingsOpen(false);
        els.settingsToggle.focus();
      }
    });
  }

  function boot() {
    bindIds({
      apiInput: 'api-base', apiApply: 'api-apply', apiReset: 'api-reset',
      settings: 'settings', settingsToggle: 'settings-toggle', settingsClose: 'settings-close',
      modelSelect: 'model-select', conn: 'conn', connText: 'conn-text', progress: 'progress',
      tripList: 'trip-list', tripCount: 'trip-count', tripEmpty: 'trip-empty',
      modelsTable: 'models-table', modelsBody: 'models-body', modelsEmpty: 'models-empty',
      modelCount: 'model-count', resultState: 'result-state', resultTrip: 'result-trip',
      resultObs: 'result-obs', resultModel: 'result-model', metricsSection: 'metrics-section',
      scoringSection: 'scoring-section', latencySection: 'latency-section', resultEmpty: 'result-empty',
      top1clf: 'm-top1-clf', top1hmm: 'm-top1-hmm', contClf: 'm-cont-clf', contHmm: 'm-cont-hmm',
      seqClf: 'm-seq-clf', seqHmm: 'm-seq-hmm', f1: 'm-f1', logloss: 'm-logloss',
      miss: 'm-miss', scored: 'm-scored', totalMs: 'm-total', latencyLine: 'm-latency',
      modeDot: 'mode-dot', modeText: 'mode-text', partitionText: 'partition-text',
      trainText: 'train-text', modelsStatus: 'models-status', endpointText: 'endpoint-text',
      tileNote: 'tile-note'
    });

    state.initial = new URLSearchParams(window.location.search);
    var fromQuery = state.initial.get('api');
    if (fromQuery) writeStorage(STORAGE.api, fromQuery);
    els.apiInput.value = state.apiBase = '';

    bindEvents();
    initMap();
    window.pathsnapUI = state;

    var startBase = fromQuery || readStorage(STORAGE.api) || DEFAULT_API;
    try { startBase = normalizeApiBase(startBase); } catch (err) { startBase = DEFAULT_API; }
    els.apiInput.value = startBase;
    connect();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
}());
