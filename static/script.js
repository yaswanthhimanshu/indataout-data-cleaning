/**
 * CanvasGrid — lightweight spreadsheet grid using HTML5 Canvas
 * Supports: read-only preview mode and full edit mode
 * No external dependencies.
 *
 * Usage:
 *   const grid = new CanvasGrid(containerEl, { columns, data, editable, onCellChange })
 *   grid.setData(columns, data)        // replace data
 *   grid.patchRow(rowIdx, colKey, val) // update single cell
 *   grid.destroy()
 */
/**
 * inDataOut browser workflow controller.
 *
 * The browser owns interaction state, busy/progress affordances, visible
 * previews, and editor paging. It does not own authoritative dataset state:
 * committed Apply, Undo/Reapply, resume, final export, and history evidence all
 * come from Flask responses backed by persisted snapshots/history.
 *
 * window._ido is the intentionally shared namespace between page subsystems
 * (upload, cleaning panels, Change Log, custom code, and Dataset Editor). Keep
 * route names, request field names, and exported function names stable because
 * templates and tests rely on them.
 */
class CanvasGrid {
  constructor(container, opts = {}) {
    this.container  = container;
    this.opts       = Object.assign({
      editable:      false,
      rowHeight:     28,
      headerHeight:  32,
      minColWidth:   80,
      maxColWidth:   300,
      fontSize:      13,
      fontFamily:    'DM Sans, system-ui, sans-serif',
      headerBg:      '#1e293b',
      headerFg:      '#94a3b8',
      rowBg:         '#0f172a',
      rowAltBg:      '#1a2235',
      rowFg:         '#e2e8f0',
      borderColor:   '#2d3f55',
      selBg:         '#2563eb33',
      selBorder:     '#3b82f6',
      editBg:        '#1e3a5f',
      changedBg:     '#052e16',
      changedFg:     '#4ade80',
      onCellChange:  null,   // fn(rowIdx, colKey, newValue)
      onReady:       null,
    }, opts);

    this.columns   = [];   // [{key, label, width}]
    this.data      = [];   // [{col: value, ...}]
    this.changes   = {};   // { "rowIdx:colKey": newValue }
    this.sel       = { row: -1, col: -1 };
    this.scrollX   = 0;
    this.scrollY   = 0;
    this._dpr      = window.devicePixelRatio || 1;
    this._input    = null;  // floating input overlay
    this._raf      = null;
    this._dirty    = true;

    this._build();
    if (opts.columns && opts.data) this.setData(opts.columns, opts.data);
  }

  // ── Build DOM ─────────────────────────────────────────────────────────
  _build() {
    this.container.style.position = 'relative';
    this.container.style.overflow = 'hidden';
    this.container.style.userSelect = 'none';

    this.canvas = document.createElement('canvas');
    this.canvas.style.cssText = 'display:block;position:absolute;top:0;left:0;cursor:default;';
    this.container.appendChild(this.canvas);
    this.ctx = this.canvas.getContext('2d');

    // Scrollbars (native) via a transparent overlay div
    this.scroller = document.createElement('div');
    this.scroller.style.cssText = 'position:absolute;top:0;left:0;right:0;bottom:0;overflow:auto;z-index:1;';
    this.inner = document.createElement('div');  // sized to full virtual grid
    this.inner.style.cssText = 'position:absolute;top:0;left:0;';
    this.scroller.appendChild(this.inner);
    this.container.appendChild(this.scroller);

    // Floating edit input — appended to body so it's above everything
    if (this.opts.editable) {
      this._input = document.createElement('input');
      this._input.type = 'text';
      this._input.style.cssText = `position:fixed;z-index:9999;display:none;box-sizing:border-box;
        border:2px solid ${this.opts.selBorder};outline:none;
        background:${this.opts.editBg};color:${this.opts.rowFg};
        font-size:${this.opts.fontSize}px;font-family:${this.opts.fontFamily};
        padding:0 6px;`;
      document.body.appendChild(this._input);
      this._input.addEventListener('keydown', e => this._onInputKey(e));
      this._input.addEventListener('blur', () => this._commitInput());
    }

    // Events
    // NOTE: scroller sits above canvas at z-index:1 and captures all pointer events.
    // _onMouseDown must live here — the canvas mousedown never fires.
    // e.preventDefault() is called unconditionally to stop the browser from
    // blurring (and immediately hiding) the floating input on every cell click.
    this.scroller.addEventListener('mousedown', e => {
      e.preventDefault();
      this._inputMousedown = true;
      this._onMouseDown(e);
    });
    this.scroller.addEventListener('scroll', () => { this._onScroll(); }, { passive: true });
    window.addEventListener('resize', this._onResize = () => this._resize());

    this._resize();
    this._startRaf();

    // If container has zero size on init (hidden tab), re-resize when it becomes visible
    const obs = new ResizeObserver(() => {
      if (this.container.clientWidth > 0 && this.container.clientHeight > 0) {
        this._resize();
        this._dirty = true;
      }
    });
    obs.observe(this.container);
    this._resizeObs = obs;
  }

  // ── Public API ────────────────────────────────────────────────────────
  setData(columns, data) {
    this.columns = columns.map((c, i) => {
      const key   = typeof c === 'string' ? c : c.key;
      const label = typeof c === 'string' ? c : (c.label || c.key);
      return { key, label, width: this._measureColWidth(key, label, data) };
    });
    this.data    = data;
    this.changes = {};
    this.sel     = { row: -1, col: -1 };
    this.scrollX = 0;
    this.scrollY = 0;
    this._updateInnerSize();
    this.scroller.scrollTop  = 0;
    this.scroller.scrollLeft = 0;
    this._dirty = true;
  }

  getChanges() {
    // Returns [{rowIdx, colKey, value}]
    return Object.entries(this.changes).map(([k, value]) => {
      const [rowIdx, ...colParts] = k.split(':');
      return { rowIdx: parseInt(rowIdx), colKey: colParts.join(':'), value };
    });
  }

  commitActiveInput() {
    this._commitInput();
  }

  patchRow(rowIdx, colKey, value) {
    if (this.data[rowIdx]) {
      this.data[rowIdx][colKey] = value;
      this._dirty = true;
    }
  }

  // Insert a full row object at a given position (used by undo/reapply
  // patches that add a row). Any recorded `changes` at/after this position
  // are shifted +1 so highlighting stays attached to the correct row.
  addRow(rowIdx, data) {
    const idx = Math.max(0, Math.min(rowIdx, this.data.length));
    this.data.splice(idx, 0, data);
    this._shiftChanges(idx, +1);
    if (this.sel.row >= idx) this.sel.row++;
    this._updateInnerSize();
    this._dirty = true;
  }

  // Remove a row at a given position (used by undo/reapply patches that
  // drop a row). Any recorded `changes` for that row are discarded; changes
  // after it are shifted -1.
  removeRow(rowIdx) {
    if (rowIdx < 0 || rowIdx >= this.data.length) return;
    this.data.splice(rowIdx, 1);
    this._shiftChanges(rowIdx, -1);
    if (this.sel.row === rowIdx) this.sel = { row: -1, col: -1 };
    else if (this.sel.row > rowIdx) this.sel.row--;
    this._updateInnerSize();
    this._dirty = true;
  }

  // Re-key this.changes ("rowIdx:colKey" -> value) after a row insertion or
  // removal at `atIdx` so highlighted/edited cells stay pinned to the right
  // row instead of silently pointing at whatever row slides into that slot.
  _shiftChanges(atIdx, delta) {
    if (!Object.keys(this.changes).length) return;
    const next = {};
    for (const [k, v] of Object.entries(this.changes)) {
      const sep = k.indexOf(':');
      const r = parseInt(k.slice(0, sep), 10);
      const colKey = k.slice(sep + 1);
      if (delta < 0 && r === atIdx) continue; // that row is gone
      const newR = r >= atIdx ? r + delta : r;
      next[`${newR}:${colKey}`] = v;
    }
    this.changes = next;
  }

  // Add a column, filling it from `valuesByPosition` (an array aligned 1:1
  // with the CURRENT this.data order — the caller is responsible for mapping
  // row labels to positions before calling this, same as addRow/removeRow).
  addColumn(key, label, valuesByPosition) {
    if (this.columns.some(c => c.key === key)) return; // already present
    this.data.forEach((row, i) => {
      row[key] = (valuesByPosition && valuesByPosition[i] !== undefined) ? valuesByPosition[i] : null;
    });
    this.columns.push({ key, label: label || key, width: this._measureColWidth(key, label || key, this.data) });
    this._updateInnerSize();
    this._dirty = true;
  }

  // Remove a column entirely — from the column list, every row's data, and
  // any pending cell-edit highlighting recorded against it.
  removeColumn(key) {
    const idx = this.columns.findIndex(c => c.key === key);
    if (idx === -1) return;
    this.columns.splice(idx, 1);
    this.data.forEach(row => { delete row[key]; });
    const next = {};
    for (const [k, v] of Object.entries(this.changes)) {
      if (!k.endsWith(':' + key)) next[k] = v;
    }
    this.changes = next;
    this._updateInnerSize();
    this._dirty = true;
  }

  destroy() {
    if (this._raf) cancelAnimationFrame(this._raf);
    if (this._resizeObs) this._resizeObs.disconnect();
    window.removeEventListener('resize', this._onResize);
    this.container.innerHTML = '';
  }

  // ── Internal ──────────────────────────────────────────────────────────
  _measureColWidth(key, label, data) {
    const ctx = this.ctx;
    ctx.font = `600 ${this.opts.fontSize}px ${this.opts.fontFamily}`;
    let w = ctx.measureText(label).width + 24;
    ctx.font = `${this.opts.fontSize}px ${this.opts.fontFamily}`;
    const sample = data.slice(0, 50);
    for (const row of sample) {
      const v = row[key];
      if (v !== null && v !== undefined) {
        w = Math.max(w, ctx.measureText(String(v)).width + 16);
      }
    }
    return Math.max(this.opts.minColWidth, Math.min(this.opts.maxColWidth, w));
  }

  _totalWidth()  { return this.columns.reduce((s,c) => s+c.width, 0); }
  _totalHeight() { return this.opts.headerHeight + this.data.length * this.opts.rowHeight; }

  _updateInnerSize() {
    this.inner.style.width  = this._totalWidth()  + 'px';
    this.inner.style.height = this._totalHeight() + 'px';
  }

  _resize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    const dpr = this._dpr;
    this.canvas.width  = w * dpr;
    this.canvas.height = h * dpr;
    this.canvas.style.width  = w + 'px';
    this.canvas.style.height = h + 'px';
    this.ctx.scale(dpr, dpr);
    this._W = w; this._H = h;
    this._dirty = true;
  }

  _onScroll() {
    this.scrollX = this.scroller.scrollLeft;
    this.scrollY = this.scroller.scrollTop;
    this._dirty = true;
    this._repositionInput();
  }

  _startRaf() {
    const loop = () => {
      if (this._dirty) { this._draw(); this._dirty = false; }
      this._raf = requestAnimationFrame(loop);
    };
    this._raf = requestAnimationFrame(loop);
  }

  // ── Drawing ───────────────────────────────────────────────────────────
  _draw() {
    const ctx = this.ctx;
    const { rowHeight, headerHeight, headerBg, headerFg, rowBg, rowAltBg,
            rowFg, borderColor, selBg, selBorder, changedBg, changedFg,
            fontSize, fontFamily } = this.opts;
    const W = this._W, H = this._H;
    const sx = this.scrollX, sy = this.scrollY;

    ctx.clearRect(0, 0, W, H);

    if (!this.columns.length) {
      ctx.fillStyle = '#334155';
      ctx.font = `${fontSize}px ${fontFamily}`;
      ctx.fillText('No data', 16, 40);
      return;
    }

    // Compute visible column range
    let x = 0, colStart = -1, colEnd = -1;
    const colOffsets = [];
    for (let i = 0; i < this.columns.length; i++) {
      colOffsets.push(x);
      const right = x + this.columns[i].width;
      if (right > sx && colStart === -1) colStart = i;
      if (x < sx + W) colEnd = i;
      x += this.columns[i].width;
    }
    if (colStart === -1) colStart = 0;
    if (colEnd === -1)   colEnd   = this.columns.length - 1;

    // Visible row range
    const rowStart = Math.max(0, Math.floor((sy - headerHeight) / rowHeight));
    const rowEnd   = Math.min(this.data.length - 1,
                              Math.ceil((sy + H - headerHeight) / rowHeight));

    // ── Draw rows ──
    ctx.font = `${fontSize}px ${fontFamily}`;
    ctx.textBaseline = 'middle';

    for (let r = rowStart; r <= rowEnd; r++) {
      const ry = headerHeight + r * rowHeight - sy;
      const row = this.data[r];

      for (let c = colStart; c <= colEnd; c++) {
        const cx = colOffsets[c] - sx;
        const cw = this.columns[c].width;
        const key = this.columns[c].key;
        const ck  = `${r}:${key}`;
        const isChanged = this.changes[ck] !== undefined;
        const isSel     = this.sel.row === r && this.sel.col === c;

        // Cell background
        const isSearch  = this.searchHighlights && this.searchHighlights.has(ck);
        let bg = r % 2 === 0 ? rowBg : rowAltBg;
        if (isSearch)   bg = '#854d0e';   // amber-900 — search match
        if (isChanged)  bg = changedBg;
        if (isSel)      bg = selBg;
        ctx.fillStyle = bg;
        ctx.fillRect(cx, ry, cw, rowHeight);

        // Selection border
        if (isSel) {
          ctx.strokeStyle = selBorder;
          ctx.lineWidth = 1.5;
          ctx.strokeRect(cx + 0.75, ry + 0.75, cw - 1.5, rowHeight - 1.5);
        }

        // Cell text
        const rawVal = this.changes[ck] !== undefined ? this.changes[ck] : (row ? row[key] : '');
        const text   = rawVal === null || rawVal === undefined ? '' : String(rawVal);
        ctx.fillStyle = isChanged ? changedFg : rowFg;

        // Clip text
        ctx.save();
        ctx.beginPath();
        ctx.rect(cx + 1, ry, cw - 10, rowHeight);
        ctx.clip();
        ctx.fillText(text, cx + 6, ry + rowHeight / 2);
        ctx.restore();
      }

      // Row bottom border
      ctx.fillStyle = borderColor;
      ctx.fillRect(colOffsets[colStart] - sx, ry + rowHeight - 1,
                   colOffsets[colEnd] + this.columns[colEnd].width - colOffsets[colStart], 1);
    }

    // ── Draw header ──
    ctx.fillStyle = headerBg;
    ctx.fillRect(0, 0, W, headerHeight);
    ctx.font = `600 ${fontSize}px ${fontFamily}`;
    ctx.fillStyle = headerFg;
    ctx.textBaseline = 'middle';

    for (let c = colStart; c <= colEnd; c++) {
      const cx = colOffsets[c] - sx;
      const cw = this.columns[c].width;
      // Column divider
      ctx.fillStyle = borderColor;
      ctx.fillRect(cx + cw - 1, 0, 1, headerHeight);
      // Header text
      ctx.fillStyle = headerFg;
      ctx.save();
      ctx.beginPath();
      ctx.rect(cx + 1, 0, cw - 10, headerHeight);
      ctx.clip();
      ctx.fillText(this.columns[c].label, cx + 8, headerHeight / 2);
      ctx.restore();
    }

    // Header bottom border
    ctx.fillStyle = borderColor;
    ctx.fillRect(0, headerHeight - 1, W, 1);
  }

  // ── Hit testing ────────────────────────────────────────────────────────
  _hitTest(e) {
    const rect = this.canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left + this.scrollX;
    const my = e.clientY - rect.top  + this.scrollY;
    if (my < this.opts.headerHeight) return null;   // header click
    const rowIdx = Math.floor((my - this.opts.headerHeight) / this.opts.rowHeight);
    if (rowIdx < 0 || rowIdx >= this.data.length) return null;
    let x = 0;
    for (let c = 0; c < this.columns.length; c++) {
      if (mx >= x && mx < x + this.columns[c].width) return { row: rowIdx, col: c };
      x += this.columns[c].width;
    }
    return null;
  }

  _onMouseDown(e) {
    const hit = this._hitTest(e);
    if (!hit) return;
    this._commitInput();
    this.sel = hit;
    this._dirty = true;
    // Single click opens input immediately in editable mode
    if (this.opts.editable) {
      this._openInput(hit.row, hit.col);
    }
  }

  _onDblClick(e) {
    // kept for compatibility but single click already handles it
  }

  _cellRect(row, col) {
    let x = 0;
    for (let c = 0; c < col; c++) x += this.columns[c].width;
    return {
      x: x - this.scrollX,
      y: this.opts.headerHeight + row * this.opts.rowHeight - this.scrollY,
      w: this.columns[col].width,
      h: this.opts.rowHeight,
    };
  }

  _openInput(row, col) {
    if (!this._input) return;
    const canvasRect = this.canvas.getBoundingClientRect();
    const r = this._cellRect(row, col);
    const key = this.columns[col].key;
    const ck  = `${row}:${key}`;
    const cur = this.changes[ck] !== undefined ? this.changes[ck] : (this.data[row] ? this.data[row][key] : '');
    this._input.value = cur === null || cur === undefined ? '' : String(cur);
    // Use fixed positioning relative to viewport
    this._input.style.left   = (canvasRect.left + r.x) + 'px';
    this._input.style.top    = (canvasRect.top  + r.y) + 'px';
    this._input.style.width  = r.w + 'px';
    this._input.style.height = r.h + 'px';
    this._input.style.display = 'block';
    this._input._row = row;
    this._input._col = col;
    this._input.focus();
    this._input.select();
  }

  _commitInput() {
    if (!this._input || this._input.style.display === 'none') return;
    const row = this._input._row;
    const col = this._input._col;
    if (row === undefined || col === undefined) { this._input.style.display = 'none'; return; }
    const key = this.columns[col].key;
    const ck  = `${row}:${key}`;
    const orig = this.data[row] ? this.data[row][key] : '';
    const val  = this._input.value;
    const origStr = orig === null || orig === undefined ? '' : String(orig);
    if (val !== origStr) {
      this.changes[ck] = val;
      if (this.opts.onCellChange) this.opts.onCellChange(row, key, val, origStr);
    } else {
      delete this.changes[ck];
      if (this.opts.onCellChange) this.opts.onCellChange(row, key, val, origStr);
    }
    this._input.style.display = 'none';
    this._input._row = undefined;
    this._input._col = undefined;
    this._dirty = true;
  }

  _repositionInput() {
    if (!this._input || this._input.style.display === 'none') return;
    const row = this._input._row, col = this._input._col;
    if (row === undefined) return;
    const canvasRect = this.canvas.getBoundingClientRect();
    const r = this._cellRect(row, col);
    this._input.style.left = (canvasRect.left + r.x) + 'px';
    this._input.style.top  = (canvasRect.top  + r.y) + 'px';
  }

  _onInputKey(e) {
    if (e.key === 'Enter' || e.key === 'Tab') {
      e.preventDefault();
      this._commitInput();
      // Move selection
      if (e.key === 'Tab') {
        const nextCol = e.shiftKey ? this.sel.col - 1 : this.sel.col + 1;
        if (nextCol >= 0 && nextCol < this.columns.length) {
          this.sel.col = nextCol;
          if (this.opts.editable) this._openInput(this.sel.row, this.sel.col);
        }
      } else {
        const nextRow = this.sel.row + 1;
        if (nextRow < this.data.length) {
          this.sel.row = nextRow;
          if (this.opts.editable) this._openInput(this.sel.row, this.sel.col);
        }
      }
    } else if (e.key === 'Escape') {
      this._input.style.display = 'none';
      this._input._row = undefined; this._input._col = undefined;
    }
  }
}

window.CanvasGrid = CanvasGrid;

/**
 * Install the shared fetch wrapper and progress announcer.
 *
 * The wrapper adds CSRF and request IDs to same-origin mutations. The progress
 * object drives the accessible operation banner; it shows indeterminate state
 * unless the backend returns real progress values.
 */
function indataoutInstallRequestAndProgress() {
  if (!window._ido) window._ido = {};
  const csrfMeta = document.querySelector('meta[name="csrf-token"]');
  const csrfToken = csrfMeta ? csrfMeta.getAttribute('content') : '';
  const nativeFetch = window.fetch.bind(window);
  window.fetch = function indataoutFetch(input, init = {}) {
    const opts = Object.assign({}, init);
    const headers = new Headers(opts.headers || {});
    if (csrfToken && !headers.has('X-CSRF-Token')) headers.set('X-CSRF-Token', csrfToken);
    if (!headers.has('X-Request-ID')) headers.set('X-Request-ID', 'req_' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2, 8));
    opts.headers = headers;
    return nativeFetch(input, opts);
  };
  let progressTimer = null;
  let progressStarted = 0;
  function updateElapsed() {
    const el = document.getElementById('operationProgressElapsed');
    if (!el || !progressStarted) return;
    el.textContent = Math.max(0, Math.round((Date.now() - progressStarted) / 1000)) + 's';
  }
  window._ido.operationProgress = {
    start(name, stage) {
      const box = document.getElementById('operationProgress');
      if (!box) return;
      progressStarted = Date.now();
      box.hidden = false;
      box.classList.remove('is-success', 'is-error');
      const n = document.getElementById('operationProgressName');
      const s = document.getElementById('operationProgressStage');
      const r = document.getElementById('operationProgressResult');
      if (n) n.textContent = name || 'Working';
      if (s) s.textContent = stage || 'Starting...';
      if (r) r.textContent = '';
      clearInterval(progressTimer);
      updateElapsed();
      progressTimer = setInterval(updateElapsed, 1000);
    },
    stage(stage, result) {
      const s = document.getElementById('operationProgressStage');
      const r = document.getElementById('operationProgressResult');
      if (s && stage) s.textContent = stage;
      if (r && result) r.textContent = result;
    },
    finish(kind, message) {
      const box = document.getElementById('operationProgress');
      if (!box) return;
      clearInterval(progressTimer);
      updateElapsed();
      box.classList.toggle('is-success', kind !== 'error');
      box.classList.toggle('is-error', kind === 'error');
      const r = document.getElementById('operationProgressResult');
      if (r) r.textContent = message || (kind === 'error' ? 'Failed' : 'Completed');
    },
    hideLater(ms = 2200) {
      const box = document.getElementById('operationProgress');
      if (!box) return;
      setTimeout(() => { box.hidden = true; }, ms);
    }
  };
}
// static/script.js

// ─── Cross-closure shared bridge ────────────────────────────────────────────
// script.js has multiple DOMContentLoaded blocks in different scopes. Variables
// declared with let/const in one block are NOT visible in another, so this is
// the stable public bridge for shared UI state and exported helpers. It mirrors
// server state for rendering only; persisted history/snapshots remain server-owned.
window._ido = {};
indataoutInstallRequestAndProgress();
window._ido.lastSummary = null;  // backstop for showCleaningPanel — see there for why

// Hide loading overlay if the browser restores this page from bfcache
// (e.g. user hits Back after a form submission).
window.addEventListener('pageshow', (e) => {
  const overlay = document.getElementById('loadingOverlay');
  if (overlay) overlay.classList.remove('active');
});

document.addEventListener('DOMContentLoaded', () => {
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  const chooseBtn = document.getElementById('chooseBtn');
  const uploadForm = document.getElementById('uploadForm');
  const advToggle = document.getElementById('advToggle');
  const advancedPanel = document.getElementById('advanced');

  const allowed = ['csv','tsv','txt','xlsx','xls','json'];

  function escapeHtml(s) {
    return String(s).replace(/[&<>"'`=\/]/g, function (c) {
      return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;','/':'&#x2F;','`':'&#x60;','=':'&#x3D;'}[c];
    });
  }

  function showFilename(name) {
    let el = dropZone.querySelector('.file-name');
    if (!el) {
      el = document.createElement('div');
      el.className = 'file-name';
      dropZone.insertBefore(el, dropZone.firstChild);
    }
    el.innerHTML = 'Uploaded: <strong>' + escapeHtml(name) + '</strong>';
  }

  function showLoading(message) {
    const overlay = document.getElementById('loadingOverlay');
    const msg = document.getElementById('loadingMsg');
    if (overlay) overlay.classList.add('active');
    if (msg) msg.textContent = message;
  }

  if (dropZone && fileInput && uploadForm) {
    // Clicks
    dropZone.addEventListener('click', () => fileInput.click());
    if (chooseBtn) chooseBtn.addEventListener('click', (e) => { e.preventDefault(); fileInput.click(); });

    // Drag styles
    ['dragenter','dragover'].forEach(ev => {
      dropZone.addEventListener(ev, (e) => { e.preventDefault(); e.stopPropagation(); dropZone.classList.add('dragover'); });
    });
    ['dragleave','drop'].forEach(ev => {
      dropZone.addEventListener(ev, (e) => { e.preventDefault(); e.stopPropagation(); dropZone.classList.remove('dragover'); dropZone.classList.remove('drag-pulse'); });
    });

    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('drag-pulse'); });

    // Handle drop
    dropZone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      if (!dt || !dt.files || dt.files.length === 0) return;
      const file = dt.files[0];
      const ext = file.name.split('.').pop().toLowerCase();
      if (!allowed.includes(ext)) {
        alert('Unsupported file type. Allowed: ' + allowed.join(', ').toUpperCase());
        return;
      }
      const data = new DataTransfer();
      data.items.add(file);
      fileInput.files = data.files;
      showFilename(file.name);
      if (window._ido.operationProgress) window._ido.operationProgress.start('Upload', 'Uploading and parsing dataset');
      showLoading('Uploading and reading your dataset...');
      uploadForm.submit();
    });

    // Handle file selection
    fileInput.addEventListener('change', () => {
      if (!fileInput.files || fileInput.files.length === 0) return;
      const file = fileInput.files[0];
      const ext = file.name.split('.').pop().toLowerCase();
      if (!allowed.includes(ext)) {
        alert('Unsupported file type. Allowed: ' + allowed.join(', ').toUpperCase());
        fileInput.value = '';
        return;
      }
      showFilename(file.name);
      if (window._ido.operationProgress) window._ido.operationProgress.start('Upload', 'Uploading and parsing dataset');
      showLoading('Uploading and reading your dataset...');
      uploadForm.submit();
    });

    // Keyboard activation
    dropZone.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        fileInput.click();
      }
    });
  }

  // Legacy advanced toggle support, if an older template renders it.
  if (advToggle && advancedPanel) {
    advToggle.addEventListener('click', () => {
      const hidden = window.getComputedStyle(advancedPanel).display === 'none';
      advancedPanel.style.display = hidden ? 'block' : 'none';
      advToggle.classList.toggle('active', hidden);
    });
  }

  // Outlier preview functionality
  const previewOutliersBtn = document.getElementById('previewOutliersBtn');
  const outlierPreviewSection = document.getElementById('outlierPreviewSection');
  const outlierTableBody = document.getElementById('outlierTableBody');
  const selectAllOutliers = document.getElementById('selectAllOutliers');
  const outlierCountInfo = document.getElementById('outlierCountInfo');
  const outlierStatus = document.getElementById('outlierStatus');

  let missingCellInputs = Array.from(document.querySelectorAll('.missing-cell-edit input'));
  const missingEditorStatus = document.getElementById('missingEditorStatus');
  function markMissingInputFromUser(event) {
    const input = event && event.currentTarget;
    if (!input) return;
    if (event.isTrusted) input.dataset.userTyped = '1';
    if (!input.value.trim()) delete input.dataset.userTyped;
  }
  function isUserTypedMissingInput(input) {
    return !!(input && input.dataset.userTyped === '1' && input.value.trim() !== '' && !input.disabled && input.name);
  }
  function syncMissingEditorStatus() {
    if (!missingEditorStatus) return;
    const typedCount = missingCellInputs.filter(isUserTypedMissingInput).length;
    missingEditorStatus.textContent = typedCount > 0
      ? `${typedCount} typed value${typedCount === 1 ? '' : 's'} ready. These update the dataset first, then cleaning continues.`
      : 'No typed values yet.';
    missingEditorStatus.classList.toggle('has-edits', typedCount > 0);
  }
  function bindMissingInput(input) {
    if (!input || input.dataset.missingInputBound === '1') return;
    input.dataset.missingInputBound = '1';
    input.addEventListener('input', markMissingInputFromUser);
    input.addEventListener('input', syncMissingEditorStatus);
  }
  if (missingCellInputs.length && missingEditorStatus) {
    missingCellInputs.forEach(bindMissingInput);
    syncMissingEditorStatus();
  }

  const cleanForm = document.getElementById('cleanForm');

  // "Clean Data" submits to /clean. The applied-history ledger
  // (window._ido.appliedControlEntries) is the single source of truth for
  // what's actually been applied — it's serialized to JSON and sent as one
  // hidden `applied_entries` field by indataoutPrepareAppliedControls /
  // onCleanSubmit (see below and index.html). We deliberately do NOT fall
  // back to "whatever's currently in the DOM" here: a control sitting in a
  // non-default state that the user never clicked Apply on must NOT be
  // included, and a previously-applied control that happens to not be
  // rendered right now (e.g. outlier checkboxes on a resumed project) must
  // still be included. Trusting live DOM state over the ledger was exactly
  // the bug that let unapplied toggles leak into the final file and let
  // applied ones (outliers) silently drop out.

  const livePreviewBtn = document.getElementById('livePreviewBtn');
  const undoPreviewBtn = document.getElementById('undoPreviewBtn');
  const livePreviewTable = document.getElementById('livePreviewTable');
  const liveMetrics = document.getElementById('liveMetrics');
  const liveActionLog = document.getElementById('liveActionLog');
  const liveCodeLog = document.getElementById('liveCodeLog');
  const applyTypedPreviewBtn = document.getElementById('applyTypedPreviewBtn');
  const missingValuesGroup = document.getElementById('missingValuesGroup');
  const missingReviewPanel = missingValuesGroup ? missingValuesGroup.querySelector('.missing-review-panel') : null;
  const missingReviewTitle = missingReviewPanel ? missingReviewPanel.querySelector('.missing-review-title') : null;
  const missingReviewSubtitle = missingReviewPanel ? missingReviewPanel.querySelector('.missing-review-subtitle') : null;
  const missingColumnSummary = missingReviewPanel ? missingReviewPanel.querySelector('.missing-column-summary') : null;
  const wideMissingEditor = missingValuesGroup ? missingValuesGroup.querySelector('.wide-missing-editor') : null;
  let missingEditorRows = Array.from(document.querySelectorAll('.missing-editor-table tbody tr'));
  const applyDatasetEditsBtn = document.getElementById('applyDatasetEditsBtn');
  const datasetEditInputs = Array.from(document.querySelectorAll('.dataset-edit-input'));
  const datasetColumnInputs = Array.from(document.querySelectorAll('.dataset-column-input'));
  const datasetEditStatus = document.getElementById('datasetEditStatus');
  const cellSearchInput = document.getElementById('cellSearchInput');
  const cellSearchColumn = document.getElementById('cellSearchColumn');
  const cellSearchRow = document.getElementById('cellSearchRow');
  const cellSearchBtn = document.getElementById('cellSearchBtn');
  const clearCellSearchBtn = document.getElementById('clearCellSearchBtn');
  const cellSearchStatus = document.getElementById('cellSearchStatus');
  const sectionApplyBtns = Array.from(document.querySelectorAll('.section-apply-btn'));
  const datasetTabs = Array.from(document.querySelectorAll('.dataset-tab, .rail-dataset-btn'));
  const panelLinks = Array.from(document.querySelectorAll('.rail-panel-link'));
  const optionGroups = Array.from(document.querySelectorAll('.opt-group'));
  let undoStack = [];
  let previewUndoStack = [];
  let currentHistoryScope = null;
  // Live on window._ido so the second DOMContentLoaded closure can access it.
  window._ido.appliedControlEntries = [];
  window._ido.datasetVersion = Number(document.body.dataset.datasetVersion || 0);
  let currentWorkingColumns = null;
  let appliedVisibleOpsCount = 0;
  // Expose getter/setter so changelog and custom code modules can keep the count in sync
  Object.defineProperty(window._ido, 'appliedVisibleOpsCount', {
    get: () => appliedVisibleOpsCount,
    set: (v) => { appliedVisibleOpsCount = v; },
    configurable: true,
  });
  let lastSnapshot = cleanForm ? snapshotForm(cleanForm) : [];

  function showToast(message, type = 'warning') {
    let region = document.getElementById('appToastRegion');
    if (!region) {
      region = document.createElement('div');
      region.id = 'appToastRegion';
      region.className = 'app-toast-region';
      document.body.appendChild(region);
    }
    const toast = document.createElement('div');
    toast.className = `app-toast app-toast-${type}`;
    toast.textContent = message;
    region.appendChild(toast);
    setTimeout(() => toast.classList.add('show'), 20);
    setTimeout(() => {
      toast.classList.remove('show');
      setTimeout(() => toast.remove(), 220);
    }, 3600);
  }
  // Bridge — called as window._ido.showToast(...) both later in this same
  // closure (_applyManifestOnServer) and potentially from others, but this
  // was never actually assigned here, so `if (window._ido.showToast)` guards
  // were always silently false and the toast never fired.
  window._ido.showToast = showToast;

  function showApplyBlocked(message) {
    showToast(message, 'warning');
    if (liveActionLog) {
      liveActionLog.innerHTML = `<div class="live-log-empty">${escapeHtml(message)}</div>`;
    }
  }

  function showDatasetSlide(slideId) {
    document.querySelectorAll('.dataset-slide').forEach(slide => {
      slide.classList.toggle('active', slide.id === slideId);
    });
    datasetTabs.forEach(tab => {
      tab.classList.toggle('active', tab.dataset.datasetSlide === slideId);
    });
    if (slideId === 'datasetEditorSlide' && window._ido.refreshCurrentEditorFromServer) {
      window._ido.refreshCurrentEditorFromServer();
    }
  }

  datasetTabs.forEach(tab => {
    tab.addEventListener('click', () => showDatasetSlide(tab.dataset.datasetSlide));
  });

  function showCleaningPanel(panelId, shouldScroll = true) {
    if (!panelId || optionGroups.length === 0) return;
    currentHistoryScope = panelId;
    window._ido.currentHistoryScope = panelId;
    optionGroups.forEach(group => {
      group.classList.toggle('active-panel', group.id === panelId);
    });
    panelLinks.forEach(link => {
      link.classList.toggle('active', link.dataset.panelTarget === panelId);
    });
    const selected = document.getElementById(panelId);
    if (selected && shouldScroll) selected.scrollIntoView({ behavior: 'smooth', block: 'start' });
    if (window._ido.renderScopedHistory) window._ido.renderScopedHistory(panelId);
    // Defensive backstop, not the fix for the row-hiding bug (that was in
    // syncMissingEditorFromPreview/syncDatasetEditorFromPreview, already
    // corrected) — updateMissingReviewState already writes to the DOM
    // regardless of visibility, so this re-application is normally a no-op.
    // Kept cheap and harmless in case any future code path updates state
    // without going through _refreshUIFromServerData.
    if (panelId === 'missingValuesGroup' && window._ido.lastSummary && window._ido.updateMissingReviewState) {
      window._ido.updateMissingReviewState(window._ido.lastSummary);
    }
  }

  panelLinks.forEach(link => {
    link.addEventListener('click', (event) => {
      event.preventDefault();
      showCleaningPanel(link.dataset.panelTarget);
      history.replaceState(null, '', `#${link.dataset.panelTarget}`);
    });
  });

  if (optionGroups.length > 0) {
    const hashPanel = window.location.hash ? window.location.hash.slice(1) : '';
    const initialPanel = optionGroups.some(group => group.id === hashPanel) ? hashPanel : optionGroups[0].id;
    showCleaningPanel(initialPanel, false);
  }

  function syncDatasetEditStatus() {
    if (!datasetEditStatus) return;
    const gridChanges = window._ido && window._ido.editorGrid
      ? window._ido.editorGrid.getChanges().length
      : 0;
    const edited = datasetEditInputs.filter(input => input.value !== input.dataset.original);
    const renamed = datasetColumnInputs.filter(input => input.value !== input.dataset.original);
    const parts = [];
    const editedCount = gridChanges + edited.length;
    if (editedCount > 0) parts.push(`${editedCount} edited cell${editedCount === 1 ? '' : 's'}`);
    if (renamed.length > 0) parts.push(`${renamed.length} renamed column${renamed.length === 1 ? '' : 's'}`);
    datasetEditStatus.textContent = editedCount > 0
      ? `${parts.join(' and ')} ready. Apply edits to update Change Log.`
      : 'No edited cells yet.';
    if (editedCount === 0 && renamed.length > 0) {
      datasetEditStatus.textContent = `${parts.join(' and ')} ready. Apply edits to update Change Log.`;
    }
    datasetEditStatus.classList.toggle('has-edits', editedCount > 0 || renamed.length > 0);
  }
  window._ido.syncDatasetEditStatus = syncDatasetEditStatus;

  function normalizeColumnName(name) {
    return String(name || '')
      .trim()
      .toLowerCase()
      .replace(/ /g, '_')
      .split('')
      .map(ch => ch.charCodeAt(0) < 128 ? ch : '_')
      .join('');
  }

  function workingColumnKeys(columns) {
    return new Set((columns || []).map(normalizeColumnName));
  }

  function currentRenameTarget(column) {
    const input = datasetColumnInputs.find(el => el.dataset.original === column);
    if (!input || !input.value || input.value === input.dataset.original) return '';
    return input.value;
  }

  function applyWorkingColumns(columns) {
    currentWorkingColumns = columns && columns.length ? columns.slice() : null;
    const keep = workingColumnKeys(currentWorkingColumns);
    const hasWorkingColumns = keep.size > 0;
    document.querySelectorAll('[data-column]').forEach(el => {
      const originalColumn = el.dataset.column || '';
      const renamedColumn = currentRenameTarget(originalColumn);
      const visible = !hasWorkingColumns ||
        keep.has(normalizeColumnName(originalColumn)) ||
        (renamedColumn && keep.has(normalizeColumnName(renamedColumn)));
      el.style.display = visible ? '' : 'none';
      el.querySelectorAll('input, select, textarea').forEach(control => {
        control.disabled = !visible;
      });
      if (el.matches('input, select, textarea')) {
        el.disabled = !visible;
      }
    });
    if (cellSearchColumn) {
      Array.from(cellSearchColumn.options).forEach(option => {
        if (!option.value) return;
        const visible = !hasWorkingColumns || keep.has(normalizeColumnName(option.value));
        option.hidden = !visible;
        option.disabled = !visible;
      });
      if (cellSearchColumn.value && cellSearchColumn.selectedOptions[0]?.disabled) {
        cellSearchColumn.value = '';
      }
    }
  }

  function findRowForInput(input, rowMap) {
    return rowMap.get(String(input.dataset.rowNumber || ''));
  }

  function valueFromWorkingRow(row, column) {
    if (!row) return undefined;
    const wanted = normalizeColumnName(currentRenameTarget(column) || column);
    const key = Object.keys(row).find(col => normalizeColumnName(col) === wanted);
    if (!key) return undefined;
    const value = row[key];
    return value === null || value === undefined ? '' : String(value);
  }

  function syncDatasetEditorFromPreview(preview, rowNumbers, columns, removedRowNumbers) {
    if (!Array.isArray(preview) || preview.length === 0 || !Array.isArray(rowNumbers)) return;
    const rowMap = new Map();
    rowNumbers.forEach((rowNumber, index) => {
      if (preview[index]) rowMap.set(String(rowNumber), preview[index]);
    });
    const availableRows = new Set(rowMap.keys());

    datasetEditInputs.forEach(input => {
      const row = findRowForInput(input, rowMap);
      const nextValue = valueFromWorkingRow(row, input.dataset.column || '');
      if (nextValue === undefined) return;
      input.value = nextValue;
      input.dataset.original = nextValue;
      input.removeAttribute('name');
      input.classList.remove('is-edited');
    });

    document.querySelectorAll('.dataset-edit-table tbody tr').forEach(rowEl => {
      const rowInput = rowEl.querySelector('.dataset-edit-input');
      if (!rowInput) return;
      const rn = String(rowInput.dataset.rowNumber || '');
      // Only show/confirm rows we actually have data for in this response —
      // a row's absence from a partial preview (e.g. the lightweight
      // undo/reapply patch, which only covers ~100 rows + whatever this one
      // action touched) is not evidence it was removed, so its current
      // display is left untouched rather than hidden by default. Removed
      // rows are hidden explicitly below instead, from the exact
      // removed_row_numbers list rather than inferred from absence.
      if (availableRows.has(rn)) {
        rowEl.style.display = '';
      } else if (Array.isArray(removedRowNumbers) && removedRowNumbers.includes(parseInt(rn, 10))) {
        rowEl.style.display = 'none';
      }
    });

    if (Array.isArray(columns) && columns.length > 0) {
      datasetColumnInputs.forEach(input => {
        const match = columns.find(col => normalizeColumnName(col) === normalizeColumnName(currentRenameTarget(input.dataset.original) || input.dataset.original));
        if (!match) return;
        input.value = match;
        input.dataset.original = match;
        input.dataset.column = match;
        input.dataset.name = `column_rename[${match}]`;
        const headerCell = input.closest('[data-column]');
        if (headerCell) headerCell.dataset.column = match;
        input.removeAttribute('name');
        input.classList.remove('is-edited');
      });
    }

    syncDatasetEditStatus();
    if (cellSearchInput && cellSearchInput.value.trim()) runCellSearch();
  }

  function rowMapFromPreview(preview, rowNumbers) {
    const rowMap = new Map();
    if (!Array.isArray(preview) || !Array.isArray(rowNumbers)) return rowMap;
    rowNumbers.forEach((rowNumber, index) => {
      if (preview[index]) rowMap.set(String(rowNumber), preview[index]);
    });
    return rowMap;
  }

  function remainingMissingCellSetFromAnalysis(analysis) {
    const cells = new Set();
    if (!analysis || !Array.isArray(analysis.missing_rows)) return cells;
    analysis.missing_rows.forEach(row => {
      const rowNumber = String(row.row_number || row.data_row_number || '');
      if (!rowNumber) return;
      (row.missing_columns || []).forEach(column => {
        cells.add(`${rowNumber}::${normalizeColumnName(column)}`);
      });
    });
    return cells;
  }

  function syncMissingEditorFromPreview(preview, rowNumbers, columns, summary, analysis) {
    if (!missingEditorRows.length) return;
    const rowMap = rowMapFromPreview(preview, rowNumbers);
    const availableRows = new Set(rowMap.keys());
    const remainingCells = remainingMissingCellSetFromAnalysis(analysis);
    const hasExactRemainingCells = remainingCells.size > 0;
    const remainingColumns = new Set(
      ((summary && summary.missing_by_column_remaining) || []).map(item => normalizeColumnName(item.column))
    );
    const noMissingRemain = summary && Number(summary.missing_remaining || 0) === 0;

    missingEditorRows.forEach(rowEl => {
      const rowNumber = rowEl.dataset.rowNumber ||
        rowEl.querySelector('[data-row-number]')?.dataset.rowNumber ||
        rowEl.querySelector('.missing-row-number')?.textContent.replace(/\D+/g, '');
      const row = rowMap.get(String(rowNumber || ''));
      if (row) {
        // We have fresh data for this row in this response — show it and
        // sync its cells below.
        rowEl.style.display = '';
      } else if (noMissingRemain) {
        // Nothing missing anywhere, confirmed by the server's full-dataset
        // count — safe to hide every row.
        rowEl.style.display = 'none';
      }
      // Otherwise: this response's preview simply didn't include this row
      // (e.g. it's outside the lightweight undo/reapply patch's touched-rows
      // + first-100 window) — that is NOT evidence the row was resolved, so
      // its current display is left exactly as it was rather than assumed
      // hidden. Only a response that actually covers a row is allowed to
      // change its visibility.
      if (!row) return;

      rowEl.querySelectorAll('[data-column]').forEach(cell => {
        const column = cell.dataset.column || '';
        const nextValue = valueFromWorkingRow(row, column);
        if (nextValue === undefined) return;
        const input = cell.querySelector('input');
        const columnStillMissing = remainingColumns.has(normalizeColumnName(column));
        const cellStillMissing = hasExactRemainingCells
          ? remainingCells.has(`${String(rowNumber || '')}::${normalizeColumnName(column)}`)
          : columnStillMissing;

        if (input) {
          delete input.dataset.userTyped;
          const hasDisplayValue = nextValue !== undefined && nextValue !== null && String(nextValue) !== '';
          const handled = noMissingRemain || !cellStillMissing;
          input.value = handled ? nextValue : '';
          if (handled) {
            input.removeAttribute('name');
            input.disabled = true;
            input.classList.remove('is-edited');
            input.classList.add('is-synced');
            input.placeholder = hasDisplayValue ? '' : 'Handled';
          } else {
            input.disabled = false;
            input.classList.remove('is-synced');
            input.placeholder = 'Type value';
          }
          return;
        }

        cell.textContent = nextValue;
        cell.title = nextValue;
      });
    });

    if (missingCellInputs.length && missingEditorStatus) {
      missingCellInputs.forEach(input => input.dispatchEvent(new Event('input')));
    }
  }

  function snapshotDatasetEditor() {
    return {
      cells: datasetEditInputs.map(input => ({
        value: input.value,
        original: input.dataset.original || '',
        column: input.dataset.column || '',
        dataName: input.dataset.name || '',
        name: input.name || '',
        disabled: !!input.disabled,
        edited: input.classList.contains('is-edited')
      })),
      columns: datasetColumnInputs.map(input => ({
        value: input.value,
        original: input.dataset.original || '',
        column: input.dataset.column || '',
        dataName: input.dataset.name || '',
        name: input.name || '',
        disabled: !!input.disabled,
        edited: input.classList.contains('is-edited'),
        headerColumn: input.closest('[data-column]')?.dataset.column || ''
      })),
      rowDisplays: Array.from(document.querySelectorAll('.dataset-edit-table tbody tr')).map(row => row.style.display || '')
    };
  }

  function restoreDatasetEditorSnapshot(snapshot) {
    if (!snapshot) return;
    (snapshot.cells || []).forEach((item, index) => {
      const input = datasetEditInputs[index];
      if (!input) return;
      input.value = item.value;
      input.dataset.original = item.original;
      input.dataset.column = item.column;
      input.dataset.name = item.dataName;
      input.name = item.name || '';
      if (!item.name) input.removeAttribute('name');
      input.disabled = !!item.disabled;
      input.classList.toggle('is-edited', !!item.edited);
    });
    (snapshot.columns || []).forEach((item, index) => {
      const input = datasetColumnInputs[index];
      if (!input) return;
      input.value = item.value;
      input.dataset.original = item.original;
      input.dataset.column = item.column;
      input.dataset.name = item.dataName;
      input.name = item.name || '';
      if (!item.name) input.removeAttribute('name');
      input.disabled = !!item.disabled;
      input.classList.toggle('is-edited', !!item.edited);
      const header = input.closest('[data-column]');
      if (header) header.dataset.column = item.headerColumn || item.column;
    });
    Array.from(document.querySelectorAll('.dataset-edit-table tbody tr')).forEach((row, index) => {
      row.style.display = (snapshot.rowDisplays || [])[index] || '';
    });
    syncDatasetEditStatus();
  }

  datasetEditInputs.forEach(input => {
    input.addEventListener('input', () => {
      if (input.value !== input.dataset.original) {
        input.name = input.dataset.name;
        input.classList.add('is-edited');
      } else {
        input.removeAttribute('name');
        input.classList.remove('is-edited');
      }
      syncDatasetEditStatus();
    });
  });
  datasetColumnInputs.forEach(input => {
    input.addEventListener('input', () => {
      if (input.value !== input.dataset.original) {
        input.name = input.dataset.name;
        input.classList.add('is-edited');
      } else {
        input.removeAttribute('name');
        input.classList.remove('is-edited');
      }
      syncDatasetEditStatus();
    });
  });
  syncDatasetEditStatus();

  let lastSearchMatches = [];

  function clearCellSearchHighlights() {
    lastSearchMatches.forEach(input => input.classList.remove('is-search-match', 'is-search-focus'));
    lastSearchMatches = [];
    const grid = window._ido && window._ido.editorGrid;
    if (grid && grid.searchHighlights) {
      grid.searchHighlights = null;
      grid._dirty = true;
    }
    if (cellSearchStatus) cellSearchStatus.textContent = '';
  }

  function runCellSearch() {
    const query     = cellSearchInput ? cellSearchInput.value.trim().toLowerCase() : '';
    const column    = cellSearchColumn ? cellSearchColumn.value : '';
    const rowNumber = cellSearchRow ? cellSearchRow.value.trim() : '';

    clearCellSearchHighlights();

    const grid = window._ido && window._ido.editorGrid;
    if (!grid || !grid.data) {
      if (cellSearchStatus) cellSearchStatus.textContent = 'Editor not loaded yet.';
      return;
    }

    const maxMatches = 500;
    const matchRows = [];  // { rowIdx, colIdx }

    for (let i = 0; i < grid.data.length; i++) {
      const row = grid.data[i];
      // Row number filter
      if (rowNumber && String(i + 1) !== rowNumber) continue;
      for (let c = 0; c < grid.columns.length; c++) {
        const col = grid.columns[c];
        if (column && col.key !== column) continue;
        const cellVal = row[col.key];
        const cellStr = cellVal === null || cellVal === undefined ? '' : String(cellVal).toLowerCase();
        if (!query || cellStr.includes(query)) {
          matchRows.push({ rowIdx: i, colIdx: c });
          if (matchRows.length >= maxMatches) break;
        }
      }
      if (matchRows.length >= maxMatches) break;
    }

    if (matchRows.length > 0) {
      // Highlight all matching cells in the canvas
      grid.searchHighlights = new Set(matchRows.map(m => `${m.rowIdx}:${grid.columns[m.colIdx].key}`));
      grid._dirty = true;
      // Scroll to first match
      const first = matchRows[0];
      grid.sel = { row: first.rowIdx, col: first.colIdx };
      grid.scrollY = Math.max(0, first.rowIdx * grid.opts.rowHeight - grid.container.clientHeight / 2);
      grid.scroller.scrollTop = grid.scrollY;
      grid._dirty = true;
    } else {
      if (grid.searchHighlights) { grid.searchHighlights = null; grid._dirty = true; }
    }

    if (cellSearchStatus) {
      cellSearchStatus.textContent = matchRows.length > 0
        ? `${matchRows.length} matching cell${matchRows.length === 1 ? '' : 's'} found${matchRows.length >= maxMatches ? ' (first 500)' : ''}.`
        : 'No matching cells found.';
    }
  }

  if (cellSearchBtn) cellSearchBtn.addEventListener('click', runCellSearch);
  if (clearCellSearchBtn) clearCellSearchBtn.addEventListener('click', () => {
    if (cellSearchInput) cellSearchInput.value = '';
    if (cellSearchColumn) cellSearchColumn.value = '';
    if (cellSearchRow) cellSearchRow.value = '';
    clearCellSearchHighlights();
  });
  [cellSearchInput, cellSearchColumn, cellSearchRow].forEach(el => {
    if (!el) return;
    el.addEventListener('keydown', event => {
      if (event.key === 'Enter') {
        event.preventDefault();
        runCellSearch();
      }
    });
  });

  document.querySelectorAll('.opt-item').forEach((item, index) => {
    if (item.querySelector('.feature-apply-btn')) return;
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'btn btn-outline btn-sm feature-apply-btn';
    btn.textContent = 'Apply Feature';
    btn.dataset.featureIndex = String(index);
    if (item.classList.contains('opt-item-select') || item.classList.contains('opt-item-block')) {
      const info = item.querySelector('.opt-info') || item;
      info.appendChild(btn);
    } else {
      item.appendChild(btn);
    }
  });

  function snapshotForm(form) {
    const controls = Array.from(form.querySelectorAll('[name], .dataset-edit-input, .dataset-column-input'));
    const seen = new Set();
    return controls.map(el => {
      const key = el.name || el.dataset.name;
      if (el.type === 'hidden' && key && form.querySelector(`[name="${CSS.escape(key)}"][type="checkbox"]`)) {
        return null;
      }
      if (!key || seen.has(key)) return null;
      seen.add(key);
      return {
        key,
        name: el.name || '',
        dataName: el.dataset.name || '',
        type: el.type,
        checked: !!el.checked,
        value: el.value,
        hasName: !!el.name
      };
    }).filter(Boolean);
  }

  function restoreSnapshot(form, snapshot) {
    snapshot.forEach(item => {
      const selector = item.name
        ? `[name="${CSS.escape(item.name)}"]`
        : `[data-name="${CSS.escape(item.dataName || item.key)}"]`;
      const el = form.querySelector(selector);
      if (!el) return;
      if (item.type === 'checkbox' || item.type === 'radio') {
        el.checked = item.checked;
      } else {
        el.value = item.value;
      }
      if (item.dataName) {
        if (item.hasName) el.name = item.dataName;
        else el.removeAttribute('name');
      }
    });
    if (missingCellInputs.length && missingEditorStatus) {
      const event = new Event('input');
      missingCellInputs.forEach(input => input.dispatchEvent(event));
    }
    datasetEditInputs.forEach(input => input.dispatchEvent(new Event('input')));
    datasetColumnInputs.forEach(input => input.dispatchEvent(new Event('input')));
    syncDatasetEditStatus();
  }

  function captureMissingReviewState() {
    if (!missingValuesGroup) return null;
    return {
      groupClass: missingValuesGroup.className,
      panelClass: missingReviewPanel ? missingReviewPanel.className : '',
      editorClass: wideMissingEditor ? wideMissingEditor.className : '',
      editorDisplay: wideMissingEditor ? wideMissingEditor.style.display : '',
      title: missingReviewTitle ? missingReviewTitle.textContent : '',
      subtitle: missingReviewSubtitle ? missingReviewSubtitle.textContent : '',
      summaryHtml: missingColumnSummary ? missingColumnSummary.innerHTML : '',
      rows: missingEditorRows.map(row => ({
        display: row.style.display || '',
        cells: Array.from(row.querySelectorAll('[data-column]')).map(cell => {
          const input = cell.querySelector('input');
          return {
            text: input ? '' : cell.textContent,
            title: input ? '' : (cell.title || ''),
            inputValue: input ? input.value : null,
            inputName: input ? (input.name || '') : null,
            inputDisabled: input ? !!input.disabled : null,
            inputClass: input ? input.className : null,
            inputPlaceholder: input ? input.placeholder : null,
          };
        })
      })),
      badgeStates: Array.from(missingValuesGroup.querySelectorAll('.badge')).map(badge => ({
        text: badge.textContent,
        className: badge.className
      }))
    };
  }

  function restoreMissingReviewState(state) {
    if (!state || !missingValuesGroup) return;
    missingValuesGroup.className = state.groupClass || missingValuesGroup.className;
    if (missingReviewPanel) missingReviewPanel.className = state.panelClass || missingReviewPanel.className;
    if (wideMissingEditor) {
      wideMissingEditor.className = state.editorClass || wideMissingEditor.className;
      wideMissingEditor.style.display = state.editorDisplay || '';
    }
    if (missingReviewTitle) missingReviewTitle.textContent = state.title || 'Missing Values Found';
    if (missingReviewSubtitle) missingReviewSubtitle.textContent = state.subtitle || '';
    if (missingColumnSummary) missingColumnSummary.innerHTML = state.summaryHtml || '';
    (state.rows || []).forEach((savedRow, rowIndex) => {
      const row = missingEditorRows[rowIndex];
      if (!row) return;
      row.style.display = savedRow.display || '';
      (savedRow.cells || []).forEach((savedCell, cellIndex) => {
        const cell = row.querySelectorAll('[data-column]')[cellIndex];
        if (!cell) return;
        const input = cell.querySelector('input');
        if (input) {
          input.value = savedCell.inputValue || '';
          input.name = savedCell.inputName || '';
          if (!savedCell.inputName) input.removeAttribute('name');
          input.disabled = !!savedCell.inputDisabled;
          input.className = savedCell.inputClass || '';
          input.placeholder = savedCell.inputPlaceholder || 'Type value';
        } else {
          cell.textContent = savedCell.text || '';
          cell.title = savedCell.title || '';
        }
      });
    });
    Array.from(missingValuesGroup.querySelectorAll('.badge')).forEach((badge, index) => {
      const saved = state.badgeStates && state.badgeStates[index];
      if (!saved) return;
      badge.textContent = saved.text;
      badge.className = saved.className;
    });
  }

  function updateMissingReviewState(summary) {
    if (!missingValuesGroup || !summary || summary.missing_remaining === undefined) return;
    const remaining = Number(summary.missing_remaining) || 0;
    const remainingRows = Number(summary.missing_rows_remaining) || 0;
    const remainingColumns = Array.isArray(summary.missing_by_column_remaining)
      ? summary.missing_by_column_remaining
      : [];

    missingValuesGroup.querySelectorAll('.badge').forEach(badge => {
      if (!/missing cells|affected rows/i.test(badge.textContent)) return;
      badge.textContent = badge.textContent.toLowerCase().includes('affected rows')
        ? `${remainingRows} affected rows`
        : `${remaining} missing cells`;
      badge.classList.toggle('badge-orange', remaining > 0);
      badge.classList.toggle('badge-green', remaining === 0);
      // Un-hide badges that were hidden at page load because the dataset had
      // zero missing values then (e.g. #missingStrategyBadge) — a later
      // action (like a datetime conversion that coerces bad strings to NaT)
      // can introduce missing values that didn't exist on upload.
      badge.style.display = remaining > 0 ? '' : badge.style.display;
    });

    // The "Missing Values Found" panel itself is hidden at page load when the
    // dataset starts with zero missing values (see index.html). If a later
    // action introduces missing values, un-hide it here so it isn't stuck
    // invisible for the rest of the session.
    if (missingReviewPanel && remaining > 0) {
      missingReviewPanel.style.display = '';
    }

    if (remaining === 0) {
      if (missingReviewTitle) missingReviewTitle.textContent = 'Missing Values Handled';
      if (missingReviewSubtitle) {
        missingReviewSubtitle.textContent =
          'Change Log has no true missing cells after the applied actions. The original blank-cell editor is hidden to avoid showing stale cells.';
      }
      if (missingColumnSummary) {
        missingColumnSummary.innerHTML = '<span class="missing-count-pill is-clear">0 missing cells remain in Change Log</span>';
      }
      if (wideMissingEditor) wideMissingEditor.style.display = 'none';
      if (missingReviewPanel) missingReviewPanel.classList.add('is-handled');
      return;
    }

    if (missingReviewTitle) missingReviewTitle.textContent = 'Remaining Missing Values';
    if (missingReviewSubtitle) {
      missingReviewSubtitle.textContent =
        `Change Log still has ${remaining} true missing cell${remaining === 1 ? '' : 's'} across ${remainingRows} row${remainingRows === 1 ? '' : 's'}.`;
    }
    if (missingColumnSummary) {
      missingColumnSummary.innerHTML = remainingColumns.length
        ? remainingColumns.map(item =>
          `<span class="missing-count-pill" data-column="${escapeHtml(item.column)}">${escapeHtml(item.column)}: ${escapeHtml(item.missing)} (${escapeHtml(item.percent)}%)</span>`
        ).join('')
        : `<span class="missing-count-pill">${remaining} missing cells remain</span>`;
    }
    if (wideMissingEditor) wideMissingEditor.style.display = '';
    if (missingReviewPanel) missingReviewPanel.classList.remove('is-handled');
  }

  // ── Dynamic panel refresh (Data Types table + Outlier column list) ─────
  // These two panels are otherwise built once from server-rendered Jinja at
  // page load and never touched again, so they go stale after any action
  // that changes the column set or dtypes (drop columns, convert dtypes).
  const dataTypesGroupEl = document.getElementById('dataTypesGroup');
  const dtypeTableBodyEl = dataTypesGroupEl ? dataTypesGroupEl.querySelector('.dtype-table-body') : null;

  function rebuildDtypeTable(analysis) {
    if (!dtypeTableBodyEl || !analysis || !analysis.dtypes) return;
    const rowsHtml = Object.entries(analysis.dtypes).map(([col, dtype]) => `
      <div class="dtype-row">
        <span class="dtype-col-name" title="${escapeHtml(col)}">${escapeHtml(col)}</span>
        <span class="dtype-current">${escapeHtml(dtype)}</span>
        <select name="dtype_map[${escapeHtml(col)}]" class="dtype-select">
          <option value="auto" selected>No manual conversion</option>
          <option value="int">Integer (Int64)</option>
          <option value="float">Float (float64)</option>
          <option value="datetime">Datetime (datetime64[ns])</option>
          <option value="string">Text (object)</option>
          <option value="category">Category (category)</option>
          <option value="bool">Boolean (boolean)</option>
        </select>
      </div>`).join('');
    dtypeTableBodyEl.innerHTML = rowsHtml;
  }

  function rebuildOutlierColumnList(analysis) {
    // NOTE: outlierColumnList/outlierColumnSelector/selectAllOutlierCols are
    // declared in a *different* DOMContentLoaded closure further down this
    // file, so they aren't reachable as closure variables from here — look
    // them up directly by id instead.
    const listEl = document.getElementById('outlierColumnList');
    const selectorEl = document.getElementById('outlierColumnSelector');
    const selectAllEl = document.getElementById('selectAllOutlierCols');
    if (!listEl || !analysis) return;
    const numericCols = analysis.numeric_columns || [];
    listEl.innerHTML = '';
    numericCols.forEach(col => {
      const label = document.createElement('label');
      label.style.cssText = 'display:inline-flex;align-items:center;gap:4px;font-size:0.82rem;cursor:pointer;white-space:nowrap;';
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.value = col;
      cb.className = 'outlier-col-checkbox';
      cb.checked = false;
      label.appendChild(cb);
      label.appendChild(document.createTextNode(col));
      listEl.appendChild(label);
    });
    if (selectorEl) selectorEl.style.display = numericCols.length ? 'block' : 'none';
    if (selectAllEl) {
      selectAllEl.checked = false;
      selectAllEl.indeterminate = false;
    }
  }

  // Called after every successful preview/apply (with fresh server analysis)
  // and after every Undo/Reapply restore (with that step's saved analysis).
  function updateUI(analysis) {
    if (!analysis) return;
    rebuildDtypeTable(analysis);
    rebuildOutlierColumnList(analysis);
    rebuildMissingEditorFromAnalysis(analysis);
    rebuildAuthoritativeCleaningCandidates(analysis);
  }

  function rebuildMissingEditorFromAnalysis(analysis) {
    if (!wideMissingEditor) return;
    const rows = Array.isArray(analysis.missing_rows) ? analysis.missing_rows : [];
    const columns = Object.keys(analysis.dtypes || {});
    const theadRow = wideMissingEditor.querySelector('thead tr');
    const tbody = wideMissingEditor.querySelector('tbody');
    if (!theadRow || !tbody) return;

    wideMissingEditor.style.display = rows.length ? '' : 'none';
    const badge = wideMissingEditor.querySelector('.wide-missing-header .badge');
    if (badge) badge.textContent = `Showing ${rows.length} of ${Number(analysis.missing_rows_total || rows.length)} affected rows`;
    theadRow.innerHTML = '<th>Uploaded Row #</th>' + columns.map(column =>
      `<th data-column="${escapeHtml(column)}">${escapeHtml(column)}</th>`
    ).join('');
    tbody.innerHTML = rows.map(row => {
      const missingColumns = new Set((row.missing_columns || []).map(String));
      const cells = columns.map(column => {
        if (missingColumns.has(String(column))) {
          return `<td class="missing-cell-edit" data-column="${escapeHtml(column)}"><input type="text" name="missing_cell[${escapeHtml(row.row_index)}][${escapeHtml(column)}]" data-column="${escapeHtml(column)}" data-row-number="${escapeHtml(row.row_number)}" placeholder="Type value"></td>`;
        }
        const value = row.row_data && row.row_data[column] != null ? row.row_data[column] : '';
        return `<td data-column="${escapeHtml(column)}" title="${escapeHtml(value)}">${escapeHtml(value)}</td>`;
      }).join('');
      return `<tr data-row-number="${escapeHtml(row.row_number)}" data-row-id="${escapeHtml(row.row_index)}"><td class="missing-row-number">Row ${escapeHtml(row.row_number)}</td>${cells}</tr>`;
    }).join('');

    missingEditorRows = Array.from(tbody.querySelectorAll('tr'));
    missingCellInputs = Array.from(tbody.querySelectorAll('.missing-cell-edit input'));
    missingCellInputs.forEach(input => {
      delete input.dataset.userTyped;
      bindMissingInput(input);
    });
    syncMissingEditorStatus();
  }

  function rebuildAuthoritativeCleaningCandidates(analysis) {
    const numeric = new Set((analysis.numeric_columns || []).map(String));
    const missing = Array.isArray(analysis.missing_by_column) ? analysis.missing_by_column : [];
    const rebuildMissing = (itemId, wantNumeric, choices) => {
      const item = document.getElementById(itemId);
      if (!item) return;
      const rows = missing.filter(entry => numeric.has(String(entry.column)) === wantNumeric);
      const tbody = item.querySelector('tbody');
      if (tbody) tbody.innerHTML = rows.map(entry => `<tr>
        <td class="per-col-fill-name">${escapeHtml(entry.column)}</td>
        <td class="per-col-fill-count">${escapeHtml(entry.missing)} (${escapeHtml(entry.percent)}%)</td>
        <td><select name="${wantNumeric ? 'fill_numeric_col' : 'fill_categorical_col'}[${escapeHtml(entry.column)}]" class="per-col-fill-select">
          <option value="skip" selected>Skip</option>${choices}
        </select></td></tr>`).join('');
      item.style.display = rows.length ? '' : 'none';
      item.querySelectorAll('.feature-apply-btn').forEach(button => { button.disabled = !rows.length; });
    };
    rebuildMissing('fillNumericItem', true,
      '<option value="mean">Mean (average)</option><option value="median">Median (middle value)</option><option value="zero">Zero</option><option value="drop">Drop rows with missing</option>');
    rebuildMissing('fillCategoricalItem', false,
      '<option value="mode">Mode (most common value)</option><option value="constant">Mark as "Unknown"</option><option value="drop">Drop rows with missing</option>');

    const configureCandidate = (name, count, detail) => {
      const input = cleanForm?.querySelector(`input[type="checkbox"][name="${name}"]`);
      const item = input?.closest('.opt-item');
      if (!item) return;
      input.checked = false;
      input.disabled = count === 0;
      item.classList.toggle('is-resolved', count === 0);
      const countBadge = Array.from(item.querySelectorAll('.badge')).find(badge => /found|resolved/i.test(badge.textContent));
      if (countBadge) countBadge.textContent = count ? `${count} found` : 'Resolved';
      const why = item.querySelector('.opt-why');
      if (why && detail) why.textContent = detail;
      item.querySelectorAll('.feature-apply-btn').forEach(button => { button.disabled = count === 0; });
    };
    configureCandidate('remove_duplicates', Number(analysis.duplicate_count || 0));
    configureCandidate('drop_empty_columns', (analysis.empty_columns || []).length,
      (analysis.empty_columns || []).length ? `Will remove: ${(analysis.empty_columns || []).join(', ')}` : 'No fully empty columns remain.');
    configureCandidate('drop_constant_columns', (analysis.constant_columns || []).length,
      (analysis.constant_columns || []).length ? `Will remove: ${(analysis.constant_columns || []).join(', ')}` : 'No constant columns remain.');
    configureCandidate('trim_whitespace', Number(analysis.whitespace_cells || 0),
      Number(analysis.whitespace_cells || 0) ? `${analysis.whitespace_cells} text cells contain outer whitespace.` : 'No leading or trailing whitespace remains.');

    const dropBody = document.querySelector('.drop-selected-columns-body');
    if (dropBody) {
      const autoEmpty = new Set(analysis.empty_columns || []);
      dropBody.innerHTML = Object.keys(analysis.dtypes || {}).filter(column => !autoEmpty.has(column)).map(column => `
        <div class="dtype-row"><span class="dtype-col-name" title="${escapeHtml(column)}">${escapeHtml(column)}</span>
        <label class="mini-check"><input type="checkbox" name="drop_column[${escapeHtml(column)}]" value="1"> Drop</label></div>`).join('');
    }
  }
  window._ido.updateUI = updateUI;
  // Seed with the dtypes the page was rendered with, so that undoing the
  // very first Apply (before any other analysisSnapshot exists) still has
  // something correct to restore the Data Types/Outlier panels to.
  (function seedInitialAnalysis() {
    const serverDataEl = document.getElementById('indataout-server-data');
    if (!serverDataEl) { window._ido.lastAnalysis = null; return; }
    try {
      const dtypes = JSON.parse(serverDataEl.dataset.initialDtypes || '{}');
      const numericColumns = JSON.parse(serverDataEl.dataset.numericColumns || '[]');
      window._ido.lastAnalysis = { dtypes, numeric_columns: numericColumns };
    } catch (e) {
      window._ido.lastAnalysis = null;
    }
  })();

  function capturePreviewState() {
    const thead = document.getElementById('livePreviewThead') || (livePreviewTable ? livePreviewTable.querySelector('thead') : null);
    const tbody  = document.getElementById('livePreviewTbody');
    // Save virtual scroll rows array so undo can restore it properly
    const savedAllRows = tbody && tbody._allRows ? tbody._allRows.slice() : null;
    // Save canvas editor state for undo
    const editorGrid = window._ido && window._ido.editorGrid;
    return {
      formSnapshot: cleanForm ? snapshotForm(cleanForm) : [],
      datasetEditorSnapshot: snapshotDatasetEditor(),
      missingReviewSnapshot: captureMissingReviewState(),
      appliedControlEntries: window._ido.appliedControlEntries.map(([key, value]) => [key, value]),
      workingColumns: currentWorkingColumns ? currentWorkingColumns.slice() : null,
      thead: thead ? thead.innerHTML : '',
      tbody: tbody ? tbody.innerHTML : '',
      savedAllRows,
      editorData: editorGrid ? editorGrid.data.slice() : [],
      editorColumns: editorGrid ? editorGrid.columns.map(c => c.key) : [],
      rowIndexMap: null, // rowIndexMap is module-scoped in the second closure, no need to snapshot
      metrics: liveMetrics ? liveMetrics.innerHTML : '',
      log: liveActionLog ? liveActionLog.innerHTML : '',
      code: liveCodeLog ? liveCodeLog.textContent : '',
      appliedVisibleOpsCount,
      // Fresh analysis of the dataset as of this snapshot (see updateUI) —
      // lets Undo/Reapply put the Data Types table and Outlier column list
      // back to what they looked like at this exact point in history,
      // instead of leaving them showing whatever the most recent apply left.
      analysisSnapshot: window._ido.lastAnalysis
    };
  }

  function restorePreviewState(state) {
    if (!state) return;
    if (cleanForm) restoreSnapshot(cleanForm, state.formSnapshot || []);
    window._ido.appliedControlEntries = (state.appliedControlEntries || []).map(([key, value]) => [key, value]);
    restoreDatasetEditorSnapshot(state.datasetEditorSnapshot);
    restoreMissingReviewState(state.missingReviewSnapshot);
    applyWorkingColumns(state.workingColumns || null);

    // Restore the progress table — must restore virtual scroll state, not just innerHTML
    const thead = document.getElementById('livePreviewThead') || (livePreviewTable ? livePreviewTable.querySelector('thead') : null);
    const tbody  = document.getElementById('livePreviewTbody');
    const scrollArea = document.getElementById('livePreviewScrollArea');

    if (thead) thead.innerHTML = state.thead || '';

    if (tbody && state.savedAllRows && state.savedAllRows.length > 0) {
      // Detach old scroll listener before restoring
      if (tbody._scrollListener && scrollArea) {
        scrollArea.removeEventListener('scroll', tbody._scrollListener);
      }
      tbody._allRows = state.savedAllRows;
      tbody._rowHeight = 32;
      tbody._visibleCount = 30;

      // Re-attach scroll listener
      function renderRestoredRows() {
        if (!scrollArea || !tbody._allRows) return;
        const scrollTop = scrollArea ? scrollArea.scrollTop : 0;
        const colCount = (state.thead.match(/<th/g) || []).length || 1;
        const startIdx = Math.max(0, Math.floor(scrollTop / tbody._rowHeight) - 5);
        const endIdx = Math.min(tbody._allRows.length, startIdx + tbody._visibleCount + 10);
        const topSpacer = startIdx * tbody._rowHeight;
        const bottomSpacer = (tbody._allRows.length - endIdx) * tbody._rowHeight;
        tbody.innerHTML =
          `<tr style="height:${topSpacer}px"><td colspan="${colCount}" style="border:none;padding:0"></td></tr>` +
          tbody._allRows.slice(startIdx, endIdx).join('') +
          `<tr style="height:${bottomSpacer}px"><td colspan="${colCount}" style="border:none;padding:0"></td></tr>`;
      }
      tbody._scrollListener = renderRestoredRows;
      if (scrollArea) scrollArea.addEventListener('scroll', renderRestoredRows, { passive: true });
      renderRestoredRows();
    } else if (tbody) {
      // Detach any lingering scroll listener
      if (tbody._scrollListener && scrollArea) {
        scrollArea.removeEventListener('scroll', tbody._scrollListener);
        delete tbody._scrollListener;
      }
      delete tbody._allRows;
      // Show placeholder when restoring to pre-apply empty state
      if (!state.tbody || state.tbody.trim() === '' || (state.savedAllRows && state.savedAllRows.length === 0)) {
        tbody.innerHTML = '<tr><td colspan="99" style="text-align:center;padding:32px;color:var(--text-muted);">Hit an <strong>Apply</strong> button on any cleaning section to see changed rows here.</td></tr>';
      } else {
        tbody.innerHTML = state.tbody || '';
      }
    }

    // Restore canvas editor to undo state
    if (window._ido && window._ido.refreshEditorCanvas && state.editorData) {
      window._ido.refreshEditorCanvas(state.editorColumns, state.editorData, state.rowIndexMap);
    }

    if (liveMetrics) liveMetrics.innerHTML = state.metrics || '';
    appliedVisibleOpsCount = Number.isFinite(state.appliedVisibleOpsCount) ? state.appliedVisibleOpsCount : appliedVisibleOpsCount;
    if (liveActionLog) {
      liveActionLog.innerHTML = '<div class="live-log-undo">↩ Undo applied — Change Log restored to previous state.</div>';
    }
    if (liveCodeLog) liveCodeLog.textContent = state.code || '# Undo restored the previous preview.';
    if (state.analysisSnapshot) {
      window._ido.lastAnalysis = state.analysisSnapshot;
      if (window._ido.updateUI) window._ido.updateUI(state.analysisSnapshot);
    }
    showDatasetSlide('progressDatasetSlide');
    const progressSlide = document.getElementById('progressDatasetSlide');
    if (progressSlide) progressSlide.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function addSelectedOutliersToFormData(formData) {
    formData.delete('selected_outlier_rows');
    formData.delete('selected_outlier_records');
    const section = document.getElementById('outlierPreviewSection');
    if (section && section.style.display !== 'none') {
      const checkedBoxes = Array.from(document.querySelectorAll('.outlier-checkbox:checked'));
      const checked = Array.from(new Set(checkedBoxes
        .map(cb => String(cb.dataset.rowId || cb.dataset.row || ''))
        .filter(Boolean)));
      if (checked.length > 0) {
        formData.append('selected_outlier_rows', JSON.stringify(checked));
        const records = checkedBoxes.map(cb => {
          try { return JSON.parse(cb.dataset.record || '{}'); } catch (e) { return {}; }
        }).filter(record => record && (record.column || record.value !== undefined || record.row_number));
        if (records.length) formData.append('selected_outlier_records', JSON.stringify(records));
      }
    }
  }

  function collectScopedChanges(scopeRoot) {
    const set = new Map();
    const remove = new Set();
    if (!scopeRoot) return { set, remove };
    scopeRoot.querySelectorAll('[name]').forEach(el => {
      if (!el.name || el.disabled || el.type === 'button' || el.type === 'submit') return;
      if (el.name.startsWith('missing_cell[') && !isUserTypedMissingInput(el)) return;
      if (el.type === 'checkbox' || el.type === 'radio') {
        if (el.checked) set.set(el.name, el.value);
        else remove.add(el.name);
        return;
      }
      set.set(el.name, el.value);
    });
    if (scopeRoot.id === 'missingValuesGroup' && set.get('missing_strategy') === 'drop') {
      const dropEmpty = cleanForm ? cleanForm.querySelector('input[type="checkbox"][name="drop_empty_columns"]') : null;
      if (dropEmpty && dropEmpty.checked && !dropEmpty.disabled) {
        set.set('drop_empty_columns', '1');
      }
    }
    return { set, remove };
  }

  function mergeEntries(baseEntries, changes) {
    const merged = new Map(baseEntries);
    changes.remove.forEach(key => merged.delete(key));
    changes.set.forEach((value, key) => merged.set(key, value));
    return merged;
  }

  // Diffs two ledger snapshots (arrays of [key, value] pairs, as stored in
  // window._ido.appliedControlEntries) and returns exactly what changed:
  // [{key, before, after}, ...] — before/after is null when the key didn't
  // exist on that side (added or removed entirely, vs. just changed value).
  //
  // This is what makes Undo/Reapply able to keep the ledger honest: an
  // action's own_entries is this diff, captured at apply time. Undo sets
  // each key back to `before` (deleting it if before is null); Reapply sets
  // each key to `after`. Since it's a precise per-key diff — not "restore
  // the whole ledger to some earlier snapshot" — an unrelated action's keys
  // are never touched, same guarantee as the dataset-level manifest.
  function _diffLedger(beforeEntries, afterEntries) {
    const beforeMap = new Map(beforeEntries || []);
    const afterMap  = new Map(afterEntries || []);
    const changed = [];
    const allKeys = new Set([...beforeMap.keys(), ...afterMap.keys()]);
    allKeys.forEach(key => {
      const hadBefore = beforeMap.has(key);
      const hasAfter  = afterMap.has(key);
      const beforeVal = hadBefore ? beforeMap.get(key) : null;
      const afterVal  = hasAfter  ? afterMap.get(key)  : null;
      if (!hadBefore && !hasAfter) return;
      if (hadBefore && hasAfter && beforeVal === afterVal) return;
      changed.push({ key, before: hadBefore ? beforeVal : null, after: hasAfter ? afterVal : null });
    });
    return changed;
  }

  // Applies an own_entries diff (from _diffLedger, or the simple form custom
  // code uses) to the current ledger, in the given direction. Returns the
  // new ledger array — caller is responsible for assigning it back to
  // window._ido.appliedControlEntries.
  function _applyLedgerDiff(ownEntries, direction) {
    const map = new Map(window._ido.appliedControlEntries || []);
    (ownEntries || []).forEach(({ key, before, after }) => {
      const targetVal = direction === 'reverse' ? before : after;
      if (targetVal === null || targetVal === undefined) {
        map.delete(key);
      } else {
        map.set(key, targetVal);
      }
    });
    return Array.from(map.entries());
  }
  // Bridge to other closures — same story as logActionToDB/window._ido.logActionToDB
  // above. _undoAction/_reapplyAction (closure 1, further down) call this
  // directly; without this bridge that's a ReferenceError thrown partway
  // through the undo, AFTER the dataset had already correctly reverted via
  // the server manifest but BEFORE action.undone got set and the card
  // re-rendered — which looks exactly like "undo reverts the data but the
  // status tag and Reapply button never update."
  window._ido._applyLedgerDiff = _applyLedgerDiff;

  function addScopedFillDefaults(entryMap, changes) {
    const hasNumericFillInScope = changes.set.has('fill_numeric');
    const hasCategoricalFillInScope = changes.set.has('fill_categorical');
    if (hasNumericFillInScope && !entryMap.has('fill_categorical')) entryMap.set('fill_categorical', 'skip');
    if (hasCategoricalFillInScope && !entryMap.has('fill_numeric')) entryMap.set('fill_numeric', 'skip');
    if ((hasNumericFillInScope || hasCategoricalFillInScope) && !changes.set.has('missing_strategy')) {
      entryMap.set('missing_strategy', 'fill');
    }
  }

  function featureApplyReadiness(item) {
    if (!item) return { ok: false, message: 'Choose a feature before applying.' };
    const featureName = item.querySelector('.opt-name')?.textContent.trim() || 'this feature';
    const toggle = item.querySelector('.toggle input[type="checkbox"]');
    if (toggle && !toggle.checked) {
      return { ok: false, message: `Turn on "${featureName}" before applying it.` };
    }

    const missingStrategy = item.querySelector('#missingStrategySelect');
    if (missingStrategy && missingStrategy.value === 'ignore') {
      return { ok: false, message: 'Choose Fill or Drop before applying Missing Value Strategy.' };
    }

    const dtypeSelects = Array.from(item.querySelectorAll('.dtype-select'));
    if (dtypeSelects.length > 0 && !dtypeSelects.some(select => select.value !== 'auto')) {
      return { ok: false, message: 'Choose at least one column type before applying Manual Column Type Conversion.' };
    }

    const dropColumns = Array.from(item.querySelectorAll('input[name^="drop_column["]'));
    if (dropColumns.length > 0 && !dropColumns.some(input => input.checked && !input.disabled)) {
      return { ok: false, message: 'Select at least one column before applying Drop Selected Columns.' };
    }

    if (item.closest('#outlierGroup') && item.querySelector('#previewOutliersBtn')) {
      const selectedOutliers = document.querySelectorAll('.outlier-checkbox:checked').length;
      if (selectedOutliers === 0) {
        return { ok: false, message: 'Preview outliers and select at least one row before applying Outlier Choices.' };
      }
    }

    return { ok: true, message: '' };
  }

  function sectionApplyReadiness(group) {
    if (!group) return { ok: false, message: 'Choose a cleaning section before applying.' };
    const sectionName = group.querySelector('.opt-group-label')?.textContent.trim() || 'this section';

    if (group.id === 'dataIntegrityGroup') {
      const hasCheckedToggle = Array.from(group.querySelectorAll('.toggle input[type="checkbox"]')).some(cb => cb.checked);
      const hasReviewedColumn = Array.from(group.querySelectorAll('input[name^="drop_column["]')).some(cb => cb.checked);
      if (!hasCheckedToggle && !hasReviewedColumn) {
        return { ok: false, message: 'Turn on at least one Data Integrity option before applying the section.' };
      }
    }

    if (group.id === 'missingValuesGroup') {
      const strategy = group.querySelector('#missingStrategySelect')?.value || 'ignore';
      const typedValues = Array.from(group.querySelectorAll('input[name^="missing_cell["]')).some(isUserTypedMissingInput);
      if (strategy === 'ignore' && !typedValues) {
        return { ok: false, message: 'Choose Fill or Drop, or type missing-cell values, before applying Missing Values.' };
      }
    }

    if (group.id === 'dataTypesGroup') {
      const autoConvert = group.querySelector('input[type="checkbox"][name="convert_dtypes"]')?.checked;
      const hasManualType = Array.from(group.querySelectorAll('.dtype-select')).some(select => select.value !== 'auto');
      if (!autoConvert && !hasManualType) {
        return { ok: false, message: 'Turn on Auto-Convert or choose at least one manual column type before applying Data Types.' };
      }
    }

    if (group.id === 'outlierGroup') {
      const selectedOutliers = document.querySelectorAll('.outlier-checkbox:checked').length;
      if (selectedOutliers === 0) {
        return { ok: false, message: 'Preview outliers and select at least one row before applying Outlier Choices.' };
      }
    }

    return { ok: true, message: `Ready to apply ${sectionName}.` };
  }

  function mapToFormData(entryMap, scopeName) {
    const formData = new FormData();
    const filename = cleanForm ? cleanForm.querySelector('[name="filename"]') : null;
    if (filename) formData.append('filename', filename.value);
    formData.append('dataset_version', String(window._ido.datasetVersion || 0));
    formData.append('preview_scope', scopeName || 'feature');
    entryMap.forEach((value, key) => formData.append(key, value));
    return formData;
  }

  function buildScopedFormData(scopeRoot, scopeName, includeAppliedControls = true) {
    const changes = collectScopedChanges(scopeRoot);
    // A committed mutation starts from the authoritative current snapshot, so
    // only this feature's controls belong in its request. Preview requests use
    // the applied ledger because they are rendered from the immutable source.
    const entryMap = mergeEntries([], changes);
    addScopedFillDefaults(entryMap, changes);
    const formData = mapToFormData(entryMap, scopeName);

    const hasNumericFill = formData.has('fill_numeric');
    const hasCategoricalFill = formData.has('fill_categorical');
    if (hasNumericFill && !hasCategoricalFill) formData.append('fill_categorical', 'skip');
    if (hasCategoricalFill && !hasNumericFill) formData.append('fill_numeric', 'skip');
    if ((hasNumericFill || hasCategoricalFill) && !formData.has('missing_strategy')) {
      formData.append('missing_strategy', 'fill');
    }

    addSelectedOutliersToFormData(formData);
    // Tell the backend whether this apply came from the outlier section.
    // scopeRoot may be the opt-group itself (section apply) OR an opt-item
    // inside it (feature apply) — check both cases.
    const isOutlierScope = scopeRoot && (
      scopeRoot.id === 'outlierGroup' ||
      (scopeRoot.closest && !!scopeRoot.closest('#outlierGroup'))
    );
    if (isOutlierScope) {
      formData.append('outlier_scope_apply', '1');
    }
    return formData;
  }

  function persistScopedControls(scopeRoot) {
    const changes = collectScopedChanges(scopeRoot);
    const entryMap = mergeEntries([], changes);
    addScopedFillDefaults(entryMap, changes);

    // Outlier row selections are checkboxes rendered dynamically inside the
    // outlier preview table — collectScopedChanges never sees them (they
    // aren't part of the section's own fields), and previewCleaning clears
    // them right after a successful apply. Without persisting them here,
    // the selection only ever existed in that one request's FormData and is
    // gone by the time "Clean Data" runs. Union with any rows already
    // recorded from earlier outlier applies (rows are stable original
    // index labels, so accumulating across rounds is safe and correct —
    // see cleaner.py's manual_outlier_removal handling).
    const isOutlierScope = scopeRoot && (
      scopeRoot.id === 'outlierGroup' ||
      (scopeRoot.closest && !!scopeRoot.closest('#outlierGroup'))
    );
    if (isOutlierScope) {
      const checkedRows = Array.from(document.querySelectorAll('.outlier-checkbox:checked'))
        .map(cb => parseInt(cb.dataset.row, 10))
        .filter(n => !isNaN(n));
      if (false && checkedRows.length > 0) {
        let existing = [];
        try { existing = JSON.parse(entryMap.get('selected_outlier_rows') || '[]'); } catch (e) { existing = []; }
        const union = Array.from(new Set([...existing, ...checkedRows]));
        entryMap.set('selected_outlier_rows', JSON.stringify(union));
        // Deliberately NOT persisting outlier_scope_apply here — that flag
        // means "this specific request is the outlier apply", used by the
        // backend only to decide whether to log a fresh history entry.
        // Baking it into the ledger would make every later, unrelated apply
        // look like an outlier apply too.
      }
    }

    entryMap.delete('selected_outlier_rows');
    entryMap.delete('selected_outlier_records');
    window._ido.appliedControlEntries = Array.from(entryMap.entries());
  }

  function previewCurrentSection() {
    const editorSlide = document.getElementById('datasetEditorSlide');
    if (editorSlide && editorSlide.classList.contains('active')) {
      previewCleaning('Previewed current dataset edits only', editorSlide, 'manual', false);
      return;
    }

    const activeGroup = document.querySelector('.opt-group.active-panel');
    if (activeGroup) {
      const label = activeGroup.querySelector('.opt-group-label')?.textContent.trim() || 'current section';
      previewCleaning(`Previewed current section only: ${label}`, activeGroup, 'section', false);
      return;
    }

    if (liveActionLog) {
      liveActionLog.innerHTML = '<div class="live-log-empty">Choose a cleaning section or Dataset Editor first.</div>';
    }
  }

  function liveCellHighlightSet(operations, rowNumbers) {
    const visibleRows = new Set((rowNumbers || []).map(row => String(row)));
    const highlights = new Set();
    const addCell = (row, column) => {
      if (row === undefined || row === null || !column) return;
      const rowKey = String(row);
      if (visibleRows.size > 0 && !visibleRows.has(rowKey)) return;
      highlights.add(`${rowKey}::${normalizeColumnName(column)}`);
    };
    (operations || []).forEach(op => {
      if (!op || !op.action) return;
      if (op.action === 'manual_cell_updates' || op.action === 'manual_missing_cell_updates') {
        (op.updates || []).forEach(update => addCell(update.row_number, update.column));
      }
      if (op.action === 'fill_numeric') {
        (op.details || []).forEach(item => (item.rows || []).forEach(row => addCell(row, item.column)));
      }
      if (op.action === 'fill_categorical') {
        Object.entries(op.details || {}).forEach(([column, item]) => {
          (item.rows || []).forEach(row => addCell(row, column));
        });
      }
      if (op.action === 'trim_whitespace') {
        (op.details || []).forEach(item => (item.rows || []).forEach(row => addCell(row, item.column)));
      }
    });
    return highlights;
  }

  let _liveClusterize = null;

  function renderLivePreview(preview, columnsFromServer, operations = [], rowNumbers = []) {
    if (!livePreviewTable) return;
    const columns = columnsFromServer && columnsFromServer.length
      ? columnsFromServer
      : (preview && preview.length ? Object.keys(preview[0]) : []);
    const highlights = liveCellHighlightSet(operations, rowNumbers);

    // Always update the thead (lightweight, no virtualization needed)
    const thead = document.getElementById('livePreviewThead') || livePreviewTable.querySelector('thead');
    if (thead) {
      thead.innerHTML = '<tr>' + columns.map(col => `<th>${escapeHtml(col)}</th>`).join('') + '</tr>';
    }

    const contentId = 'livePreviewTbody';
    const scrollId  = 'livePreviewScrollArea';

    if (!preview || preview.length === 0) {
      const emptyRow = `<tr><td colspan="${Math.max(columns.length, 1)}">No rows remain after this preview.</td></tr>`;
      if (_liveClusterize) {
        _liveClusterize.update([emptyRow]);
      } else {
        const tbody = document.getElementById(contentId);
        if (tbody) tbody.innerHTML = emptyRow;
      }
      return;
    }

    // Build ALL rows as HTML strings — Clusterize only renders ~30 visible ones at a time
    const rowsHtml = preview.map((row, rowIndex) => {
      const rowNumber = rowNumbers[rowIndex];
      const cells = columns.map(col => {
        const value = row[col] === null || row[col] === undefined ? '' : row[col];
        const updated = highlights.has(`${String(rowNumber)}::${normalizeColumnName(col)}`);
        return `<td class="${updated ? 'live-cell-updated' : ''}" title="${escapeHtml(String(value))}">${escapeHtml(String(value))}</td>`;
      }).join('');
      return `<tr>${cells}</tr>`;
    });

    // Pure JS virtual scroll - no external libraries
    const scrollArea = document.getElementById(scrollId);
    const tbody = document.getElementById(contentId);
    if (!tbody) return;

    // Store all rows for virtual scrolling
    tbody._allRows = rowsHtml;
    tbody._rowHeight = 32; // approximate row height in pixels
    tbody._visibleCount = 30; // number of rows to render at once

    function renderVisibleRows() {
      if (!scrollArea || !tbody._allRows) return;
      const scrollTop = scrollArea.scrollTop;
      const startIdx = Math.max(0, Math.floor(scrollTop / tbody._rowHeight) - 5);
      const endIdx = Math.min(tbody._allRows.length, startIdx + tbody._visibleCount + 10);
      const visibleRows = tbody._allRows.slice(startIdx, endIdx);

      // Add spacer for total height
      const topSpacer = startIdx * tbody._rowHeight;
      const bottomSpacer = (tbody._allRows.length - endIdx) * tbody._rowHeight;

      tbody.innerHTML = 
        `<tr style="height:${topSpacer}px"><td colspan="${columns.length || 1}" style="border:none;padding:0"></td></tr>` +
        visibleRows.join('') +
        `<tr style="height:${bottomSpacer}px"><td colspan="${columns.length || 1}" style="border:none;padding:0"></td></tr>`;
    }

    // Remove old listener if exists
    if (tbody._scrollListener) {
      scrollArea.removeEventListener('scroll', tbody._scrollListener);
    }
    tbody._scrollListener = renderVisibleRows;
    scrollArea.addEventListener('scroll', renderVisibleRows, { passive: true });

    // Initial render
    renderVisibleRows();

    _liveClusterize = {
      update(rows) {
        tbody._allRows = rows;
        renderVisibleRows();
      },
      destroy() {
        if (scrollArea && tbody._scrollListener) {
          scrollArea.removeEventListener('scroll', tbody._scrollListener);
        }
        tbody._allRows = null;
        tbody._scrollListener = null;
      }
    };
  }

  function describeOperation(op) {
    if (!op || !op.action) return 'Cleaning step applied.';
    if (op.action === 'manual_missing_cell_updates') return `Updated ${op.count} typed missing cell(s).`;
    if (op.action === 'manual_cell_updates') return `Updated ${op.count} edited dataset cell(s).`;
    if (op.action === 'manual_column_renames') return `Renamed ${op.count} dataset column(s).`;
    if (op.action === 'drop_empty_columns') return `Removed ${op.count || 0} fully empty column(s).`;
    if (op.action === 'drop_constant_columns') return `Removed ${op.count || 0} constant column(s).`;
    if (op.action === 'drop_duplicates') return `Removed ${op.removed_rows || 0} duplicate row(s).`;
    if (op.action === 'drop_missing_rows') return `Dropped ${op.dropped_row_count || 0} row(s) with remaining missing values.`;
    if (op.action === 'fill_numeric') {
      const dropped = op.rows_dropped || 0;
      const filled = op.filled_cells || 0;
      if (dropped > 0 && filled > 0) return `Filled ${filled} numeric missing cell(s) and dropped ${dropped} row(s) with missing values.`;
      if (dropped > 0) return `Dropped ${dropped} row(s) with missing values in numeric column(s).`;
      return `Filled ${filled} numeric missing cell(s).`;
    }
    if (op.action === 'fill_categorical') return `Filled ${op.filled_cells || 0} text/category missing cell(s).`;
    if (op.action === 'manual_outlier_removal') return `Removed ${op.rows_removed || 0} selected outlier row(s).`;
    if (op.action === 'convert_dtypes') return `Auto-converted ${(op.changes || []).length} column type(s).`;
    if (op.action === 'manual_convert_dtypes') return `Applied ${Object.keys(op.changes || {}).length} manual column type conversion(s).`;
    if (op.action === 'trim_whitespace') return `Trimmed whitespace in ${(op.columns_processed || []).length} text column(s).`;
    if (op.action === 'rename_columns') return `Renamed ${op.count || 0} column(s).`;
    if (op.action === 'label_encode') return `Label-encoded ${(op.columns_encoded || []).length} categorical column(s).`;
    return op.action.replace(/_/g, ' ');
  }

  function formatList(values, limit = 18) {
    const list = Array.isArray(values) ? values : [];
    if (list.length === 0) return 'none';
    const shown = list.slice(0, limit).map(value => escapeHtml(value));
    const suffix = list.length > limit ? `, +${list.length - limit} more` : '';
    return shown.join(', ') + suffix;
  }

  function operationDetailHtml(op) {
    if (!op || !op.action) return '';
    if (op.action === 'manual_cell_updates' || op.action === 'manual_missing_cell_updates') {
      return (op.updates || []).map(update =>
        `<div class="live-log-detail">Row ${escapeHtml(update.row_number)} / column <code>${escapeHtml(update.column)}</code> -> ${update.blank ? '<code>blank / missing</code>' : `<code>${escapeHtml(update.value)}</code>`}</div>`
      ).join('');
    }
    if (op.action === 'manual_column_renames') {
      return Object.entries(op.renamed_map || {}).map(([from, to]) =>
        `<div class="live-log-detail">Column <code>${escapeHtml(from)}</code> renamed to <code>${escapeHtml(to)}</code></div>`
      ).join('');
    }
    if (op.action === 'drop_empty_columns' || op.action === 'drop_constant_columns' || op.action === 'drop_selected_columns') {
      return `<div class="live-log-detail">Columns: <code>${formatList(op.columns_dropped || [])}</code></div>`;
    }
    if (op.action === 'drop_duplicates') {
      return `<div class="live-log-detail">Dropped uploaded rows: ${formatList(op.rows_dropped || [])}</div>`;
    }
    if (op.action === 'drop_missing_rows') {
      const rows = `<div class="live-log-detail">Dropped uploaded rows: ${formatList(op.row_numbers || [])}</div>`;
      const rowDetails = (op.missing_columns_by_row || []).slice(0, 12).map(item =>
        `<div class="live-log-detail">Row ${escapeHtml(item.row)} had missing columns: <code>${formatList(item.columns || [], 12)}</code></div>`
      ).join('');
      return rows + rowDetails;
    }
    if (op.action === 'fill_numeric') {
      const filled = (op.details || []).filter(item => item.strategy !== 'drop').map(item =>
        `<div class="live-log-detail">Column <code>${escapeHtml(item.column)}</code>: filled ${escapeHtml(item.filled)} cell(s) on rows ${formatList(item.rows || [])} with <code>${escapeHtml(item.value)}</code></div>`
      ).join('');
      const dropped = (op.details || []).filter(item => item.strategy === 'drop' && (item.dropped || 0) > 0).map(item =>
        `<div class="live-log-detail">Column <code>${escapeHtml(item.column)}</code>: dropped ${escapeHtml(item.dropped)} row(s) with missing values — rows ${formatList(item.rows || [])}</div>`
      ).join('');
      const skipped = (op.skipped || []).map(item =>
        `<div class="live-log-detail">Column <code>${escapeHtml(item.column)}</code>: ${escapeHtml(item.missing)} cell(s) still missing because ${escapeHtml(item.reason)}</div>`
      ).join('');
      return filled + dropped + skipped || `<div class="live-log-detail">${escapeHtml(op.reason || 'No numeric cells filled.')}</div>`;
    }
    if (op.action === 'fill_categorical') {
      return Object.entries(op.details || {}).map(([column, item]) =>
        `<div class="live-log-detail">Column <code>${escapeHtml(column)}</code>: filled ${escapeHtml(item.filled || 0)} cell(s) on rows ${formatList(item.rows || [])} with <code>${escapeHtml(item.value || 'Unknown')}</code> (${escapeHtml(item.method || op.strategy)})</div>`
      ).join('');
    }
    if (op.action === 'trim_whitespace') {
      return (op.details || []).map(item =>
        `<div class="live-log-detail">Column <code>${escapeHtml(item.column)}</code>: trimmed rows ${formatList(item.rows || [])}</div>`
      ).join('');
    }
    if (op.action === 'convert_dtypes') {
      return (op.changes || []).map(item =>
        `<div class="live-log-detail">Column <code>${escapeHtml(item.column)}</code> converted to <code>${escapeHtml(item.new_dtype)}</code></div>`
      ).join('');
    }
    if (op.action === 'manual_convert_dtypes') {
      return Object.entries(op.changes || {}).map(([column, item]) =>
        `<div class="live-log-detail">Column <code>${escapeHtml(column)}</code>: requested <code>${escapeHtml(item.requested || item.to)}</code>, result <code>${escapeHtml(item.actual_dtype || item.to)}</code>, coerced ${escapeHtml(item.coerced || 0)} existing value(s) to missing.</div>`
      ).join('');
    }
    if (op.action === 'manual_outlier_removal') {
      // Use the human-readable row_numbers (1-based), not the raw 0-based
      // index labels — see the note in _viewFullChanges for why.
      return `<div class="live-log-detail">Removed uploaded rows: ${formatList(op.row_numbers || op.selected_rows || op.valid_indices_dropped || [])}</div>`;
    }
    return '';
  }

  function pyString(value) {
    return `'${String(value).replace(/\\/g, '\\\\').replace(/'/g, "\\'")}'`;
  }

  function pyValue(value, blank = false) {
    if (blank) return 'np.nan';
    const raw = String(value ?? '');
    if (raw.trim() === '') return "''";
    if (/^-?(?:0|[1-9]\d*)(?:\.\d+)?$/.test(raw.trim())) return raw.trim();
    return pyString(raw);
  }

  function pyList(values) {
    const list = Array.isArray(values) ? values : [];
    return `[${list.map(value => typeof value === 'number' ? String(value) : pyString(value)).join(', ')}]`;
  }

  function pyDict(obj) {
    return `{${Object.entries(obj || {}).map(([key, value]) => `${pyString(key)}: ${pyString(value)}`).join(', ')}}`;
  }

  function uploadedRowsToIndexList(rows) {
    const list = (Array.isArray(rows) ? rows : [])
      .map(row => Number(row))
      .filter(row => Number.isFinite(row))
      .map(row => row - 1);
    return pyList(list);
  }

  function codeLinesForCellUpdates(op) {
    const updates = op.updates || [];
    if (updates.length === 0) return '# No typed cell updates were applied';
    return updates.map(update => {
      const rowIndex = Number.isFinite(Number(update.row_index)) ? Number(update.row_index) : update.row_index;
      return `df.loc[${typeof rowIndex === 'number' ? rowIndex : pyString(rowIndex)}, ${pyString(update.column)}] = ${pyValue(update.value, update.blank)}`;
    }).join('\n');
  }

  function codeLinesForFillNumeric(op) {
    const details = op.details || [];
    if (details.length === 0) return `# ${op.reason || 'No numeric cells were filled.'}`;
    return details.map(item => {
      const rows = uploadedRowsToIndexList(item.rows || []);
      return [
        `# uploaded rows ${pyList(item.rows || [])}`,
        `df.loc[${rows}, ${pyString(item.column)}] = ${pyValue(item.value)}`
      ].join('\n');
    }).join('\n\n');
  }

  function codeLinesForFillCategorical(op) {
    const entries = Object.entries(op.details || {}).filter(([, item]) => (item.filled || 0) > 0);
    if (entries.length === 0) return '# No text/category cells were filled.';
    return entries.map(([column, item]) => {
      const rows = uploadedRowsToIndexList(item.rows || []);
      return [
        `# uploaded rows ${pyList(item.rows || [])}`,
        `df.loc[${rows}, ${pyString(column)}] = ${pyValue(item.value || 'Unknown')}`
      ].join('\n');
    }).join('\n\n');
  }

  function codeLinesForDtypeChanges(op) {
    const changes = op.changes || [];
    if (changes.length === 0) return '# No data type conversions were needed.';
    return changes.map(item => `df[${pyString(item.column)}] = pd.to_numeric(df[${pyString(item.column)}], errors='coerce')`).join('\n');
  }

  function codeForOperation(op) {
    if (!op || !op.action) return '# Cleaning step applied';
    if (op.action === 'manual_missing_cell_updates') return codeLinesForCellUpdates(op);
    if (op.action === 'manual_cell_updates') return codeLinesForCellUpdates(op);
    if (op.action === 'manual_column_renames') return `df = df.rename(columns=${pyDict(op.renamed_map || {})})`;
    if (op.action === 'rename_columns') return 'df.columns = df.columns.str.strip().str.lower().str.replace(\" \", \"_\")';
    if (op.action === 'trim_whitespace') {
      const columns = (op.details || []).map(item => item.column);
      return columns.length
        ? `df[${pyList(columns)}] = df[${pyList(columns)}].apply(lambda s: s.str.strip())`
        : `df[${pyList(op.columns_processed || [])}] = df[${pyList(op.columns_processed || [])}].apply(lambda s: s.str.strip())`;
    }
    if (op.action === 'convert_dtypes') return codeLinesForDtypeChanges(op);
    if (op.action === 'manual_convert_dtypes') {
      return Object.entries(op.changes || {}).map(([column, item]) => {
        const requested = item.requested || item.to;
        if (requested === 'datetime' || item.to === 'datetime64[ns]') return `df[${pyString(column)}] = pd.to_datetime(df[${pyString(column)}], errors='coerce')`;
        if (requested === 'numeric' || item.to === 'float64' || item.to === 'int64') return `df[${pyString(column)}] = pd.to_numeric(df[${pyString(column)}], errors='coerce')`;
        if (requested === 'bool') {
          return [
            'bool_map = {',
            "    'true': True, 't': True, 'yes': True, 'y': True, '1': True,",
            "    'false': False, 'f': False, 'no': False, 'n': False, '0': False,",
            '}',
            `df[${pyString(column)}] = df[${pyString(column)}].astype(str).str.lower().map(bool_map)`
          ].join('\n');
        }
        if (requested === 'string') return `df[${pyString(column)}] = df[${pyString(column)}].astype(object)`;
        return `df[${pyString(column)}] = df[${pyString(column)}].astype(${pyString(requested)})`;
      }).join('\n') || '# No manual type conversions were applied.';
    }
    if (op.action === 'drop_empty_columns') return `df = df.drop(columns=${pyList(op.columns_dropped || [])})`;
    if (op.action === 'drop_selected_columns') return `df = df.drop(columns=${pyList(op.columns_dropped || [])})`;
    if (op.action === 'drop_constant_columns') return `df = df.drop(columns=${pyList(op.columns_dropped || [])})`;
    if (op.action === 'drop_duplicates') {
      const subset = op.subset && op.subset.length ? `subset=${pyList(op.subset)}, ` : '';
      const rows = op.rows_dropped && op.rows_dropped.length ? `\n# uploaded duplicate rows removed: ${pyList(op.rows_dropped)}` : '';
      return `df = df.drop_duplicates(${subset}keep='first')${rows}`;
    }
    if (op.action === 'manual_outlier_removal') return `df = df.drop(index=${pyList(op.valid_indices_dropped || op.selected_positions || [])})`;
    if (op.action === 'drop_missing_rows') return `df = df.drop(index=${uploadedRowsToIndexList(op.row_numbers || [])})\n# uploaded rows removed: ${pyList(op.row_numbers || [])}`;
    if (op.action === 'fill_numeric') return codeLinesForFillNumeric(op);
    if (op.action === 'fill_categorical') return codeLinesForFillCategorical(op);
    if (op.action === 'label_encode') return (op.columns_encoded || []).map(col => `df[${pyString(col)}] = LabelEncoder().fit_transform(df[${pyString(col)}].astype(str))`).join('\n') || '# No categorical columns were label encoded.';
    return `# ${op.action}`;
  }

  function hasMeaningfulChange(op) {
    if (!op || !op.action || ['replace_placeholders', 'rename_columns', 'skip_missing_values'].includes(op.action)) return false;
    if (op.action === 'drop_duplicates') return (op.removed_rows || 0) > 0;
    if (op.action === 'drop_empty_columns' || op.action === 'drop_constant_columns' || op.action === 'drop_selected_columns') {
      return (op.count || 0) > 0 || (op.columns_dropped || op.columns || []).length > 0;
    }
    if (op.action === 'trim_whitespace') return (op.details || []).some(item => (item.count || 0) > 0 || (item.rows || []).length > 0);
    // fill_numeric can act via a per-column "drop rows" strategy, which doesn't fill any
    // cells (filled_cells stays 0) but does remove rows — that still counts as meaningful.
    if (op.action === 'fill_numeric') return (op.filled_cells || 0) > 0 || (op.rows_dropped || 0) > 0;
    if (op.action === 'fill_categorical') return (op.filled_cells || 0) > 0;
    if (op.action === 'manual_missing_cell_updates' || op.action === 'manual_cell_updates') return (op.count || 0) > 0 || (op.updates || []).length > 0;
    if (op.action === 'drop_missing_rows') return (op.dropped_row_count || 0) > 0 || (op.row_numbers || []).length > 0;
    if (op.action === 'manual_outlier_removal') return (op.rows_removed || 0) > 0 || (op.selected_rows || op.valid_indices_dropped || []).length > 0;
    if (op.action === 'convert_dtypes') return (op.changes || []).length > 0;
    if (op.action === 'manual_convert_dtypes') return Object.keys(op.changes || {}).length > 0;
    if (op.action === 'manual_column_renames') return false;
    if (op.action === 'label_encode') return false;
    if (op.action === 'custom_pandas_code') return true;
    return true;
  }

  function visibleOperations(operations) {
    return (operations || []).filter(hasMeaningfulChange);
  }
  // Expose so other modules (changelog, custom code) can call it
  window._ido.visibleOperations = visibleOperations;
  // Bridged for closure #2 (Change Log system) — hasMeaningfulChange is defined
  // here in closure #1 and is not visible there directly. Without this line,
  // every pushAppliedHistory() call throws "hasMeaningfulChange is not defined"
  // and the Change Log silently never updates for any feature.
  window._ido.hasMeaningfulChange = hasMeaningfulChange;

  // Single canonical action->scope membership map. Bridged so both this
  // closure's previewCleaning() (scoped Apply filtering) and closure #2's
  // scopeForOperations() (Change Log filing / Resume) read the exact same
  // object — one definition, not two that can silently drift apart.
  window._ido.SCOPE_OP_MAP = {
    missingValuesGroup: new Set(['fill_numeric','fill_categorical','drop_missing_rows','manual_missing_cell_updates']),
    outlierGroup:       new Set(['manual_outlier_removal','outlier_removal','outlier_clip']),
    dataIntegrityGroup: new Set(['drop_duplicates','drop_empty_columns','drop_constant_columns','trim_whitespace']),
    dataTypesGroup:     new Set(['convert_dtypes','manual_convert_dtypes']),
    datasetEditorSlide: new Set(['manual_cell_updates']),
    dropSelectedColumnsGroup: new Set(['drop_selected_columns']),
    customCodeGroup:    new Set(['custom_pandas_code']),
    // typed missing-cell values use 'manual' scope
    manual:             new Set(['manual_missing_cell_updates','fill_numeric','fill_categorical']),
  };

  // Per-action-type signature of the operation content that was last actually
  // logged to the Change Log. Used (via _newOrChangedOperations below) to detect
  // genuinely new/changed work even when the backend's recomputed-from-scratch
  // operations array doesn't grow — e.g. re-applying the Outliers panel after
  // selecting MORE rows updates the existing manual_outlier_removal entry in
  // place rather than adding a new one, so array length alone can't tell "new"
  // from "unchanged".
  let _lastLoggedOpSignature = {};
  function _newOrChangedOperations(ops) {
    return (ops || []).filter(op => _lastLoggedOpSignature[op.action] !== JSON.stringify(op));
  }
  function _markOperationsLogged(ops) {
    (ops || []).forEach(op => { _lastLoggedOpSignature[op.action] = JSON.stringify(op); });
  }
  window._ido._markOperationsLogged = _markOperationsLogged;
  window._ido._newOrChangedOperations = _newOrChangedOperations;

  function operationsForLatestApply(operations, persistApply) {
    const visible = visibleOperations(operations);
    if (!persistApply) return visible;
    return _newOrChangedOperations(visible);
  }

  // ── Step-by-step before/after preview (Design Feature) ──────────────
  // Renders a clearly labelled before/after block for each applied operation.
  // Bulk ops (trim, drop empty/constant cols, data types) → 5-row sample
  // Targeted ops (missing values, duplicates, outliers, custom code) → all affected rows
  // ── Step-by-step before/after preview ───────────────────────────────
  // Called after every real Apply. Builds one clearly-labelled block per operation,
  // with accurate before/after rows and changed cells highlighted in both sides.
  // Undo and Reapply buttons live on each block — not in a separate history panel.

  function buildPreviewTable(columns, rows, highlightCells) {
    // highlightCells = Set of "colName" strings for cells to highlight in this row set
    if (!columns || !columns.length || !rows || !rows.length) {
      return '<div class="step-preview-empty">No rows to display.</div>';
    }
    const visibleCols = columns.slice(0, 12);
    const hl = highlightCells || new Set();
    return `<div class="step-preview-table-scroll"><table class="step-preview-table">
      <thead><tr>${visibleCols.map(c => `<th>${escapeHtml(String(c))}</th>`).join('')}</tr></thead>
      <tbody>${rows.map(row => `<tr>${visibleCols.map(c => {
        const v = row[c] === null || row[c] === undefined ? '' : row[c];
        const isMissing = v === '' && hl.has(c);
        const isChanged = hl.has(c);
        const style = isMissing
          ? 'background:#3d1a1a;color:#f87171;font-style:italic;'
          : isChanged
            ? 'background:#052e16;color:#4ade80;font-weight:600;'
            : '';
        const display = isMissing ? '(missing)' : escapeHtml(String(v));
        return `<td title="${escapeHtml(String(v))}" style="${style}">${display}</td>`;
      }).join('')}</tr>`).join('')}</tbody>
    </table></div>`;
  }

  function stepPreviewHeaderText(op) {
    if (!op) return 'Cleaning step applied';
    if (op.action === 'fill_numeric') {
      const dropped = op.rows_dropped || 0, filled = op.filled_cells || 0;
      if (dropped > 0 && filled > 0) return `Missing values — ${filled} numeric cell${filled!==1?'s':''} filled, ${dropped} row${dropped!==1?'s':''} dropped`;
      if (dropped > 0) return `Missing values — ${dropped} row${dropped!==1?'s':''} dropped (numeric column${dropped!==1?'s':''})`;
      return `Missing values — ${filled} numeric cell${filled!==1?'s':''} filled`;
    }
    if (op.action === 'fill_categorical')           return `Missing values — ${op.filled_cells||0} text/category cell${(op.filled_cells||0)!==1?'s':''} filled`;
    if (op.action === 'manual_missing_cell_updates')return `Missing values — ${op.count||0} cell${(op.count||0)!==1?'s':''} filled manually`;
    if (op.action === 'drop_duplicates')            return `Drop duplicates — ${op.removed_rows||0} row${(op.removed_rows||0)!==1?'s':''} removed`;
    if (op.action === 'drop_empty_columns')         return `Drop empty columns — ${op.count||0} column${(op.count||0)!==1?'s':''} removed, showing 5-row sample`;
    if (op.action === 'drop_constant_columns')      return `Drop constant columns — ${op.count||0} column${(op.count||0)!==1?'s':''} removed, showing 5-row sample`;
    if (op.action === 'trim_whitespace')            return `Trim whitespace — applied to all text columns, showing 5-row sample`;
    if (op.action === 'convert_dtypes')             return `Data types — ${(op.changes||[]).length} column${(op.changes||[]).length!==1?'s':''} converted, showing 5-row sample`;
    if (op.action === 'manual_convert_dtypes')      return `Data types — ${Object.keys(op.changes||{}).length} column${Object.keys(op.changes||{}).length!==1?'s':''} converted, showing 5-row sample`;
    if (op.action === 'manual_outlier_removal')     return `Outliers — ${op.rows_removed||0} row${(op.rows_removed||0)!==1?'s':''} removed`;
    if (op.action === 'drop_missing_rows')          return `Missing values — ${op.dropped_row_count||0} row${(op.dropped_row_count||0)!==1?'s':''} dropped`;
    if (op.action === 'custom_pandas_code')         return `Custom code — applied successfully, showing all affected rows`;
    if (op.action === 'manual_cell_updates')        return `Dataset edits — ${op.count||0} cell${(op.count||0)!==1?'s':''} updated`;
    if (op.action === 'drop_selected_columns')      return `Drop columns — ${op.count||0} column${(op.count||0)!==1?'s':''} removed`;
    return op.action.replace(/_/g,' ');
  }

  const SHOW_ALL_OPS = new Set(['fill_numeric','fill_categorical','manual_missing_cell_updates',
    'drop_duplicates','drop_missing_rows','manual_outlier_removal','custom_pandas_code','manual_cell_updates']);

  // stepBlocks stores per-step data so Undo/Reapply buttons can reference their entry
  // Each entry: { opLabel, histIdx (index into appliedHistory at time of render) }
  // We don't store histIdx directly — instead we match by label from appliedHistory

  function renderStepPreview(operations, changedRows, changedRowNums, removedRowNums, columnsAfter, allBeforeRows, allBeforeRowNums) {
    const stepLog     = document.getElementById('stepPreviewLog');
    const stepContent = document.getElementById('stepPreviewContent');
    if (!stepLog || !stepContent) return;

    const visible = (operations || []).filter(op =>
      op && op.action && !['replace_placeholders','rename_columns','skip_missing_values'].includes(op.action)
    );

    // stepPreviewLog is permanently hidden — content is embedded inside each action card instead.
    // We still build the HTML so pushAppliedHistory can attach it to the action card.
    if (!visible.length) { window._ido._lastStepPreviewHTML = ''; return; }

    function cloneRow(row) { return Object.assign({}, row); }

    // rowNum → before-row (editor canvas data captured before this apply)
    const beforeRowMap = new Map();
    if (Array.isArray(allBeforeRows) && Array.isArray(allBeforeRowNums)) {
      allBeforeRowNums.forEach((rn, i) => { if (allBeforeRows[i]) beforeRowMap.set(String(rn), allBeforeRows[i]); });
    }

    // rowNum → after-row (changed rows from server response)
    const afterRowMap = new Map();
    if (Array.isArray(changedRows) && Array.isArray(changedRowNums)) {
      changedRowNums.forEach((rn, i) => { if (changedRows[i]) afterRowMap.set(String(rn), changedRows[i]); });
    }

    const cols = columnsAfter || [];

    const blocks = visible.map((op, opIdx) => {
      const isBulk = !SHOW_ALL_OPS.has(op.action);
      let beforeRows = [], afterRows = [];
      // changedCols = set of column names whose cells changed — used for highlighting
      let changedCols = new Set();

      if (isBulk) {
        if (op.action === 'drop_empty_columns' || op.action === 'drop_constant_columns' || op.action === 'drop_selected_columns') {
          const dropped = new Set((op.columns_dropped || []).map(c => String(c)));
          dropped.forEach(c => changedCols.add(c));
          // For column-level ops: show the dropped columns as a before table, no after rows needed
          const sampleNums = allBeforeRowNums ? allBeforeRowNums.slice(0, 5) : [];
          beforeRows = sampleNums.map(rn => beforeRowMap.get(String(rn))).filter(Boolean);
          if (!beforeRows.length && allBeforeRows) beforeRows = allBeforeRows.slice(0, 5);
          afterRows = beforeRows.map(row => {
            const r = cloneRow(row);
            dropped.forEach(col => delete r[col]);
            return r;
          });

        } else if (op.action === 'trim_whitespace') {
          // Show only rows that actually had whitespace in the trimmed column(s)
          const trimmedCols = new Set((op.details || []).map(d => d.column));
          trimmedCols.forEach(c => changedCols.add(c));
          // Collect affected row numbers from op.details if available
          const affectedNums = new Set();
          (op.details || []).forEach(d => {
            (d.rows || []).forEach(rn => affectedNums.add(String(rn)));
          });
          const candidateNums = affectedNums.size > 0
            ? Array.from(affectedNums).slice(0, 5)
            : (allBeforeRowNums || []).slice(0, 5);
          beforeRows = candidateNums.map(rn => beforeRowMap.get(String(rn))).filter(Boolean);
          if (!beforeRows.length && allBeforeRows) beforeRows = allBeforeRows.slice(0, 5);
          afterRows = beforeRows.map(row => {
            const r = cloneRow(row);
            trimmedCols.forEach(col => { if (r[col] != null) r[col] = String(r[col]).trim(); });
            return r;
          });

        } else if (op.action === 'convert_dtypes' || op.action === 'manual_convert_dtypes') {
          (op.changes ? Object.keys(op.changes) : []).forEach(c => changedCols.add(c));
          (op.changes || []).forEach && (op.changes || []).forEach(item => changedCols.add(item.column));
          // Use changedRows (affected rows from server) — these are the rows that changed
          const affectedNums = Array.from(afterRowMap.keys()).slice(0, 5);
          afterRows  = affectedNums.map(rn => afterRowMap.get(rn)).filter(Boolean);
          beforeRows = affectedNums.map(rn => beforeRowMap.get(rn)).filter(Boolean);
          if (!afterRows.length && changedRows) afterRows = changedRows.slice(0, 5);
          if (!beforeRows.length && allBeforeRows) beforeRows = allBeforeRows.slice(0, 5);

        } else {
          // Generic bulk: use changedRows from server as the affected rows
          const affectedNums = Array.from(afterRowMap.keys()).slice(0, 5);
          afterRows  = affectedNums.map(rn => afterRowMap.get(rn)).filter(Boolean);
          beforeRows = affectedNums.map(rn => beforeRowMap.get(rn)).filter(Boolean);
          if (!afterRows.length && changedRows) afterRows = changedRows.slice(0, 5);
          if (!beforeRows.length && allBeforeRows) beforeRows = allBeforeRows.slice(0, 5);
        }

      } else {
        // Targeted ops — reconstruct before by reversing just this op's changes on the after rows

        if (op.action === 'drop_duplicates') {
          const droppedNums = (op.rows_dropped || []).map(String);
          beforeRows = droppedNums.map(rn => beforeRowMap.get(rn)).filter(Boolean);
          afterRows  = [];

        } else if (op.action === 'manual_outlier_removal' || op.action === 'drop_missing_rows') {
          const droppedNums = (op.row_numbers || op.selected_rows || op.valid_indices_dropped || []).map(String);
          beforeRows = droppedNums.map(rn => beforeRowMap.get(rn)).filter(Boolean);
          afterRows  = [];

        } else if (op.action === 'manual_missing_cell_updates' || op.action === 'manual_cell_updates') {
          // updates[]: {row_number, column, value}
          // Before = after-row with those exact columns set back to '' (they were missing/blank)
          const updatesByRow = new Map();
          (op.updates || []).forEach(u => {
            changedCols.add(u.column);
            const rn = String(u.row_number);
            if (!updatesByRow.has(rn)) updatesByRow.set(rn, []);
            updatesByRow.get(rn).push(u.column);
          });
          const nums = Array.from(updatesByRow.keys());
          afterRows  = nums.map(rn => afterRowMap.get(rn)).filter(Boolean);
          beforeRows = nums.map(rn => {
            const after = afterRowMap.get(rn);
            if (!after) return null;
            const r = cloneRow(after);
            (updatesByRow.get(rn) || []).forEach(col => { r[col] = ''; }); // was missing
            return r;
          }).filter(Boolean);

        } else if (op.action === 'fill_numeric') {
          // Per-column "drop" rows no longer exist in the after-dataset at all — they
          // must be reconstructed from beforeRowMap, separately from filled cells
          // (which DO still exist in afterRowMap, just with a value instead of blank).
          const filledByRow = new Map();
          const droppedRowNums = new Set();
          (op.details || []).forEach(item => {
            changedCols.add(item.column);
            if (item.strategy === 'drop') {
              (item.rows || []).forEach(rn => droppedRowNums.add(String(rn)));
              return;
            }
            (item.rows || []).forEach(rn => {
              const key = String(rn);
              if (!filledByRow.has(key)) filledByRow.set(key, []);
              filledByRow.get(key).push(item.column);
            });
          });
          const nums = Array.from(filledByRow.keys());
          afterRows  = nums.map(rn => afterRowMap.get(rn)).filter(Boolean);
          beforeRows = nums.map(rn => {
            const after = afterRowMap.get(rn);
            if (!after) return null;
            const r = cloneRow(after);
            (filledByRow.get(rn) || []).forEach(col => { r[col] = ''; }); // was missing
            return r;
          }).filter(Boolean);
          // Dropped rows: show them as before-only (removed), same convention as
          // drop_duplicates / drop_missing_rows below.
          const droppedBefore = Array.from(droppedRowNums).map(rn => beforeRowMap.get(rn)).filter(Boolean);
          beforeRows = beforeRows.concat(droppedBefore);

        } else if (op.action === 'fill_categorical') {
          const filledByRow = new Map();
          Object.entries(op.details || {}).forEach(([col, item]) => {
            changedCols.add(col);
            (item.rows || []).forEach(rn => {
              const key = String(rn);
              if (!filledByRow.has(key)) filledByRow.set(key, []);
              filledByRow.get(key).push(col);
            });
          });
          const nums = Array.from(filledByRow.keys());
          afterRows  = nums.map(rn => afterRowMap.get(rn)).filter(Boolean);
          beforeRows = nums.map(rn => {
            const after = afterRowMap.get(rn);
            if (!after) return null;
            const r = cloneRow(after);
            (filledByRow.get(rn) || []).forEach(col => { r[col] = ''; }); // was missing
            return r;
          }).filter(Boolean);

        } else if (op.action === 'custom_pandas_code') {
          const changedByRow = new Map();
          (op.changed_cell_details || []).forEach(detail => {
            changedCols.add(detail.column);
            const rn = String(detail.row_number);
            if (!changedByRow.has(rn)) changedByRow.set(rn, []);
            changedByRow.get(rn).push({ column: detail.column, before: detail.before });
          });
          const nums = Array.from(changedByRow.keys());
          afterRows  = nums.map(rn => afterRowMap.get(rn)).filter(Boolean);
          beforeRows = nums.map(rn => {
            const after = afterRowMap.get(rn);
            if (!after) return null;
            const r = cloneRow(after);
            (changedByRow.get(rn) || []).forEach(d => { r[d.column] = d.before; });
            return r;
          }).filter(Boolean);

        } else {
          const nums = Array.from(afterRowMap.keys());
          beforeRows = nums.map(rn => beforeRowMap.get(rn)).filter(Boolean);
          afterRows  = nums.map(rn => afterRowMap.get(rn)).filter(Boolean);
        }
      }

      const header = stepPreviewHeaderText(op);
      // A pure per-column "drop" fill_numeric (no cells filled, only rows dropped) behaves
      // like a remove-op for preview purposes: before rows exist, after rows don't.
      const isPureNumericDrop = op.action === 'fill_numeric' && (op.rows_dropped || 0) > 0 && !afterRows.length;
      const isRemoveOp = ['drop_duplicates','manual_outlier_removal','drop_missing_rows'].includes(op.action) || isPureNumericDrop;
      const isBulkNote = isBulk;

      // Use changedCols for highlighting in both before and after
      // For "before" of fill ops: the changed col should show (missing) style — handled inside buildPreviewTable
      // For "after" of fill ops: the changed col should show green highlight
      const beforeHl = changedCols;  // highlight same cols in before (shows what was blank/different)
      const afterHl  = changedCols;  // highlight what changed in after

      // Build the before table — for fill ops the changed cells show as (missing) since value is ''
      const beforeTable = beforeRows.length
        ? buildPreviewTable(cols, beforeRows, beforeHl)
        : '<div class="step-preview-empty">No data available.</div>';

      let afterContent;
      if (isRemoveOp && beforeRows.length) {
        afterContent = `<div class="step-preview-empty">${beforeRows.length} row${beforeRows.length!==1?'s':''} removed from dataset.</div>`;
      } else if (!afterRows.length) {
        if (op.action === 'drop_empty_columns' || op.action === 'drop_constant_columns') {
          afterContent = op.count === 0
            ? `<div class="step-preview-empty">No columns removed — none qualified.</div>`
            : buildPreviewTable(cols.filter(c => !(op.columns_dropped||[]).includes(c)), afterRows.length ? afterRows : beforeRows.map(r => { const x = cloneRow(r); (op.columns_dropped||[]).forEach(c => delete x[c]); return x; }), new Set());
        } else {
          afterContent = `<div class="step-preview-empty">No row-level changes for this operation.</div>`;
        }
      } else {
        afterContent = buildPreviewTable(cols, afterRows, afterHl);
      }

      const isMixedNumericDrop = op.action === 'fill_numeric' && (op.rows_dropped || 0) > 0 && afterRows.length > 0;
      const note = (isBulkNote && ['drop_empty_columns','drop_constant_columns'].includes(op.action))
        ? '<div class="step-preview-note">Showing sample — this operation affects columns across all rows.</div>'
        : isMixedNumericDrop
          ? `<div class="step-preview-note">Also dropped ${op.rows_dropped} row${op.rows_dropped!==1?'s':''} with missing values (shown in Before, removed from dataset).</div>`
          : '';

      // Undo/Reapply buttons — wire up via data-step-idx matched to applied/undo history
      return `<div class="step-preview-block">
        <div class="step-preview-header">
          <span class="step-preview-header-text">${escapeHtml(header)}</span>
        </div>
        <div class="step-preview-tables">
          <div class="step-preview-col">
            <div class="step-preview-col-label before" title="The dataset exactly as it was right before this action ran — already includes any earlier actions you applied. Undoing this action only reverts to this state; earlier actions are untouched.">Before this action</div>
            ${beforeTable}
          </div>
          <div class="step-preview-col">
            <div class="step-preview-col-label after" title="The dataset immediately after this action ran.">After this action</div>
            ${afterContent}
          </div>
        </div>
        ${note}
      </div>`;
    });

    // Do NOT write to the hidden stepPreviewContent DOM.
    // Expose HTML so pushAppliedHistory can embed it inside each action card.
    window._ido._lastStepPreviewHTML = blocks.join('');
    if (stepContent) stepContent.innerHTML = ''; // keep the hidden container empty

  }

  function renderLiveLog(operations, actionLabel) {
    if (!liveActionLog) return;
    const visible = (operations || []).filter(op =>
      !['replace_placeholders', 'rename_columns', 'skip_missing_values'].includes(op.action)
    );
    const header = actionLabel
      ? `<div class="live-log-applied">${escapeHtml(actionLabel)}</div>`
      : '';
    if (visible.length === 0) {
      liveActionLog.innerHTML = header + `<div class="live-log-empty">${escapeHtml(actionLabel || 'Preview updated')}: no visible changes for this preview.</div>`;
      return;
    }
    liveActionLog.innerHTML = header + visible.map(op =>
      `<div class="live-log-item"><div class="live-log-summary">${escapeHtml(describeOperation(op))}</div>${operationDetailHtml(op)}</div>`
    ).join('');
    if (liveCodeLog) {
      liveCodeLog.textContent = visible.map(codeForOperation).join('\n\n');
    }
  }

  function renderLiveMetrics(summary) {
    if (!liveMetrics || !summary) return;
    liveMetrics.innerHTML = [
      `Rows: ${summary.rows_before} -> ${summary.rows_after}`,
      `Columns: ${summary.cols_after}`,
      `Filled: ${summary.filled_cells}`,
      `Duplicates removed: ${summary.duplicates_removed}`,
      `Outliers removed: ${summary.outliers_removed}`,
      `Cleaning completeness: ${summary.cleaning_completeness || 100}`,
      `Data quality: ${summary.quality_before} -> ${summary.quality_after}`
    ].map(text => `<span>${escapeHtml(text)}</span>`).join('');
  }

  // --- DB action logger — fires after every successful Apply ------------------
  function logActionToDB(actionName, summary, extraConfig) {
    // Fire-and-forget for the caller (never throws), but returns a Promise<id|null>
    // so callers that need the DB row id (e.g. to sync Undo/Reapply status later)
    // can chain onto it without blocking the UI.
    try {
      // Save full appliedControlEntries so resume can replay exact state
      const savedEntries = (window._ido.appliedControlEntries || []).map(([k, v]) => [k, v]);
      const formSnap = cleanForm ? snapshotForm(cleanForm) : [];
      const activePanel = document.querySelector('.opt-group.active-panel');
      // Consumed by pushAppliedHistory() just before this normally runs — captures the
      // same before/after preview HTML shown in the live Change Log card, so Resume
      // can rebuild the exact same "View Full Changes" detail instead of a blank stub.
      const previewHTML = window._ido._lastLoggedPreviewHTML || '';
      window._ido._lastLoggedPreviewHTML = '';
      const appliedAtMs = Number(extraConfig && extraConfig.applied_at_ms) || Date.now();
      return fetch('/log_action', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action_name:   actionName,
          action_config: {
            label:                 actionName,
            appliedControlEntries: savedEntries,
            formSnapshot:          formSnap,
            activePanelId:         activePanel ? activePanel.id : null,
            previewHTML,
            applied_at_ms:         appliedAtMs,
            ...(extraConfig || {}),
          },
          rows_before:   summary ? (summary.rows_before || 0) : 0,
          rows_after:    summary ? (summary.rows_after  || 0) : 0,
          cells_changed: summary ? (summary.filled_cells || summary.changed_rows || 0) : 0,
          // Cache the already-computed DQ score for this action so resume
          // doesn't need to recompute data_quality_score() from scratch —
          // see app.py's get_cached_dq_after / resume_replay.
          dq_after:      summary && typeof summary.quality_after === 'number' ? summary.quality_after : null,
        })
      }).then(r => r.json()).then(d => {
        if (!d.ok) { console.warn('log_action soft-fail:', d.reason); showChangeLogSyncWarning(); return null; }
        return d.id || null;
      }).catch(e => { console.warn('log_action network error:', e); showChangeLogSyncWarning(); return null; });
    } catch(e) { /* never crash the UI */ return Promise.resolve(null); }
  }
  // Bridge to other closures — logActionToDB is only defined here (closure 1),
  // but applyCell (custom code Apply, closure 2 further down) calls it
  // directly. Without this, that call is a plain ReferenceError: "logActionToDB
  // is not defined", which is exactly what surfaced as the Custom Pandas Code
  // Apply failure. See _diffLedger/_captureSnap above for the same pattern —
  // window._ido is this file's established way of sharing functions across
  // its several separate DOMContentLoaded closures.
  window._ido.logActionToDB = logActionToDB;

  // A8 — surface DB-sync failures instead of only logging to console.
  // Shown once per session; the app keeps working locally either way.
  let _changeLogSyncWarned = false;
  function showChangeLogSyncWarning() {
    if (_changeLogSyncWarned) return;
    _changeLogSyncWarned = true;
    // changeLogBlocksContainer is declared with `let` inside the SECOND
    // DOMContentLoaded closure further down this file — this function lives
    // in the first closure, which can never see it (separate closures don't
    // share let/const bindings, regardless of load order). Looking the
    // element up directly avoids the ReferenceError entirely.
    const host = document.getElementById('liveActionLog') || document.getElementById('changeLogBlocks');
    if (!host) return;
    const banner = document.createElement('div');
    banner.className = 'live-log-empty';
    banner.style.cssText = 'color:#b91c1c;background:#fef2f2;border:1px solid #fecaca;padding:8px 12px;border-radius:6px;margin-bottom:8px;';
    banner.textContent = "Couldn't save to the Change Log on the server — your changes still apply to this session, but may not be there after you reload or resume.";
    host.parentNode ? host.parentNode.insertBefore(banner, host) : host.prepend(banner);
  }

  // --- Show a real visible error in the progress log -----------------------
  function showPreviewError(msg, isSessionExpired) {
    if (msg && msg.message) msg = msg.message;
    const html = isSessionExpired
      ? `<div class="live-log-error">⚠️ ${escapeHtml(msg)} <a href="/projects" style="color:inherit;font-weight:600;text-decoration:underline">Go to My Projects</a> to reopen this file.</div>`
      : `<div class="live-log-error">❌ ${escapeHtml(msg)}</div>`;
    if (liveActionLog) liveActionLog.innerHTML = html;
    if (window._ido.operationProgress) window._ido.operationProgress.finish('error', String(msg || 'Operation failed'));
    // Also scroll progress into view so the user sees the error
    showDatasetSlide('progressDatasetSlide');
    const slide = document.getElementById('progressDatasetSlide');
    if (slide) slide.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  let _applyInProgress = false;

  function setApplyBusy(busy, label, activeButton = null) {
    _applyInProgress = busy;
    document.querySelectorAll(
      '.section-apply-btn, .feature-apply-btn, #applyTypedPreviewBtn, #applyDatasetEditsBtn'
    ).forEach(btn => {
      if (busy && !btn.dataset.originalLabel) btn.dataset.originalLabel = btn.textContent;
      if (busy && label && (!activeButton || btn === activeButton)) btn.textContent = label;
      if (!busy && btn.dataset.originalLabel) {
        btn.textContent = btn.dataset.originalLabel;
        delete btn.dataset.originalLabel;
      }
      btn.disabled = busy;
      const isActive = busy && (!activeButton || btn === activeButton);
      btn.setAttribute('aria-busy', isActive ? 'true' : 'false');
      btn.classList.toggle('is-busy', isActive);
      btn.style.opacity = busy && !isActive ? '0.55' : '';
      btn.style.cursor  = isActive ? 'wait' : '';
    });
  }

  /**
   * Preview or commit the current cleaning section.
   *
   * Preview requests update visible proposed rows only. When persistApply is
   * true, the server must return an authoritative history entry before the UI
   * inserts or updates a Change Log card.
   */
  async function previewCleaning(actionLabel, scopeRoot = null, scopeName = 'all', persistApply = false, activeButton = null) {
    if (!cleanForm) return;
    if (_applyInProgress) return;   // block double-click / concurrent applies
    _applyInProgress = true;
    setApplyBusy(true, persistApply ? 'Applying...' : 'Previewing...', activeButton);
    if (window._ido.operationProgress) window._ido.operationProgress.start(persistApply ? 'Apply' : 'Preview', actionLabel || 'Preparing request');
    const formData = scopeRoot ? buildScopedFormData(scopeRoot, scopeName, !persistApply) : new FormData(cleanForm);
    if (!scopeRoot) addSelectedOutliersToFormData(formData);
    // Tells the backend this call is a real Apply, not a live preview while
    // the user is still adjusting a toggle — only real Applies touch the
    // persistent current-state snapshot and generate an undo/redo manifest.
    if (persistApply) {
      const appliedAtMs = Date.now();
      formData.append('persist_apply', '1');
      formData.append('operation_token', 'op_' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2, 10));
      formData.append('client_applied_at_ms', String(appliedAtMs));
      formData.append('action_name', actionLabel || 'Apply cleaning action');
      formData.append('action_scope', window._ido.currentHistoryScope || scopeName || 'all');
    }
    const previousPreviewState = capturePreviewState();
    // ── Capture the active panel scope NOW before any await — this is the panel the user clicked Apply on ──
    const _scopeAtApply = window._ido.currentHistoryScope || 'dataIntegrityGroup';
    if (livePreviewBtn) livePreviewBtn.disabled = true;
    if (livePreviewBtn) livePreviewBtn.textContent = 'Updating...';
    try {
      const response = await fetch('/preview_clean', { method: 'POST', body: formData });
      const data = await response.json();
      if (!data.success) {
        if (data.error?.code === 'STALE_DATASET_VERSION' && window._ido.refreshCurrentProjectFromServer) {
          await window._ido.refreshCurrentProjectFromServer(data.error?.details?.current_version);
        }
        showPreviewError(data.error?.message || data.error || 'Preview failed', data.session_expired);
        return;
      }
      if (persistApply && window._ido.refreshAuthoritativeProjectState) {
        if (!window._ido.refreshAuthoritativeProjectState(data)) return;
      }
      if (window._ido.operationProgress) window._ido.operationProgress.stage('Rendering authoritative server result');
      // Snapshot the ledger before persisting this apply's controls into it.
      // If this apply turns out to produce nothing loggable (see the
      // actionOps.length check below), we roll back to this snapshot —
      // otherwise a toggle that was on but did nothing visible (e.g.
      // trim_whitespace on an already-clean column) could get permanently
      // baked into the ledger with zero trace in the Change Log, only to
      // resurface later when the ledger is replayed for the final clean.
      const _ledgerBeforeThisApply = window._ido.appliedControlEntries.slice();
      let _ownLedgerEntries = [];
      if (persistApply && scopeRoot) {
        persistScopedControls(scopeRoot);
        _ownLedgerEntries = _diffLedger(_ledgerBeforeThisApply, window._ido.appliedControlEntries);
        // Section-level applies (Apply button on a whole .opt-group) sweep
        // every field currently set in that section, not just the one the
        // user meant to change. If an unrelated toggle in the same section
        // was left checked from earlier browsing, it would otherwise get
        // silently folded into the ledger on the next, unrelated apply
        // click. Wiping the section's live controls back to defaults right
        // after a successful apply means the only way something ends up in
        // a future apply from this section is if the user deliberately set
        // it up again beforehand. (Feature-level applies are already
        // narrowly scoped to a single opt-item, so this isn't needed there.)
        if (scopeRoot.classList && (
          scopeRoot.classList.contains('opt-group') || scopeRoot.classList.contains('opt-item')
        )) {
          resetSectionControls(scopeRoot);
        }
      }
      // Exactly what this apply added/changed in the ledger — stored as this
      // action's own_entries so Undo/Reapply can precisely add/remove just
      // these keys later (see _diffLedger), instead of leaving the ledger
      // out of sync with the dataset snapshot after an Undo.
      // Outlier checkboxes are never rebuilt except by an explicit "Preview
      // Outliers" click — so a row left checked from a previous Apply would
      // silently ride along into the next Apply's selected_outlier_rows,
      // making it look like an already-removed row was being "reapplied" and
      // inflating the next action's row count. Clear the selection state
      // every time an outlier apply succeeds, and tell the user to re-preview
      // before picking more rows.
      if (persistApply && _scopeAtApply === 'outlierGroup') {
        document.querySelectorAll('.outlier-checkbox:checked').forEach(cb => { cb.checked = false; });
        if (selectAllOutliers) selectAllOutliers.checked = false;
        // Clearing the checkboxes isn't enough — the table of rows itself
        // was still visible after Apply, so it still looked like those rows
        // needed a decision. Hide it outright; the user must click "Preview
        // Outliers" again to see the current (post-apply) outlier list.
        if (outlierPreviewSection) outlierPreviewSection.style.display = 'none';
        if (outlierStatus) {
          outlierStatus.textContent = 'Applied. Click "Preview Outliers" again to refresh the list before selecting more rows.';
        }
      }
      const responseOperations =
        (data.report && Array.isArray(data.report.operations) && data.report.operations.length ? data.report.operations : null) ||
        (Array.isArray(data.operations) && data.operations.length ? data.operations : null) ||
        (data.report && data.report.manifest && Array.isArray(data.report.manifest.operations) ? data.report.manifest.operations : []);
      const displayOperations = operationsForLatestApply(responseOperations, persistApply);
      // Progress view: only changed rows, highlighted
      const changedRows = data.changed_preview || data.preview;
      const changedNums = (data.report && data.report.changed_row_numbers) || (data.report && data.report.preview_row_numbers);
      renderLivePreview(
        changedRows,
        data.report && data.report.cleaned_columns,
        responseOperations,
        changedNums
      );
      // Highlight changed rows in progress table
      if (data.report && data.report.changed_row_numbers) {
        highlightChangedRows(data.report.changed_row_numbers, data.report.removed_row_numbers || []);
      }
      applyWorkingColumns(data.report && data.report.cleaned_columns);
      if (persistApply && window._ido && window._ido.refreshEditorCanvas) {
        window._ido.refreshEditorCanvas(
          data.editor_columns || (data.report && data.report.cleaned_columns),
          data.editor_data || [],
          data.row_index_map,
          data.dataset_version
        );
      }
      if (persistApply) {
        syncDatasetEditorFromPreview(
          data.preview,
          data.report && data.report.preview_row_numbers,
          data.report && data.report.cleaned_columns
        );
      }
      renderLiveMetrics(data.summary);
      // Keep the Data Types table and Outlier column list in sync with
      // whatever the dataset actually looks like right now (see updateUI).
      window._ido.lastAnalysis = data.analysis || window._ido.lastAnalysis;
      if (data.analysis) updateUI(data.analysis);
      syncMissingEditorFromPreview(
        data.preview,
        data.report && data.report.preview_row_numbers,
        data.report && data.report.cleaned_columns,
        data.summary,
        data.analysis
      );
      updateMissingReviewState(data.summary);
      if (data.summary) window._ido.lastSummary = data.summary;
      renderLiveLog(displayOperations, actionLabel);
      let scopedDisplayOps = displayOperations;
      // Before/after step preview — capture before-state from editor grid
      if (persistApply) {
        const edGrid = window._ido && window._ido.editorGrid;
        const beforeRowData   = previousPreviewState && previousPreviewState.editorData   ? previousPreviewState.editorData   : (edGrid ? edGrid.data.slice() : []);
        const beforeColKeys   = previousPreviewState && previousPreviewState.editorColumns ? previousPreviewState.editorColumns : (edGrid ? edGrid.columns.map(c => c.key) : []);
        // Build row numbers 1..N for before data
        const beforeRowNums   = beforeRowData.map((_, i) => i + 1);

        // Filter displayOperations to only those that belong to the current apply scope.
        // This prevents ops from previous applies (which the backend carries forward cumulatively)
        // from being mis-filed into the wrong feature block.
        const allowedForScope = window._ido.SCOPE_OP_MAP[_scopeAtApply];
        scopedDisplayOps = allowedForScope
          ? displayOperations.filter(op => allowedForScope.has(op.action))
          : displayOperations.filter(op => op.action !== 'manual_outlier_removal');
        // Use scoped ops for the step preview (so only the right rows appear in the right block)
        const opsForPreview = scopedDisplayOps.length ? scopedDisplayOps : displayOperations.slice(-1);
        renderStepPreview(
          opsForPreview,
          data.changed_preview || data.preview,
          (data.report && data.report.changed_row_numbers) || [],
          (data.report && data.report.removed_row_numbers) || [],
          (data.report && data.report.cleaned_columns) || (data.editor_columns) || beforeColKeys,
          beforeRowData,
          beforeRowNums
        );
      }
      previewUndoStack.push(previousPreviewState);
      showDatasetSlide('progressDatasetSlide');
      const progressSlide = document.getElementById('progressDatasetSlide');
      if (progressSlide) progressSlide.scrollIntoView({ behavior: 'smooth', block: 'start' });
      lastSnapshot = snapshotForm(cleanForm);
      cleanForm.querySelectorAll('[data-undo-pending="1"]').forEach(el => delete el.dataset.undoPending);
      // Log to DB if this was a real Apply (not just a preview)
      if (persistApply) {
        if (data.no_op) {
          window._ido.appliedControlEntries = _ledgerBeforeThisApply;
          const cleanLabel = actionLabel.replace(/^Applied( section only)?:\s*/i, '');
          const noChangeMsg = `${cleanLabel}: ${data.no_op_message || 'no changes needed.'}`;
          if (liveActionLog) {
            liveActionLog.innerHTML = `<div class="live-log-empty">${escapeHtml(noChangeMsg)}</div>`;
          }
          showToast(noChangeMsg, 'success');
          if (window._ido.operationProgress) {
            window._ido.operationProgress.finish('success', 'No changes needed');
            window._ido.operationProgress.hideLater();
          }
          return;
        }
        const serverHistories = Array.isArray(data.history_entries) && data.history_entries.length
          ? data.history_entries : (data.history_entry ? [data.history_entry] : []);
        const serverHistory = serverHistories[0] || null;
        const serverConfig = serverHistory && serverHistory.action_config ? serverHistory.action_config : null;
        const actionOps = serverHistories.length
          ? serverHistories.flatMap(entry => (entry.action_config?.operations || []).filter(hasMeaningfulChange))
          : scopedDisplayOps;
        if (!actionOps.length) {
          // Nothing worth showing in the Change Log — so nothing should have
          // been permanently committed to the ledger either. persistScopedControls
          // ran unconditionally above (before we knew this); undo that here so
          // the ledger can't silently diverge from what the Change Log shows.
          window._ido.appliedControlEntries = _ledgerBeforeThisApply;
          // liveActionLog is display:none in the markup — writing here alone
          // is invisible, which is why "Apply" on a section with nothing to
          // do looked like it silently did nothing. Surface it via the
          // existing toast system instead. 'success' because "nothing needed
          // changing" is a good outcome, not a warning.
          const cleanLabel = actionLabel.replace(/^Applied( section only)?:\s*/i, '');
          const noChangeMsg = `${cleanLabel}: no changes needed — dataset already clean for this section.`;
          if (liveActionLog) {
            liveActionLog.innerHTML = `<div class="live-log-empty">${escapeHtml(noChangeMsg)}</div>`;
          }
          showToast(noChangeMsg, 'success');
          if (window._ido.operationProgress) {
            window._ido.operationProgress.finish('success', 'No changes needed');
            window._ido.operationProgress.hideLater();
          }
          return;
        }
        // Mark exactly the ops we're about to log as "seen" — scoped to what
        // actually got applied here, so a later apply of a DIFFERENT section
        // isn't wrongly suppressed, and a later re-apply of THIS section only
        // logs again if its content actually changed.
        _markOperationsLogged(actionOps);
        let createdActions = [];
        if (window._ido && window._ido.pushAppliedHistory) {
          serverHistories.forEach(entry => {
            const config = entry.action_config || {};
            const operations = (config.operations || []).filter(hasMeaningfulChange);
            const added = window._ido.pushAppliedHistory(
              config.label || actionLabel,
              previousPreviewState,
              capturePreviewState(),
              operations,
              config.activePanelId || _scopeAtApply,
              {
                ownEntries: _ownLedgerEntries,
                preview: config.preview,
                ts: Number(config.applied_at_ms) || undefined,
                dbHistoryId: entry.id,
                hasManifest: !!entry.has_manifest,
              }
            ) || [];
            added.forEach(action => {
              action.dbHistoryId = entry.id;
              action.hasManifest = !!entry.has_manifest;
            });
            createdActions.push(...added);
          });
        }
        if (!serverHistories.length || serverHistories.some(entry => !entry.id)) {
          showChangeLogSyncWarning();
          throw new Error('The server applied the dataset change but did not return a persisted history entry.');
        }
      }
      if (window._ido.operationProgress) {
        window._ido.operationProgress.finish('success', persistApply ? 'Applied' : 'Preview completed');
        window._ido.operationProgress.hideLater();
      }
    } catch (error) {
      showPreviewError(error.message || 'Unknown error', false);
    } finally {
      _applyInProgress = false;
      setApplyBusy(false);
      if (livePreviewBtn) livePreviewBtn.disabled = false;
      if (livePreviewBtn) livePreviewBtn.textContent = 'Preview Current Section';
      if (undoPreviewBtn) undoPreviewBtn.disabled = previewUndoStack.length === 0 && undoStack.length === 0;
    }
  }

  function rememberUndoState(event) {
    if (!cleanForm || !event.target.name) return;
    if (event.target.type === 'hidden' || event.target.type === 'submit' || event.target.type === 'button') return;
    if (!event.target.dataset.undoPending) {
      undoStack.push(lastSnapshot);
      event.target.dataset.undoPending = '1';
      if (undoPreviewBtn) undoPreviewBtn.disabled = false;
    }
    if (liveActionLog) {
      liveActionLog.innerHTML = '<div class="live-log-empty">Pending changes. Click an Apply button to update the live spreadsheet.</div>';
    }
  }

  if (cleanForm) {
    cleanForm.addEventListener('change', rememberUndoState);
    cleanForm.addEventListener('input', rememberUndoState);
  }
  if (cleanForm && livePreviewBtn) {
    livePreviewBtn.addEventListener('click', previewCurrentSection);
  }

  if (applyTypedPreviewBtn) {
    applyTypedPreviewBtn.addEventListener('click', () => {
      const editor = document.querySelector('.wide-missing-editor');
      const typedValues = editor
        ? Array.from(editor.querySelectorAll('input[name^="missing_cell["]')).filter(isUserTypedMissingInput)
        : [];
      if (typedValues.length === 0) {
        if (liveActionLog) {
          liveActionLog.innerHTML = '<div class="live-log-empty">No typed missing-cell values to apply yet. Type into a highlighted blank cell first.</div>';
        }
        showDatasetSlide('progressDatasetSlide');
        return;
      }
      // Typed missing values always belong to the Missing Values block
      currentHistoryScope = 'missingValuesGroup';
      window._ido.currentHistoryScope = 'missingValuesGroup';
      previewCleaning('Applied: typed missing-cell values', editor, 'manual', true);
    });
  }

  if (applyDatasetEditsBtn) {
    applyDatasetEditsBtn.addEventListener('click', () => {
      const grid = window._ido.editorGrid;
      if (grid && typeof grid.commitActiveInput === 'function') {
        grid.commitActiveInput();
      }
      const dirtyChanges = grid ? grid.getChanges() : [];
      const editedCells = datasetEditInputs.filter(input => input.value !== input.dataset.original);
      const renamedColumns = [];

      if (dirtyChanges.length === 0 && editedCells.length === 0 && renamedColumns.length === 0) {
        if (liveActionLog) {
          liveActionLog.innerHTML = '<div class="live-log-empty">No dataset edits to apply yet. Double-click a cell in the Dataset Editor to edit it.</div>';
        }
        showDatasetSlide('progressDatasetSlide');
        return;
      }

      const filename = cleanForm ? cleanForm.querySelector('[name="filename"]')?.value : null;
      const fd = new FormData();
      if (filename) fd.append('filename', filename);
      fd.append('dataset_version', String(window._ido.datasetVersion || 0));
      fd.append('preview_scope', 'manual');
      fd.append('persist_apply', '1');
      fd.append('operation_token', 'op_' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2, 10));
      fd.append('client_applied_at_ms', String(Date.now()));
      fd.append('action_name', 'Dataset edits');
      fd.append('action_scope', 'datasetEditorSlide');
      // Read directly from the grid's own dirty-cell tracking at click time —
      // the simplest, self-contained source of truth for "what's pending
      // right now", instead of diffing a separately-tracked cross-closure
      // ledger baseline (a repeated source of bugs in this feature).
      dirtyChanges.forEach(({ rowIdx, colKey, value }) => {
        const rowIndexMapRef = window._ido.rowIndexMap || {};
        const dfIdx = rowIndexMapRef[String(rowIdx)];
        fd.append(`cell_update[${dfIdx !== undefined ? dfIdx : rowIdx}][${colKey}]`, value);
      });
      renamedColumns.forEach(input => { if (input.name) fd.append(input.name, input.value); });

      const previousPreviewState = capturePreviewState();
      if (liveActionLog) {
        liveActionLog.innerHTML = '<div class="live-log-empty">Applying dataset edits...</div>';
      }
      applyDatasetEditsBtn.disabled = true;

      fetch('/preview_clean', { method: 'POST', body: fd })
        .then(r => r.json())
        .then(data => {
          applyDatasetEditsBtn.disabled = false;
          if (!data.success) { showPreviewError(data.error || 'Apply failed', data.session_expired); return; }
          if (window._ido.refreshAuthoritativeProjectState &&
              !window._ido.refreshAuthoritativeProjectState(data)) return;
          const changedRows = data.changed_preview || data.preview;
          const changedNums = (data.report && data.report.changed_row_numbers) || (data.report && data.report.preview_row_numbers);
          renderLivePreview(changedRows, data.report && data.report.cleaned_columns, data.report && data.report.operations, changedNums);
          if (data.report && data.report.changed_row_numbers) {
            highlightChangedRows(data.report.changed_row_numbers, data.report.removed_row_numbers || []);
          }
          applyWorkingColumns(data.report && data.report.cleaned_columns);
          // Resets the canvas grid's own dirty-tracking baseline to this
          // apply's result. Without this call, grid.getChanges() would keep
          // reporting these same edits as "still pending" forever.
          if (window._ido && window._ido.refreshEditorCanvas) {
            window._ido.refreshEditorCanvas(
              data.editor_columns || (data.report && data.report.cleaned_columns),
              data.editor_data || [],
              data.row_index_map,
              data.dataset_version
            );
          }
          renderLiveMetrics(data.summary);
          syncMissingEditorFromPreview(data.preview, data.report && data.report.preview_row_numbers, data.report && data.report.cleaned_columns, data.summary, data.analysis);
          updateMissingReviewState(data.summary);
          if (data.summary) window._ido.lastSummary = data.summary;
          renderLiveLog(data.report.operations, 'Applied: edited dataset cells');
          previewUndoStack.push(previousPreviewState);
          if (undoPreviewBtn) undoPreviewBtn.disabled = false;
          showDatasetSlide('progressDatasetSlide');
          const progressSlide = document.getElementById('progressDatasetSlide');
          if (progressSlide) progressSlide.scrollIntoView({ behavior: 'smooth', block: 'start' });
          // Restrict what gets logged here to what Dataset Editor actually
          // owns — same SCOPE_OP_MAP used everywhere else — so this action's
          // label or before/after preview can never pick up another panel's
          // operation.
          const datasetEditorOps = window._ido.SCOPE_OP_MAP && window._ido.SCOPE_OP_MAP.datasetEditorSlide;
          let editOps = visibleOperations(data.report && data.report.operations);
          if (datasetEditorOps) editOps = editOps.filter(op => datasetEditorOps.has(op.action));
          const serverHistory = data.history_entry || null;
          const serverConfig = serverHistory && serverHistory.action_config ? serverHistory.action_config : null;
          if (serverConfig && Array.isArray(serverConfig.operations)) {
            editOps = serverConfig.operations.filter(op => datasetEditorOps ? datasetEditorOps.has(op.action) : true);
          }
          if (editOps.length && serverHistory && serverHistory.id) {
            let createdActions = [];
            if (window._ido && window._ido.pushAppliedHistory) {
              createdActions = window._ido.pushAppliedHistory('Dataset edits', previousPreviewState, capturePreviewState(), editOps, 'datasetEditorSlide', {
                preview: serverConfig && serverConfig.preview,
                ts: Number(serverConfig && serverConfig.applied_at_ms) || undefined,
                dbHistoryId: serverHistory.id,
                hasManifest: !!(serverHistory && serverHistory.has_manifest),
              }) || [];
            }
            createdActions.forEach(a => {
              a.dbHistoryId = serverHistory.id;
              a.hasManifest = !!serverHistory.has_manifest;
            });
          }
        })
        .catch(err => { applyDatasetEditsBtn.disabled = false; showPreviewError(err.message || 'Apply failed', false); });
    });
  }

  window.indataoutPrepareAppliedControls = function () {
    if (!cleanForm) return;

    cleanForm.querySelectorAll('[data-final-disabled="1"]').forEach(el => {
      el.disabled = false;
      delete el.dataset.finalDisabled;
    });
    cleanForm.querySelectorAll('[data-applied-control="1"]').forEach(el => el.remove());

    // Belt-and-braces: disable every live control except filename, so that
    // even if something goes wrong parsing applied_entries server-side, a
    // legacy request.form read still can't pick up unapplied toggle states.
    cleanForm.querySelectorAll('[name]').forEach(el => {
      if (['filename', 'csrf_token', 'dataset_version', 'project_id', 'version_id', 'cleaned_filename', 'report_filename'].includes(el.name)) return;
      el.disabled = true;
      el.dataset.finalDisabled = '1';
    });

    // The applied-history ledger is the single source of truth for the
    // final clean — send it as one JSON field (same shape /resume_replay
    // already accepts and replays), rather than reconstructing individual
    // hidden inputs whose names could collide with disabled-but-still-
    // present original fields.
    // Final Clean reads the current server snapshot; no browser ledger is sent.
  };

  sectionApplyBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const group = btn.closest('.opt-group');
      const readiness = sectionApplyReadiness(group);
      if (!readiness.ok) {
        showApplyBlocked(readiness.message);
        return;
      }
      // Update scope to this panel BEFORE the async apply so it's captured correctly
      if (group && group.id) {
        currentHistoryScope = group.id;
        window._ido.currentHistoryScope = group.id;
      }
      if (liveActionLog) {
        liveActionLog.innerHTML = `<div class="live-log-empty">Applying ${escapeHtml(btn.dataset.sectionName || 'section')}...</div>`;
      }
      previewCleaning(`Applied section only: ${btn.dataset.sectionName || 'section'}`, group, 'section', true, btn);
    });
  });

  // ── "Reset section" — clears a section's toggles/selects/inputs back to
  // page-load defaults, so re-applying with a different set of options
  // doesn't require manually unchecking everything left on from a previous
  // apply. This does not touch anything already applied to the dataset —
  // it only resets the *form controls* for what to apply next.
  function resetSectionControls(section) {
    if (!section) return;
    section.querySelectorAll('.toggle input[type="checkbox"]').forEach(cb => { cb.checked = false; });
    section.querySelectorAll('select').forEach(sel => { sel.selectedIndex = 0; });
    section.querySelectorAll('input[type="text"], input[type="number"]').forEach(inp => { inp.value = ''; });
    // Re-fire change on every reset control so any existing show/hide
    // logic tied to these controls (e.g. the missing-values fill
    // sub-options toggling with missing_strategy) updates immediately.
    section.querySelectorAll('select, input[type="checkbox"]').forEach(el => {
      el.dispatchEvent(new Event('change', { bubbles: true }));
    });
  }
  window._ido.resetSectionControls = resetSectionControls;

  document.querySelectorAll('.section-reset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const section = document.getElementById(btn.dataset.section);
      if (!section) return;
      resetSectionControls(section);
      const sectionLabel = btn.dataset.sectionLabel || btn.dataset.section;
      if (liveActionLog) {
        liveActionLog.innerHTML = `<div class="live-log-empty">${escapeHtml(sectionLabel)} controls reset to defaults. Nothing already applied to the dataset was changed.</div>`;
      }
      showToast(`${sectionLabel} reset to defaults.`, 'success');
    });
  });

  function applyFeatureButton(btn) {
    const item = btn.closest('.opt-item');
    const group = item && item.closest('.opt-group');
    const featureName = item ? (item.querySelector('.opt-name')?.textContent || 'feature') : 'feature';
    const readiness = featureApplyReadiness(item);
    if (!readiness.ok) {
      showApplyBlocked(readiness.message);
      return;
    }
    // Update scope to the parent panel BEFORE the async apply
    if (group && group.id) {
      currentHistoryScope = group.id;
      window._ido.currentHistoryScope = group.id;
    }
    if (liveActionLog) {
      liveActionLog.innerHTML = `<div class="live-log-empty">Applying ${escapeHtml(featureName)}...</div>`;
    }
    previewCleaning(`Applied feature only: ${featureName}`, item, 'feature', true, btn);
  }

  document.addEventListener('click', event => {
    const btn = event.target && event.target.closest && event.target.closest('.feature-apply-btn');
    if (!btn) return;
    event.preventDefault();
    event.stopPropagation();
    applyFeatureButton(btn);
  }, true);
  window._ido.applyFeatureButton = applyFeatureButton;

  if (cleanForm && undoPreviewBtn) {
    undoPreviewBtn.addEventListener('click', () => {
      if (previewUndoStack.length > 0) {
        restorePreviewState(previewUndoStack.pop());
        lastSnapshot = snapshotForm(cleanForm);
        undoPreviewBtn.disabled = previewUndoStack.length === 0 && undoStack.length === 0;
        // Notify the Applied History system so it stays in sync
        window.dispatchEvent(new CustomEvent('indataout:undoFromProgressBtn'));
        return;
      }
      const snapshot = undoStack.pop();
      if (!snapshot) return;
      restoreSnapshot(cleanForm, snapshot);
      lastSnapshot = snapshotForm(cleanForm);
      undoPreviewBtn.disabled = previewUndoStack.length === 0 && undoStack.length === 0;
      window.dispatchEvent(new CustomEvent('indataout:undoFromProgressBtn'));
      previewCleaning('Undo applied: restored the previous option state');
    });
  }

  // ── Populate outlier column selector from server-provided numeric columns ──
  const outlierColumnSelector = document.getElementById('outlierColumnSelector');
  const outlierColumnList     = document.getElementById('outlierColumnList');
  const selectAllOutlierCols  = document.getElementById('selectAllOutlierCols');

  // Read numeric columns from the hidden server-data element (avoids Jinja-in-script linter errors)
  const _serverDataEl = document.getElementById('indataout-server-data');
  const _numericCols = _serverDataEl
    ? JSON.parse(_serverDataEl.dataset.numericColumns || '[]')
    : [];

  if (outlierColumnList && _numericCols.length > 0) {
    outlierColumnSelector.style.display = 'block';
    _numericCols.forEach(col => {
      const label = document.createElement('label');
      label.style.cssText = 'display:inline-flex;align-items:center;gap:4px;font-size:0.82rem;cursor:pointer;white-space:nowrap;';
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.value = col;
      cb.className = 'outlier-col-checkbox';
      cb.checked = true; // all selected by default
      label.appendChild(cb);
      label.appendChild(document.createTextNode(col));
      outlierColumnList.appendChild(label);
    });
    if (selectAllOutlierCols) {
      selectAllOutlierCols.checked = true;
      selectAllOutlierCols.addEventListener('change', function() {
        document.querySelectorAll('.outlier-col-checkbox').forEach(cb => { cb.checked = this.checked; });
      });
      // Keep select-all in sync when individual boxes are toggled
      outlierColumnList.addEventListener('change', () => {
        const all = document.querySelectorAll('.outlier-col-checkbox');
        const checked = document.querySelectorAll('.outlier-col-checkbox:checked');
        selectAllOutlierCols.indeterminate = checked.length > 0 && checked.length < all.length;
        selectAllOutlierCols.checked = checked.length === all.length;
      });
    }
  }

  if (previewOutliersBtn) {
    previewOutliersBtn.addEventListener('click', async () => {
      const filename = document.querySelector('input[name="filename"]').value;
      if (!filename) {
        alert('Please upload a file first');
        return;
      }

      try {
        previewOutliersBtn.disabled = true;
        previewOutliersBtn.textContent = 'Detecting outliers...';
        if (outlierCountInfo) outlierCountInfo.textContent = '';
        if (outlierStatus) outlierStatus.textContent = 'Scanning current working dataset...';

        const formData = mapToFormData(new Map(window._ido.appliedControlEntries), 'section');
        formData.delete('selected_outlier_rows');
        if (!formData.has('filename')) formData.append('filename', filename);
        formData.append('multiplier', '1.5');

        // Send only the user-selected columns (comma-separated)
        const selectedCols = Array.from(document.querySelectorAll('.outlier-col-checkbox:checked')).map(cb => cb.value);
        if (selectedCols.length > 0) {
          formData.append('outlier_columns', JSON.stringify(selectedCols));
        }

        const response = await fetch('/preview_outliers', {
          method: 'POST',
          body: formData
        });

        const data = await response.json();

        if (data.success && Number(data.dataset_version) === Number(window._ido.datasetVersion || 0)) {
          outlierPreviewSection.style.display = 'block';
          outlierTableBody.innerHTML = '';

          data.outliers.outlier_rows.forEach(row => {
            const outlierColumn = String(row.column);
            const context = Object.entries(row.row_data || {})
              .filter(([key]) => String(key) !== outlierColumn)
              .slice(0, 3)
              .map(([key, value]) => `${escapeHtml(key)}: ${escapeHtml(value)}`)
              .join(' | ');
            const displayValue = row.value === null || row.value === undefined ? 'N/A' : escapeHtml(row.value);
            const recordPayload = escapeHtml(JSON.stringify({
              row_index: row.row_index,
              row_number: row.row_number || (row.row_position + 1),
              column: row.column,
              value: row.value,
              row_data: row.row_data || {}
            }));
            const tr = document.createElement('tr');
            tr.innerHTML = `
              <td><input type="checkbox" class="outlier-checkbox" data-row-id="${escapeHtml(String(row.row_index))}" data-row="${escapeHtml(String(row.row_index))}" data-record="${recordPayload}"></td>
              <td>${row.row_number || (row.row_position + 1)}</td>
              <td>${escapeHtml(row.column)}</td>
              <td>${displayValue}</td>
              <td>${context || 'Exact row selected'}</td>
            `;
            outlierTableBody.appendChild(tr);
          });

          outlierCountInfo.textContent =
            `Showing ${data.outliers.outlier_rows.length} of ${data.outliers.total_count} outlier value(s) across ${data.outliers.columns_checked.length} column(s). Selecting multiple values from the same row removes that row once.`;
          if (outlierStatus) {
            outlierStatus.textContent = data.outliers.total_count > 0
              ? `${data.outliers.total_count} outlier candidate${data.outliers.total_count === 1 ? '' : 's'} found.`
              : 'No outliers found in the selected columns.';
          }

          if (selectAllOutliers) selectAllOutliers.checked = false;
          if (selectAllOutliers && !selectAllOutliers.dataset.bound) {
            selectAllOutliers.dataset.bound = '1';
            selectAllOutliers.addEventListener('change', function() {
              document.querySelectorAll('.outlier-checkbox').forEach(cb => { cb.checked = this.checked; });
            });
          }

        } else {
          if (outlierStatus) outlierStatus.textContent = 'Outlier preview failed.';
          alert('Error detecting outliers: ' + data.error);
        }
      } catch (error) {
        console.error('Error:', error);
        if (outlierStatus) outlierStatus.textContent = 'Outlier preview failed.';
        alert('Error detecting outliers');
      } finally {
        previewOutliersBtn.disabled = false;
        previewOutliersBtn.textContent = '🔍 Preview Outliers';
      }
    });
  }

  // ── Register everything closure 2 needs on window._ido ──────────────────
  // Closure 2 (history + notebook) is a separate DOMContentLoaded scope and
  // cannot see any let/const/function from this closure. window._ido is the
  // only reliable bridge between them.
  window._ido.mapToFormData                = mapToFormData;
  window._ido.addSelectedOutliersToFormData = addSelectedOutliersToFormData;
  window._ido.capturePreviewState           = capturePreviewState;
  window._ido.restorePreviewState           = restorePreviewState;
  window._ido.cleanForm                     = cleanForm;
  window._ido.snapshotForm                  = snapshotForm;
  window._ido.restoreSnapshot               = restoreSnapshot;
  window._ido.renderLivePreview             = renderLivePreview;
  window._ido.renderLiveMetrics             = renderLiveMetrics;
  window._ido.renderLiveLog                 = renderLiveLog;
  window._ido.syncDatasetEditorFromPreview  = syncDatasetEditorFromPreview;
  window._ido.syncMissingEditorFromPreview  = syncMissingEditorFromPreview;
  window._ido.updateMissingReviewState      = updateMissingReviewState;
  window._ido.applyWorkingColumns           = applyWorkingColumns;
  window._ido.showDatasetSlide              = showDatasetSlide;
  window._ido.showCleaningPanel             = showCleaningPanel;
  window._ido.previewUndoStack              = previewUndoStack;
  window._ido.undoPreviewBtn                = undoPreviewBtn;
  window._ido.triggerPreviewCleaning        = previewCleaning;
  // pushAppliedHistory is set by closure 2 once it initialises — placeholder until then
  window._ido.pushAppliedHistory            = window._ido.pushAppliedHistory || null;

});

// ─────────────────────────────────────────────────────────────────────────────
// SECTION: Applied / Undo History Log + Notebook Cell Workspace
// BUG FIXES:
//   Bug 4 — Reapply works in any order (each step has its own snapAfter)
//   Bug 5 — Undo only affects the exact step clicked (no cascade)
//   Bug 8 — Every Apply/Undo/Reapply is completely isolated from others
// ─────────────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {

  // hasMeaningfulChange lives in the other DOMContentLoaded closure (above) —
  // bridged via window._ido since local functions aren't shared across the
  // two closures. Aliased locally so call sites below don't need the prefix.
  const hasMeaningfulChange = window._ido.hasMeaningfulChange;

  // ── helpers ──────────────────────────────────────────────────────────
  function _esc(s) {
    return String(s === null || s === undefined ? '' : s)
      .replace(/[&<>"'`=\/]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;','/':'&#x2F;','`':'&#x60;','=':'&#x3D;'}[c]));
  }

  // ── snap helpers (bridge to closure 1) ───────────────────────────────
  function _captureSnap() {
    return window._ido.capturePreviewState ? window._ido.capturePreviewState() : null;
  }
  function _restoreSnap(snap) {
    if (snap && window._ido.restorePreviewState) window._ido.restorePreviewState(snap);
  }

  // Refreshes the core dataset views (canvas grid, progress table, metrics,
  // Data Types/Outlier panels) from a /preview_clean-shaped server response.
  // Used by Undo/Reapply-via-manifest, which changes the dataset on the
  // server without going through the normal previewCleaning() apply flow —
  // this reuses the same render calls that flow does, minus the
  // action-logging side effects (the action card's own status is updated
  // separately, since this isn't a new action).
  function _refreshUIFromServerData(data) {
    if (!data) return false;
    const responseVersion = Number(data.dataset_version);
    if (Number.isFinite(responseVersion)) {
      if (responseVersion < Number(window._ido.datasetVersion || 0)) return false;
      window._ido.datasetVersion = responseVersion;
      document.body.dataset.datasetVersion = String(responseVersion);
      document.querySelectorAll('[name="dataset_version"]').forEach(input => {
        input.value = String(responseVersion);
      });
    }
    if (data.patch && window._ido && window._ido.patchEditorCanvas) {
      // Fast path: Undo/Reapply sent a manifest-derived patch instead of the
      // full dataset — apply it in place rather than re-rendering the grid.
      window._ido.patchEditorCanvas(data.patch, data.row_index_map);
    } else if (window._ido && window._ido.refreshEditorCanvas) {
      // Full path: a real Apply, or the rare conflict fallback, which sends
      // the complete editor_data because a partial patch can't be trusted.
      window._ido.refreshEditorCanvas(
        data.editor_columns || (data.report && data.report.cleaned_columns),
        data.editor_data || [],
        data.row_index_map,
        responseVersion
      );
    }
    const changedRows = data.changed_preview || data.preview;
    const changedNums = (data.report && data.report.changed_row_numbers) || (data.report && data.report.preview_row_numbers);
    if (window._ido.renderLivePreview) {
      window._ido.renderLivePreview(changedRows, data.report && data.report.cleaned_columns, data.report && data.report.operations, changedNums);
    }
    if (data.report && data.report.changed_row_numbers && window._ido.highlightChangedRows) {
      window._ido.highlightChangedRows(data.report.changed_row_numbers, data.report.removed_row_numbers || []);
    }
    if (window._ido.applyWorkingColumns) window._ido.applyWorkingColumns(data.report && data.report.cleaned_columns);
    if (window._ido.syncDatasetEditorFromPreview) window._ido.syncDatasetEditorFromPreview(data.preview, data.report && data.report.preview_row_numbers, data.report && data.report.cleaned_columns, data.report && data.report.removed_row_numbers);
    if (window._ido.renderLiveMetrics) window._ido.renderLiveMetrics(data.summary);
    window._ido.lastAnalysis = data.analysis || window._ido.lastAnalysis;
    if (data.analysis && window._ido.updateUI) window._ido.updateUI(data.analysis);
    if (window._ido.syncMissingEditorFromPreview) window._ido.syncMissingEditorFromPreview(data.preview, data.report && data.report.preview_row_numbers, data.report && data.report.cleaned_columns, data.summary, data.analysis);
    if (window._ido.updateMissingReviewState) window._ido.updateMissingReviewState(data.summary);
    if (data.summary) window._ido.lastSummary = data.summary;
    return true;
  }
  window._ido.refreshAuthoritativeProjectState = _refreshUIFromServerData;
  window._ido.refreshCurrentProjectFromServer = async function(currentVersion) {
    if (Number.isFinite(Number(currentVersion))) window._ido.datasetVersion = Number(currentVersion);
    const response = await fetch('/resume_replay', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset_version: window._ido.datasetVersion || 0 }),
    });
    const data = await response.json();
    if (data.success) _refreshUIFromServerData(data);
    return data;
  };

  // Undo/Reapply changes status on the server, which rebuilds the current
  // dataset from the immutable source and active actions in commit order.
  // The browser updates the card only after that replay succeeds.
  /**
   * Ask Flask to Undo/Reapply by ordered replay and refresh from the response.
   *
   * The card status changes only after this call succeeds; stale client
   * snapshots are deliberately not used as committed state.
   */
  async function _applyManifestOnServer(action, direction) {
    if (!action.dbHistoryId) return false;
    try {
      const resp = await fetch('/apply_manifest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          history_id: action.dbHistoryId,
          direction,
          dataset_version: window._ido.datasetVersion || 0,
        }),
      });
      const data = await resp.json();
      if (!data.ok) {
        // Surfacing the exact reason here (rather than silently falling back
        // to a no-op restore, which is invisible for resumed actions since
        // they have no in-memory snapBefore/snapAfter) is what actually lets
        // us tell apart no_manifest / no_snapshot / apply_failed / save_failed
        // instead of guessing.
        console.warn('apply_manifest returned not-ok:', data.reason);
        if (data.error?.code === 'STALE_DATASET_VERSION') {
          await window._ido.refreshCurrentProjectFromServer(data.error?.details?.current_version);
        }
        if (window._ido.showToast) {
          window._ido.showToast(`Undo/Reapply couldn't reach the server state (${data.reason || 'unknown reason'}) — no changes were made.`, 'error');
        }
        return false;  // no_manifest / no_snapshot / etc. — fall back
      }
      try {
        _refreshUIFromServerData(data);
      } catch (refreshError) {
        console.warn('server state changed but UI refresh hit a recoverable error:', refreshError);
      }
      if (data.conflicts && data.conflicts.length && window._ido.showToast) {
        window._ido.showToast('Some earlier or later changes overlapped — applied what was still consistent.', 'warning');
      }
      return true;
    } catch (e) {
      console.warn('apply_manifest request failed:', e);
      if (window._ido.showToast) {
        window._ido.showToast('Undo/Reapply could not reach the server. Nothing was changed.', 'error');
      }
      return false;
    }
  }

  // ══════════════════════════════════════════════════════════════════════════
  // CHANGE LOG SYSTEM — One collapsible section per feature, actions append
  // ══════════════════════════════════════════════════════════════════════════
  //
  // Structure:
  //   _featureBlocks = { scope → { el, actionsEl, actions: [] } }
  //   Each scope = one collapsible section (Missing Values, Outliers, etc.)
  //   Each committed operation appends a card inside its matching section.
  //   Actions are NEVER replaced — only appended.
  //
  // Action card contains:
  //   - Action #N header with time, method, column, changed count
  //   - Before / After table side by side
  //   - Undo   Reapply   View Full Changes buttons

  const SCOPE_META = {
    missingValuesGroup: { label: 'Missing Values',     icon: '🔢' },
    outlierGroup:       { label: 'Outliers',           icon: '📊' },
    dataIntegrityGroup: { label: 'Data Integrity',     icon: '🔧' },
    dataTypesGroup:     { label: 'Data Types',         icon: '🔄' },
    datasetEditorSlide: { label: 'Dataset Cell Edits', icon: '✏️' },
    dropSelectedColumnsGroup: { label: 'Drop Selected Columns', icon: '🧹' },
    customCodeGroup:    { label: 'Custom Pandas Code', icon: '🐍' },
  };

  const CHANGE_LOG_SUBFEATURES = {
    dataIntegrityGroup: {
      drop_duplicates:       { key: 'removeDuplicateRows', label: 'Remove Duplicate Rows' },
      drop_empty_columns:    { key: 'dropEmptyColumns', label: 'Drop Fully Empty Columns' },
      drop_constant_columns: { key: 'dropConstantColumns', label: 'Drop Constant Columns' },
      trim_whitespace:       { key: 'trimWhitespace', label: 'Trim Whitespace' },
    },
  };

  const PERMANENT_SUBFEATURES = {
    dataIntegrityGroup: ['removeDuplicateRows','dropEmptyColumns','dropConstantColumns','trimWhitespace'],
  };

  function subfeatureMeta(scope, key) {
    return Object.values(CHANGE_LOG_SUBFEATURES[scope] || {}).find(item => item.key === key) || null;
  }

  function canonicalHistoryScope(scope) {
    return scope;
  }

  function scopeLabel(scope) {
    return (SCOPE_META[scope] || {}).label || 'Cleaning Action';
  }
  function scopeIcon(scope) {
    return (SCOPE_META[scope] || {}).icon || '⚙️';
  }

  // Same scope-priority order the original hardcoded version used, for the rare
  // case where a single operations array contains MORE than one action type and
  // there's no forcedScope/activePanelId to disambiguate (legacy resumed rows
  // only). Membership itself now comes from SCOPE_OP_MAP — one list, not two.
  const SCOPE_PRIORITY_ORDER = [
    'outlierGroup', 'missingValuesGroup', 'dataTypesGroup',
    'customCodeGroup', 'datasetEditorSlide', 'dropSelectedColumnsGroup', 'dataIntegrityGroup',
  ];

  function scopeForOperations(operations, fallbackLabel = '') {
    const SCOPE_OP_MAP = window._ido.SCOPE_OP_MAP;
    const actions = new Set((operations || []).map(op => op && op.action).filter(Boolean));
    for (const scope of SCOPE_PRIORITY_ORDER) {
      if ([...SCOPE_OP_MAP[scope]].some(a => actions.has(a))) return scope;
    }
    const lbl = String(fallbackLabel).toLowerCase();
    if (lbl.includes('outlier')) return 'outlierGroup';
    if (lbl.includes('missing') || lbl.includes('fill')) return 'missingValuesGroup';
    if (lbl.includes('type')) return 'dataTypesGroup';
    if (lbl.includes('drop selected') || lbl.includes('drop column')) return 'dropSelectedColumnsGroup';
    if (lbl.includes('edit') || lbl.includes('rename')) return 'datasetEditorSlide';
    return window._ido.currentHistoryScope || 'dataIntegrityGroup';
  }
  window._ido.scopeForOperations = scopeForOperations; // exposed for testing/verification only

  function buildSmartLabel(fallbackLabel, operations) {
    const visible = (operations||[]).filter(hasMeaningfulChange);
    if (!visible.length) return fallbackLabel;
    const parts = visible.slice(0,3).map(op => {
      if (op.action==='manual_missing_cell_updates') return `Filled ${op.count||0} missing cell${op.count!==1?'s':''}`;
      if (op.action==='manual_cell_updates')         return `Edited ${op.count||0} cell${op.count!==1?'s':''}`;
      if (op.action==='manual_column_renames')       return `Renamed ${op.count||0} column${op.count!==1?'s':''}`;
      if (op.action==='drop_empty_columns')          return op.count>0?`Dropped ${op.count} empty col${op.count!==1?'s':''}` : 'No empty columns found';
      if (op.action==='drop_constant_columns')       return op.count>0?`Dropped ${op.count} constant col${op.count!==1?'s':''}` : 'No constant columns found';
      if (op.action==='drop_duplicates')             return (op.removed_rows||0)>0?`Removed ${op.removed_rows} duplicate row${op.removed_rows!==1?'s':''}` : 'No duplicates found';
      if (op.action==='drop_missing_rows')           return `Dropped ${op.dropped_row_count||0} row${(op.dropped_row_count||0)!==1?'s':''} with missing`;
      if (op.action==='fill_numeric') {
        const dropped = op.rows_dropped||0, filled = op.filled_cells||0;
        if (dropped > 0 && filled > 0) return `Filled ${filled} numeric cell${filled!==1?'s':''} & dropped ${dropped} row${dropped!==1?'s':''}`;
        if (dropped > 0) return `Dropped ${dropped} row${dropped!==1?'s':''} (numeric missing)`;
        return `Filled ${filled} numeric cell${filled!==1?'s':''}`;
      }
      if (op.action==='fill_categorical')            return `Filled ${op.filled_cells||0} text cell${(op.filled_cells||0)!==1?'s':''}`;
      if (op.action==='manual_outlier_removal')      return `Removed ${op.rows_removed||0} outlier row${(op.rows_removed||0)!==1?'s':''}`;
      if (op.action==='convert_dtypes')              return `Auto-converted ${(op.changes||[]).length} type${(op.changes||[]).length!==1?'s':''}`;
      if (op.action==='manual_convert_dtypes')       return `Converted ${Object.keys(op.changes||{}).length} type${Object.keys(op.changes||{}).length!==1?'s':''}`;
      if (op.action==='trim_whitespace')             return (op.columns_processed||[]).length>0?`Trimmed ${(op.columns_processed||[]).length} col${(op.columns_processed||[]).length!==1?'s':''}` : 'No whitespace found';      return op.action.replace(/_/g,' ');
    }).filter(Boolean);
    if (!parts.length) return fallbackLabel;
    const label = parts.join(' · ');
    return visible.length > 3 ? label + ` (+${visible.length-3} more)` : label;
  }

  function _fmt(ts) {
    if (!ts) return '';
    return new Date(ts).toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
  }
  function _historyTs(entry, config) {
    const ms = Number(config && config.applied_at_ms);
    if (Number.isFinite(ms) && ms > 0) return ms;
    return entry && entry.applied_at ? new Date(entry.applied_at).getTime() : Date.now();
  }
  function _esc(s) {
    return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }

  // Container
  let changeLogBlocksContainer = document.getElementById('changeLogBlocks');
  if (!changeLogBlocksContainer) {
    const parent = document.querySelector('.live-log-section') || document.getElementById('liveActionLog');
    if (parent) {
      changeLogBlocksContainer = document.createElement('div');
      changeLogBlocksContainer.id = 'changeLogBlocks';
      parent.appendChild(changeLogBlocksContainer);
    }
  }
  // Hide old grid
  document.querySelectorAll('.history-log-grid').forEach(el => el.style.display = 'none');
  // Hide old Cleaning Timeline — blocks replace it
  const _oldTimeline = document.getElementById('stepPreviewLog');
  if (_oldTimeline) _oldTimeline.style.display = 'none';

  // _featureBlocks: scope → { el, actionsEl, actions[] }
  const _featureBlocks = {};
  let _actionSeq = 0;

  function _getOrCreateFeatureBlock(scope) {
    if (_featureBlocks[scope]) return _featureBlocks[scope];

    const label = scopeLabel(scope);
    const icon  = scopeIcon(scope);
    const fbId  = `fb-${scope}`;

    const el = document.createElement('div');
    el.className = 'cl-feature-block';
    el.id = fbId;
    el.innerHTML = `
      <div class="cl-feature-header" data-scope="${scope}">
        <span class="cl-feature-toggle">▼</span>
        <span class="cl-feature-icon">${icon}</span>
        <span class="cl-feature-label">${_esc(label)}</span>
        <span class="cl-feature-count" id="${fbId}-count">0 Actions</span>
      </div>
      <div class="cl-feature-body" id="${fbId}-body">
        <div class="cl-feature-empty">No actions yet</div>
      </div>`;

    // Toggle collapse
    el.querySelector('.cl-feature-header').addEventListener('click', () => {
      const body   = el.querySelector('.cl-feature-body');
      const toggle = el.querySelector('.cl-feature-toggle');
      const open   = body.style.display !== 'none';
      body.style.display  = open ? 'none' : '';
      toggle.textContent  = open ? '▶' : '▼';
    });

    const target = changeLogBlocksContainer || document.body;
    target.appendChild(el);

    // Hide the "No cleaning actions yet" empty state once the first block appears
    const emptyState = document.getElementById('clEmptyState');
    if (emptyState) emptyState.style.display = 'none';

    _featureBlocks[scope] = {
      el,
      actionsEl: el.querySelector(`#${fbId}-body`),
      countEl:   el.querySelector(`#${fbId}-count`),
      emptyEl:   el.querySelector('.cl-feature-empty'),
      actions:   [],
    };
    return _featureBlocks[scope];
  }

  function _getOrCreateSubfeatureBlock(fb, scope, subKey) {
    if (!subKey || !fb || !fb.actionsEl) return null;
    fb.subBlocks = fb.subBlocks || {};
    if (fb.subBlocks[subKey]) return fb.subBlocks[subKey];

    const meta = subfeatureMeta(scope, subKey) || { label: 'Cleaning Tool' };
    const subId = `${fb.el.id}-${subKey}`;
    const el = document.createElement('div');
    el.className = 'cl-subfeature-block';
    el.id = subId;
    el.innerHTML = `
      <div class="cl-subfeature-header">
        <span class="cl-subfeature-toggle">▼</span>
        <span class="cl-subfeature-label">${_esc(meta.label)}</span>
        <span class="cl-subfeature-count">0 Actions</span>
      </div>
      <div class="cl-subfeature-body">
        <div class="cl-subfeature-empty">No actions yet</div>
      </div>`;

    el.querySelector('.cl-subfeature-header').addEventListener('click', () => {
      const body = el.querySelector('.cl-subfeature-body');
      const toggle = el.querySelector('.cl-subfeature-toggle');
      const open = body.style.display !== 'none';
      body.style.display = open ? 'none' : '';
      toggle.textContent = open ? '▶' : '▼';
    });

    fb.actionsEl.appendChild(el);
    fb.subBlocks[subKey] = {
      el,
      actionsEl: el.querySelector('.cl-subfeature-body'),
      countEl: el.querySelector('.cl-subfeature-count'),
      emptyEl: el.querySelector('.cl-subfeature-empty'),
      actions: [],
    };
    return fb.subBlocks[subKey];
  }

  function _updateFeatureCount(fb) {
    if (fb.countEl) {
      const n = fb.actions.length;
      fb.countEl.textContent = `${n} Action${n !== 1 ? 's' : ''}`;
      if (fb.emptyEl) fb.emptyEl.style.display = n ? 'none' : '';
    }
  }

  function _updateSubFeatureCount(sub) {
    if (!sub || !sub.countEl) return;
    const n = sub.actions.length;
    sub.countEl.textContent = `${n} Action${n !== 1 ? 's' : ''}`;
    if (sub.emptyEl) sub.emptyEl.style.display = n ? 'none' : '';
  }

  // ── Build a per-operation Before/After table from raw operation data ──────
  // Helper: 2-col Before/After table — pink left, green right
  function _baTable(headerLeft, headerRight, pairs, extra) {
    if (!pairs || !pairs.length) return '';
    const ths = `<th style="width:50%">${_esc(headerLeft)}</th><th style="width:50%">${_esc(headerRight)}</th>`;
    const trs = pairs.map(([b, a]) =>
      `<tr><td class="cl-ba-before">${b}</td><td class="cl-ba-after">${a}</td></tr>`
    ).join('');
    const moreRow = extra
      ? `<tr><td colspan="2" style="padding:7px 16px;color:#9ca3af;font-size:12px;font-style:italic;">${_esc(extra)}</td></tr>`
      : '';
    return `<div class="cl-ba-wrap"><table class="cl-ba-table">
      <thead><tr>${ths}</tr></thead>
      <tbody>${trs}${moreRow}</tbody>
    </table></div>`;
  }

  function _buildOpBeforeAfterHtml(op) {
    if (!op || !op.action) return '';
    const act = op.action;

    // ── Fill numeric ──────────────────────────────────────────────────
    if (act === 'fill_numeric') {
      const details = (op.details || []).slice(0, 5);
      if (!details.length) return '';
      const pairs = details.map(d => {
        if (d.strategy === 'drop') {
          const dropped = d.dropped || 0;
          const label = ` <span style="color:#9ca3af;font-size:11px;">(${_esc(d.column||'')} · ${dropped} row${dropped!==1?'s':''})</span>`;
          return [`Missing${label}`, 'Removed'];
        }
        const val   = d.value !== undefined && d.value !== null ? String(d.value) : '—';
        const count = (d.rows || []).length;
        const label = count > 0 ? ` <span style="color:#9ca3af;font-size:11px;">(${_esc(d.column||'')} · ${count} cell${count!==1?'s':''})</span>` : ` <span style="color:#9ca3af;font-size:11px;">(${_esc(d.column||'')})</span>`;
        return [`NaN${label}`, _esc(val)];
      });
      const extra = (op.details||[]).length > 5 ? `+${(op.details||[]).length-5} more columns…` : '';
      return _baTable('Before', 'After', pairs, extra);
    }

    // ── Fill categorical ──────────────────────────────────────────────
    if (act === 'fill_categorical') {
      const entries = Object.entries(op.details || {}).slice(0, 5);
      if (!entries.length) return '';
      const pairs = entries.map(([col, d]) => {
        const val   = d.value !== undefined && d.value !== null ? String(d.value) : '—';
        const count = (d.rows || []).length;
        const label = ` <span style="color:#9ca3af;font-size:11px;">(${_esc(col)} · ${count} cell${count!==1?'s':''})</span>`;
        return [`NaN${label}`, _esc(val)];
      });
      const extra = Object.keys(op.details||{}).length > 5 ? `+${Object.keys(op.details||{}).length-5} more…` : '';
      return _baTable('Before', 'After', pairs, extra);
    }

    // ── Manual missing cell fills (typed values) ──────────────────────
    if (act === 'manual_missing_cell_updates') {
      const allChanges = op.changes || op.updates || [];
      const changes = allChanges.slice(0, 5);
      if (!changes.length) return op.count ? `<div class="cl-ba-info">${op.count} cell${op.count!==1?'s':''} filled.</div>` : '';
      const pairs = changes.map(c => [
        `NaN <span style="color:#9ca3af;font-size:11px;">(${_esc(c.column||'')})</span>`,
        _esc(String(c.value !== undefined ? c.value : '—'))
      ]);
      const extra = (op.changes||[]).length > 5 ? `+${(op.changes||[]).length-5} more…` : '';
      return _baTable('Before', 'After', pairs, extra);
    }

    // ── Drop missing rows ─────────────────────────────────────────────
    if (act === 'drop_missing_rows') {
      const rowNums = (op.row_numbers || []).slice(0, 5);
      const dropped = op.dropped_row_count || rowNums.length;
      const pairs = rowNums.length
        ? rowNums.map(r => [`Row ${_esc(String(r))}`, 'Removed'])
        : [[`${dropped} row${dropped!==1?'s':''} with missing values`, 'Removed']];
      const extra = (op.row_numbers||[]).length > 5 ? `+${(op.row_numbers||[]).length-5} more rows…` : '';
      return _baTable('Before', 'After', pairs, extra);
    }

    // ── Drop duplicates ───────────────────────────────────────────────
    if (act === 'drop_duplicates') {
      const removed = op.removed_rows || op.duplicate_count || 0;
      const droppedRows = op.dropped_rows_preview || [];
      if (droppedRows.length) {
        const cols = Object.keys(droppedRows[0].data || {}).slice(0, 8);
        const beforeRows = droppedRows.map(r => `<tr><td class="cl-row-label">Row ${_esc(String(Number(r.row) + 1 || r.row))}</td>${cols.map(c => `<td>${_esc(String((r.data || {})[c] ?? ''))}</td>`).join('')}</tr>`).join('');
        const afterRows = droppedRows.map(r => `<tr><td class="cl-row-label">Row ${_esc(String(Number(r.row) + 1 || r.row))}</td><td colspan="${Math.max(cols.length, 1)}">Removed</td></tr>`).join('');
        const head = `<thead><tr><th>Row</th>${cols.map(c => `<th>${_esc(c)}</th>`).join('')}</tr></thead>`;
        const extra = (op.dropped_rows_total || droppedRows.length) > droppedRows.length
          ? `<div class="cl-ba-info">Showing ${droppedRows.length} of ${op.dropped_rows_total} removed duplicate rows.</div>`
          : '';
        return `<div class="cl-ba-grid">
          <div><div class="cl-ba-title">Before</div><table class="cl-ba-table">${head}<tbody>${beforeRows}</tbody></table></div>
          <div><div class="cl-ba-title">After</div><table class="cl-ba-table"><tbody>${afterRows}</tbody></table></div>
        </div>${extra}`;
      }
      return _baTable('Before', 'After', [
        [`${removed} duplicate row${removed!==1?'s':''}`, '0 duplicates']
      ]);
    }

    // ── Manual outlier removal ────────────────────────────────────────
    if (act === 'manual_outlier_removal') {
      const vals = (op.removed_values || []).slice(0, 5);
      const pairs = vals.length
        ? vals.map(v => [_esc(String(v)), 'Removed'])
        : [[`${op.rows_removed||0} outlier row${(op.rows_removed||0)!==1?'s':''}`, 'Removed']];
      const extra = (op.removed_values||[]).length > 5 ? `+${(op.removed_values||[]).length-5} more…` : '';
      return _baTable('Before', 'After', pairs, extra);
    }

    // ── Auto dtype conversion ─────────────────────────────────────────
    if (act === 'convert_dtypes') {
      const changes = (op.changes || []).slice(0, 6);
      if (!changes.length) return '';
      const pairs = changes.map(c => [
        `<code class="cl-dtype">${_esc(c.original_dtype||c.from||'—')}</code> <span style="color:#9ca3af;font-size:11px;">${_esc(c.column||'')}</span>`,
        `<code class="cl-dtype cl-dtype-new">${_esc(c.new_dtype||c.to||'—')}</code>`
      ]);
      const extra = (op.changes||[]).length > 6 ? `+${(op.changes||[]).length-6} more…` : '';
      return _baTable('Before', 'After', pairs, extra);
    }

    // ── Manual dtype conversion ───────────────────────────────────────
    if (act === 'manual_convert_dtypes') {
      const entries = Object.entries(op.changes || {}).slice(0, 6);
      if (!entries.length) return '';
      const pairs = entries.map(([col, info]) => {
        const from = typeof info === 'object' ? (info.from || info.original_dtype || '—') : '—';
        const to   = typeof info === 'object' ? (info.to   || info.new_dtype      || '—') : String(info);
        return [
          `<code class="cl-dtype">${_esc(from)}</code> <span style="color:#9ca3af;font-size:11px;">${_esc(col)}</span>`,
          `<code class="cl-dtype cl-dtype-new">${_esc(to)}</code>`
        ];
      });
      const extra = Object.keys(op.changes||{}).length > 6 ? `+${Object.keys(op.changes||{}).length-6} more…` : '';
      return _baTable('Before', 'After', pairs, extra);
    }

    // ── Trim whitespace ───────────────────────────────────────────────
    if (act === 'trim_whitespace') {
      const cols = (op.columns_processed || []).slice(0, 5);
      if (!cols.length) return '';
      const pairs = cols.map(c => [
        `"value " <span style="color:#9ca3af;font-size:11px;">${_esc(c)}</span>`,
        '"value"'
      ]);
      const extra = (op.columns_processed||[]).length > 5 ? `+${(op.columns_processed||[]).length-5} more…` : '';
      return _baTable('Before', 'After', pairs, extra);
    }

    // ── Drop empty / constant / selected columns ──────────────────────
    if (['drop_empty_columns','drop_constant_columns','drop_selected_columns'].includes(act)) {
      const src     = op.columns_dropped || op.columns || [];
      const dropped = src.slice(0, 6);
      if (!dropped.length) return `<div class="cl-ba-info">No columns qualified — none removed.</div>`;
      const pairs = dropped.map(c => [_esc(c), 'Dropped']);
      const extra = src.length > 6 ? `+${src.length-6} more…` : '';
      return _baTable('Before', 'After', pairs, extra);
    }

    // ── Manual cell edits ─────────────────────────────────────────────
    if (act === 'manual_cell_updates') {
      const updates = (op.updates || []).slice(0, 5);
      if (!updates.length) return op.count ? `<div class="cl-ba-info">${op.count} cell${op.count!==1?'s':''} edited.</div>` : '';
      const pairs = updates.map(u => [
        `${_esc(u.old !== undefined && u.old !== '' ? u.old : '(blank)')} <span style="color:#9ca3af;font-size:11px;">row ${_esc(String(u.row_number||''))}, ${_esc(u.column||'')}</span>`,
        u.blank ? '<em style="color:#9ca3af">blank/missing</em>' : _esc(String(u.value !== undefined ? u.value : '—'))
      ]);
      const extra = (op.updates||[]).length > 5 ? `+${(op.updates||[]).length-5} more…` : '';
      return _baTable('Before', 'After', pairs, extra);
    }

    // ── Column renames ────────────────────────────────────────────────
    if (act === 'manual_column_renames') {
      const changes = (op.changes || []).slice(0, 6);
      if (!changes.length) return op.count ? `<div class="cl-ba-info">${op.count} column${op.count!==1?'s':''} renamed.</div>` : '';
      const pairs = changes.map(c => [_esc(c.from||c.old||'—'), _esc(c.to||c.new||'—')]);
      const extra = (op.changes||[]).length > 6 ? `+${(op.changes||[]).length-6} more…` : '';
      return _baTable('Before', 'After', pairs, extra);
    }

    // ── Custom pandas code ────────────────────────────────────────────
    if (act === 'custom_pandas_code') {
      const details = (op.changed_cell_details || []).slice(0, 5);
      if (details.length) {
        const pairs = details.map(d => [
          `${_esc(String(d.before!==undefined?d.before:'—'))} <span style="color:#9ca3af;font-size:11px;">${_esc(d.column||'')}</span>`,
          _esc(String(d.after!==undefined?d.after:'—'))
        ]);
        const extra = (op.changed_cell_details||[]).length > 5 ? `+${(op.changed_cell_details||[]).length-5} more…` : '';
        return _baTable('Before', 'After', pairs, extra);
      }
      if (op.code) {
        const firstLine = op.code.trim().split('\n')[0].slice(0, 80);
        return `<div class="cl-ba-info"><code>${_esc(firstLine)}</code></div>`;
      }
      return '';
    }

    return '';
  }


  // ── Build metadata row (Strategy / Affected / Changed cells) ────────────
  function _buildOpMetaHtml(op) {
    if (!op || !op.action) return '';
    const act = op.action;
    const parts = [];

    if (act === 'fill_numeric') {
      if (op.strategy) parts.push(['Strategy', op.strategy]);
      const cols = (op.details||[]).map(d=>d.column).slice(0,4);
      if (cols.length) parts.push(['Affected', cols.join(', ') + (cols.length < (op.details||[]).length ? '…' : '')]);
      if (op.filled_cells) parts.push(['Changed', `${op.filled_cells} cells`]);
    } else if (act === 'fill_categorical') {
      if (op.strategy) parts.push(['Strategy', op.strategy]);
      const cols = Object.keys(op.details||{}).slice(0,4);
      if (cols.length) parts.push(['Affected', cols.join(', ')]);
      if (op.filled_cells) parts.push(['Changed', `${op.filled_cells} cells`]);
    } else if (act === 'drop_duplicates') {
      parts.push(['Removed', `${op.removed_rows || 0} rows`]);
    } else if (act === 'drop_missing_rows') {
      parts.push(['Dropped', `${op.dropped_row_count || 0} rows`]);
    } else if (act === 'manual_outlier_removal') {
      parts.push(['Removed', `${op.rows_removed || 0} rows`]);
      if ((op.columns||[]).length) parts.push(['Columns', op.columns.slice(0,3).join(', ')]);
    } else if (act === 'convert_dtypes') {
      const n = (op.changes||[]).length;
      if (n) parts.push(['Converted', `${n} column${n!==1?'s':''}`]);
    } else if (act === 'manual_convert_dtypes') {
      const n = Object.keys(op.changes||{}).length;
      if (n) parts.push(['Converted', `${n} column${n!==1?'s':''}`]);
    } else if (act === 'trim_whitespace') {
      const n = (op.columns_processed||[]).length;
      if (n) parts.push(['Columns', `${n} trimmed`]);
    } else if (act === 'drop_empty_columns' || act === 'drop_constant_columns') {
      const n = (op.columns_dropped||[]).length;
      parts.push([act === 'drop_empty_columns' ? 'Empty cols' : 'Constant cols', `${n} dropped`]);
    } else if (act === 'drop_selected_columns') {
      const n = (op.columns||op.columns_dropped||[]).length;
      if (n) parts.push(['Dropped', `${n} column${n!==1?'s':''}`]);
    } else if (act === 'manual_cell_updates' || act === 'manual_missing_cell_updates') {
      if (op.count) parts.push(['Changed', `${op.count} cell${op.count!==1?'s':''}`]);
    } else if (act === 'manual_column_renames') {
      if (op.count) parts.push(['Renamed', `${op.count} column${op.count!==1?'s':''}`]);
    } else if (act === 'custom_pandas_code') {
      if (op.changed_cells) parts.push(['Changed', `${op.changed_cells} cells`]);
      if (op.rows_removed)  parts.push(['Removed', `${op.rows_removed} rows`]);
      if (op.rows_added)    parts.push(['Added',   `${op.rows_added} rows`]);
    }

    if (!parts.length) return '';
    return `<div class="cl-meta-row">${parts.map(([k,v]) =>
      `<span class="cl-meta-pill"><span class="cl-meta-key">${_esc(k)}</span><span class="cl-meta-val">${_esc(v)}</span></span>`
    ).join('')}</div>`;
  }

  function _operationTitle(op) {
    const titles = {
      drop_duplicates: 'Remove Duplicate Rows',
      drop_empty_columns: 'Drop Fully Empty Columns',
      drop_constant_columns: 'Drop Constant Columns',
      trim_whitespace: 'Trim Whitespace',
      fill_numeric: 'Fill Missing Numeric Values',
      fill_categorical: 'Fill Missing Categorical Values',
      drop_missing_rows: 'Drop Rows With Missing Values',
      manual_missing_cell_updates: 'Fill Missing Cells',
      convert_dtypes: 'Convert Data Types',
      manual_convert_dtypes: 'Convert Data Types',
      manual_outlier_removal: 'Remove Outlier Rows',
      outlier_removal: 'Remove Outlier Rows',
      outlier_clip: 'Clip Outlier Values',
      label_encode: 'Label Encoding',
      manual_cell_updates: 'Dataset Cell Edit',
      manual_column_renames: 'Column Rename',
      drop_selected_columns: 'Drop Selected Columns',
      custom_pandas_code: 'Execute Custom Pandas Code',
    };
    return titles[op && op.action] || String((op && op.action) || 'Cleaning Action').replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  }

  function _formatEvidenceValue(value) {
    if (value === null || value === undefined) return '<span class="cl-null-value">Missing</span>';
    if (Array.isArray(value)) return _esc(value.map(item => String(item)).slice(0, 4).join(', '));
    if (typeof value === 'object') return _esc(_compactRowEvidence(value));
    return _esc(String(value));
  }

  function _compactRowEvidence(data) {
    const values = Object.entries(data || {})
      .filter(([, value]) => value !== null && value !== undefined && value !== '')
      .slice(0, 4)
      .map(([column, value]) => {
        const text = String(value);
        return `${column}: ${text.length > 36 ? `${text.slice(0, 33)}...` : text}`;
      });
    return values.join(' | ') || 'Row present';
  }

  function _outlierRecordsHtml(operations) {
    const records = (operations || [])
      .filter(op => op && op.action === 'manual_outlier_removal')
      .flatMap(op => (op.selected_records || []).map(record => ({ op, record })));
    if (!records.length) return '';

    const byRow = new Map();
    records.forEach(({ op, record }) => {
      const key = String(record.row_index ?? record.row_number ?? byRow.size);
      if (!byRow.has(key)) byRow.set(key, { op, records: [], rowData: record.row_data || {}, rowNumber: record.row_number });
      byRow.get(key).records.push(record);
    });

    const rows = Array.from(byRow.values()).slice(0, 8).map(item => {
      const values = item.records.map(record => {
        const column = record.column || 'value';
        const value = record.value === null || record.value === undefined ? 'Missing' : String(record.value);
        return `<span class="cl-outlier-value"><strong>${_esc(column)}</strong>: ${_esc(value)}</span>`;
      }).join('');
      const rowData = item.rowData || {};
      const context = Object.entries(rowData)
        .filter(([column, value]) =>
          value !== null && value !== undefined && value !== '' &&
          !item.records.some(record => String(record.column) === String(column))
        )
        .slice(0, 3)
        .map(([column, value]) => `<span><strong>${_esc(column)}</strong>: ${_esc(String(value))}</span>`)
        .join('');
      const method = String(item.op.method || 'iqr').toLowerCase() === 'user_selection'
        ? 'Selected from IQR outlier preview'
        : 'IQR outlier rule';
      return `<tr>
        <td>Row ${_esc(item.rowNumber || '')}</td>
        <td>${values || '<span class="cl-muted">Outlier value recorded</span>'}</td>
        <td>${_esc(method)}</td>
        <td><div class="cl-outlier-context">${context || '<span class="cl-muted">Exact row removed</span>'}</div></td>
        <td class="cl-ba-after">Removed</td>
      </tr>`;
    }).join('');
    const total = byRow.size;
    const note = total > 8 ? `<div class="cl-preview-note">Showing 8 of ${total} removed outlier rows. Use View Full Changes for more detail.</div>` : '';
    return `<div class="cl-outlier-preview">
      <div class="cl-outlier-title">Removed outlier rows</div>
      <table class="cl-outlier-table">
        <thead><tr><th>Uploaded row</th><th>Outlier value</th><th>Why removed</th><th>Row context</th><th>After</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      ${note}
    </div>`;
  }

  function _operationColumns(op) {
    const columns = new Set();
    ['columns', 'columns_processed', 'columns_dropped', 'columns_encoded'].forEach(key => {
      (Array.isArray(op && op[key]) ? op[key] : []).forEach(column => columns.add(String(column)));
    });
    const details = (op && (op.details || op.changes)) || {};
    if (Array.isArray(details)) details.forEach(item => { if (item && item.column) columns.add(String(item.column)); });
    else if (details && typeof details === 'object') Object.keys(details).forEach(column => columns.add(column));
    return columns;
  }

  function _previewForOperation(preview, op) {
    const source = preview && typeof preview === 'object' ? preview : {};
    const action = String((op && op.action) || '');
    const trimDetailRows = action === 'trim_whitespace'
      ? new Set((op.details || []).flatMap(item => (item.rows || []).map(String)))
      : null;
    const trimCellCount = action === 'trim_whitespace'
      ? (op.details || []).reduce((total, item) => total + Number(item.count || (item.rows || []).length || 0), 0)
      : 0;
    const result = {
      before_shape: source.before_shape,
      after_shape: source.after_shape,
      before_columns: source.before_columns,
      after_columns: source.after_columns,
      limit: source.limit,
      total_affected_rows: source.total_affected_rows,
      changed_cells: [], removed_rows: [], added_rows: [], removed_columns: [], added_columns: [], dtype_changes: [],
    };
    const rowRemovals = new Set(['drop_duplicates', 'drop_missing_rows', 'manual_outlier_removal', 'outlier_removal']);
    const columnRemovals = new Set(['drop_empty_columns', 'drop_constant_columns', 'drop_selected_columns']);
    if (rowRemovals.has(action)) {
      const rows = (op.rows_dropped || op.row_numbers || op.selected_rows || []).map(String);
      const requested = new Set(rows);
      result.removed_rows = (source.removed_rows || []).filter(item => !requested.size || requested.has(String(item.row_number)) || requested.has(String(item.row_id)));
    } else if (columnRemovals.has(action)) {
      const columns = _operationColumns(op);
      result.removed_columns = (source.removed_columns || []).filter(item => !columns.size || columns.has(String(item.column)));
    } else if (action === 'custom_pandas_code') {
      ['changed_cells', 'removed_rows', 'added_rows', 'removed_columns', 'added_columns'].forEach(key => {
        result[key] = (source[key] || []).slice();
      });
    } else {
      const columns = _operationColumns(op);
      result.changed_cells = (source.changed_cells || []).filter(item => !columns.size || columns.has(String(item.column)));
      result.dtype_changes = (source.dtype_changes || []).filter(item => !columns.size || columns.has(String(item.column)));
      if (action === 'manual_column_renames') {
        result.removed_columns = (source.removed_columns || []).slice();
        result.added_columns = (source.added_columns || []).slice();
      }
    }

    const counts = {
      total_changed_cells: Number(source.total_changed_cells || result.changed_cells.length || 0),
      total_removed_rows: Number(source.total_removed_rows || result.removed_rows.length || 0),
      total_added_rows: Number(source.total_added_rows || result.added_rows.length || 0),
      total_removed_columns: Number(source.total_removed_columns || result.removed_columns.length || 0),
      total_added_columns: Number(source.total_added_columns || result.added_columns.length || 0),
      total_dtype_changes: Number(source.total_dtype_changes || result.dtype_changes.length || 0),
    };
    if (action === 'drop_duplicates') counts.total_removed_rows = Number(op.removed_rows || result.removed_rows.length);
    else if (action === 'drop_missing_rows') counts.total_removed_rows = Number(op.dropped_row_count || result.removed_rows.length);
    else if (rowRemovals.has(action)) counts.total_removed_rows = Number(op.rows_removed || result.removed_rows.length);
    else if (columnRemovals.has(action)) counts.total_removed_columns = Number(op.count || result.removed_columns.length);
    else if (action === 'trim_whitespace') {
      counts.total_changed_cells = trimCellCount || Number(source.total_changed_cells || result.changed_cells.length || 0);
      result.total_affected_rows = trimDetailRows && trimDetailRows.size
        ? trimDetailRows.size
        : Number(source.total_affected_rows || 0);
    }
    Object.assign(result, counts);
    result.displayed_change_count = ['changed_cells', 'removed_rows', 'added_rows', 'removed_columns', 'added_columns', 'dtype_changes']
      .reduce((total, key) => total + result[key].length, 0);
    result.total_change_count = Object.values(counts).reduce((total, value) => total + value, 0);
    result.is_sample = result.displayed_change_count < result.total_change_count;
    return result;
  }

  function _buildPersistedPreviewHtml(preview, rowLimit = 3) {
    if (!preview || typeof preview !== 'object') return '';
    const removedColumns = Array.isArray(preview.removed_columns) ? preview.removed_columns : [];
    const addedColumns = Array.isArray(preview.added_columns) ? preview.added_columns : [];
    const dtypeChanges = Array.isArray(preview.dtype_changes) ? preview.dtype_changes : [];
    const dtypeOnly = dtypeChanges.length
      && !(preview.changed_cells || []).length
      && !(preview.removed_rows || []).length
      && !(preview.added_rows || []).length
      && !removedColumns.length
      && !addedColumns.length;
    if (dtypeOnly) {
      const body = dtypeChanges.slice(0, Math.max(rowLimit, 8)).map(item => `<tr>
        <td class="cl-ba-context">${_esc(item.column || '')}</td>
        <td class="cl-ba-before"><code class="cl-dtype">${_esc(item.before || '-')}</code></td>
        <td class="cl-ba-direction" aria-hidden="true">-&gt;</td>
        <td class="cl-ba-after"><code class="cl-dtype cl-dtype-new">${_esc(item.after || '-')}</code></td>
      </tr>`).join('');
      const totalTypes = preview.total_dtype_changes || dtypeChanges.length;
      const note = dtypeChanges.length < totalTypes
        ? `<div class="cl-preview-note">Showing ${dtypeChanges.length} of ${totalTypes} type changes.</div>`
        : '';
      return `<div class="cl-ba-wrap cl-authoritative-preview">
        <table class="cl-ba-table">
          <thead><tr><th>Column</th><th>Before type</th><th aria-label="Direction"></th><th>After type</th></tr></thead>
          <tbody>${body}</tbody>
        </table>${note}
      </div>`;
    }
    const structuralColumnOnly = (removedColumns.length || addedColumns.length)
      && !(preview.changed_cells || []).length
      && !(preview.removed_rows || []).length
      && !(preview.added_rows || []).length;
    if (structuralColumnOnly) {
      const rows = [];
      removedColumns.forEach(item => rows.push({
        context: String(item.column),
        before: 'Column existed',
        after: 'Dropped',
      }));
      addedColumns.forEach(item => rows.push({
        context: String(item.column),
        before: 'Not present',
        after: 'Added',
      }));
      const body = rows.slice(0, Math.max(rowLimit, 8)).map(item => `<tr>
        <td class="cl-ba-context">${_esc(item.context)}</td>
        <td class="cl-ba-before">${_formatEvidenceValue(item.before)}</td>
        <td class="cl-ba-direction" aria-hidden="true">-&gt;</td>
        <td class="cl-ba-after">${_formatEvidenceValue(item.after)}</td>
      </tr>`).join('');
      const totalColumns = (preview.total_removed_columns || removedColumns.length) + (preview.total_added_columns || addedColumns.length);
      const note = rows.length < totalColumns
        ? `<div class="cl-preview-note">Showing ${rows.length} of ${totalColumns} affected columns.</div>`
        : '';
      return `<div class="cl-ba-wrap cl-authoritative-preview">
        <table class="cl-ba-table">
          <thead><tr><th>Column</th><th>Before</th><th aria-label="Direction"></th><th>After</th></tr></thead>
          <tbody>${body}</tbody>
        </table>${note}
      </div>`;
    }
    const rowEvidence = Array.isArray(preview.row_evidence) ? preview.row_evidence.slice(0, rowLimit) : [];
    if (rowEvidence.length) {
      const duplicateMatches = new Map();
      (preview.duplicate_matches || []).forEach(match => {
        if (match && match.removed_uploaded_row != null) duplicateMatches.set(String(match.removed_uploaded_row), match);
        if (match && match.removed_row_id != null) duplicateMatches.set(String(match.removed_row_id), match);
      });
      const duplicateMatchHtml = (row) => {
        const match = duplicateMatches.get(String(row.original_row_number)) || duplicateMatches.get(String(row.row_id));
        if (!match) return '';
        const basis = match.normalization_used ? 'Matches after normalization' : 'Exact duplicate';
        const compared = Array.isArray(match.comparison_columns) && match.comparison_columns.length
          ? `${match.comparison_columns.length} column${match.comparison_columns.length === 1 ? '' : 's'} compared`
          : 'All columns compared';
        return `<div class="cl-duplicate-match">
          <strong>${_esc(basis)}</strong>
          <span>Matched retained uploaded row ${_esc(match.retained_uploaded_row)}${match.retained_record_id == null ? '' : ` (record_id ${_esc(match.retained_record_id)})`}</span>
          <span>${_esc(compared)}</span>
          ${match.reason ? `<span>${_esc(match.reason)}</span>` : ''}
        </div>`;
      };
      const renderSide = (row, side) => {
        const values = row[side];
        if (!values) return `<div class="cl-row-removed">${side === 'after' ? 'Removed' : 'Not present'}</div>`;
        const changed = new Set((row.changed_columns || []).map(String));
        return `<table class="cl-context-table"><thead><tr>${row.columns.map(col => `<th>${_esc(col)}</th>`).join('')}</tr></thead>
          <tbody><tr>${row.columns.map(col => {
            const value = values[col];
            const text = typeof value === 'string' && (/^\s|\s$/.test(value))
              ? JSON.stringify(value) : value;
            return `<td class="${changed.has(String(col)) ? `is-${side}` : ''}">${_formatEvidenceValue(text)}</td>`;
          }).join('')}</tr></tbody></table>`;
      };
      const rowsHtml = rowEvidence.map(row => `<section class="cl-row-evidence">
        <div class="cl-row-identity">Uploaded Row ${_esc(row.original_row_number)}${row.record_id == null ? '' : ` | record_id ${_esc(row.record_id)}`}</div>
        ${duplicateMatchHtml(row)}
        <div class="cl-row-sides"><div><strong>Before</strong>${renderSide(row, 'before')}</div><div><strong>After</strong>${renderSide(row, 'after')}</div></div>
      </section>`).join('');
      return `<div class="cl-complete-row-preview">${rowsHtml}<div class="cl-preview-note">Showing ${rowEvidence.length} of ${preview.total_affected_rows || rowEvidence.length} affected rows.</div></div>`;
    }
    const rows = [];
    (preview.changed_cells || []).forEach(item => rows.push({
      context: `Row ${item.row_number} | ${item.column}`,
      before: item.before,
      after: item.after,
    }));
    (preview.removed_rows || []).forEach(item => rows.push({
      context: `Row ${item.row_number}`,
      before: _compactRowEvidence(item.data),
      after: 'Removed',
    }));
    (preview.added_rows || []).forEach(item => rows.push({
      context: `Row ${item.row_number}`,
      before: 'Not present',
      after: _compactRowEvidence(item.data),
    }));
    removedColumns.forEach(item => rows.push({
      context: String(item.column),
      before: 'Column existed',
      after: 'Dropped',
    }));
    addedColumns.forEach(item => rows.push({
      context: String(item.column),
      before: 'Not present',
      after: 'Added',
    }));
    if (!rows.length) return '';
    const body = rows.slice(0, 8).map(item => `<tr>
      <td class="cl-ba-context">${_esc(item.context)}</td>
      <td class="cl-ba-before">${_formatEvidenceValue(item.before)}</td>
      <td class="cl-ba-direction" aria-hidden="true">-&gt;</td>
      <td class="cl-ba-after">${_formatEvidenceValue(item.after)}</td>
    </tr>`).join('');
    const sampled = preview.is_sample
      ? `<div class="cl-preview-note">Showing ${preview.displayed_change_count} of ${preview.total_change_count} persisted changes.</div>`
      : '';
    return `<div class="cl-ba-wrap cl-authoritative-preview">
      <table class="cl-ba-table">
        <thead><tr><th>Location</th><th>Before</th><th aria-label="Direction"></th><th>After</th></tr></thead>
        <tbody>${body}</tbody>
      </table>${sampled}
    </div>`;
  }

  function _buildActionFactsHtml(action) {
    const ops = (action.operations || []).filter(Boolean);
    const facts = [];
    const titles = [...new Set(ops.map(_operationTitle))];
    if (titles.length) facts.push(['Operation', titles.join(', ')]);
    const methods = [...new Set(ops.map(op => op.method || op.strategy).filter(Boolean))];
    if (methods.length) facts.push(['Method', methods.join(', ')]);
    const perColumnMethods = [];
    ops.forEach(op => {
      Object.entries(op.column_strategies || {}).forEach(([column, method]) => {
        if (method && method !== 'skip') perColumnMethods.push(`${column}: ${method}`);
      });
      if (op.details && !Array.isArray(op.details)) {
        Object.entries(op.details).forEach(([column, detail]) => {
          if (detail && detail.method) perColumnMethods.push(`${column}: ${detail.method}`);
        });
      }
    });
    if (perColumnMethods.length) facts.push(['Methods by column', [...new Set(perColumnMethods)].join(', ')]);
    const columns = new Set();
    ops.forEach(op => {
      (op.columns || op.columns_processed || op.columns_dropped || op.columns_encoded || []).forEach(col => columns.add(col));
      if (Array.isArray(op.details)) {
        op.details.forEach(item => { if (item && item.column) columns.add(item.column); });
      } else if (op.details && typeof op.details === 'object') {
        Object.keys(op.details).forEach(col => columns.add(col));
      }
      if (Array.isArray(op.changes)) {
        op.changes.forEach(item => { if (item && item.column) columns.add(item.column); });
      } else if (op.changes && typeof op.changes === 'object') {
        Object.keys(op.changes).forEach(col => columns.add(col));
      }
    });
    if (columns.size) facts.push(['Affected columns', [...columns].slice(0, 6).join(', ') + (columns.size > 6 ? '...' : '')]);
    const preview = action.preview || {};
    if (Array.isArray(preview.before_shape) && Array.isArray(preview.after_shape)) {
      facts.push(['Immediate shape', `${preview.before_shape[0]} x ${preview.before_shape[1]} -> ${preview.after_shape[0]} x ${preview.after_shape[1]}`]);
      const actionName = String((ops[0] && ops[0].action) || '');
      if (actionName.includes('column')) facts.push(['Columns', `${preview.before_shape[1]} to ${preview.after_shape[1]}`]);
      if (['drop_duplicates', 'drop_missing_rows', 'manual_outlier_removal', 'outlier_removal'].includes(actionName)) {
        facts.push(['Rows', `${preview.before_shape[0]} to ${preview.after_shape[0]}`]);
      }
    }
    const affectedRowIds = new Set([
      ...(preview.changed_cells || []).map(item => item.row_id),
      ...(preview.removed_rows || []).map(item => item.row_id),
      ...(preview.added_rows || []).map(item => item.row_id),
    ]);
    const structuralRows = (preview.total_removed_rows || 0) + (preview.total_added_rows || 0);
    const affectedRows = Number(preview.total_affected_rows || 0) || Math.max(affectedRowIds.size, structuralRows);
    if (affectedRows) facts.push(['Affected rows', String(affectedRows)]);
    if (preview.total_changed_cells) facts.push(['Changed cells', String(preview.total_changed_cells)]);
    if (preview.total_dtype_changes) {
      const dtypeList = (preview.dtype_changes || [])
        .map(item => `${item.column}: ${item.before} -> ${item.after}`)
        .slice(0, 6);
      facts.push(['Type changes', dtypeList.length ? dtypeList.join(', ') : String(preview.total_dtype_changes)]);
    }
    if (preview.total_removed_columns) facts.push(['Removed columns', String(preview.total_removed_columns)]);
    if (preview.total_added_columns) facts.push(['Added columns', String(preview.total_added_columns)]);
    return facts.length ? `<dl class="cl-action-facts">${facts.map(([key, value]) =>
      `<div><dt>${_esc(key)}</dt><dd>${_esc(value)}</dd></div>`).join('')}</dl>` : '';
  }

  function _previewIsSampled(preview) {
    if (!preview || typeof preview !== 'object') return false;
    const displayed = Number(preview.displayed_change_count || 0);
    const total = Number(preview.total_change_count || 0);
    return !!preview.is_sample || (total > 0 && displayed > 0 && displayed < total);
  }

  function _buildActionCard(action, actionIndex, fb) {
    const id = `cl-action-${action.id}`;
    const undone     = !!action.undone;
    // A manifest-capable action can always be attempted via the server
    // (in-session actions get a manifest computed at apply time; resumed
    // actions carry hasManifest from their stored action_config) — this is
    // what makes Undo/Reapply work on resumed actions too, not just ones
    // created in the current session. Actions without any manifest (legacy
    // rows from before this feature, or the rare case where the DB write is
    // still in flight) fall back to the original in-memory snapshot check.
    const serverCapable = !!action.dbHistoryId && !!action.hasManifest;
    const canUndo = serverCapable && !undone;
    const canReapply = serverCapable && undone;
    const hasSyntheticOnlyFallback = !(action.preview && typeof action.preview === 'object') &&
      (action.operations || []).some(op => op.action === 'trim_whitespace' && !(op.changed_cell_details || []).length);
    const detailButtonLabel = (_previewIsSampled(action.preview) || hasSyntheticOnlyFallback) ? 'View Persisted Preview' : 'View Full Changes';

    const visibleOps = (action.operations || []).filter(hasMeaningfulChange);

    // If we have the rich step-preview HTML (detailed row-level before/after from renderStepPreview),
    // show it inside the card. Fall back to the compact _buildOpBeforeAfterHtml only if absent.
    let previewSection = '';
    const outlierPreview = _outlierRecordsHtml(visibleOps);
    const persistedPreview = _buildPersistedPreviewHtml(action.preview);
    if (outlierPreview) {
      previewSection = outlierPreview;
    } else if (persistedPreview) {
      previewSection = persistedPreview;
    } else if (action.previewHTML && action.previewHTML.trim()) {
      // Wrap in a container that strips the Undo/Reapply buttons (the card's own buttons handle those)
      previewSection = `<div class="changelog-block-preview">${action.previewHTML}</div>`;
    } else {
      // Compact meta + ba fallback (compact NaN → value table)
      const opsHtml = visibleOps.map(op => {
        const meta = _buildOpMetaHtml(op);
        const ba   = _buildOpBeforeAfterHtml(op);
        if (!meta && !ba) return '';
        return `<div class="cl-op-block">${meta}${ba}</div>`;
      }).filter(Boolean).join('');
      previewSection = opsHtml;
    }

    const el = document.createElement('div');
    el.className = 'cl-action-card' + (undone ? ' cl-action-undone' : '');
    el.id = id;
    el.innerHTML = `
      <div class="cl-action-header">
        <span class="cl-action-num">Action #${actionIndex + 1}</span>
        <span class="cl-action-status ${undone ? 'is-undone' : 'is-applied'}">${undone ? 'Undone' : 'Applied'}</span>
        <span class="cl-action-time">${_fmt(action.ts)}</span>
      </div>
      <div class="cl-action-label">${_esc(action.label)}</div>
      ${_buildActionFactsHtml(action)}
      ${previewSection}
      <div class="cl-action-btns">
        ${action.hasManifest ? `<button type="button" class="cl-btn cl-btn-undo"    data-action-id="${action.id}" ${canUndo ? '' : 'disabled'}>Undo</button>` : ''}
        ${action.hasManifest ? `<button type="button" class="cl-btn cl-btn-reapply" data-action-id="${action.id}" ${canReapply ? '' : 'disabled'}>Reapply</button>` : ''}
        <button type="button" class="cl-btn cl-btn-view" data-action-id="${action.id}">${detailButtonLabel}</button>
        ${!action.hasManifest ? `<span class="cl-resumed-badge">Undo/Reapply unavailable · no persisted manifest</span>` : ''}
      </div>`;

    el.querySelectorAll('.cl-btn-undo').forEach(btn =>
      btn.addEventListener('click', () => { if (!btn.disabled) _undoAction(action.id, btn); }));
    el.querySelectorAll('.cl-btn-reapply').forEach(btn =>
      btn.addEventListener('click', () => { if (!btn.disabled) _reapplyAction(action.id, btn); }));
    el.querySelectorAll('.cl-btn-view').forEach(btn =>
      btn.addEventListener('click', () => _viewFullChanges(action, actionIndex + 1)));

    return el;
  }

  function _dtypeChangesFromOperations(operations) {
    const rows = [];
    (operations || []).forEach(op => {
      if (!op || op.action === undefined) return;
      if (op.action === 'convert_dtypes') {
        (op.changes || []).forEach(item => {
          rows.push({
            column: item.column || '',
            before: item.original_dtype || item.from || '-',
            after: item.new_dtype || item.to || '-',
          });
        });
      } else if (op.action === 'manual_convert_dtypes') {
        Object.entries(op.changes || {}).forEach(([column, item]) => {
          const info = item && typeof item === 'object' ? item : {};
          rows.push({
            column,
            before: info.from || info.original_dtype || '-',
            after: info.to || info.new_dtype || info.actual_dtype || String(item || '-'),
          });
        });
      }
    });
    return rows.filter(item => item.column || item.before !== '-' || item.after !== '-');
  }

  function _openPersistedFullChanges(action, displayNum) {
    const preview = Object.assign({}, action.preview || {});
    if (!(preview.dtype_changes || []).length) {
      const synthesizedDtypes = _dtypeChangesFromOperations(action.operations || []);
      if (synthesizedDtypes.length) {
        preview.dtype_changes = synthesizedDtypes;
        preview.total_dtype_changes = synthesizedDtypes.length;
      }
    }
    const previewSampled = _previewIsSampled(preview);
    const detailTitle = previewSampled ? 'Persisted Preview Sample' : 'Full Changes';
    const removedColumns = Array.isArray(preview.removed_columns) ? preview.removed_columns : [];
    const addedColumns = Array.isArray(preview.added_columns) ? preview.added_columns : [];
    const dtypeChanges = Array.isArray(preview.dtype_changes) ? preview.dtype_changes : [];
    const dtypeOnly = dtypeChanges.length
      && !(preview.changed_cells || []).length
      && !(preview.removed_rows || []).length
      && !(preview.added_rows || []).length
      && !removedColumns.length
      && !addedColumns.length;
    const structuralColumnOnly = (removedColumns.length || addedColumns.length)
      && !(preview.changed_cells || []).length
      && !(preview.removed_rows || []).length
      && !(preview.added_rows || []).length;
    const entries = [];
    (preview.changed_cells || []).forEach(item => entries.push({ location: `Row ${item.row_number} | ${item.column}`, before: item.before, after: item.after }));
    (preview.removed_rows || []).forEach(item => entries.push({ location: `Row ${item.row_number}`, before: _compactRowEvidence(item.data), after: 'Removed' }));
    (preview.added_rows || []).forEach(item => entries.push({ location: `Row ${item.row_number}`, before: 'Not present', after: _compactRowEvidence(item.data) }));
    removedColumns.forEach(item => entries.push({ location: String(item.column), before: 'Column existed', after: 'Dropped' }));
    addedColumns.forEach(item => entries.push({ location: String(item.column), before: 'Not present', after: 'Added' }));
    dtypeChanges.forEach(item => entries.push({
      location: `Column ${item.column} | Data type`,
      before: item.before,
      after: item.after,
    }));

    const operations = action.operations || [];
    const warnings = operations.flatMap(op => op.warnings || []);
    const skipped = operations.flatMap(op => op.skipped || []);
    const conflicts = operations.flatMap(op => op.conflicts || []);
    const failures = operations.filter(op => op.error || /failed$/i.test(op.action || '')).map(op => op.error || _operationTitle(op));
    const modal = document.createElement('div');
    modal.className = 'cl-full-modal';
    modal.setAttribute('role', 'dialog');
    modal.setAttribute('aria-modal', 'true');
    modal.setAttribute('aria-label', `Action ${displayNum} ${detailTitle.toLowerCase()}`);
    modal.innerHTML = `<div class="cl-full-dialog">
      <div class="cl-full-header">
        <span>Action #${displayNum} - ${detailTitle}</span>
        <button type="button" class="cl-full-close" aria-label="Close details">x</button>
      </div>
      <div class="cl-full-summary">
        <span>${preview.total_changed_cells || 0} affected cells</span>
        <span>${preview.total_removed_rows || 0} rows removed</span>
        <span>${preview.total_added_rows || 0} rows added</span>
        <span>${preview.total_removed_columns || 0} columns removed</span>
        <span>${preview.total_added_columns || 0} columns added</span>
        <span>${preview.total_dtype_changes || 0} type changes</span>
      </div>
      <div class="cl-full-body">
        <div class="cl-full-table-wrap"><table class="cl-full-table">
          <thead><tr><th>Location</th><th>Before</th><th>After</th></tr></thead>
          <tbody></tbody>
        </table></div>
        <div class="cl-full-page"></div>
        ${previewSampled ? `<p class="cl-full-note">Stored preview shows ${preview.displayed_change_count || 0} of ${preview.total_change_count || 0} persisted changes. This is a bounded preview; Undo/Reapply replay still uses the complete manifest.</p>` : ''}
        ${skipped.length ? `<div class="cl-full-message"><strong>Skipped rows</strong><pre>${_esc(JSON.stringify(skipped, null, 2))}</pre></div>` : ''}
        ${conflicts.length ? `<div class="cl-full-message"><strong>Conflicts</strong><pre>${_esc(JSON.stringify(conflicts, null, 2))}</pre></div>` : ''}
        ${failures.length ? `<div class="cl-full-message is-error"><strong>Failed operations</strong><pre>${_esc(JSON.stringify(failures, null, 2))}</pre></div>` : ''}
        ${warnings.length ? `<div class="cl-full-message"><strong>Warnings</strong><pre>${_esc(JSON.stringify(warnings, null, 2))}</pre></div>` : ''}
      </div>
    </div>`;
    document.body.appendChild(modal);

    const pageSize = 10;
    let page = 0;
    const tbody = modal.querySelector('tbody');
    const pageHost = modal.querySelector('.cl-full-page');
    const tableWrap = modal.querySelector('.cl-full-table-wrap');
    let serverEvidence = false;
    async function loadEvidencePage(pageNumber) {
      if (!action.dbHistoryId) return false;
      const response = await fetch(`/api/history/${encodeURIComponent(action.dbHistoryId)}/evidence?page=${pageNumber}&per_page=25`);
      const data = await response.json();
      if (!data.success || data.limited_evidence || !data.rows.length) return false;
      serverEvidence = true;
      page = pageNumber - 1;
      tableWrap.innerHTML = _buildPersistedPreviewHtml({
        row_evidence: data.rows,
        total_affected_rows: data.total,
      }, 25);
      const pages = Math.max(1, Math.ceil(data.total / data.per_page));
      pageHost.innerHTML = `<button type="button" data-page="prev" ${pageNumber <= 1 ? 'disabled' : ''}>Previous</button>
        <span>Page ${pageNumber} of ${pages}</span>
        <button type="button" data-page="next" ${pageNumber >= pages ? 'disabled' : ''}>Next</button>`;
      const headerSpan = modal.querySelector('.cl-full-header span');
      if (headerSpan) headerSpan.textContent = `Action #${displayNum} - Full Changes`;
      const note = modal.querySelector('.cl-full-note');
      if (note) note.remove();
      return true;
    }
    function renderPage() {
      const start = page * pageSize;
      const visible = entries.slice(start, start + pageSize);
      tbody.innerHTML = visible.length ? visible.map(item => `<tr>
        <td>${_esc(item.location)}</td><td>${_formatEvidenceValue(item.before)}</td><td>${_formatEvidenceValue(item.after)}</td>
      </tr>`).join('') : '<tr><td colspan="3" class="cl-full-empty">No detailed persisted values are available for this action.</td></tr>';
      const pages = Math.max(1, Math.ceil(entries.length / pageSize));
      pageHost.innerHTML = pages > 1 ? `<button type="button" data-page="prev" ${page === 0 ? 'disabled' : ''}>Previous</button>
        <span>Page ${page + 1} of ${pages}</span>
        <button type="button" data-page="next" ${page >= pages - 1 ? 'disabled' : ''}>Next</button>` : '';
    }
    pageHost.addEventListener('click', event => {
      const button = event.target.closest('button[data-page]');
      if (!button || button.disabled) return;
      if (serverEvidence) {
        loadEvidencePage(page + (button.dataset.page === 'next' ? 2 : 0));
        return;
      }
      page += button.dataset.page === 'next' ? 1 : -1;
      renderPage();
    });
    const close = () => modal.remove();
    modal.querySelector('.cl-full-close').addEventListener('click', close);
    modal.addEventListener('click', event => { if (event.target === modal) close(); });
    if (dtypeOnly || structuralColumnOnly) {
      tableWrap.innerHTML = _buildPersistedPreviewHtml(preview, 25)
        || '<div class="cl-full-empty">No detailed persisted values are available for this action.</div>';
      pageHost.innerHTML = '';
    } else {
      renderPage();
      loadEvidencePage(1).catch(() => {});
    }
    modal.querySelector('.cl-full-close').focus();
  }

  function _viewFullChanges(action, displayNum) {
    if (action.preview && typeof action.preview === 'object') {
      _openPersistedFullChanges(action, displayNum);
      return;
    }
    const ops = (action.operations || []).filter(hasMeaningfulChange);

    // Build rows from changed_cell_details (custom code / manual edits) OR synthesise from op data
    let rows = '';

    ops.forEach(op => {
      // ── Use changed_cell_details when present (custom code, manual edits) ──
      if ((op.changed_cell_details || []).length) {
        op.changed_cell_details.forEach(d => {
          rows += `<tr>
            <td style="padding:7px 14px;color:#6b7280;font-size:12px;">${_esc(String(d.row_number||''))}</td>
            <td style="padding:7px 14px;font-size:13px;font-weight:600;color:#374151;">${_esc(String(d.column||''))}</td>
            <td style="padding:7px 14px;background:#fff5f5;font-size:13px;color:#374151;">${_esc(String(d.before!==undefined?d.before:''))}</td>
            <td style="padding:7px 14px;background:#f0fdf4;font-size:13px;color:#374151;">${_esc(String(d.after!==undefined?d.after:''))}</td>
          </tr>`;
        });
        return;
      }

      // ── Synthesise from fill_numeric ──
      if (op.action === 'fill_numeric') {
        (op.details || []).forEach(item => {
          const isDrop = item.strategy === 'drop';
          (item.rows || []).forEach(rowNum => {
            rows += `<tr>
              <td style="padding:7px 14px;color:#6b7280;font-size:12px;">${_esc(String(rowNum))}</td>
              <td style="padding:7px 14px;font-size:13px;font-weight:600;color:#374151;">${_esc(item.column||'')}</td>
              <td style="padding:7px 14px;background:#fff5f5;font-size:13px;color:#9ca3af;font-style:italic;">NaN (missing)</td>
              <td style="padding:7px 14px;background:${isDrop ? '#fff5f5' : '#f0fdf4'};font-size:13px;color:${isDrop ? '#b91c1c' : '#374151'};${isDrop ? 'font-weight:600;' : ''}">${isDrop ? 'Row removed' : _esc(String(item.value!==undefined?item.value:'—'))}</td>
            </tr>`;
          });
        });
        return;
      }

      // ── Synthesise from fill_categorical ──
      if (op.action === 'fill_categorical') {
        Object.entries(op.details || {}).forEach(([col, item]) => {
          (item.rows || []).forEach(rowNum => {
            rows += `<tr>
              <td style="padding:7px 14px;color:#6b7280;font-size:12px;">${_esc(String(rowNum))}</td>
              <td style="padding:7px 14px;font-size:13px;font-weight:600;color:#374151;">${_esc(col)}</td>
              <td style="padding:7px 14px;background:#fff5f5;font-size:13px;color:#9ca3af;font-style:italic;">NaN (missing)</td>
              <td style="padding:7px 14px;background:#f0fdf4;font-size:13px;color:#374151;">${_esc(String(item.value!==undefined?item.value:'—'))}</td>
            </tr>`;
          });
        });
        return;
      }

      // ── Synthesise from manual_outlier_removal ──
      if (op.action === 'manual_outlier_removal') {
        // op.row_numbers is the human-readable, 1-based row number cleaner.py
        // already computes (matches what the user saw/checked in the outlier
        // picker). op.selected_rows / op.valid_indices_dropped are the raw
        // 0-based pandas index labels used internally for the actual drop —
        // showing those here was the source of the "off by one" row-number
        // mismatch users saw between what they selected and what the log reported.
        const rowNums = op.row_numbers || op.selected_rows || op.valid_indices_dropped || [];
        const vals    = op.removed_values || [];
        rowNums.forEach((rn, i) => {
          const valSnip = vals[i] !== undefined ? _esc(String(vals[i])) : '—';
          rows += `<tr>
            <td style="padding:7px 14px;color:#6b7280;font-size:12px;">${_esc(String(rn))}</td>
            <td style="padding:7px 14px;font-size:13px;font-weight:600;color:#374151;">${_esc((op.columns||[])[0]||'(row)')}</td>
            <td style="padding:7px 14px;background:#fff5f5;font-size:13px;color:#374151;">${valSnip}</td>
            <td style="padding:7px 14px;background:#fef2f2;font-size:13px;color:#ef4444;">Removed</td>
          </tr>`;
        });
        if (!rowNums.length && op.rows_removed) {
          rows += `<tr>
            <td colspan="4" style="padding:14px;color:#6b7280;font-size:13px;text-align:center;">${op.rows_removed} outlier row${op.rows_removed!==1?'s':''} removed.</td>
          </tr>`;
        }
        return;
      }

      // ── Synthesise from drop_missing_rows ──
      if (op.action === 'drop_missing_rows') {
        (op.row_numbers || []).forEach(rn => {
          rows += `<tr>
            <td style="padding:7px 14px;color:#6b7280;font-size:12px;">${_esc(String(rn))}</td>
            <td style="padding:7px 14px;font-size:13px;font-weight:600;color:#374151;">(row with missing values)</td>
            <td style="padding:7px 14px;background:#fff5f5;font-size:13px;color:#374151;">Present</td>
            <td style="padding:7px 14px;background:#fef2f2;font-size:13px;color:#ef4444;">Dropped</td>
          </tr>`;
        });
        return;
      }

      // ── Synthesise from drop_duplicates ──
      if (op.action === 'drop_duplicates') {
        (op.rows_dropped || []).forEach(rn => {
          rows += `<tr>
            <td style="padding:7px 14px;color:#6b7280;font-size:12px;">${_esc(String(rn))}</td>
            <td style="padding:7px 14px;font-size:13px;font-weight:600;color:#374151;">(duplicate row)</td>
            <td style="padding:7px 14px;background:#fff5f5;font-size:13px;color:#374151;">Duplicate</td>
            <td style="padding:7px 14px;background:#fef2f2;font-size:13px;color:#ef4444;">Removed</td>
          </tr>`;
        });
        return;
      }

      // ── Synthesise from dtype conversions ──
      if (op.action === 'convert_dtypes') {
        (op.changes || []).forEach(item => {
          rows += `<tr>
            <td style="padding:7px 14px;color:#6b7280;font-size:12px;">(all rows)</td>
            <td style="padding:7px 14px;font-size:13px;font-weight:600;color:#374151;">${_esc(item.column||'')}</td>
            <td style="padding:7px 14px;background:#fff5f5;font-size:13px;color:#374151;">${_esc(item.original_dtype||item.from||'—')}</td>
            <td style="padding:7px 14px;background:#f0fdf4;font-size:13px;color:#374151;">${_esc(item.new_dtype||item.to||'—')}</td>
          </tr>`;
        });
        return;
      }
      if (op.action === 'manual_convert_dtypes') {
        Object.entries(op.changes || {}).forEach(([col, item]) => {
          const from = typeof item==='object' ? (item.from||item.original_dtype||'—') : '—';
          const to   = typeof item==='object' ? (item.to||item.new_dtype||item.actual_dtype||'—') : String(item);
          rows += `<tr>
            <td style="padding:7px 14px;color:#6b7280;font-size:12px;">(all rows)</td>
            <td style="padding:7px 14px;font-size:13px;font-weight:600;color:#374151;">${_esc(col)}</td>
            <td style="padding:7px 14px;background:#fff5f5;font-size:13px;color:#374151;">${_esc(from)}</td>
            <td style="padding:7px 14px;background:#f0fdf4;font-size:13px;color:#374151;">${_esc(to)}</td>
          </tr>`;
        });
        return;
      }

      // ── Synthesise from trim_whitespace ──
      if (op.action === 'trim_whitespace') {
        (op.columns_processed || []).forEach(col => {
          rows += `<tr>
            <td style="padding:7px 14px;color:#6b7280;font-size:12px;">(text rows)</td>
            <td style="padding:7px 14px;font-size:13px;font-weight:600;color:#374151;">${_esc(col)}</td>
            <td style="padding:7px 14px;background:#fff5f5;font-size:13px;color:#9ca3af;font-style:italic;">"value " (with trailing space)</td>
            <td style="padding:7px 14px;background:#f0fdf4;font-size:13px;color:#374151;">"value" (trimmed)</td>
          </tr>`;
        });
        return;
      }

      // ── Synthesise from column drops ──
      if (['drop_empty_columns','drop_constant_columns','drop_selected_columns'].includes(op.action)) {
        const dropped = op.columns_dropped || op.columns || [];
        dropped.forEach(col => {
          rows += `<tr>
            <td style="padding:7px 14px;color:#6b7280;font-size:12px;">(column)</td>
            <td style="padding:7px 14px;font-size:13px;font-weight:600;color:#374151;">${_esc(col)}</td>
            <td style="padding:7px 14px;background:#fff5f5;font-size:13px;color:#374151;">Present</td>
            <td style="padding:7px 14px;background:#fef2f2;font-size:13px;color:#ef4444;">Dropped</td>
          </tr>`;
        });
        return;
      }

      // ── Synthesise from manual_missing_cell_updates / manual_cell_updates ──
      if (op.action === 'manual_missing_cell_updates' || op.action === 'manual_cell_updates') {
        (op.updates || op.changes || []).forEach(u => {
          rows += `<tr>
            <td style="padding:7px 14px;color:#6b7280;font-size:12px;">${_esc(String(u.row_number||''))}</td>
            <td style="padding:7px 14px;font-size:13px;font-weight:600;color:#374151;">${_esc(u.column||'')}</td>
            <td style="padding:7px 14px;background:#fff5f5;font-size:13px;color:#9ca3af;font-style:italic;">${_esc(String(u.old||u.before||'(missing)'))}</td>
            <td style="padding:7px 14px;background:#f0fdf4;font-size:13px;color:#374151;">${u.blank ? '<em style="color:#9ca3af">blank/missing</em>' : _esc(String(u.value||u.after||''))}</td>
          </tr>`;
        });
        return;
      }

      // ── Synthesise from manual_column_renames ──
      if (op.action === 'manual_column_renames') {
        Object.entries(op.renamed_map || {}).forEach(([from, to]) => {
          rows += `<tr>
            <td style="padding:7px 14px;color:#6b7280;font-size:12px;">(column)</td>
            <td style="padding:7px 14px;font-size:13px;font-weight:600;color:#374151;">${_esc(from)}</td>
            <td style="padding:7px 14px;background:#fff5f5;font-size:13px;color:#374151;">${_esc(from)}</td>
            <td style="padding:7px 14px;background:#f0fdf4;font-size:13px;color:#374151;">${_esc(to)}</td>
          </tr>`;
        });
      }
    });

    const modal = document.createElement('div');
    modal.style.cssText = 'position:fixed;inset:0;z-index:10000;background:rgba(0,0,0,.5);display:flex;align-items:center;justify-content:center;padding:24px;';
    modal.innerHTML = `
      <div style="background:#fff;border-radius:12px;max-width:780px;width:100%;max-height:80vh;display:flex;flex-direction:column;box-shadow:0 20px 60px rgba(0,0,0,.3);overflow:hidden;">
        <div style="display:flex;align-items:center;justify-content:space-between;padding:16px 22px;background:#1a2233;">
          <span style="font-weight:700;font-size:15px;color:#fff;">Action #${displayNum || action.id} \u2014 Change Details</span>
          <button id="cl-modal-close" style="background:transparent;border:none;color:#9ca3af;font-size:22px;cursor:pointer;line-height:1;padding:0 4px;">&times;</button>
        </div>
        <div style="overflow-y:auto;flex:1;">
          ${rows ? `<table style="width:100%;border-collapse:collapse;font-family:inherit;">
            <thead><tr style="background:#f9fafb;border-bottom:1px solid #e5e7eb;">
              <th style="padding:10px 14px;text-align:left;font-size:11px;font-weight:700;color:#6b7280;text-transform:uppercase;letter-spacing:.06em;">Row</th>
              <th style="padding:10px 14px;text-align:left;font-size:11px;font-weight:700;color:#6b7280;text-transform:uppercase;letter-spacing:.06em;">Column</th>
              <th style="padding:10px 14px;text-align:left;font-size:11px;font-weight:700;color:#6b7280;text-transform:uppercase;letter-spacing:.06em;">Before</th>
              <th style="padding:10px 14px;text-align:left;font-size:11px;font-weight:700;color:#6b7280;text-transform:uppercase;letter-spacing:.06em;">After</th>
            </tr></thead>
            <tbody>${rows}</tbody>
          </table>` : '<div style="padding:32px;text-align:center;color:#9ca3af;font-size:14px;">No detailed cell changes recorded for this action.</div>'}
        </div>
      </div>`;
    document.body.appendChild(modal);
    modal.addEventListener('click', e => { if (e.target === modal) modal.remove(); });
    modal.querySelector('#cl-modal-close').addEventListener('click', () => modal.remove());
  }

  // ── Main push function — called after every Apply ─────────────────────

  function _rebuildActionStore(store, fb) {
    const targetEl = store.actionsEl || fb.actionsEl;
    if (!targetEl) return;
    const directActions = store === fb ? fb.actions.filter(action => !action.subKey) : store.actions;
    const subBlocks = store === fb && fb.subBlocks ? Object.values(fb.subBlocks) : [];
    targetEl.innerHTML = '';
    if (store.emptyEl) targetEl.appendChild(store.emptyEl);
    directActions.forEach((action, i) => {
      targetEl.appendChild(_buildActionCard(action, i, fb));
    });
    subBlocks.forEach(sub => targetEl.appendChild(sub.el));
  }

  function _forceActionButtonState(action) {
    document.querySelectorAll(`[data-action-id="${action.id}"]`).forEach(el => {
      if (el.classList.contains('cl-btn-undo')) el.disabled = !!action.undone;
      if (el.classList.contains('cl-btn-reapply')) el.disabled = !action.undone;
    });
  }

  let _historyMutationInProgress = false;

  function _setHistoryMutationBusy(button, busy, label) {
    document.querySelectorAll('.cl-btn-undo, .cl-btn-reapply').forEach(control => {
      if (busy) control.disabled = true;
    });
    if (!button) return;
    if (busy) {
      button.dataset.idleLabel = button.textContent;
      button.style.minWidth = `${Math.ceil(button.getBoundingClientRect().width)}px`;
      button.textContent = label;
      button.classList.add('is-busy');
      button.setAttribute('aria-busy', 'true');
    } else {
      button.textContent = button.dataset.idleLabel || button.textContent;
      button.classList.remove('is-busy');
      button.removeAttribute('aria-busy');
      button.style.minWidth = '';
    }
  }

  function _restoreHistoryButtonStates() {
    Object.values(_featureBlocks).forEach(fb => fb.actions.forEach(_forceActionButtonState));
  }

  function _linkedHistoryActions(action) {
    if (!action.dbHistoryId) return [action];
    return Object.values(_featureBlocks)
      .flatMap(fb => fb.actions)
      .filter(item => String(item.dbHistoryId) === String(action.dbHistoryId));
  }

  function _rebuildLinkedHistoryActions(actions) {
    const stores = new Map();
    actions.forEach(action => {
      const fb = _featureBlocks[action.scope];
      if (!fb) return;
      const store = action.subKey && fb.subBlocks ? fb.subBlocks[action.subKey] : fb;
      stores.set(`${action.scope}:${action.subKey || 'parent'}`, { store, fb });
    });
    stores.forEach(({ store, fb }) => {
      _rebuildActionStore(store, fb);
      _updateFeatureCount(fb);
      if (store !== fb) _updateSubFeatureCount(store);
    });
  }

  async function _undoAction(actionId, button) {
    if (_historyMutationInProgress) return;
    _historyMutationInProgress = true;
    _setHistoryMutationBusy(button, true, 'Undoing…');
    if (window._ido.operationProgress) window._ido.operationProgress.start('Undo', 'Replaying active actions from the immutable source');
    let succeeded = false;
    try {
    for (const scope in _featureBlocks) {
      const fb = _featureBlocks[scope];
      const action = fb.actions.find(a => a.id === actionId);
      if (!action) continue;
      const handledByServer = await _applyManifestOnServer(action, 'reverse');
      if (!handledByServer) {
        if (window._ido.operationProgress) window._ido.operationProgress.finish('error', 'Undo failed — nothing changed');
        return;
      }

      // Keep the ledger honest: remove exactly the entries THIS action
      // contributed (or restore whatever value was there before it, if it
      // changed an existing key) — otherwise a later, unrelated Apply would
      // recompute from raw + the full ledger and silently resurrect this
      // action's effect, even though the dataset snapshot and the Change Log
      // card both correctly show it as undone.
      if (action.ownEntries && action.ownEntries.length) {
        try {
          window._ido.appliedControlEntries = window._ido._applyLedgerDiff(action.ownEntries, 'reverse');
        } catch (e) {
          console.warn('ledger reverse update failed after server undo:', e);
        }
      }

      // Status flips in place — the action stays at its original position
      // and keeps its original Action #N. Nothing is removed or renumbered.
      const linkedActions = _linkedHistoryActions(action);
      linkedActions.forEach(item => { item.undone = true; });
      _rebuildLinkedHistoryActions(linkedActions);
      window.dispatchEvent(new CustomEvent('indataout:undoFromProgressBtn'));
      succeeded = true;
      if (window._ido.operationProgress) {
        window._ido.operationProgress.finish('success', 'Undone — dataset and Change Log refreshed');
        window._ido.operationProgress.hideLater();
      }
      return;
    }
    } finally {
      _historyMutationInProgress = false;
      _setHistoryMutationBusy(button, false);
      _restoreHistoryButtonStates();
      if (!succeeded && button && button.isConnected) button.focus();
    }
  }

  async function _reapplyAction(actionId, button) {
    if (_historyMutationInProgress) return;
    _historyMutationInProgress = true;
    _setHistoryMutationBusy(button, true, 'Reapplying…');
    if (window._ido.operationProgress) window._ido.operationProgress.start('Reapply', 'Replaying active actions from the immutable source');
    let succeeded = false;
    try {
    for (const scope in _featureBlocks) {
      const fb = _featureBlocks[scope];
      const action = fb.actions.find(a => a.id === actionId);
      if (!action) continue;
      const handledByServer = await _applyManifestOnServer(action, 'forward');
      if (!handledByServer) {
        if (window._ido.operationProgress) window._ido.operationProgress.finish('error', 'Reapply failed — nothing changed');
        return;
      }

      if (action.ownEntries && action.ownEntries.length) {
        try {
          window._ido.appliedControlEntries = window._ido._applyLedgerDiff(action.ownEntries, 'forward');
        } catch (e) {
          console.warn('ledger forward update failed after server reapply:', e);
        }
      }

      const linkedActions = _linkedHistoryActions(action);
      linkedActions.forEach(item => { item.undone = false; });
      _rebuildLinkedHistoryActions(linkedActions);
      succeeded = true;
      if (window._ido.operationProgress) {
        window._ido.operationProgress.finish('success', 'Reapplied — dataset and Change Log refreshed');
        window._ido.operationProgress.hideLater();
      }
      return;
    }
    } finally {
      _historyMutationInProgress = false;
      _setHistoryMutationBusy(button, false);
      _restoreHistoryButtonStates();
      if (!succeeded && button && button.isConnected) button.focus();
    }
  }

  // Build one operation card under its owning feature or sub-feature.
  function _pushSingleAction(fb, subKey, label, snapBefore, snapAfter, ops, previewHTML, opts) {
    opts = opts || {};
    const smartLabel  = buildSmartLabel(label, ops);
    const actionStore = subKey ? _getOrCreateSubfeatureBlock(fb, opts.scope, subKey) : fb;

    const action = {
      id:          ++_actionSeq,
      label:       smartLabel,
      previewHTML: previewHTML || '',
      operations:  ops,
      snapBefore:  snapBefore || null,
      snapAfter:   snapAfter  || null,
      ts:          opts.ts || Date.now(),
      resumed:     !!opts.resumed,
      undone:      !!opts.undone,
      dbHistoryId: opts.dbHistoryId || null,
      hasManifest: !!opts.hasManifest,
      ownEntries:  opts.ownEntries || null,
      preview:     opts.preview || null,
      scope:       opts.scope,
      subKey,
    };

    fb.actions.push(action);
    _updateFeatureCount(fb);
    if (actionStore !== fb) {
      actionStore.actions.push(action);
      _updateSubFeatureCount(actionStore);
    }

    const actionIndex = actionStore === fb ? fb.actions.length - 1 : actionStore.actions.length - 1;
    const card = _buildActionCard(action, actionIndex, fb);
    const targetEl = actionStore.actionsEl || fb.actionsEl;
    const firstSubBlock = actionStore === fb && fb.subBlocks ? Object.values(fb.subBlocks)[0]?.el : null;
    if (firstSubBlock) targetEl.insertBefore(card, firstSubBlock);
    else targetEl.appendChild(card);
    return { action, card };
  }

  // Shared by the live Apply path AND Resume rehydration — the ONE place that
  // decides how a set of operations becomes Change Log action card(s). Used to
  // live duplicated (and drifting) between pushAppliedHistory() and
  // rehydrateResumedHistory(); now there is exactly one implementation.
  //
  // A section Apply can contain several operations. Each operation gets its
  // own visual card, while cards from that Apply retain the same server history
  // ID so Undo/Reapply continues through the authoritative replay path.
  function _fileOperationsIntoChangeLog(scope, label, snapBefore, snapAfter, operations, previewHTML, opts) {
    opts = opts || {};
    scope = canonicalHistoryScope(scope);
    const operationList = (operations || []).filter(Boolean);
    if (!operationList.length) {
      if (!SCOPE_META[scope]) scope = 'dataIntegrityGroup';
      const fb = _getOrCreateFeatureBlock(scope);
      const res = _pushSingleAction(fb, null, label, snapBefore, snapAfter, [], previewHTML, Object.assign({}, opts, { scope }));
      return { cards: [res.card], actions: [res.action] };
    }

    const cards = [];
    const actions = [];
    operationList.forEach(op => {
      const detectedScope = canonicalHistoryScope(scopeForOperations([op], label));
      const activeScope = opts.activePanelId ? canonicalHistoryScope(opts.activePanelId) : null;
      const activeOwnsOperation = !!(activeScope && window._ido.SCOPE_OP_MAP &&
        window._ido.SCOPE_OP_MAP[activeScope] &&
        window._ido.SCOPE_OP_MAP[activeScope].has(op.action));
      let operationScope = activeOwnsOperation ? activeScope : detectedScope;
      if (!SCOPE_META[operationScope]) operationScope = scope;
      if (!SCOPE_META[operationScope]) operationScope = 'dataIntegrityGroup';
      const fb = _getOrCreateFeatureBlock(operationScope);
      const meta = (CHANGE_LOG_SUBFEATURES[operationScope] || {})[op.action];
      const actionOpts = Object.assign({}, opts, {
        scope: operationScope,
        preview: _previewForOperation(opts.preview, op),
      });
      const res = _pushSingleAction(
        fb,
        meta ? meta.key : null,
        _operationTitle(op),
        snapBefore,
        snapAfter,
        [op],
        operationList.length === 1 ? previewHTML : '',
        actionOpts,
      );
      cards.push(res.card);
      actions.push(res.action);
    });
    return { cards, actions };
  }

  // Returns the array of action objects created. Callers use this to attach the
  // DB row id once logActionToDB() resolves (needed for per-action Undo/Reapply sync).
  /**
   * Render server-authored history evidence into the Change Log.
   *
   * The function accepts legacy snapshot arguments for display compatibility,
   * but committed cards are driven by backend history IDs and stored preview
   * metadata returned by Apply/Resume.
   */
  function pushAppliedHistory(label, snapBefore, snapAfter, operations, forcedScope, opts) {
    const actionOperations = (operations || []).filter(hasMeaningfulChange);
    if (!actionOperations.length) return [];
    const previewHTML = window._ido._lastStepPreviewHTML || '';
    window._ido._lastStepPreviewHTML = '';
    // Stash it here too — logActionToDB() runs immediately after this call and
    // needs the same HTML to persist it to the DB, but by then _lastStepPreviewHTML
    // has already been cleared above.
    window._ido._lastLoggedPreviewHTML = previewHTML;
    const actualSnapAfter = snapAfter || _captureSnap();

    // forcedScope = the panel the user actually clicked Apply on (captured before the async call).
    // Falls back to operation-based detection, then currentHistoryScope.
    // This prevents wrong-panel filing when operations is empty or contains only carry-over ops.
    const scope = forcedScope || scopeForOperations(actionOperations, label);
    // opts.ownEntries (when the caller already knows exactly what this apply
    // added/changed in the ledger) is threaded straight into the action at
    // creation time — NOT patched in later once logActionToDB()'s promise
    // resolves. Without this, action.ownEntries stayed null for however long
    // that DB round-trip took, so an Undo click in that window would find no
    // ownEntries to work from and silently fail to revert the ledger.
    const { cards, actions } = _fileOperationsIntoChangeLog(scope, label, snapBefore, actualSnapAfter, actionOperations, previewHTML, opts);

    const lastCard = cards[cards.length - 1];
    if (lastCard) lastCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    return actions;
  }

  // ── Compatibility stubs ───────────────────────────────────────────────
  function _initializeChangeLogTree() {
    if (!changeLogBlocksContainer) return;
    ['dataIntegrityGroup','missingValuesGroup','dataTypesGroup','outlierGroup','datasetEditorSlide','dropSelectedColumnsGroup','customCodeGroup'].forEach(scope => {
      const fb = _getOrCreateFeatureBlock(scope);
      if (PERMANENT_SUBFEATURES[scope]) {
        if (fb.emptyEl) fb.emptyEl.style.display = 'none';
        PERMANENT_SUBFEATURES[scope].forEach(subKey => _getOrCreateSubfeatureBlock(fb, scope, subKey));
      }
    });
  }

  _initializeChangeLogTree();

  // ── Expose on bridge ──────────────────────────────────────────────────
  window._ido.pushAppliedHistory      = pushAppliedHistory;
  window._ido.renderScopedHistory     = function() {};
  window._ido._featureBlocks          = _featureBlocks;
  window._ido.getAppliedHistoryLength = () => Object.values(_featureBlocks).reduce((s,fb) => s + fb.actions.length, 0);
  window._ido.getUndoneHistoryLength  = () => Object.values(_featureBlocks).reduce((s,fb) => s + fb.actions.filter(a => a.undone).length, 0);

  // ── Notebook Cell Workspace ───────────────────────────────────────────
  const nbCellList      = document.getElementById('nbCellList');
  const nbAddCellBtn    = document.getElementById('nbAddCellBtn');
  const nbWorkspaceStat = document.getElementById('nbWorkspaceStatus');

  if (!nbCellList) return; // custom code not enabled

  let cells = [];
  let nextCellId = 1;

  function getFilename() {
    const inp = document.querySelector('input[name="filename"]');
    return inp ? inp.value : '';
  }

  function getAppliedFd() {
    const fn = getFilename();
    if (window._ido.mapToFormData) {
      const fd = window._ido.mapToFormData(new Map(window._ido.appliedControlEntries), 'section');
      if (fn && !fd.has('filename')) fd.append('filename', fn);
      if (window._ido.addSelectedOutliersToFormData) window._ido.addSelectedOutliersToFormData(fd);
      return fd;
    }
    const fd = new FormData();
    if (fn) fd.append('filename', fn);
    return fd;
  }

  // ── output rendering ─────────────────────────────────────────────────
  function renderBlock(block) {
    if (!block) return '';
    if (block.type === 'print') {
      return `<div class="nb-out-print"><pre class="nb-out-pre">${_esc(block.text)}</pre></div>`;
    }
    if (block.type === 'scalar') {
      const v = block.value === null ? 'NaN' : String(block.value);
      return `<div class="nb-out-scalar"><span class="nb-out-scalar-val">${_esc(v)}</span></div>`;
    }
    if (block.type === 'series' || block.type === 'dataframe') {
      const cols = block.columns || [];
      const rows = block.rows || [];
      const total = block.total_rows || rows.length;
      const typeLabel = block.type === 'series'
        ? `Series <strong>${_esc(block.name || '')}</strong> — ${total} row${total !== 1 ? 's' : ''}`
        : `DataFrame — ${total} row${total !== 1 ? 's' : ''} × ${block.total_cols || cols.length} col${(block.total_cols||cols.length)!==1?'s':''}`;
      const trunc = rows.length < total ? ` <span class="nb-out-trunc">(showing first ${rows.length})</span>` : '';
      return `<div class="nb-out-table-wrap">
        <div class="nb-out-table-label">${typeLabel}${trunc}</div>
        <div class="nb-out-table-scroll">
          <table class="data-table nb-out-table">
            <thead><tr>${cols.map(c=>`<th>${_esc(c)}</th>`).join('')}</tr></thead>
            <tbody>${rows.map(row=>'<tr>'+cols.map(c=>{
              const v = row[c]===null||row[c]===undefined?'':row[c];
              return `<td title="${_esc(v)}">${_esc(v)}</td>`;
            }).join('')+'</tr>').join('')}</tbody>
          </table>
        </div>
      </div>`;
    }
    return '';
  }

  // ── cell DOM builder ─────────────────────────────────────────────────
  function buildCellEl(cell) {
    const el = document.createElement('div');
    el.className = 'nb-cell';
    el.dataset.cellId = cell.id;
    el.innerHTML = `
      <div class="nb-cell-gutter">
        <span class="nb-cell-num" id="nb-num-${cell.id}">[ ]</span>
      </div>
      <div class="nb-cell-body">
        <div class="nb-cell-toolbar">
          <div class="nb-cell-toolbar-left">
            <button type="button" class="nb-btn-run"  data-action="run"   title="Run cell (Shift+Enter)">▶ Run</button>
            <button type="button" class="nb-btn-apply" data-action="apply" title="Apply cell — commits df changes">⬆ Apply</button>
          </div>
          <div class="nb-cell-toolbar-right">
            <button type="button" class="nb-btn-icon" data-action="move-up"   title="Move up">↑</button>
            <button type="button" class="nb-btn-icon" data-action="move-down" title="Move down">↓</button>
            <button type="button" class="nb-btn-icon nb-btn-delete" data-action="delete" title="Delete cell">✕</button>
          </div>
        </div>
        <textarea class="nb-cell-textarea" spellcheck="false" placeholder="# Write Pandas code here
df.isna().sum()">${_esc(cell.code)}</textarea>
        <div class="nb-cell-output" id="nb-out-${cell.id}"></div>
      </div>`;

    const ta = el.querySelector('.nb-cell-textarea');
    ta.addEventListener('input', () => autoResize(ta));
    autoResize(ta);

    ta.addEventListener('keydown', e => {
      if (e.key === 'Enter' && e.shiftKey) { e.preventDefault(); runCell(cell.id); }
    });

    el.querySelectorAll('[data-action]').forEach(btn => {
      btn.addEventListener('click', () => {
        const action = btn.dataset.action;
        if (action === 'run')        runCell(cell.id);
        else if (action === 'apply') applyCell(cell.id);
        else if (action === 'move-up')   moveCellUp(cell.id);
        else if (action === 'move-down') moveCellDown(cell.id);
        else if (action === 'delete')    deleteCell(cell.id);
      });
    });

    return el;
  }

  function autoResize(ta) {
    ta.style.height = 'auto';
    ta.style.height = Math.max(80, ta.scrollHeight) + 'px';
  }

  function getCellCode(id) {
    const el = nbCellList.querySelector(`[data-cell-id="${id}"] .nb-cell-textarea`);
    return el ? el.value.trim() : '';
  }

  function setCellState(id, state, numLabel) {
    const num = document.getElementById(`nb-num-${id}`);
    if (!num) return;
    if (state === 'running') { num.textContent = '[*]'; num.className = 'nb-cell-num nb-num-running'; }
    else if (state === 'done')  { num.textContent = numLabel || '[✓]'; num.className = 'nb-cell-num nb-num-done'; }
    else if (state === 'error') { num.textContent = '[!]'; num.className = 'nb-cell-num nb-num-error'; }
    else { num.textContent = '[ ]'; num.className = 'nb-cell-num'; }
  }

  function setOutput(id, html) {
    const el = document.getElementById(`nb-out-${id}`);
    if (el) el.innerHTML = html;
  }

  function setCellApplied(id, applied) {
    const el = nbCellList.querySelector(`[data-cell-id="${id}"]`);
    if (!el) return;
    el.classList.toggle('nb-cell-applied', applied);
    const applyBtn = el.querySelector('[data-action="apply"]');
    if (applyBtn) {
      applyBtn.textContent = applied ? '✓ Applied' : '⬆ Apply';
      applyBtn.classList.toggle('nb-btn-applied', applied);
    }
  }

  function setWorkspaceStatus(msg, type) {
    if (!nbWorkspaceStat) return;
    nbWorkspaceStat.textContent = msg;
    nbWorkspaceStat.className = 'nb-workspace-status' + (type ? ' nb-ws-' + type : '');
  }

  function addCell(code) {
    const cell = { id: nextCellId++, code: code || '', outputs: [], state: 'idle', applied: false };
    cells.push(cell);
    const el = buildCellEl(cell);
    nbCellList.appendChild(el);
    el.querySelector('.nb-cell-textarea').focus();
    return cell.id;
  }

  function deleteCell(id) {
    if (cells.length <= 1) return;
    cells = cells.filter(c => c.id !== id);
    const el = nbCellList.querySelector(`[data-cell-id="${id}"]`);
    if (el) el.remove();
  }

  function moveCellUp(id) {
    const idx = cells.findIndex(c => c.id === id);
    if (idx <= 0) return;
    [cells[idx - 1], cells[idx]] = [cells[idx], cells[idx - 1]];
    rebuildCellList();
  }

  function moveCellDown(id) {
    const idx = cells.findIndex(c => c.id === id);
    if (idx < 0 || idx >= cells.length - 1) return;
    [cells[idx], cells[idx + 1]] = [cells[idx + 1], cells[idx]];
    rebuildCellList();
  }

  function rebuildCellList() {
    nbCellList.innerHTML = '';
    cells.forEach(cell => {
      cell.code = getCellCode(cell.id) || cell.code;
      nbCellList.appendChild(buildCellEl(cell));
      setOutput(cell.id, cell._outputHtml || '');
      if (cell.state) setCellState(cell.id, cell.state);
      if (cell.applied) setCellApplied(cell.id, true);
    });
  }

  async function runCell(id) {
    const code = getCellCode(id);
    if (!code) return;
    setCellState(id, 'running');
    setOutput(id, '<div class="nb-out-running">Running…</div>');
    try {
      const fd = window._ido.mapToFormData(new Map(), 'feature');
      fd.append('code', code);
      const resp = await fetch('/preview_custom_code', { method: 'POST', body: fd });
      const data = await resp.json();
      if (!data.success) throw new Error(data.error || 'Run failed');
      const blocks = data.output_blocks || [];
      const rowsBefore = data.rows_before || 0;
      const rowsAfter  = data.rows_after  || 0;
      const changed    = data.changed_cells || 0;
      let outHtml = blocks.length ? blocks.map(renderBlock).join('') :
        `<div class="nb-out-empty">Code ran — no output.</div>`;
      if (rowsBefore !== rowsAfter) {
        outHtml += `<div class="nb-out-notice nb-out-notice-info">ℹ️ Would change df: ${rowsBefore} → ${rowsAfter} rows. Click <strong>Apply</strong> to commit.</div>`;
      } else if (changed > 0) {
        outHtml += `<div class="nb-out-notice nb-out-notice-info">ℹ️ Would change ${changed} cell(s). Click <strong>Apply</strong> to commit.</div>`;
      }
      const cell = cells.find(c => c.id === id);
      if (cell) { cell.state = 'done'; cell._outputHtml = outHtml; }
      setOutput(id, outHtml);
      setCellState(id, 'done', '[✓]');
    } catch (err) {
      const errHtml = `<div class="nb-out-error"><strong>Error:</strong> ${_esc(err.message)}</div>`;
      const cell = cells.find(c => c.id === id);
      if (cell) { cell.state = 'error'; cell._outputHtml = errHtml; }
      setOutput(id, errHtml);
      setCellState(id, 'error');
    }
  }

  /**
   * Apply one trusted-local custom-code cell through the same Apply path.
   *
   * Custom code runs exactly as written on the backend. This function only
   * presents the returned structural/value preview and authoritative history.
   */
  async function applyCell(id) {
    const code = getCellCode(id);
    if (!code) { setWorkspaceStatus('Cell is empty.', 'error'); return; }
    setCellState(id, 'running');
    setWorkspaceStatus('Applying cell…', '');
    const firstLine = code.split('\n').find(l => l.trim() && !l.trim().startsWith('#')) || code.split('\n')[0];
    const histLabel = firstLine.trim().slice(0, 60);
    // Unique per-block key (not the old fixed 'custom_code_block' slot) — lets
    // multiple custom-code actions coexist in the ledger independently, and
    // gives Undo/Reapply something precise to remove/restore (see own_entries
    // below) instead of clobbering whichever custom code happened to run last.
    const codeKey = `custom_code_block_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    try {
      window._ido.appliedControlEntries = [...window._ido.appliedControlEntries, [codeKey, code]];
      const fd = window._ido.mapToFormData(new Map([[codeKey, code]]), 'feature');
      fd.append('persist_apply', '1');
      fd.append('operation_token', 'op_' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2, 10));
      fd.append('client_applied_at_ms', String(Date.now()));
      fd.append('action_name', 'Custom code: ' + histLabel);
      fd.append('action_scope', 'customCodeGroup');
      const snap = _captureSnap();
      const resp = await fetch('/preview_clean', { method: 'POST', body: fd });
      const data = await resp.json();
      if (!data.success) throw new Error(data.error || 'Apply failed');
      if (window._ido.refreshAuthoritativeProjectState &&
          !window._ido.refreshAuthoritativeProjectState(data)) return;
      if (window._ido.renderLivePreview)           window._ido.renderLivePreview(data.preview, data.report?.cleaned_columns, data.report?.operations, data.report?.preview_row_numbers);
      if (window._ido.applyWorkingColumns)         window._ido.applyWorkingColumns(data.report?.cleaned_columns);
      if (window._ido.syncDatasetEditorFromPreview) window._ido.syncDatasetEditorFromPreview(data.preview, data.report?.preview_row_numbers, data.report?.cleaned_columns);
      // syncDatasetEditorFromPreview (above) only updates the lightweight
      // preview table — the actual interactive CanvasGrid in the Dataset
      // Editor tab is a separate thing entirely and only refreshEditorCanvas
      // touches it. Every other Apply path (previewCleaning, applyDatasetEditsBtn)
      // already calls this; applyCell (custom code) was the one path that
      // didn't, so rows/values a custom code cell changed or removed never
      // showed up in the Dataset Editor until some other panel's apply
      // happened to trigger a refresh.
      if (window._ido.refreshEditorCanvas) {
        window._ido.refreshEditorCanvas(
          data.editor_columns || data.report?.cleaned_columns,
          data.editor_data || [],
          data.row_index_map,
          data.dataset_version
        );
      }
      if (window._ido.renderLiveMetrics)           window._ido.renderLiveMetrics(data.summary);
      if (window._ido.syncMissingEditorFromPreview) window._ido.syncMissingEditorFromPreview(data.preview, data.report?.preview_row_numbers, data.report?.cleaned_columns, data.summary, data.analysis);
      if (window._ido.updateMissingReviewState)    window._ido.updateMissingReviewState(data.summary);
      if (data.summary) window._ido.lastSummary = data.summary;
      if (window._ido.renderLiveLog)               window._ido.renderLiveLog(data.report?.operations, 'Applied: ' + histLabel);
      if (snap && window._ido.previewUndoStack)    window._ido.previewUndoStack.push(snap);
      if (window._ido.showDatasetSlide)            window._ido.showDatasetSlide('progressDatasetSlide');
      const prog = document.getElementById('progressDatasetSlide');
      if (prog) prog.scrollIntoView({ behavior: 'smooth', block: 'start' });
      if (window._ido.undoPreviewBtn) window._ido.undoPreviewBtn.disabled = false;
      const cell = cells.find(c => c.id === id);
      if (cell) { cell.applied = true; cell.state = 'done'; }
      setCellState(id, 'done', '[✓]');
      setCellApplied(id, true);
      const successHtml = `<div class="nb-out-notice nb-out-notice-success">✓ Applied to Change Log.</div>`;
      setOutput(id, successHtml);
      const c = cells.find(c => c.id === id);
      if (c) c._outputHtml = successHtml;
      setWorkspaceStatus('Applied.', 'success');
      // Exactly what this action added to the ledger — Undo removes this key,
      // Reapply restores it. See _diffLedger() for the general case; custom
      // code is simpler since the key is brand new (before: null). Computed
      // once and passed straight into pushAppliedHistory below so
      // action.ownEntries is populated immediately, not only once the DB
      // round-trip resolves.
      const customCodeOwnEntries = [{ key: codeKey, before: null, after: code }];
      // pushAppliedHistory with before+after snaps for independent reapply
      const serverHistory = data.history_entry || null;
      const serverConfig = serverHistory && serverHistory.action_config ? serverHistory.action_config : null;
      const createdActions = pushAppliedHistory(histLabel, snap, _captureSnap(), data.report?.operations, 'customCodeGroup', {
        ownEntries: customCodeOwnEntries,
        preview: serverConfig && serverConfig.preview,
        ts: Number(serverConfig && serverConfig.applied_at_ms) || undefined,
        dbHistoryId: serverHistory && serverHistory.id,
        hasManifest: !!(serverHistory && serverHistory.has_manifest),
      }) || [];
      // Persist to the DB-backed Change Log — without this call, custom code steps
      // are shown in the UI as "Applied" but vanish on resume/reload (never written
      // to cleaning_history). Mirrors what previewCleaning() does for every other Apply.
      if (!serverHistory || !serverHistory.id) throw new Error('Custom code action was not persisted to authoritative history.');
      createdActions.forEach(a => {
        a.dbHistoryId = serverHistory.id;
        a.hasManifest = !!serverHistory.has_manifest;
      });
    } catch (err) {
      window._ido.appliedControlEntries = window._ido.appliedControlEntries.filter(([k]) => k !== codeKey);
      setCellState(id, 'error');
      const errHtml = `<div class="nb-out-error"><strong>Apply error:</strong> ${_esc(err.message)}</div>`;
      setOutput(id, errHtml);
      const cell = cells.find(c => c.id === id);
      if (cell) { cell.state = 'error'; cell._outputHtml = errHtml; }
      setWorkspaceStatus('Apply failed: ' + err.message, 'error');
    }
  }

  // init
  addCell('');
  nbAddCellBtn.addEventListener('click', () => addCell(''));

  function renderLiveLog(operations, actionLabel) {
    if (window._ido.renderLiveLog) window._ido.renderLiveLog(operations, actionLabel);
  }

  // ── Resume: fully rehydrate previous session state ──────────────────
  /**
   * Rebuild visible UI state from server-rendered project history on resume.
   *
   * The server remains authoritative for action status, ordered replay, and the
   * current snapshot; this function only restores controls and rendered cards.
   */
  async function rehydrateResumedHistory() {
    const serverData = document.getElementById('indataout-server-data');
    if (!serverData) return;

    let history = [];
    try {
      const raw = serverData.getAttribute('data-resumed-history');
      if (raw) history = JSON.parse(raw);
    } catch (e) { return; }

    if (!Array.isArray(history) || !history.length) return;

    if (window._ido.operationProgress) {
      window._ido.operationProgress.start('Resume', `Restoring ${history.length} persisted action${history.length !== 1 ? 's' : ''}`);
    }

    // Reset state for fresh resume (handled in Step 4 below)

    const lastEntry   = history[history.length - 1];
    const lastConfig  = lastEntry.action_config || {};
    const finalEntries = [];
    const hasEntries   = false;

    const cleanForm = window._ido.cleanForm || document.getElementById('cleanForm');
    const filenameInput = cleanForm && cleanForm.querySelector('[name="filename"]');
    const fn = filenameInput ? filenameInput.value : '';

    // ── Step 1: Restore form toggles ────────────────────────────────
    if (lastConfig.formSnapshot && cleanForm && window._ido.restoreSnapshot) {
      window._ido.restoreSnapshot(cleanForm, lastConfig.formSnapshot);
    } else if (hasEntries && cleanForm) {
      const entryMap = new Map(finalEntries.map(([k,v]) => [k,v]));
      cleanForm.querySelectorAll('input[type="checkbox"], input[type="radio"], select, input[type="text"], input[type="number"], textarea').forEach(el => {
        if (!el.name) return;
        const val = entryMap.get(el.name);
        if (val === undefined) return;
        if (el.type === 'checkbox') el.checked = (val === 'on' || val === '1' || val === 'true');
        else if (el.type === 'radio') el.checked = (el.value === val);
        else el.value = val;
      });
    }

    if (hasEntries) {
      window._ido.appliedControlEntries = finalEntries.map(([k,v]) => [k,v]);
    }

    if (lastConfig.activePanelId && window._ido.showCleaningPanel) {
      window._ido.showCleaningPanel(lastConfig.activePanelId, false);
      window._ido.currentHistoryScope = lastConfig.activePanelId;
    }

    // ── Step 2: Get cleaned data state ──────────────────────────────
    // Always ask the server for the current persisted snapshot. New history
    // uses ordered action replay/snapshots; entries are only a legacy input.
    let replayData = null;

    if (fn) {
      try {
        const resp = await fetch('/resume_replay', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            filename: fn,
            dataset_version: window._ido.datasetVersion || 0,
          })
        });
        replayData = await resp.json();
        if (!replayData.ok) replayData = null;
      } catch(e) { console.warn('resume_replay failed:', e); }
    }

    // ── Step 3: Apply cleaned data state to UI ──────────────────────
    if (replayData && replayData.ok) {
      if (window._ido.refreshAuthoritativeProjectState) {
        window._ido.refreshAuthoritativeProjectState(replayData);
      }
      if (window._ido.refreshEditorCanvas) {
        window._ido.refreshEditorCanvas(replayData.editor_columns, replayData.editor_data, replayData.row_index_map, replayData.dataset_version);
      }
      if (window._ido.syncDatasetEditorFromPreview) window._ido.syncDatasetEditorFromPreview(replayData.preview, replayData.preview_row_numbers, replayData.editor_columns);
      if (window._ido.renderLiveMetrics) window._ido.renderLiveMetrics(replayData.summary);
      if (window._ido.syncMissingEditorFromPreview) window._ido.syncMissingEditorFromPreview(replayData.preview, replayData.preview_row_numbers, replayData.editor_columns, replayData.summary, replayData.analysis);
      if (window._ido.updateMissingReviewState) window._ido.updateMissingReviewState(replayData.summary);
      if (replayData.summary) window._ido.lastSummary = replayData.summary;
      renderLiveLog(replayData.report?.operations, 'Resumed — previous state restored');
    }

    // ── Step 4: Create per-feature action cards from DB history ─────────
    // Reset feature blocks for fresh resume
    Object.keys(_featureBlocks).forEach(k => delete _featureBlocks[k]);
    _actionSeq = 0;
    if (changeLogBlocksContainer) changeLogBlocksContainer.innerHTML = '';
    _initializeChangeLogTree();

    history.forEach((entry) => {
      const config     = entry.action_config || {};
      const label      = config.label || (entry.action_name || '').replace(/_/g, ' ');
      const rowsBefore = entry.rows_before || 0;
      const rowsAfter  = entry.rows_after  || 0;
      const cells      = entry.cells_changed || 0;
      const ts         = _historyTs(entry, config);
      const badge      = [];
      if (rowsBefore !== rowsAfter) badge.push(`${rowsBefore}→${rowsAfter} rows`);
      if (cells > 0) badge.push(`${cells} cells changed`);
      const smartLabel = label + (badge.length ? ` (${badge.join(', ')})` : '');
      const previewHTML = config.previewHTML || '';
      const allOps = (Array.isArray(config.operations) ? config.operations : []).filter(hasMeaningfulChange);
      const opts = {
        ts,
        resumed: true,
        undone: entry.status === 'undone',
        dbHistoryId: entry.id || null,
        // Manifests now live in their own DB table, keyed by history_id —
        // the server already resolved whether one actually exists (checking
        // the DB first, then falling back to legacy file/inline storage for
        // older rows — see app.py's get_cleaning_history JOIN and
        // _prune_missing_manifests) and puts that answer directly on the
        // entry itself. No need to guess from action_config here, which no
        // longer holds manifest data at all for anything logged after this change.
        hasManifest: !!entry.has_manifest,
        ownEntries: Array.isArray(config.own_entries) ? config.own_entries : null,
        preview: config.preview || null,
        activePanelId: config.activePanelId || null,
      };

      const scope = config.activePanelId || scopeForOperations(allOps, label);
      _fileOperationsIntoChangeLog(scope, smartLabel, null, null, allOps, previewHTML, opts);
    });

    // ── Step 5: Show resume note in liveActionLog ────────────────────────
    if (replayData?.ok) {
      renderLiveLog(replayData.report?.operations, `✓ Session resumed — ${history.length} step${history.length!==1?'s':''} restored`);
      if (window._ido.operationProgress) {
        window._ido.operationProgress.finish('success', 'Resumed — dataset and Change Log restored');
        window._ido.operationProgress.hideLater();
      }
    } else if (window._ido.operationProgress) {
      window._ido.operationProgress.finish('error', 'Resume failed — existing project state was not changed');
    }
  }

  // Restore previous session state if this is a resume
  rehydrateResumedHistory();

});
// ─────────────────────────────────────────────────────────────────────────────
// CANVAS GRID INIT + PROGRESS ROW HIGHLIGHTER
// CanvasGrid class is at top of this file — no external dependency needed
// ─────────────────────────────────────────────────────────────────────────────

// Called after each Apply to highlight which rows changed in progress table
function highlightChangedRows(changedRowNums, removedRowNums) {
  const tbody = document.getElementById('livePreviewTbody');
  if (!tbody) return;
  const rows = tbody.querySelectorAll('tr');
  rows.forEach(tr => {
    const cells = tr.querySelectorAll('td');
    // First td is the row number cell
    const rowNumCell = cells[0];
    if (!rowNumCell) return;
    const rn = parseInt(rowNumCell.textContent.trim(), 10);
    if (changedRowNums && changedRowNums.includes(rn)) {
      tr.style.background = '#052e16';
      tr.querySelectorAll('td').forEach(td => td.style.color = '#4ade80');
    }
    if (removedRowNums && removedRowNums.includes(rn)) {
      tr.style.background = '#2d0a0a';
      tr.querySelectorAll('td').forEach(td => {
        td.style.color = '#f87171';
        td.style.textDecoration = 'line-through';
      });
    }
  });
}
window.highlightChangedRows = highlightChangedRows;

document.addEventListener('DOMContentLoaded', () => {
  const serverEl = document.getElementById('indataout-server-data');
  if (!serverEl) return;

  function safeParse(attr, fallback) {
    try { const v = JSON.parse(serverEl.dataset[attr] || ''); return (v && (Array.isArray(v) ? v.length : Object.keys(v).length)) ? v : fallback; }
    catch { return fallback; }
  }

  const originalPreview = safeParse('originalPreview', []);
  const editorColumns   = safeParse('editorColumns', []);
  const rowIndexMap     = safeParse('rowIndexMap', {});
  // Bridge by reference (not a copy) — refreshEditorCanvas mutates this same
  // object in place via Object.assign, so window._ido.rowIndexMap always
  // reflects whatever's currently loaded, without needing a separate sync
  // point every time it changes.
  window._ido.rowIndexMap = rowIndexMap;

  // ── 1. Original Dataset — plain scrollable HTML table, no CanvasGrid ──────
  const origTableWrap = document.getElementById('originalCanvasWrap');
  if (origTableWrap && originalPreview.length && editorColumns.length) {
    const table = document.createElement('table');
    table.style.cssText = 'width:100%;border-collapse:collapse;font-size:13px;';
    const thead = document.createElement('thead');
    thead.style.cssText = 'position:sticky;top:0;background:var(--surface-2,#1e293b);z-index:1;';
    thead.innerHTML = '<tr>' + editorColumns.map(c =>
      `<th style="padding:7px 12px;text-align:left;font-weight:500;color:var(--text-muted);border-bottom:1px solid var(--border);white-space:nowrap;">${c}</th>`
    ).join('') + '</tr>';
    const tbody = document.createElement('tbody');
    originalPreview.forEach((row, i) => {
      const tr = document.createElement('tr');
      tr.style.background = i % 2 === 0 ? 'transparent' : 'var(--surface-1,rgba(255,255,255,0.02))';
      tr.innerHTML = editorColumns.map(c => {
        const v = row[c] === null || row[c] === undefined ? '' : row[c];
        return `<td style="padding:6px 12px;border-bottom:1px solid var(--border);color:var(--text-primary);white-space:nowrap;max-width:200px;overflow:hidden;text-overflow:ellipsis;" title="${String(v).replace(/"/g,'&quot;')}">${String(v)}</td>`;
      }).join('');
      tbody.appendChild(tr);
    });
    table.appendChild(thead);
    table.appendChild(tbody);
    origTableWrap.innerHTML = '';
    origTableWrap.appendChild(table);
  }

  // ── 2. Dataset Editor — CanvasGrid, loads all rows on page load ───────────
  const editorWrap    = document.getElementById('editorCanvasWrap');
  const hiddenInputs  = document.getElementById('editorHiddenInputs');
  const changeCountEl = document.getElementById('editorChangeCount');

  if (!editorWrap) return;

  function syncChanges(grid) {
    if (!hiddenInputs) return;
    hiddenInputs.innerHTML = '';
    const changes = grid.getChanges();
    changes.forEach(({ rowIdx, colKey, value }) => {
      const dfIdx = rowIndexMap[String(rowIdx)];
      const inp = document.createElement('input');
      inp.type  = 'hidden';
      inp.name  = `cell_update[${dfIdx !== undefined ? dfIdx : rowIdx}][${colKey}]`;
      inp.value = value;
      hiddenInputs.appendChild(inp);
    });
    if (changeCountEl) {
      changeCountEl.textContent = changes.length
        ? `${changes.length} change${changes.length !== 1 ? 's' : ''} pending`
        : '0 changes pending';
      changeCountEl.style.color = changes.length ? 'var(--accent,#f97316)' : '';
    }
    if (window._ido.syncDatasetEditStatus) window._ido.syncDatasetEditStatus();
    // Merge, don't replace: keep every cell_update[...] entry already
    // committed to the ledger from a PRIOR Apply. The cleaning pipeline
    // always recomputes from the raw uploaded file (never incrementally
    // from the last snapshot), so any previously-applied cell edit that
    // isn't resubmitted here would silently revert to its raw value the
    // next time anything is applied. Only the keys touched by THIS
    // session's dirty changes get added or overwritten.
    const map = new Map(window._ido.appliedControlEntries || []);
    changes.forEach(({ rowIdx, colKey, value }) => {
      const dfIdx = rowIndexMap[String(rowIdx)];
      map.set(`cell_update[${dfIdx !== undefined ? dfIdx : rowIdx}][${colKey}]`, value);
    });
    window._ido.appliedControlEntries = Array.from(map.entries());

    // Continuously sync Change Log: debounced live preview after each cell edit
    clearTimeout(syncChanges._debounceTimer);
    if (changes.length > 0) {
      syncChanges._debounceTimer = setTimeout(() => {
        if (window._ido.triggerPreviewCleaning) {
          const editorSlide = document.getElementById('datasetEditorSlide');
          window._ido.triggerPreviewCleaning('Dataset edits preview', editorSlide, 'manual', false);
        }
      }, 600);
    }
  }
  window._ido.syncEditorGridChanges = () => syncChanges(window._ido.editorGrid);

  // Init canvas with empty data — will be populated immediately below
  window._ido.editorGrid = new CanvasGrid(editorWrap, {
    editable: true,
    columns:  editorColumns,
    data:     [],
    onCellChange: () => syncChanges(window._ido.editorGrid),
  });

  // Load full dataset into editor on page load via get_editor_data
  (function loadEditorOnInit() {
    const filename = document.querySelector('input[name="filename"]')?.value;
    if (!filename || !editorColumns.length) return;
    const fd = new FormData();
    fd.append('filename', filename);
    const requestVersion = Number(window._ido.datasetVersion || 0);
    fd.append('dataset_version', String(requestVersion));
    fetch('/get_editor_data', { method: 'POST', body: fd })
      .then(r => r.json())
      .then(d => {
        if (!d.success || !d.data) return;
        const responseVersion = Number(d.dataset_version);
        if (Number.isFinite(responseVersion) && responseVersion < Number(window._ido.datasetVersion || 0)) return;
        window._ido.refreshEditorCanvas(d.columns || editorColumns, d.data, d.row_index_map, responseVersion);
      })
      .catch(e => console.warn('Editor init load failed:', e));
  })();

  // ── 3. After preview/Apply → update editor directly from response ─────────
  // Applies an /apply_manifest "patch" (changed_cells, removed_rows,
  // added_rows, removed_columns, added_columns — see build_manifest_patch in
  // app.py) directly to the live grid, instead of resending/re-rendering the
  // entire dataset. `newRowIndexMap` is the server's authoritative post-patch
  // position->label map (always O(rows), never O(rows x cols) — it's just
  // row labels, no cell data) and is what removed/added-row target positions
  // are computed from, since it's guaranteed correct regardless of how many
  // other actions are independently applied/undone.
  /**
   * Patch the Dataset Editor after a server-confirmed small mutation.
   *
   * ``newRowIndexMap`` is the authoritative mapping from displayed row position
   * to DataFrame row identity after the mutation.
   */
  window._ido.patchEditorCanvas = function (patch, newRowIndexMap) {
    const grid = window._ido.editorGrid;
    if (!grid || !patch) return;

    // Reverse-lookup (label -> CURRENT position) built from the map as it
    // stands BEFORE this patch — used for changed_cells and removed_rows,
    // which reference positions in the grid as it currently is.
    const labelToPos = {};
    Object.entries(rowIndexMap).forEach(([pos, label]) => { labelToPos[String(label)] = parseInt(pos, 10); });

    // 1. Cell value changes — cheap positional overwrite, no row count change.
    (patch.changed_cells || []).forEach(({ row, col, value }) => {
      const pos = labelToPos[String(row)];
      if (pos !== undefined) grid.patchRow(pos, col, value);
    });

    // 2. Column removals/additions (rare — structural manifests only).
    (patch.removed_columns || []).forEach(col => grid.removeColumn(col));
    (patch.added_columns || []).forEach(({ col, data }) => {
      // `data` is keyed by row label (string) — align to CURRENT grid order.
      const valuesByPosition = grid.data.map((_, pos) => {
        const label = rowIndexMap[String(pos)];
        return data[String(label)];
      });
      grid.addColumn(col, col, valuesByPosition);
    });

    // 3. Row removals — resolve every position from the PRE-patch map first,
    // then remove highest-position-first so earlier removals don't shift the
    // positions still queued up.
    const removePositions = (patch.removed_rows || [])
      .map(row => labelToPos[String(row)])
      .filter(p => p !== undefined)
      .sort((a, b) => b - a);
    removePositions.forEach(pos => grid.removeRow(pos));

    // 4. Row additions — resolve each row's FINAL position from the server's
    // authoritative new map, ascending order (see patchEditorCanvas doc above
    // for why ascending order is safe to insert at final target indices).
    if (newRowIndexMap && patch.added_rows && patch.added_rows.length) {
      const newLabelToPos = {};
      Object.entries(newRowIndexMap).forEach(([pos, label]) => { newLabelToPos[String(label)] = parseInt(pos, 10); });
      const toAdd = patch.added_rows
        .map(r => ({ pos: newLabelToPos[String(r.row)], data: r.data }))
        .filter(r => r.pos !== undefined)
        .sort((a, b) => a.pos - b.pos);
      toAdd.forEach(({ pos, data }) => grid.addRow(pos, data));
    }

    // Row identity has shifted — the position->label map is only valid going
    // forward once fully rebuilt from the server's authoritative version.
    if (newRowIndexMap) {
      Object.keys(rowIndexMap).forEach(k => delete rowIndexMap[k]);
      Object.assign(rowIndexMap, newRowIndexMap);
    }

    if (hiddenInputs) hiddenInputs.innerHTML = '';
    if (changeCountEl) { changeCountEl.textContent = '0 changes pending'; changeCountEl.style.color = ''; }
    window._ido._datasetEditorLedgerBaseline = (window._ido.appliedControlEntries || []).slice();
  };

  /**
   * Replace the Dataset Editor with server-provided page/current-state data.
   *
   * Used after Apply, Undo/Reapply, resume, and server paging. Browser rows are
   * editable UI state only until committed through the backend.
   */
  window._ido.refreshEditorCanvas = function (columns, data, newRowIndexMap, responseVersion) {
    if (!window._ido.editorGrid) return;
    const incomingVersion = Number(responseVersion);
    if (Number.isFinite(incomingVersion)) {
      const currentVersion = Number(window._ido.datasetVersion || 0);
      if (incomingVersion < currentVersion) return;
      window._ido.datasetVersion = incomingVersion;
      document.body.dataset.datasetVersion = String(incomingVersion);
      document.querySelectorAll('[name="dataset_version"]').forEach(input => {
        input.value = String(incomingVersion);
      });
    }
    if (newRowIndexMap) Object.assign(rowIndexMap, newRowIndexMap);
    const cols = columns || editorColumns;
    const rows = data || [];
    window._ido.editorGrid.setData(cols, rows);
    const rowCount = rows.length;
    const rowMeta = document.getElementById('editorRowMeta');
    const rowsLoaded = document.getElementById('editorRowsLoaded');
    if (rowMeta) rowMeta.textContent = `Editing ${rowCount} row${rowCount === 1 ? '' : 's'}`;
    if (rowsLoaded) rowsLoaded.textContent = `${rowCount} row${rowCount === 1 ? '' : 's'} loaded`;
    if (hiddenInputs) hiddenInputs.innerHTML = '';
    if (changeCountEl) { changeCountEl.textContent = '0 changes pending'; changeCountEl.style.color = ''; }
    // cell_update[...] entries are deliberately NOT stripped here anymore —
    // see syncChanges() above for why doing so silently lost previously-
    // committed cell edits. What DOES need to happen at this point (right
    // after a commit/undo/reapply/resume) is recording where the ledger
    // stands right now, so the *next* Dataset Editor apply's own_entries can
    // be computed as a diff against this exact baseline — not against
    // whatever's in the ledger at click time, which by then already
    // includes this session's new values and would produce an empty diff.
    window._ido._datasetEditorLedgerBaseline = (window._ido.appliedControlEntries || []).slice();
  };

  // ── 4. Edit toggle ───────────────────────────────────────────────────────
  window._ido.refreshCurrentEditorFromServer = async function () {
    const filename = document.querySelector('input[name="filename"]')?.value;
    if (!filename || !window._ido.editorGrid) return false;
    const fd = new FormData();
    fd.append('filename', filename);
    fd.append('dataset_version', String(window._ido.datasetVersion || 0));
    try {
      const response = await fetch('/get_editor_data', { method: 'POST', body: fd });
      const data = await response.json();
      if (!data.success || !data.data) return false;
      const responseVersion = Number(data.dataset_version);
      if (Number.isFinite(responseVersion) && responseVersion < Number(window._ido.datasetVersion || 0)) return false;
      window._ido.refreshEditorCanvas(data.columns || editorColumns, data.data, data.row_index_map, responseVersion);
      return true;
    } catch (err) {
      console.warn('Editor authoritative refresh failed:', err);
      return false;
    }
  };

  const editToggleBtn = document.getElementById('editorEditToggle');
  let editingEnabled = false;

  function setEditingEnabled(enabled) {
    editingEnabled = enabled;
    const grid = window._ido.editorGrid;
    if (grid) grid.opts.editable = enabled;
    if (editToggleBtn) {
      editToggleBtn.textContent = enabled ? '🔓 Editing ON' : '🔒 Enable Editing';
      editToggleBtn.style.background = enabled ? 'var(--accent,#f97316)' : '';
      editToggleBtn.style.color = enabled ? '#fff' : '';
    }
    const subtitle = document.querySelector('.dataset-editor-subtitle');
    if (subtitle) subtitle.textContent = enabled
      ? 'Click any cell to edit. Tab/Enter to navigate. Toggle off when done.'
      : 'Editing is off — toggle on to make changes.';
  }

  if (editToggleBtn) {
    editToggleBtn.addEventListener('click', () => setEditingEnabled(!editingEnabled));
  }
  // Start with editing off
  setEditingEnabled(false);

  // NOTE: "Apply Edited Cells" is wired up once, above, in the
  // `applyDatasetEditsBtn` handler (~line 2545). A second listener used to be
  // registered here on the same button (by id), which fired a *second*,
  // unfiltered /preview_clean request + Change Log push on every click —
  // that stray action (built without any SCOPE_OP_MAP filtering) is what let
  // fill_numeric bleed into the Dataset Editor's action card. Removed.

  // ── 6. Search bar — full highlight version ───────────────────────────────
  const searchInput  = document.getElementById('cellSearchInput');
  const searchColSel = document.getElementById('cellSearchColumn');
  const searchRowInp = document.getElementById('cellSearchRow');
  const searchBtn    = document.getElementById('cellSearchBtn');
  const clearBtn     = document.getElementById('clearCellSearchBtn');
  const searchStatus = document.getElementById('cellSearchStatus');

  function runSearch() {
    const grid = window._ido.editorGrid;
    if (!grid || !grid.data.length) return;
    const query  = (searchInput?.value || '').trim().toLowerCase();
    const colKey = searchColSel?.value || '';
    const rowNum = parseInt(searchRowInp?.value || '0', 10);

    // Clear previous highlights
    grid.searchHighlights = null;
    grid._dirty = true;

    if (!query && !colKey && !rowNum) {
      if (searchStatus) searchStatus.textContent = '';
      return;
    }

    const matches = [];  // { rowIdx, colIdx }
    const MAX = 500;

    for (let r = 0; r < grid.data.length; r++) {
      if (rowNum && r + 1 !== rowNum) continue;
      for (let c = 0; c < grid.columns.length; c++) {
        if (colKey && grid.columns[c].key !== colKey) continue;
        const val = String(grid.data[r][grid.columns[c].key] ?? '').toLowerCase();
        if (!query || val.includes(query)) {
          matches.push({ rowIdx: r, colIdx: c });
          if (matches.length >= MAX) break;
        }
      }
      if (matches.length >= MAX) break;
    }

    if (matches.length > 0) {
      // Highlight all matches
      grid.searchHighlights = new Set(matches.map(m => `${m.rowIdx}:${grid.columns[m.colIdx].key}`));
      grid._dirty = true;
      // Scroll to and select first match
      const first = matches[0];
      grid.sel = { row: first.rowIdx, col: first.colIdx };
      grid.scroller.scrollTop = Math.max(0, grid.opts.headerHeight + first.rowIdx * grid.opts.rowHeight - 120);
      if (searchStatus) searchStatus.textContent = `${matches.length} match${matches.length !== 1 ? 'es' : ''}${matches.length >= MAX ? ' (first 500)' : ''}.`;
    } else {
      if (searchStatus) searchStatus.textContent = 'No match found.';
    }
  }

  if (searchBtn) searchBtn.addEventListener('click', runSearch);
  if (clearBtn)  clearBtn.addEventListener('click', () => {
    if (searchInput)  searchInput.value  = '';
    if (searchColSel) searchColSel.value = '';
    if (searchRowInp) searchRowInp.value = '';
    if (searchStatus) searchStatus.textContent = '';
    const grid = window._ido.editorGrid;
    if (grid) { grid.searchHighlights = null; grid._dirty = true; }
  });
  [searchInput, searchColSel, searchRowInp].forEach(el => {
    if (el) el.addEventListener('keydown', e => { if (e.key === 'Enter') runSearch(); });
  });
});
