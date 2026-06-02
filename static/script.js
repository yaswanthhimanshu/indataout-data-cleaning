// static/script.js
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

  const missingCellInputs = Array.from(document.querySelectorAll('.missing-cell-edit input'));
  const missingEditorStatus = document.getElementById('missingEditorStatus');
  if (missingCellInputs.length && missingEditorStatus) {
    function syncMissingEditorStatus() {
      const typedCount = missingCellInputs.filter(input => input.value.trim() !== '').length;
      missingEditorStatus.textContent = typedCount > 0
        ? `${typedCount} typed value${typedCount === 1 ? '' : 's'} ready. These update the dataset first, then cleaning continues.`
        : 'No typed values yet.';
      missingEditorStatus.classList.toggle('has-edits', typedCount > 0);
    }
    missingCellInputs.forEach(input => input.addEventListener('input', syncMissingEditorStatus));
    syncMissingEditorStatus();
  }

  const cleanForm = document.getElementById('cleanForm');
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
  const missingEditorRows = Array.from(document.querySelectorAll('.missing-editor-table tbody tr'));
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
  let appliedControlEntries = [];
  let currentWorkingColumns = null;
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
  }

  datasetTabs.forEach(tab => {
    tab.addEventListener('click', () => showDatasetSlide(tab.dataset.datasetSlide));
  });

  function showCleaningPanel(panelId, shouldScroll = true) {
    if (!panelId || optionGroups.length === 0) return;
    optionGroups.forEach(group => {
      group.classList.toggle('active-panel', group.id === panelId);
    });
    panelLinks.forEach(link => {
      link.classList.toggle('active', link.dataset.panelTarget === panelId);
    });
    const selected = document.getElementById(panelId);
    if (selected && shouldScroll) selected.scrollIntoView({ behavior: 'smooth', block: 'start' });
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
    const edited = datasetEditInputs.filter(input => input.value !== input.dataset.original);
    const renamed = datasetColumnInputs.filter(input => input.value !== input.dataset.original);
    const parts = [];
    if (edited.length > 0) parts.push(`${edited.length} edited cell${edited.length === 1 ? '' : 's'}`);
    if (renamed.length > 0) parts.push(`${renamed.length} renamed column${renamed.length === 1 ? '' : 's'}`);
    datasetEditStatus.textContent = edited.length > 0
      ? `${parts.join(' and ')} ready. Apply edits to update Cleaning Work Progress.`
      : 'No edited cells yet.';
    if (edited.length === 0 && renamed.length > 0) {
      datasetEditStatus.textContent = `${parts.join(' and ')} ready. Apply edits to update Cleaning Work Progress.`;
    }
    datasetEditStatus.classList.toggle('has-edits', edited.length > 0 || renamed.length > 0);
  }

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

  function syncDatasetEditorFromPreview(preview, rowNumbers, columns) {
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
      rowEl.style.display = availableRows.size === 0 || availableRows.has(String(rowInput.dataset.rowNumber || '')) ? '' : 'none';
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

  function syncMissingEditorFromPreview(preview, rowNumbers, columns, summary) {
    if (!missingEditorRows.length) return;
    const rowMap = rowMapFromPreview(preview, rowNumbers);
    const availableRows = new Set(rowMap.keys());
    const remainingColumns = new Set(
      ((summary && summary.missing_by_column_remaining) || []).map(item => normalizeColumnName(item.column))
    );
    const noMissingRemain = summary && Number(summary.missing_remaining || 0) === 0;

    missingEditorRows.forEach(rowEl => {
      const rowNumber = rowEl.dataset.rowNumber ||
        rowEl.querySelector('[data-row-number]')?.dataset.rowNumber ||
        rowEl.querySelector('.missing-row-number')?.textContent.replace(/\D+/g, '');
      const row = rowMap.get(String(rowNumber || ''));
      rowEl.style.display = availableRows.size === 0 || row ? '' : 'none';
      if (!row) return;

      rowEl.querySelectorAll('[data-column]').forEach(cell => {
        const column = cell.dataset.column || '';
        const nextValue = valueFromWorkingRow(row, column);
        if (nextValue === undefined) return;
        const input = cell.querySelector('input');
        const columnStillMissing = remainingColumns.has(normalizeColumnName(column));

        if (input) {
          input.value = nextValue;
          if (nextValue || noMissingRemain || !columnStillMissing) {
            input.removeAttribute('name');
            input.disabled = true;
            input.classList.remove('is-edited');
            input.classList.add('is-synced');
            input.placeholder = nextValue ? '' : 'Handled';
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

  function clearCellSearchHighlights() {
    datasetEditInputs.forEach(input => input.classList.remove('is-search-match', 'is-search-focus'));
    if (cellSearchStatus) cellSearchStatus.textContent = 'Search cleared.';
  }

  function runCellSearch() {
    const query = cellSearchInput ? cellSearchInput.value.trim().toLowerCase() : '';
    const column = cellSearchColumn ? cellSearchColumn.value : '';
    const rowNumber = cellSearchRow ? cellSearchRow.value.trim() : '';
    clearCellSearchHighlights();
    const matches = datasetEditInputs.filter(input => {
      const valueOk = !query || input.value.toLowerCase().includes(query);
      const columnOk = !column || input.dataset.column === column;
      const rowOk = !rowNumber || input.dataset.rowNumber === rowNumber;
      return valueOk && columnOk && rowOk;
    });
    matches.forEach(input => input.classList.add('is-search-match'));
    if (matches.length > 0) {
      matches[0].classList.add('is-search-focus');
      matches[0].scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'center' });
      matches[0].focus({ preventScroll: true });
    }
    if (cellSearchStatus) {
      cellSearchStatus.textContent = matches.length > 0
        ? `${matches.length} matching cell${matches.length === 1 ? '' : 's'} highlighted.`
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
    });

    if (remaining === 0) {
      if (missingReviewTitle) missingReviewTitle.textContent = 'Missing Values Handled';
      if (missingReviewSubtitle) {
        missingReviewSubtitle.textContent =
          'Cleaning Work Progress has no true missing cells after the applied actions. The original blank-cell editor is hidden to avoid showing stale cells.';
      }
      if (missingColumnSummary) {
        missingColumnSummary.innerHTML = '<span class="missing-count-pill is-clear">0 missing cells remain in Cleaning Work Progress</span>';
      }
      if (wideMissingEditor) wideMissingEditor.style.display = 'none';
      if (missingReviewPanel) missingReviewPanel.classList.add('is-handled');
      return;
    }

    if (missingReviewTitle) missingReviewTitle.textContent = 'Remaining Missing Values';
    if (missingReviewSubtitle) {
      missingReviewSubtitle.textContent =
        `Cleaning Work Progress still has ${remaining} true missing cell${remaining === 1 ? '' : 's'} across ${remainingRows} row${remainingRows === 1 ? '' : 's'}.`;
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

  function capturePreviewState() {
    return {
      formSnapshot: cleanForm ? snapshotForm(cleanForm) : [],
      datasetEditorSnapshot: snapshotDatasetEditor(),
      missingReviewSnapshot: captureMissingReviewState(),
      appliedControlEntries: appliedControlEntries.map(([key, value]) => [key, value]),
      workingColumns: currentWorkingColumns ? currentWorkingColumns.slice() : null,
      thead: livePreviewTable ? livePreviewTable.querySelector('thead').innerHTML : '',
      tbody: livePreviewTable ? livePreviewTable.querySelector('tbody').innerHTML : '',
      metrics: liveMetrics ? liveMetrics.innerHTML : '',
      log: liveActionLog ? liveActionLog.innerHTML : '',
      code: liveCodeLog ? liveCodeLog.textContent : ''
    };
  }

  function restorePreviewState(state) {
    if (!state) return;
    if (cleanForm) restoreSnapshot(cleanForm, state.formSnapshot || []);
    appliedControlEntries = (state.appliedControlEntries || []).map(([key, value]) => [key, value]);
    restoreDatasetEditorSnapshot(state.datasetEditorSnapshot);
    restoreMissingReviewState(state.missingReviewSnapshot);
    applyWorkingColumns(state.workingColumns || null);
    if (livePreviewTable) {
      livePreviewTable.querySelector('thead').innerHTML = state.thead || '';
      livePreviewTable.querySelector('tbody').innerHTML = state.tbody || '';
    }
    if (liveMetrics) liveMetrics.innerHTML = state.metrics || '';
    if (liveActionLog) {
      liveActionLog.innerHTML = '<div class="live-log-undo">Undo applied: restored the previous Cleaning Work Progress preview.</div>' + (state.log || '');
    }
    if (liveCodeLog) liveCodeLog.textContent = state.code || '# Undo restored the previous preview.';
    showDatasetSlide('progressDatasetSlide');
    const progressSlide = document.getElementById('progressDatasetSlide');
    if (progressSlide) progressSlide.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function addSelectedOutliersToFormData(formData) {
    formData.delete('selected_outlier_rows');
    const section = document.getElementById('outlierPreviewSection');
    if (section && section.style.display !== 'none') {
      const checked = Array.from(document.querySelectorAll('.outlier-checkbox:checked'))
        .map(cb => parseInt(cb.dataset.row, 10))
        .filter(n => !isNaN(n));
      if (checked.length > 0) {
        formData.append('selected_outlier_rows', JSON.stringify(checked));
      }
    }
  }

  function collectScopedChanges(scopeRoot) {
    const set = new Map();
    const remove = new Set();
    if (!scopeRoot) return { set, remove };
    scopeRoot.querySelectorAll('[name]').forEach(el => {
      if (!el.name || el.disabled || el.type === 'button' || el.type === 'submit') return;
      if (el.name.startsWith('missing_cell[') && String(el.value || '').trim() === '') return;
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
      const typedValues = Array.from(group.querySelectorAll('input[name^="missing_cell["]')).some(input => input.value.trim() !== '');
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

    if (group.id === 'advancedGroup') {
      const hasCheckedToggle = Array.from(group.querySelectorAll('.toggle input[type="checkbox"]')).some(cb => cb.checked);
      if (!hasCheckedToggle) {
        return { ok: false, message: 'Turn on at least one Advanced Option before applying the section.' };
      }
    }

    return { ok: true, message: `Ready to apply ${sectionName}.` };
  }

  function mapToFormData(entryMap, scopeName) {
    const formData = new FormData();
    const filename = cleanForm ? cleanForm.querySelector('[name="filename"]') : null;
    if (filename) formData.append('filename', filename.value);
    formData.append('preview_scope', scopeName || 'feature');
    entryMap.forEach((value, key) => formData.append(key, value));
    return formData;
  }

  function buildScopedFormData(scopeRoot, scopeName) {
    const changes = collectScopedChanges(scopeRoot);
    const entryMap = mergeEntries(appliedControlEntries, changes);
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
    return formData;
  }

  function persistScopedControls(scopeRoot) {
    const changes = collectScopedChanges(scopeRoot);
    const entryMap = mergeEntries(appliedControlEntries, changes);
    addScopedFillDefaults(entryMap, changes);
    appliedControlEntries = Array.from(entryMap.entries());
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

  function renderLivePreview(preview, columnsFromServer, operations = [], rowNumbers = []) {
    if (!livePreviewTable) return;
    const columns = columnsFromServer && columnsFromServer.length
      ? columnsFromServer
      : (preview && preview.length ? Object.keys(preview[0]) : []);
    const highlights = liveCellHighlightSet(operations, rowNumbers);
    livePreviewTable.querySelector('thead').innerHTML = '<tr>' +
      columns.map(col => `<th>${escapeHtml(col)}</th>`).join('') +
      '</tr>';
    if (!preview || preview.length === 0) {
      livePreviewTable.querySelector('tbody').innerHTML =
        `<tr><td colspan="${Math.max(columns.length, 1)}">No rows remain after this preview.</td></tr>`;
      return;
    }
    livePreviewTable.querySelector('tbody').innerHTML = preview.slice(0, 100).map((row, rowIndex) => (
      '<tr>' + columns.map(col => {
        const value = row[col] === null || row[col] === undefined ? '' : row[col];
        const rowNumber = rowNumbers[rowIndex];
        const updated = highlights.has(`${String(rowNumber)}::${normalizeColumnName(col)}`);
        return `<td class="${updated ? 'live-cell-updated' : ''}" title="${escapeHtml(value)}">${escapeHtml(value)}</td>`;
      }).join('') + '</tr>'
    )).join('');
  }

  function describeOperation(op) {
    if (!op || !op.action) return 'Cleaning step applied.';
    if (op.action === 'manual_missing_cell_updates') return `Updated ${op.count} typed missing cell(s).`;
    if (op.action === 'manual_cell_updates') return `Updated ${op.count} edited dataset cell(s).`;
    if (op.action === 'manual_column_renames') return `Renamed ${op.count} dataset column(s).`;
    if (op.action === 'drop_empty_columns') return `Removed ${op.count || 0} fully empty column(s).`;
    if (op.action === 'drop_constant_columns') return `Removed ${op.count || 0} constant column(s).`;
    if (op.action === 'drop_duplicates') return `Removed ${op.removed_rows || 0} duplicate row(s).`;
    if (op.action === 'drop_missing_rows') return `Dropped ${op.rows_dropped || 0} row(s) with remaining missing values.`;
    if (op.action === 'fill_numeric') return `Filled ${op.filled_cells || 0} numeric missing cell(s).`;
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
      const filled = (op.details || []).map(item =>
        `<div class="live-log-detail">Column <code>${escapeHtml(item.column)}</code>: filled ${escapeHtml(item.filled)} cell(s) on rows ${formatList(item.rows || [])} with <code>${escapeHtml(item.value)}</code></div>`
      ).join('');
      const skipped = (op.skipped || []).map(item =>
        `<div class="live-log-detail">Column <code>${escapeHtml(item.column)}</code>: ${escapeHtml(item.missing)} cell(s) still missing because ${escapeHtml(item.reason)}</div>`
      ).join('');
      return filled + skipped || `<div class="live-log-detail">${escapeHtml(op.reason || 'No numeric cells filled.')}</div>`;
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
      return `<div class="live-log-detail">Removed uploaded rows: ${formatList(op.selected_rows || op.valid_indices_dropped || [])}</div>`;
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

  async function previewCleaning(actionLabel, scopeRoot = null, scopeName = 'all', persistApply = false) {
    if (!cleanForm || !livePreviewBtn) return;
    const formData = scopeRoot ? buildScopedFormData(scopeRoot, scopeName) : new FormData(cleanForm);
    if (!scopeRoot) addSelectedOutliersToFormData(formData);
    const previousPreviewState = capturePreviewState();
    livePreviewBtn.disabled = true;
    livePreviewBtn.textContent = 'Updating...';
    try {
      const response = await fetch('/preview_clean', { method: 'POST', body: formData });
      const data = await response.json();
      if (!data.success) throw new Error(data.error || 'Preview failed');
      if (persistApply && scopeRoot) persistScopedControls(scopeRoot);
      renderLivePreview(
        data.preview,
        data.report && data.report.cleaned_columns,
        data.report && data.report.operations,
        data.report && data.report.preview_row_numbers
      );
      applyWorkingColumns(data.report && data.report.cleaned_columns);
      syncDatasetEditorFromPreview(
        data.preview,
        data.report && data.report.preview_row_numbers,
        data.report && data.report.cleaned_columns
      );
      renderLiveMetrics(data.summary);
      syncMissingEditorFromPreview(
        data.preview,
        data.report && data.report.preview_row_numbers,
        data.report && data.report.cleaned_columns,
        data.summary
      );
      updateMissingReviewState(data.summary);
      renderLiveLog(data.report.operations, actionLabel);
      previewUndoStack.push(previousPreviewState);
      showDatasetSlide('progressDatasetSlide');
      const progressSlide = document.getElementById('progressDatasetSlide');
      if (progressSlide) progressSlide.scrollIntoView({ behavior: 'smooth', block: 'start' });
      lastSnapshot = snapshotForm(cleanForm);
      cleanForm.querySelectorAll('[data-undo-pending="1"]').forEach(el => delete el.dataset.undoPending);
    } catch (error) {
      if (liveActionLog) {
        liveActionLog.innerHTML = `<div class="live-log-empty">${escapeHtml(error.message)}</div>`;
      }
    } finally {
      livePreviewBtn.disabled = false;
      livePreviewBtn.textContent = 'Preview Current Section';
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

  if (cleanForm && livePreviewBtn) {
    cleanForm.addEventListener('change', rememberUndoState);
    cleanForm.addEventListener('input', rememberUndoState);
    livePreviewBtn.addEventListener('click', previewCurrentSection);
  }

  if (applyTypedPreviewBtn) {
    applyTypedPreviewBtn.addEventListener('click', () => {
      const editor = document.querySelector('.wide-missing-editor');
      const typedValues = editor
        ? Array.from(editor.querySelectorAll('input[name^="missing_cell["]')).filter(input => input.value.trim() !== '')
        : [];
      if (typedValues.length === 0) {
        if (liveActionLog) {
          liveActionLog.innerHTML = '<div class="live-log-empty">No typed missing-cell values to apply yet. Type into a highlighted blank cell first.</div>';
        }
        showDatasetSlide('progressDatasetSlide');
        return;
      }
      previewCleaning('Applied: typed missing-cell values', editor, 'manual', true);
    });
  }

  if (applyDatasetEditsBtn) {
    applyDatasetEditsBtn.addEventListener('click', () => {
      const editor = document.getElementById('datasetEditorSlide');
      const editedCells = datasetEditInputs.filter(input => input.value !== input.dataset.original);
      const renamedColumns = datasetColumnInputs.filter(input => input.value !== input.dataset.original);
      if (editedCells.length === 0 && renamedColumns.length === 0) {
        if (liveActionLog) {
          liveActionLog.innerHTML = '<div class="live-log-empty">No dataset edits to apply yet. Change a cell or column name first.</div>';
        }
        showDatasetSlide('progressDatasetSlide');
        return;
      }
      previewCleaning('Applied: edited dataset cells', editor, 'manual', true);
    });
  }

  window.indataoutPrepareAppliedControls = function () {
    if (!cleanForm) return;

    cleanForm.querySelectorAll('[data-final-disabled="1"]').forEach(el => {
      el.disabled = false;
      delete el.dataset.finalDisabled;
    });
    cleanForm.querySelectorAll('[data-applied-control="1"]').forEach(el => el.remove());

    cleanForm.querySelectorAll('[name]').forEach(el => {
      if (el.name === 'filename' || el.name === 'selected_outlier_rows') return;
      el.disabled = true;
      el.dataset.finalDisabled = '1';
    });

    appliedControlEntries.forEach(([key, value]) => {
      const input = document.createElement('input');
      input.type = 'hidden';
      input.name = key;
      input.value = value;
      input.dataset.appliedControl = '1';
      cleanForm.appendChild(input);
    });
  };

  sectionApplyBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const group = btn.closest('.opt-group');
      const readiness = sectionApplyReadiness(group);
      if (!readiness.ok) {
        showApplyBlocked(readiness.message);
        return;
      }
      if (liveActionLog) {
        liveActionLog.innerHTML = `<div class="live-log-empty">Applying ${escapeHtml(btn.dataset.sectionName || 'section')}...</div>`;
      }
      previewCleaning(`Applied section only: ${btn.dataset.sectionName || 'section'}`, group, 'section', true);
    });
  });

  document.querySelectorAll('.feature-apply-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const item = btn.closest('.opt-item');
      const featureName = item ? (item.querySelector('.opt-name')?.textContent || 'feature') : 'feature';
      const readiness = featureApplyReadiness(item);
      if (!readiness.ok) {
        showApplyBlocked(readiness.message);
        return;
      }
      if (liveActionLog) {
        liveActionLog.innerHTML = `<div class="live-log-empty">Applying ${escapeHtml(featureName)}...</div>`;
      }
      previewCleaning(`Applied feature only: ${featureName}`, item, 'feature', true);
    });
  });

  if (cleanForm && undoPreviewBtn) {
    undoPreviewBtn.addEventListener('click', () => {
      if (previewUndoStack.length > 0) {
        restorePreviewState(previewUndoStack.pop());
        lastSnapshot = snapshotForm(cleanForm);
        undoPreviewBtn.disabled = previewUndoStack.length === 0 && undoStack.length === 0;
        return;
      }
      const snapshot = undoStack.pop();
      if (!snapshot) return;
      restoreSnapshot(cleanForm, snapshot);
      lastSnapshot = snapshotForm(cleanForm);
      undoPreviewBtn.disabled = previewUndoStack.length === 0 && undoStack.length === 0;
      previewCleaning('Undo applied: restored the previous option state');
    });
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

        const formData = mapToFormData(new Map(appliedControlEntries), 'section');
        formData.delete('selected_outlier_rows');
        if (!formData.has('filename')) formData.append('filename', filename);
        formData.append('multiplier', '1.5'); // Default IQR multiplier

        const response = await fetch('/preview_outliers', {
          method: 'POST',
          body: formData
        });

        const data = await response.json();

        if (data.success) {
          // Show the outlier preview section
          outlierPreviewSection.style.display = 'block';
          
          // Populate the table with outlier data
          outlierTableBody.innerHTML = '';
          
          data.outliers.outlier_rows.forEach(row => {
            const context = Object.entries(row.row_data || {})
              .slice(0, 3)
              .map(([key, value]) => `${escapeHtml(key)}: ${escapeHtml(value)}`)
              .join(' | ');
            const tr = document.createElement('tr');
            tr.innerHTML = `
              <td><input type="checkbox" class="outlier-checkbox" data-row="${row.row_index}"></td>
              <td>${row.row_number || (row.row_position + 1)}</td>
              <td>${escapeHtml(row.column)}</td>
              <td>${row.value !== null ? row.value : 'N/A'}</td>
              <td>${context}</td>
            `;
            outlierTableBody.appendChild(tr);
          });
          
          // Update count info
          outlierCountInfo.textContent = 
            `Showing ${data.outliers.outlier_rows.length} of ${data.outliers.total_count} total outliers. Select rows to remove.`;
          if (outlierStatus) {
            outlierStatus.textContent = data.outliers.total_count > 0
              ? `${data.outliers.total_count} outlier candidate${data.outliers.total_count === 1 ? '' : 's'} found.`
              : 'No outliers found in the current working dataset.';
          }
          
          if (selectAllOutliers) selectAllOutliers.checked = false;
          if (selectAllOutliers && !selectAllOutliers.dataset.bound) {
            selectAllOutliers.dataset.bound = '1';
            selectAllOutliers.addEventListener('change', function() {
            const checkboxes = document.querySelectorAll('.outlier-checkbox');
            checkboxes.forEach(checkbox => checkbox.checked = this.checked);
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
        previewOutliersBtn.textContent = 'Preview Outliers';
      }
    });
  }


});
