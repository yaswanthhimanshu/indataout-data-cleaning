// static/script.js
document.addEventListener('DOMContentLoaded', () => {
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  const chooseBtn = document.getElementById('chooseBtn');
  const uploadForm = document.getElementById('uploadForm');
  const advToggle = document.getElementById('advToggle');
  const advancedPanel = document.getElementById('advanced');

  if (!dropZone || !fileInput || !uploadForm) return;

  const allowed = ['csv','tsv','txt','xlsx','xls','json'];

  function escapeHtml(s) {
    return String(s).replace(/[&<>"'`=\/]/g, function (c) {
      return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;','/':'&#x2F;','`':'&#x60;','=':'&#x3D;'}[c];
    });
  }

  function showFilename(name) {
    const content = dropZone.querySelector('.drop-content');
    if (!content) return;
    let el = content.querySelector('.file-name');
    if (!el) {
      el = document.createElement('div');
      el.className = 'file-name';
      content.insertBefore(el, content.firstChild);
    }
    el.innerHTML = 'Uploaded: <strong>' + escapeHtml(name) + '</strong>';
  }

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
    uploadForm.submit();
  });

  // Keyboard activation
  dropZone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      fileInput.click();
    }
  });

  // Advanced toggle
  if (advToggle && advancedPanel) {
    advToggle.addEventListener('click', () => {
      const hidden = window.getComputedStyle(advancedPanel).display === 'none';
      advancedPanel.style.display = hidden ? 'block' : 'none';
      advToggle.classList.toggle('active', hidden);
      // Enable inputs when shown
      advancedPanel.querySelectorAll('input, select').forEach(el => {
        if (hidden) el.removeAttribute('disabled');
      });
    });
  }
});
// static/script.js