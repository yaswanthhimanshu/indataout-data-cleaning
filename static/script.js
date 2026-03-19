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

  // Outlier preview functionality
  const previewOutliersBtn = document.getElementById('previewOutliersBtn');
  const outlierPreviewSection = document.getElementById('outlierPreviewSection');
  const outlierTableBody = document.getElementById('outlierTableBody');
  const selectAllOutliers = document.getElementById('selectAllOutliers');
  const removeSelectedOutliersBtn = document.getElementById('removeSelectedOutliersBtn');
  const outlierCountInfo = document.getElementById('outlierCountInfo');

  if (previewOutliersBtn) {
    previewOutliersBtn.addEventListener('click', async () => {
      const filename = document.querySelector('input[name="filename"]').value;
      if (!filename) {
        alert('Please upload a file first');
        return;
      }

      try {
        const formData = new FormData();
        formData.append('filename', filename);
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
            const tr = document.createElement('tr');
            tr.innerHTML = `
              <td><input type="checkbox" class="outlier-checkbox" data-row="${row.row_position}" data-position="${row.row_position}"></td>
              <td>${row.row_position}</td>
              <td>${row.column}</td>
              <td>${row.value !== null ? row.value : 'N/A'}</td>
            `;
            outlierTableBody.appendChild(tr);
          });
          
          // Update count info
          outlierCountInfo.textContent = 
            `Showing ${data.outliers.outlier_rows.length} of ${data.outliers.total_count} total outliers. Select rows to remove.`;
          
          // Add event listener for select all
          selectAllOutliers.addEventListener('change', function() {
            const checkboxes = document.querySelectorAll('.outlier-checkbox');
            checkboxes.forEach(checkbox => checkbox.checked = this.checked);
            
            // Update the selected rows in the form immediately
            updateSelectedOutlierRows();
          });
          
          // Add event listener for individual checkboxes
          document.addEventListener('change', function(e) {
            if (e.target.classList.contains('outlier-checkbox')) {
              updateSelectedOutlierRows();
            }
          });
          
          // Function to update selected outlier rows in the form
          function updateSelectedOutlierRows() {
            const selectedCheckboxes = document.querySelectorAll('.outlier-checkbox:checked');
            if (selectedCheckboxes.length > 0) {
              const selectedRows = Array.from(selectedCheckboxes).map(checkbox => 
                parseInt(checkbox.getAttribute('data-row'))
              );
              
              // Store selected rows in a hidden input field
              let selectedRowsInput = document.querySelector('input[name=\'selected_outlier_rows\']');
              if (!selectedRowsInput) {
                selectedRowsInput = document.createElement('input');
                selectedRowsInput.type = 'hidden';
                selectedRowsInput.name = 'selected_outlier_rows';
                document.getElementById('cleanForm').appendChild(selectedRowsInput);
              }
              selectedRowsInput.value = JSON.stringify(selectedRows);
              
              console.log('Updated selected outlier rows in form:', selectedRows);
            } else {
              // If no rows selected, remove the input field if it exists
              const existingInput = document.querySelector('input[name=\'selected_outlier_rows\']');
              if (existingInput) {
                existingInput.remove();
              }
              console.log('Removed selected outlier rows input field');
            }
          }
          
          // The main clean button will handle selected outliers via form submission handler
        
        } else {
          alert('Error detecting outliers: ' + data.error);
        }
      } catch (error) {
        console.error('Error:', error);
        alert('Error detecting outliers');
      }
    });
  }


});
