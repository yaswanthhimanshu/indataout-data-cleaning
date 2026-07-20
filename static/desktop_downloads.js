(function () {
  function hasDesktopBridge() {
    return !!(window.pywebview && window.pywebview.api &&
      typeof window.pywebview.api.begin_export === 'function');
  }

  function isExportLink(anchor) {
    if (!anchor || !anchor.href) return false;
    let url;
    try {
      url = new URL(anchor.href, window.location.href);
    } catch (_err) {
      return false;
    }
    if (url.origin !== window.location.origin) return false;
    return (
      url.pathname.startsWith('/download/') ||
      /^\/projects\/\d+\/download\/cleaned$/.test(url.pathname) ||
      /^\/projects\/\d+\/versions\/\d+\/download$/.test(url.pathname)
    );
  }

  function filenameFromDisposition(disposition, fallbackUrl) {
    const fallback = decodeURIComponent((fallbackUrl.pathname.split('/').pop() || 'indataout_export.csv'));
    if (!disposition) return fallback;
    const utfMatch = disposition.match(/filename\*=UTF-8''([^;]+)/i);
    if (utfMatch) {
      try { return decodeURIComponent(utfMatch[1].replace(/"/g, '')); } catch (_err) {}
    }
    const plainMatch = disposition.match(/filename="?([^";]+)"?/i);
    return plainMatch ? plainMatch[1] : fallback;
  }

  function bytesToBase64(bytes) {
    let binary = '';
    const stride = 0x8000;
    for (let i = 0; i < bytes.length; i += stride) {
      binary += String.fromCharCode.apply(null, bytes.subarray(i, i + stride));
    }
    return btoa(binary);
  }

  async function writeResponseChunks(response, exportId) {
    if (response.body && response.body.getReader) {
      const reader = response.body.getReader();
      while (true) {
        const next = await reader.read();
        if (next.done) break;
        if (next.value && next.value.length) {
          const written = await window.pywebview.api.append_export_chunk(exportId, bytesToBase64(next.value));
          if (!written || !written.ok) throw new Error((written && written.error) || 'Could not write export data.');
        }
      }
      return;
    }

    const buffer = await response.arrayBuffer();
    const bytes = new Uint8Array(buffer);
    const chunkSize = 256 * 1024;
    for (let offset = 0; offset < bytes.length; offset += chunkSize) {
      const written = await window.pywebview.api.append_export_chunk(
        exportId,
        bytesToBase64(bytes.subarray(offset, offset + chunkSize))
      );
      if (!written || !written.ok) throw new Error((written && written.error) || 'Could not write export data.');
    }
  }

  function setBusy(anchor, busy) {
    if (!anchor) return;
    if (busy) {
      if (!anchor.dataset.desktopDownloadLabel) anchor.dataset.desktopDownloadLabel = anchor.textContent;
      anchor.textContent = 'Saving...';
      anchor.setAttribute('aria-busy', 'true');
      anchor.style.pointerEvents = 'none';
      anchor.style.opacity = '0.65';
    } else {
      if (anchor.dataset.desktopDownloadLabel) {
        anchor.textContent = anchor.dataset.desktopDownloadLabel;
        delete anchor.dataset.desktopDownloadLabel;
      }
      anchor.removeAttribute('aria-busy');
      anchor.style.pointerEvents = '';
      anchor.style.opacity = '';
    }
  }

  async function saveDesktopExport(anchor) {
    const url = new URL(anchor.href, window.location.href);
    setBusy(anchor, true);
    let exportId = null;
    try {
      const response = await fetch(url.href, { credentials: 'same-origin' });
      if (!response.ok) {
        throw new Error(`Export failed (${response.status}).`);
      }
      if (response.redirected && new URL(response.url).pathname !== url.pathname) {
        throw new Error('Export is no longer available. Reopen the project or run Clean Data again.');
      }
      const filename = filenameFromDisposition(response.headers.get('Content-Disposition'), url);
      const started = await window.pywebview.api.begin_export(filename);
      if (!started || !started.ok) {
        if (started && started.cancelled) return;
        throw new Error((started && started.error) || 'Could not start desktop export.');
      }
      exportId = started.export_id;
      await writeResponseChunks(response, exportId);
      const finished = await window.pywebview.api.finish_export(exportId);
      exportId = null;
      if (!finished || !finished.ok) throw new Error((finished && finished.error) || 'Could not finish export.');
      window.alert(`Saved export:\n${finished.path}`);
    } catch (err) {
      if (exportId) {
        try { await window.pywebview.api.abort_export(exportId); } catch (_cleanupErr) {}
      }
      window.alert((err && err.message) || 'Could not save export.');
    } finally {
      setBusy(anchor, false);
    }
  }

  document.addEventListener('click', function (event) {
    if (!hasDesktopBridge()) return;
    const anchor = event.target && event.target.closest ? event.target.closest('a[href]') : null;
    if (!isExportLink(anchor)) return;
    event.preventDefault();
    saveDesktopExport(anchor);
  }, true);
})();
