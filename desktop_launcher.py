# desktop_launcher.py — entry point for the packaged desktop build.
#
# Runs the existing Flask app (unchanged) on a background thread with waitress,
# waits for it to actually accept connections, then opens it in a native
# window via pywebview instead of a browser tab. Closing the window shuts
# down the whole process — no leftover server, no terminal needed.
#
# Dev run:    python desktop_launcher.py
# Packaged:   see build_exe.bat / indataout.spec

"""Desktop entry point for the packaged inDataOut app.

The launcher imports the same Flask app used in development, serves it on
127.0.0.1 with waitress, waits for readiness, then opens a pywebview window.
It owns desktop process startup only; Flask routes, persistence, cleaning, and
security behavior remain in ``app.py``.
"""

import logging
import base64
import os
import re
import socket
import sys
import threading
import time
import uuid

import webview

# Import the existing Flask app object as-is — no changes to app.py's logic,
# just importing it as a module instead of running it as __main__.
import app as flask_app_module

APP_TITLE = "inDataOut — Data Cleaning"
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900


class DesktopExportApi:
    """Native save bridge for pywebview, whose WebView2 shell has no download UI."""

    def __init__(self) -> None:
        self._open_exports: dict[str, dict] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _safe_filename(name: str) -> str:
        cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(name or "").strip())
        cleaned = cleaned.strip(" .")
        return cleaned or "indataout_export.csv"

    @staticmethod
    def _downloads_dir() -> str:
        home = os.path.expanduser("~")
        path = os.path.join(home, "Downloads")
        return path if os.path.isdir(path) else home

    @staticmethod
    def _file_types(filename: str) -> tuple[str, ...]:
        ext = os.path.splitext(filename)[1].lower()
        if ext == ".json":
            return ("JSON report (*.json)", "All files (*.*)")
        if ext == ".csv":
            return ("CSV file (*.csv)", "All files (*.*)")
        return ("All files (*.*)",)

    def begin_export(self, filename: str) -> dict:
        window = webview.active_window()
        if window is None:
            return {"ok": False, "error": "Desktop window is not ready."}

        safe_name = self._safe_filename(filename)
        try:
            selected = window.create_file_dialog(
                webview.FileDialog.SAVE,
                directory=self._downloads_dir(),
                save_filename=safe_name,
                file_types=self._file_types(safe_name),
            )
        except Exception as exc:
            logging.exception("Desktop save dialog failed")
            return {"ok": False, "error": f"Could not open Save As dialog: {exc}"}

        if not selected:
            return {"ok": False, "cancelled": True}
        path = selected[0] if isinstance(selected, (list, tuple)) else selected
        if not path:
            return {"ok": False, "cancelled": True}

        default_ext = os.path.splitext(safe_name)[1]
        if default_ext and not os.path.splitext(path)[1]:
            path += default_ext

        export_id = uuid.uuid4().hex
        try:
            handle = open(path, "wb")
        except Exception as exc:
            logging.exception("Desktop export open failed")
            return {"ok": False, "error": f"Could not write to that location: {exc}"}

        with self._lock:
            self._open_exports[export_id] = {"handle": handle, "path": path, "bytes": 0}
        return {"ok": True, "export_id": export_id}

    def append_export_chunk(self, export_id: str, chunk_base64: str) -> dict:
        with self._lock:
            entry = self._open_exports.get(export_id)
        if not entry:
            return {"ok": False, "error": "Export session expired."}
        try:
            data = base64.b64decode(chunk_base64)
            entry["handle"].write(data)
            entry["bytes"] += len(data)
            return {"ok": True, "bytes": entry["bytes"]}
        except Exception as exc:
            logging.exception("Desktop export chunk write failed")
            return {"ok": False, "error": f"Could not save export data: {exc}"}

    def finish_export(self, export_id: str) -> dict:
        with self._lock:
            entry = self._open_exports.pop(export_id, None)
        if not entry:
            return {"ok": False, "error": "Export session expired."}
        try:
            entry["handle"].close()
            return {"ok": True, "path": entry["path"], "bytes": entry["bytes"]}
        except Exception as exc:
            logging.exception("Desktop export close failed")
            return {"ok": False, "error": f"Could not finish saving export: {exc}"}

    def abort_export(self, export_id: str) -> dict:
        with self._lock:
            entry = self._open_exports.pop(export_id, None)
        if not entry:
            return {"ok": True}
        path = entry.get("path")
        try:
            entry["handle"].close()
        except Exception:
            pass
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except Exception:
            logging.exception("Desktop export cleanup failed")
        return {"ok": True}


def _find_free_port(preferred: int = 5000) -> int:
    """Use the preferred port if free; otherwise let the OS pick one.

    Avoids the classic "port already in use" crash if a previous instance
    (or a leftover terminal session) is still holding port 5000.
    """
    for port in (preferred,):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_until_up(host: str, port: int, timeout: float = 15.0) -> bool:
    """Poll the local server until it accepts TCP connections or times out."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.15)
    return False


def _run_server(port: int) -> None:
    """Serve Flask locally; exceptions are logged for desktop diagnostics."""
    try:
        from waitress import serve
        serve(flask_app_module.app, host="127.0.0.1", port=port, threads=8)
    except Exception:
        logging.exception("Server thread crashed")


def main() -> None:
    """Start the local server and native WebView shell."""
    port = _find_free_port(5000)

    server_thread = threading.Thread(target=_run_server, args=(port,), daemon=True)
    server_thread.start()

    if not _wait_until_up("127.0.0.1", port):
        # Server never came up (e.g. a genuinely fatal startup error). Show a
        # small window with the error instead of a window that just spins
        # forever on a blank page.
        webview.create_window(
            APP_TITLE,
            html="<body style='font-family:sans-serif;padding:2em;'>"
                 "<h2>inDataOut failed to start</h2>"
                 "<p>The local server didn't respond in time. Check that no other "
                 "copy of the app is already running, then try again.</p></body>",
            width=600, height=300,
        )
        webview.start()
        sys.exit(1)

    webview.create_window(
        APP_TITLE,
        url=f"http://127.0.0.1:{port}/",
        js_api=DesktopExportApi(),
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
        min_size=(900, 600),
    )
    # gui="edgechromium" is the default and best choice on Windows (uses the
    # OS-bundled WebView2 runtime — no extra install needed on Windows 10/11).
    webview.start()


if __name__ == "__main__":
    main()
