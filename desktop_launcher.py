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
import socket
import sys
import threading
import time

import webview

# Import the existing Flask app object as-is — no changes to app.py's logic,
# just importing it as a module instead of running it as __main__.
import app as flask_app_module

APP_TITLE = "inDataOut — Data Cleaning"
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900


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
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
        min_size=(900, 600),
    )
    # gui="edgechromium" is the default and best choice on Windows (uses the
    # OS-bundled WebView2 runtime — no extra install needed on Windows 10/11).
    webview.start()


if __name__ == "__main__":
    main()
