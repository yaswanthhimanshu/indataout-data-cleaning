# inDataOut Setup

This guide covers local installation, configuration, database setup, run commands, tests, storage, logs, and cleanup.

## Prerequisites

- Python 3.10 or newer.
- MySQL server with a user allowed to create and modify the configured database.
- A writable project directory.
- Node.js only if running the JavaScript syntax check.
- Windows is required for the provided `build_exe.bat` executable build path.

## Database Requirements

The app uses MySQL through `mysql-connector-python`. `setup_database.py` creates the configured database and tables if they do not exist, then applies additive migrations.

The schema uses these main tables:

- `projects`
- `cleaning_history`
- `manifests`
- `dataset_versions`

## Create a Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bat
python -m venv .venv
.\.venv\Scripts\activate.bat
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` does not pin exact package versions. Pin versions before repeatable deployment or packaging.

## Configure Environment Variables

`.env.example` is only a safe template. Copy it to `.env`, then put your real
local MySQL credentials in `.env`.

Windows PowerShell:

```powershell
copy .env.example .env
```

Windows Command Prompt:

```bat
copy .env.example .env
```

macOS/Linux:

```bash
cp .env.example .env
```

Edit `.env` with local values. Do not put real credentials in `.env.example`,
and do not commit `.env`, `.secret_key`, database passwords, or session secrets.

| Variable | Required | Safe Example | Purpose | Default |
| --- | --- | --- | --- | --- |
| `DB_HOST` | Optional | `localhost` | MySQL host | `localhost` |
| `DB_PORT` | Optional | `3306` | MySQL port | `3306` |
| `DB_USER` | Optional | `indataout_user` | MySQL user | `root` |
| `DB_PASSWORD` | Required | `replace-with-your-local-password` | MySQL password | none |
| `DB_NAME` | Optional | `indataout` | MySQL database name | `indataout` |
| `HOST` | Optional | `127.0.0.1` | Flask bind host for `app.py` | `127.0.0.1` |
| `PORT` | Optional | `5000` | Flask port for `app.py` | `5000` |
| `INDATAOUT_TRUSTED_LOCAL` | Optional | `1` | Enables trusted local custom Pandas code | `1` |
| `UPLOAD_FOLDER` | Optional | `uploads` | Root for uploads, snapshots, reports, and versions | `uploads` next to the app or executable |
| `INDATAOUT_LOG_DIR` | Optional | `logs` | Diagnostic log directory | persistent local app-data log folder |

Set `INDATAOUT_TRUSTED_LOCAL=0` before running a network-facing server.

## Initialize the Database

Run this once after `.env` is configured:

```bash
python setup_database.py
```

The script is safe to re-run. It creates missing objects and adds missing columns/indexes. It does not delete projects or files.

## Run the Flask Application

```bash
python app.py
```

Default URL:

```text
http://127.0.0.1:5000/
```

Use `HOST` and `PORT` in `.env` to change the bind address or port.

## Run the Desktop Launcher

```bash
python desktop_launcher.py
```

The desktop launcher starts the Flask app with waitress on `127.0.0.1`, waits for the local server to respond, then opens a pywebview window. If port `5000` is busy, it selects another local port.

## Build the Executable

Windows:

```bat
.\build_exe.bat
```

The PyInstaller configuration is `indataout.spec`. It bundles `templates/` and `static/`. It does not bundle `.env`; provide configuration beside the executable or through environment variables.

## Run Tests

Python compilation:

```powershell
.\.venv\Scripts\python.exe -B -m py_compile app.py db.py cleaner.py snapshot_diff.py custom_code_runner.py setup_database.py desktop_launcher.py smoke_correctness_workflow.py tests\test_correctness_core.py tests\test_app_safety_systems.py tests\test_quality_canonical.py
```

Unit and integration tests:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_correctness_core tests.test_app_safety_systems tests.test_quality_canonical
```

Smoke workflow:

```powershell
.\.venv\Scripts\python.exe smoke_correctness_workflow.py
```

JavaScript syntax validation:

```bash
node --check static/script.js
```

Do not treat the smoke workflow as a rendered browser test. It uses Flask's test client.

## Storage Locations

- Uploaded source files are stored under `UPLOAD_FOLDER`.
- Current snapshots are stored under the project upload folder.
- Cleaned CSV files are stored under the project `cleaned/` folder.
- JSON reports are stored under the project `reports/` folder.
- Project, history, manifest, and version metadata are stored in MySQL.
- If `UPLOAD_FOLDER` is unset, the app uses an `uploads/` folder next to the Python files in development or next to the executable when packaged.

## Logs

Diagnostic logs are written to `INDATAOUT_LOG_DIR` when set. Otherwise the app uses a persistent local application-data directory.

Logs include request IDs, route names, project IDs when available, status codes, durations, and exception details for unexpected server errors. Logs should not contain database passwords, session cookies, full datasets, or full custom-code contents.

## Troubleshooting

### Database Unavailable

- Confirm MySQL is running.
- Check `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, and `DB_NAME`.
- Run `python setup_database.py`.
- Confirm the MySQL user can create, alter, select, insert, update, and delete in the configured database.

### Port Already in Use

- Set a different `PORT` for `python app.py`.
- Use `desktop_launcher.py` if you want automatic local port selection.

### Permission or Access Denied

- Use a writable `UPLOAD_FOLDER`.
- Avoid protected system directories.
- Check filesystem permissions for uploads, snapshots, reports, and logs.

### Missing Dependency

- Activate the virtual environment.
- Run `pip install -r requirements.txt` again.

### CSV or File Parsing Failure

- Check delimiter, encoding, malformed rows, and unsupported file contents.
- Try a small sample of the same file to isolate parsing problems.

### Custom Code Disabled

- Set `INDATAOUT_TRUSTED_LOCAL=1` for trusted local use.
- Keep it disabled for network-facing use.

### Packaged Executable Resource Paths

- `indataout.spec` includes `templates/` and `static/`.
- Keep `.env` outside the bundle.
- Confirm the executable can write to the configured upload and log folders.

## Safe Backup and Cleanup

Back up both storage layers:

- the MySQL database,
- the `UPLOAD_FOLDER` tree.

Use the app's project deletion path to remove a project. That path deletes the project-scoped files and DB rows together.

Do not delete the whole upload root unless every project in it can be removed. If cleaning temporary files manually, remove only known abandoned `*.tmp` files.

## Local Security Limitations

The default configuration is local-first. Before exposing the app on a network:

- set `HOST=127.0.0.1` or add proper authentication and authorization,
- set `INDATAOUT_TRUSTED_LOCAL=0`,
- review CSRF/session settings,
- use environment-managed secrets,
- pin dependencies,
- configure log storage and rotation,
- review upload size and memory limits.
