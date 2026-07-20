# inDataOut

inDataOut is a local desktop app for cleaning tabular datasets. It helps inspect messy data, apply common cleaning steps, review a Change Log, resume saved projects, and export cleaned files.

It can run as a Python app during development or as a packaged Windows `.exe`.

## Features

- Upload CSV, TSV, TXT, Excel, or JSON files
- Preview data before applying changes
- Clean duplicates, empty columns, constant columns, whitespace, placeholders, missing values, data types, and outliers
- Edit individual dataset cells
- Run trusted local Pandas code for custom cleanup
- Review applied changes in a grouped Change Log
- Undo and reapply actions
- Save and resume projects
- Export cleaned CSV files and JSON reports
- Use native desktop downloads in the packaged app

## Cleaning Tools

### Data Integrity

Use this section for duplicate rows, empty columns, constant columns, whitespace cleanup, placeholder replacement, and dropping selected columns.

### Missing Values

Use automatic fill strategies or type values manually into missing cells. Manual entries are only applied when the user actually types a value.

### Data Types

Detect column types automatically, or convert selected columns to Integer, Float, Datetime, Text, Category, or Boolean.

### Outliers

Review IQR outlier candidates and remove selected rows. The Change Log shows the removed row, outlier column, outlier value, reason, and result.

### Dataset Editor

Edit individual cells directly from the dataset grid. Saved edits are recorded in the Change Log.

### Custom Pandas Code

Run trusted local Pandas code against the current dataset for cleanup that is not covered by the built-in tools.

## Project Workflow

1. Upload a dataset.
2. Review the data and detected issues.
3. Choose a cleaning tool.
4. Preview the result.
5. Apply the action.
6. Review the Change Log.
7. Final Clean when ready.
8. Download the cleaned output.

Saved projects can be opened later from My Projects.

## Change Log Workflow

Every applied action is saved in the Change Log. Actions are grouped by the tool that created them, so missing-value fixes, data type conversions, outlier removals, cell edits, selected column drops, and custom code changes stay easy to review.

For each action, the Change Log can show the operation, affected rows or columns, Before/After values, shape changes, and a fuller detail view when more evidence is available.

The Change Log is also interactive:

- undo an applied action
- reapply an undone action
- open full change details
- review what happened after resuming a project
- keep later actions in order when earlier actions are undone or restored

This makes it possible to experiment with cleaning steps without losing track of what changed.

## Supported Files

- `.csv`
- `.tsv`
- `.txt`
- `.xlsx`
- `.xls`
- `.json`

Maximum upload size: `100 MB`.

## Run From Python

Run the Flask app:

```powershell
.\.venv\Scripts\python.exe app.py
```

Run the desktop launcher:

```powershell
.\.venv\Scripts\python.exe desktop_launcher.py
```

## Build The Desktop App

Build the Windows executable:

```powershell
.\build_exe.bat
```

The output is created at:

```text
dist\inDataOut.exe
```

Close any old running app window before opening a newly rebuilt version.

## Project Files

- `app.py` - Flask routes, projects, resume, downloads
- `cleaner.py` - cleaning logic and data-quality helpers
- `desktop_launcher.py` - desktop app launcher and native download bridge
- `snapshot_diff.py` - replay and Change Log support
- `custom_code_runner.py` - trusted Pandas code execution
- `templates/` - HTML pages
- `static/` - JavaScript and CSS
- `tests/` - regression tests

## Checks

Compile Python files:

```powershell
.\.venv\Scripts\python.exe -B -m py_compile app.py db.py cleaner.py snapshot_diff.py custom_code_runner.py setup_database.py desktop_launcher.py
```

Run tests:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_correctness_core tests.test_app_safety_systems tests.test_quality_canonical
```

Check JavaScript:

```powershell
node --check static/script.js
```

## Notes

- The app is designed for local use.
- Custom Pandas code should only be used on a trusted machine.
- Large files may take longer in preview or export steps.
- CSV export does not preserve Excel formatting, formulas, or workbook sheets.

## Setup

See [SETUP.md](SETUP.md) for installation, database setup, troubleshooting, and cleanup.
