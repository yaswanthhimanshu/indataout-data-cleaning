# inDataOut - Controlled Data Cleaning Workflow

Live project: https://indataout-data-cleaning.onrender.com

inDataOut is a controlled data-cleaning workflow that helps users inspect, correct, and validate messy datasets before analysis or modeling.

I built this project because data cleaning is usually either too automatic or too manual. Fully automatic tools can over-clean data without showing what changed, while spreadsheet-only cleaning becomes hard to track as the dataset grows. inDataOut is designed to keep the user in control: every important cleaning decision can be reviewed, applied, logged, and undone before creating the final cleaned file.

---

## What I Built

I built an interactive preprocessing workflow for messy tabular data. The workflow is delivered through a browser interface, but the main focus is the cleaning logic, user control, and validation process.

The workflow has three main dataset views:

- **Original Uploaded Dataset**: shows the untouched uploaded file.
- **Dataset Editor**: lets the user manually edit visible cells and column names.
- **Cleaning Work Progress**: shows the live working dataset after applied cleaning actions.

The main idea is that users should always understand what changed, why it changed, and whether the cleaned result still needs review.

---

## Why I Built It

Most beginner data-cleaning workflows have a few problems:

- Users select cleaning options but cannot clearly see what changed.
- Missing values may be filled without explaining which cells were affected.
- Rows or columns can be removed without enough context.
- A final "cleaned" score can look perfect even when the dataset still has review risks.
- Users may accidentally apply options they did not mean to apply.

This project solves those problems by making data cleaning visible and controlled. Users apply one feature or section at a time, then check the live progress dataset, the operation log, and the generated Pandas-style code before finalizing.

---

## Core Workflow

1. **Upload a dataset**
   - Supports CSV, Excel, JSON, TSV, and TXT files.
   - Handles common encoding issues so messy CSV files can still load correctly.

2. **Review the original data**
   - The original uploaded preview stays unchanged.
   - This gives the user a safe reference point while cleaning.

3. **Choose a cleaning section**
   - Data Integrity
   - Missing Values
   - Data Types
   - Outliers
   - Advanced Options

4. **Apply selected actions**
   - Each feature has an Apply button.
   - Options must be turned on or selected before applying.
   - If the user clicks Apply without selecting the needed option, the workflow shows a toast warning.

5. **Watch Cleaning Work Progress**
   - The live working dataset updates after every Apply.
   - Changed cells are highlighted.
   - The log explains exactly what was changed, filled, removed, or skipped.
   - Undo restores the previous preview and removes highlights.

6. **Finalize**
   - Clean Data uses the applied controls, not random toggles left on the page.
   - The user downloads a cleaned CSV and a JSON cleaning report.

---

## Data Cleaning Features Implemented

### 1. Original Dataset Preview

The original dataset preview shows the uploaded file before cleaning. I kept this separate from the cleaning preview so users can compare the original data with the working version.

Why it matters:

- Prevents confusion between original and cleaned data.
- Helps users verify that the workflow is not silently changing the source view.
- Gives a stable reference while applying multiple cleaning actions.

---

### 2. Dataset Editor

The Dataset Editor lets users manually edit visible cells and column names. It also includes search controls for finding values by text, column, or row number.

Why it matters:

- Some data issues cannot be fixed automatically, such as spelling mistakes, wrong values, or domain-specific errors.
- Users need a way to manually correct visible problems before applying automated cleaning.
- Column names should be editable because messy headers are common in real datasets.

---

### 3. Missing Value Editor

The Missing Value Editor shows rows that contain missing values and lets users type values directly into blank cells before applying fill or drop rules.

Why it matters:

- Some missing values should be manually entered instead of filled with mean, median, mode, or "Unknown".
- Users can decide which missing values deserve exact replacement.
- The editor updates after Apply so stale missing cells do not keep showing after the working dataset changes.

Important behavior:

- Fully empty columns cannot be filled with mean or median because there are no values to calculate from.
- The workflow reports those cases honestly instead of pretending they were filled.
- For fully empty columns, the recommended action is to drop the column before filling remaining missing values.

---

### 4. Cleaning Work Progress

Cleaning Work Progress is the live spreadsheet that updates after each applied action.

Implemented behavior:

- Shows the current working dataset.
- Highlights updated cells.
- Updates after each Apply.
- Syncs with the Dataset Editor and Missing Value Editor.
- Supports Undo for the previous preview state.

Why it matters:

- Users can see cleaning results before final download.
- Highlighting makes changed cells easier to notice.
- Undo reduces fear of trying a cleaning option.

---

### 5. Apply-First Cleaning Decisions

The workflow uses Apply buttons for individual features and sections. Final Clean Data uses the applied options rather than every visible form control.

Why it matters:

- Users may turn on a toggle while exploring but forget to turn it off.
- Without an Apply-first workflow, final cleaning could run actions the user did not intentionally confirm.
- This design keeps the cleaning decision deliberate and user-controlled.

---

### 6. Data Integrity Tools

Implemented data integrity features:

- Remove duplicate rows.
- Drop fully empty columns.
- Review unnamed columns.
- Drop constant columns.
- Trim whitespace.

Why it matters:

- Duplicate rows can affect counts and averages.
- Fully empty columns add no information.
- Constant columns do not help analysis or modeling.
- Whitespace can cause false mismatches during filtering, grouping, or joining.

Important distinction:

- A fully empty column is not treated as a constant column.
- Empty columns and constant columns are handled separately because they mean different things.

---

### 7. Missing Value Handling

Implemented missing-value features:

- Manual missing-cell entry.
- Fill numeric values with mean, median, or zero.
- Fill text/category values with mode or Unknown.
- Drop rows with remaining missing values.

Why it matters:

- Different columns need different missing-value strategies.
- Filling with mean is not always correct.
- Dropping rows can remove useful data, so the user should decide deliberately.
- The workflow logs what was filled, skipped, or left remaining.

---

### 8. Data Type Conversion

Implemented data type tools:

- Auto-convert safe numeric-looking columns.
- Manually convert selected columns to numeric, datetime, string, category, or boolean.
- Report conversion results and coercion behavior.

Why it matters:

- Messy datasets often store numbers as text.
- Manual override is needed when automatic detection is wrong.
- Invalid conversions can create missing values, so the workflow reports what happened instead of hiding it.

---

### 9. Outlier Preview And Removal

The workflow detects potential outliers using the IQR method and lets the user select which detected rows to remove.

Why it matters:

- Outliers are not always bad data.
- Automatically removing all outliers can over-clean a dataset.
- Users should preview outlier rows and choose what to remove.

---

### 10. Operation Log And Pandas-Style Code

Each applied action shows a log of what happened and a Pandas-style code explanation.

Examples of logged details:

- Which columns were dropped.
- Which uploaded rows were removed.
- Which cells were filled.
- Which rows and columns were edited.
- Which numeric fills were skipped and why.

Why it matters:

- Cleaning should be explainable.
- Users can learn what each cleaning action does.
- The final report gives a record of the cleaning process.

---

### 11. Data Quality And Cleaning Completeness

The project separates two ideas:

- **Cleaning Completeness**: whether selected cleaning actions finished successfully.
- **Data Quality Score**: how much review risk remains in the dataset.

Why it matters:

- A cleaning action can finish successfully while the dataset still has issues.
- A score should not pretend the data is perfect just because selected actions ran.
- This makes the result more honest and useful.

---

### 12. Temporary Session-Based Dataset Handling

Each upload is stored in a separate temporary session folder.

Implemented behavior:

- Each user upload gets a random session folder.
- Routes check the user's session before previewing, cleaning, or downloading files.
- Original uploaded files are deleted after cleaned outputs are created.
- Temporary folders are cleaned automatically.

Why it matters:

- Multiple users should not control each other's data.
- Uploaded files should not stay permanently on the server.
- Temporary storage works well for the upload-clean-download workflow.

---

## What Makes This Workflow Different

The main goal of inDataOut is not just cleaning data. The goal is controlled cleaning.

The workflow avoids over-cleaning by:

- Keeping the original dataset separate.
- Requiring Apply actions before final cleaning.
- Showing live progress.
- Highlighting changed cells.
- Logging exact changes.
- Letting users undo preview changes.
- Reporting skipped or impossible operations honestly.

This makes the project useful for users who want to prepare data without losing track of what happened.

---

## Implementation

- **Data processing**: Pandas, NumPy
- **Cleaning logic**: missing-value handling, duplicate checks, empty/constant column handling, outlier review, type conversion, quality scoring
- **User workflow**: apply-first controls, live progress preview, operation log, undo, manual dataset correction
- **Interface**: Flask, HTML, CSS, JavaScript
- **Deployment**: Render, Gunicorn

---

## Final Output

After cleaning, users can download:

- A cleaned CSV file.
- A JSON report with cleaning operations, summary metrics, and preview data.

---

## Future Improvements

Possible future additions:

- More advanced cell-level diff view.
- Better pagination for very large datasets.
- Saved cleaning recipes.
- More visual charts for column-level data quality.
- Stronger semantic validation for domain-specific datasets.

---

## Developer

Built by Yaswanth Himanshu.

Live project: https://indataout-data-cleaning.onrender.com

GitHub: https://github.com/YaswanthHimanshu
