# 🧹 inDataOut — Advanced Data Cleaning Web App

A powerful yet simple **Flask-based web app** that lets you upload, clean, analyze, and download datasets instantly — all inside your browser.  
Perfect for **data preprocessing, cleaning, and quality improvement**.

🌐 **Live App:** [https://indataout-data-cleaning.onrender.com](https://indataout-data-cleaning.onrender.com)

---

## 🚀 Features

- 📂 Supports **CSV, Excel (XLSX/XLS), JSON, TSV, TXT** formats with robust encoding detection (UTF-8, Latin1, CP1252, Chardet)
- ⚙️ **Automatic & Manual Data Cleaning** with customizable options
- 🧽 **Data Preprocessing**: Removes duplicates, outliers, empty & constant columns
- 🔢 **Smart Numeric Handling**: Fill missing values (Mean / Median / Zero)
- 🔤 **Categorical Data Treatment**: Fill missing text values (Mode / Constant)
- 🧩 **Manual Data Type Conversion**: Select specific type for each column (Auto, Numeric, Datetime, String, Category, Boolean)
- 📉 **Interactive Outlier Detection**: Preview detected outliers (IQR method) and manually select which rows to remove
- 📈 **Data Quality Score Dashboard**: Visual comparison before vs after cleaning with improvement metrics
- 📊 Download **cleaned dataset (CSV)** & **detailed cleaning report (JSON)**
- 🔒 **Privacy-Focused**: Files processed temporarily and deleted automatically after download
- ⚡ Built with **Flask + Pandas**, optimized for performance, deployed on **Render**

---

## 🧩 Cleaning Operations

### Automatic Cleaning Steps
1. **Replace Placeholder Values**: Converts common placeholders (N/A, NULL, NaN, etc.) to proper missing values
2. **Column Name Sanitization**: Cleans and normalizes column names (lowercase, underscores, removes special characters)
3. **Whitespace Trimming**: Removes leading/trailing spaces from text values
4. **Data Type Conversion**: Automatically detects and converts numeric columns
5. **Manual Type Mapping**: User-specified data type conversion for each column
6. **Empty Column Removal**: Drops columns with only missing values
7. **Constant Column Removal**: Removes columns with single repeated value
8. **Duplicate Row Removal**: Eliminates identical rows
9. **Outlier Handling**: 
   - **Manual Selection**: Preview detected outliers (IQR method) and choose which rows to remove
10. **Missing Value Treatment**:
    - **Numeric**: Mean, Median, or Zero imputation
    - **Categorical**: Mode or Constant filling
11. **Label Encoding**: Optional encoding of categorical variables

### Advanced Options

| Option | Description |
|--------|--------------|
| ✅ **Remove duplicates** | Delete identical rows |
| ✅ **Drop empty columns** | Remove columns with only missing values |
| ✅ **Drop constant columns** | Remove columns that contain one repeated value |
| ✅ **Trim whitespace** | Remove spaces around text values |
| ✅ **Convert data types** | Auto-detect numeric/date-like text and convert to proper types |
| 🎯 **Manual data type conversion** | Choose datatype for each column: Auto, Numeric, Datetime, String, Category, Boolean |
| 🔢 **Fill numeric** | Fill missing numeric values: Mean, Median, or Zero |
| 🔤 **Fill categorical** | Fill missing text values: Mode or Constant (blank) |
| 🔧 **Missing strategy** | Choose between `Fill` (recommended) or `Drop rows` |
| 📉 **Preview Outliers** | Detect outliers using IQR method and manually select which rows to remove |
| 💯 **Data Quality Score** | Visual dashboard showing quality improvement (Before → After) |

---

## 📊 Data Quality Score

The app computes a comprehensive **Data Quality Score (0-100)** based on:
- **Missing Values Percentage**: Proportion of empty cells
- **Duplicates Percentage**: Proportion of duplicate rows
- **Outliers Percentage**: Proportion of extreme values (IQR method)
- **Constant Columns Percentage**: Proportion of non-informative columns

**Weighted Formula**: `Score = 100 - (0.5×missing% + 0.2×dup% + 0.2×outlier% + 0.1×constant%)`

### Quality Metrics Displayed
| Metric | Description |
|--------|--------------|
| **Before Score** | Initial dataset quality score |
| **After Score** | Quality score after cleaning operations |
| **Improvement** | Point increase in quality score |
| **Visual Progress Bar** | Graphical representation of final quality percentage |

---

## 🧠 Example Workflow

1. **Upload Dataset** (`.csv`, `.xlsx`, `.xls`, `.json`, `.tsv`, `.txt`)
   - Drag-and-drop or click to browse
   - Automatic encoding detection for problematic files
2. **Review Dataset Summary**
   - Rows × Columns count
   - Missing values count and percentage
   - Duplicate row count
   - Empty columns identified
   - Interactive data preview table
3. **Configure Cleaning Options** (Optional)
   - Toggle "Advanced Options" panel
   - Select data type for each column (Auto/Numeric/Datetime/String/Category/Boolean)
   - Choose outlier handling: Click "Preview Outliers" to interactively select rows
   - Set missing value strategies (numeric and categorical)
4. **Execute Cleaning**
   - Click "Clean Data" button
   - System applies all selected operations sequentially
5. **Review Results**
   - **Cleaning Summary**: Rows processed, duplicates/outliers removed, filled cells, renamed columns
   - **Data Quality Dashboard**: Before/After scores with visual progress bar and improvement percentage
   - **Download Options**: 
     - ⬇️ Cleaned CSV (UTF-8 encoded, no index)
     - 📄 Detailed JSON Report (timestamp, operations log, preview data)

---

## 🛠️ Tech Stack

### Backend
- **Framework**: Flask (Python web framework)
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (LabelEncoder)
- **File Handling**: OpenPYXL (Excel), xlrd (legacy Excel), Chardet (encoding detection)
- **Production Server**: Gunicorn WSGI server

### Frontend
- **Structure**: HTML5 with Jinja2 templating
- **Styling**: Custom CSS3 with gradients, shadows, responsive design
- **Interactivity**: Vanilla JavaScript (ES6+ with async/await)
- **Features**: Drag-and-drop upload, dynamic form updates, AJAX outlier preview

### Infrastructure
- **Hosting**: Render Cloud Platform
- **Process Management**: Procfile for production deployment
- **Configuration**: Environment variables for port and debug mode

---

## 🔧 Installation & Local Development

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup Steps

1. **Clone the Repository**
```bash
git clone <repository-url>
cd indataout-DATA-CLEANING
```

2. **Create Virtual Environment** (recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the Application**
```bash
python app.py
```

The app will start on `http://localhost:5000` (or use `PORT` environment variable if set).

### Environment Variables (Optional)
- `PORT`: Override default port 5000
- `FLASK_ENV`: Set to `development` to enable debug mode

---

## 📁 Project Structure

```
indataout - DATA CLEANING/
├── app.py                  # Main Flask application
├── cleaner.py              # Data cleaning logic and helpers
├── requirements.txt        # Python dependencies
├── Procfile               # Deployment configuration
├── render.yaml            # Render platform config
├── templates/
│   └── index.html         # Main UI template
├── static/
│   ├── style.css          # Custom styles
│   └── script.js          # Client-side interactions
└── uploads/               # Temporary file storage (auto-created)
    ├── cleaned/           # Processed CSV files
    └── reports/           # JSON cleaning reports
```

---

## 📋 API Endpoints

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Home page with upload interface |
| `/upload` | POST | Handle file upload and display analysis |
| `/clean` | POST | Execute cleaning pipeline with options |
| `/download/cleaned/<filename>` | GET | Download cleaned CSV file |
| `/download/report/<filename>` | GET | Download JSON cleaning report |
| `/files` | GET | List all uploaded files (JSON response) |
| `/preview_outliers` | POST | Detect and return outlier information (JSON) |

---

## 🎨 UI Components

### Header
- Sticky topbar with logo and branding
- Quick navigation CTA button

### Hero Section
- Upload card with drag-and-drop zone
- File type indicators and size limits
- Privacy assurance message

### Features Grid
- Six feature cards highlighting core capabilities
- Responsive layout for mobile/tablet/desktop

### Analysis Panel
- Dataset summary statistics cards
- Interactive data preview table (up to 1000 rows)
- Collapsible advanced options panel
- Real-time outlier preview with checkbox selection

### Results Section
- Cleaning operation summary grid
- Data Quality Score dashboard with animated progress bar
- Download buttons for CSV and JSON report
- "Upload Another" reset option

---

## 🔒 Security & Privacy

- **Secure File Handling**: Werkzeug secure_filename for sanitization
- **Temporary Storage**: Files stored only during session
- **Automatic Cleanup**: No permanent data retention
- **Encoding Robustness**: Multiple fallback encodings prevent crashes
- **Error Handling**: Comprehensive logging and user-friendly error messages

---

## 🐛 Known Limitations

- Maximum file size depends on hosting platform (typically 100MB on Render)
- Outlier preview limited to first 100 detected rows for performance
- Some Excel features (formulas, formatting) not preserved in CSV output
- Very large datasets (>50k rows) may experience slower processing

---

## 👨‍💻 Developer

**Yaswanth Himanshu**  
📘 GitHub: [https://github.com/YaswanthHimanshu](https://github.com/YaswanthHimanshu)  
🌐 Live App: [https://indataout-data-cleaning.onrender.com](https://indataout-data-cleaning.onrender.com)
