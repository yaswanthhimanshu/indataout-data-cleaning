# 🧹 inDataOut — Advanced Data Cleaning Web App

A powerful yet simple **Flask-based web app** that lets you upload, clean, analyze, and download datasets instantly — all inside your browser.  
Perfect for **data preprocessing, cleaning, and quality improvement**.

🌐 **Live App:** [https://indataout-data-cleaning.onrender.com](https://indataout-data-cleaning.onrender.com)

---

## 🚀 Features

- 📂 Supports **CSV, Excel (XLSX/XLS), JSON, TSV, TXT** formats  
- ⚙️ One-click **Automatic & Manual Data Cleaning**  
- 🧽 Removes duplicates, outliers, empty & constant columns  
- 🔢 **Fill Numeric** (Mean / Median / Zero)  
- 🔤 **Fill Categorical** (Mode / Constant)  
- 🧠 **KNN Imputation** for smart numeric filling  
- 🧩 **Manual Data Type Conversion** (select type for each column)  
- 🔍 **Fuzzy Text Clustering** (merges similar strings / typos)  
- 📈 **Data Quality Score Dashboard** (before vs after cleaning)  
- 📊 Download **cleaned dataset (CSV)** & **cleaning report (JSON)**  
- ⚡ Built for speed using **Flask + Pandas**, hosted on **Render**

---

## 🧩 Advanced Cleaning Options

| Option | Description |
|--------|--------------|
| ✅ **Remove duplicates** | Delete identical rows |
| ✅ **Drop empty columns** | Remove columns with only missing values |
| ✅ **Drop constant columns** | Remove columns that contain one repeated value |
| ✅ **Trim whitespace** | Remove spaces around text values |
| ✅ **Convert data types** | Convert numeric/date-like text to proper types |
| 🔄 **Manual data type conversion** | Choose datatype for each column manually |
| 🔢 **Fill numeric** | Fill missing numeric values (Mean / Median / Zero) |
| 🔤 **Fill categorical** | Fill missing text values (Mode / Constant) |
| 🔧 **Missing strategy** | Choose between `Fill`, `Drop`, or `KNN` |
| 📉 **Handle outliers** | Detect and cap or remove extreme values |
| 🧠 **KNN Imputation** | Estimate numeric values using nearest rows |
| 🔍 **Fuzzy text clustering** | Merge similar text values (typos, abbreviations) |
| 💯 **Data Quality Score** | See before/after quality improvement instantly |

---

## 🧠 Example Workflow

1. Upload your dataset (`.csv`, `.xlsx`, `.xls`, `.json`, `.tsv`, `.txt`)  
2. Review detected columns and select datatypes (optional)  
3. Pick advanced options like KNN, Fuzzy Matching, Fill Strategy  
4. Click **Clean Data**  
5. Instantly view:
   - Cleaned data preview  
   - Data Quality Score (Before vs After)  
   - Download links for CSV + JSON report  

---

## 🛠️ Tech Stack

- **Backend:** Flask (Python)  
- **Libraries:** Pandas, NumPy, Scikit-learn, OpenPyXL, Chardet  
- **Frontend:** HTML, CSS, Vanilla JS  
- **Server:** Gunicorn  
- **Hosting:** Render  

---

## 📊 Data Quality Metrics

| Metric | Description |
|--------|--------------|
| **Before Score** | Measures initial dataset quality |
| **After Score** | Evaluates quality after cleaning |
| **Improvement** | % increase in overall data consistency |

---

## 👨‍💻 Developer

**Yaswanth Himanshu**  
📘 GitHub: [https://github.com/YaswanthHimanshu](https://github.com/YaswanthHimanshu)  
🌐 Live App: [https://indataout-data-cleaning.onrender.com](https://indataout-data-cleaning.onrender.com)

