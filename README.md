# 🧹 inDataOut — Data Cleaning Web App

A simple and smart **Flask-based data cleaning tool** that lets you upload, clean, and download datasets instantly — all in the browser.

🌐 **Live App:** [https://indataout-data-cleaning.onrender.com](https://indataout-data-cleaning.onrender.com)

---

## 🚀 Features

- 📂 Supports multiple file formats: **CSV, XLSX, XLS, JSON, TSV, TXT**
- 🧽 One-click automatic data cleaning  
- ⚙️ Advanced cleaning options for full control  
- 📊 Handles missing data, duplicates, and formatting issues  
- 🧠 Includes **KNN Imputation** and **Fuzzy Text Clustering**  
- 📉 Instant download of cleaned dataset (CSV) and JSON report  
- ⚡ Fast, reliable & optimized using **Flask + Pandas**  
- ☁️ Hosted seamlessly on **Render** with **Gunicorn**

---

## 🧩 Advanced Cleaning Options

| Option | Description |
|--------|--------------|
| ✅ **Remove duplicates** | Delete identical rows |
| ✅ **Drop empty columns** | Remove columns with only missing values |
| ✅ **Drop constant columns** | Remove columns that contain one repeated value |
| ✅ **Trim whitespace** | Remove spaces around text values |
| ✅ **Convert data types** | Convert numeric/date-like text to proper types |
| ⬜ **Handle outliers** | Detect and cap extreme numeric values |
| 🔧 **Missing strategy** | Decide how to handle missing data (`Fill`, `Drop`, or `KNN`) |
| 🔢 **Fill numeric** | Choose strategy for numeric data (`Mean`, `Median`, `Zero`) |
| 🧠 **KNN Imputation** | Estimate numeric missing values using nearest neighbors |
| 🔍 **Fuzzy text clustering** | Automatically merge similar text values (typos, abbreviations) |

*(All options are applied when you click **Clean Data** in the app.)*

---

## 🛠️ Tech Stack

- **Backend:** Flask (Python)  
- **Libraries:** Pandas, NumPy, Scikit-learn, OpenPyXL, Chardet  
- **Server:** Gunicorn  
- **Hosting:** Render  
- **Frontend:** HTML + CSS (Bootstrap)

---

## 🧠 Example Workflow

1. Upload your file (`.csv`, `.xlsx`, `.xls`, `.json`, `.tsv`, `.txt`)  
2. Choose desired cleaning options  
3. Click **Clean Data**  
4. Download the cleaned dataset instantly  

---

**Developer:** Yaswanth Himanshu  
🌐 GitHub: [https://github.com/YaswanthHimanshu](https://github.com/YaswanthHimanshu)  
💻 App: [https://indataout-data-cleaning.onrender.com](https://indataout-data-cleaning.onrender.com)
