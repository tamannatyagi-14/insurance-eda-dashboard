# 🏥 Insurance Charges — EDA Dashboard

An interactive Exploratory Data Analysis dashboard built with **Streamlit** and **Plotly**,
analyzing the Medical Cost Personal Dataset before model creation.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=flat-square&logo=streamlit)
![Plotly](https://img.shields.io/badge/Plotly-5.22-blueviolet?style=flat-square&logo=plotly)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📊 Dashboard Preview

The dashboard covers:
- KPI cards (avg charge, smoker premium, correlations)
- Charges by smoker status, age group, BMI category, region
- Interactive scatter plot (age vs charges, colored by smoker)
- Feature distributions (charges, age, BMI histograms)
- Correlation heatmap
- Dataset composition (sex, smoker, region pie charts)
- Sidebar filters (age, smoker, sex, region, BMI category)

---

## 🚀 Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/insurance-eda-dashboard.git
cd insurance-eda-dashboard
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📁 Project Structure

```
insurance-eda-dashboard/
├── app.py               # Main Streamlit dashboard
├── insurance.csv        # Dataset (Medical Cost Personal Dataset)
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## 📌 Key Findings

| Finding | Value |
|---|---|
| Average Insurance Charge | $13,270 |
| Smoker Premium | **3.8×** more than non-smokers |
| Obese + Smoker avg charge | $41,693 |
| Age correlation with charges | r = 0.30 |
| BMI correlation with charges | r = 0.20 |
| Missing values | **0** |

---

## 🛠 Tech Stack

- **Python** — Data processing
- **Pandas / NumPy** — Data manipulation
- **Plotly** — Interactive charts
- **Streamlit** — Web dashboard framework
- **Scikit-learn** — Preprocessing utilities

---

## 📦 Dataset

**Medical Cost Personal Dataset**
- 1,338 rows, 7 features
- Features: age, sex, bmi, children, smoker, region, charges
- Source: insurance.csv


---

## 👤 Author

Made by **Tamanna Tyagi** · [LinkedIn](https://linkedin.com/in/yourprofile) · [GitHub](https://github.com/yourusername)

---

## ⭐ If you found this useful, give it a star!
