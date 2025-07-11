# HR Attrition Analytics Dashboard

A ready‑to‑run Streamlit & Plotly dashboard for exploring employee attrition, satisfaction and demographics. The app connects to a MySQL database (or optionally a CSV file) and lets you slice, dice and visualise the classic HR‑Analytics data set in seconds.

---

## 📂 Repository structure

```
├── dashboard.py            # Streamlit application
├── HRDB.sql                # MySQL dump – creates the `hrdb` database
├── final_project_v2.ipynb  # Notebook used for EDA & dashboard prototyping
└── data.csv (optional)     # CSV fallback loaded when DB is unavailable
```

## ✨ Features

| View                              | What it shows                                                                      |
| --------------------------------- | ---------------------------------------------------------------------------------- |
| **Attrition rate by Department**  | Bar chart with a dotted company‑average line for context                           |
| **Job Satisfaction × Attrition**  | Crosstab heat‑tables per selected department                                       |
| **Sales Attrition % by Job Role** | Side‑by‑side bars (Yes/No) for each sales role                                     |
| **Years at Company – Sales Reps** | Distribution with attrition % annotated above each bar                             |
| **Education Level – Sales Reps**  | Employee counts with attrition % annotations                                       |
| **Distance from Home Histogram**  | Company‑wide distances, annotated with mean job satisfaction & attrition % per bin |

All plots are fully interactive thanks to Plotly Express/Graph Objects, and every view reacts instantly to sidebar filters for **Department** and **Job Role**.

## 🚀 Quick start

### 1 . Clone & enter the repo

```bash
git clone https://github.com/<your‑org>/hr‑attrition‑dashboard.git
cd hr‑attrition‑dashboard
```

### 2 . Create a virtual env & install requirements

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Tip:** If you don’t have a `requirements.txt` yet, start with:
>
> ```
> streamlit pandas plotly mysql‑connector‑python numpy
> ```

### 3 . Import the database

**Option A – MySQL (recommended)**

```bash
mysql -u root -p < HRDB.sql
```

The dump creates a database named **`hrdb`** and populates `table1` and `table2`. fileciteturn0file1
Configure credentials for the app via environment variables or a `.env` file:

| Variable      | Default     | Description   |
| ------------- | ----------- | ------------- |
| `DB_HOST`     | `localhost` | MySQL host    |
| `DB_NAME`     | `hrdb`      | Database name |
| `DB_USER`     | `root`      | Username      |
| `DB_PASSWORD` | *(none)*    | Password      |

**Option B – CSV fallback**
If the database connection fails the app automatically loads `data.csv` instead and shows a warning. fileciteturn0file0

### 4 . Run the dashboard

```bash
streamlit run dashboard.py
```

The app will open in your browser at [http://localhost:8501](http://localhost:8501).

---

## 🏗️ Code overview (dashboard.py)

* **`load_data()`** – Tries MySQL first, falls back to CSV and caches the DataFrame. fileciteturn0file0
* **Page layout** – Wide mode, white background, centred container for a clean look.
* **Filters** – Multiselects for Department & Job Role (context‑aware).
* **Visualisations** – Six dynamic Plotly views (see *Features* above).
* **Annotations** – Key metrics (attrition %, avg satisfaction) are drawn directly on the charts for at‑a‑glance insights.

The code is heavily commented, so feel free to dive in and tweak colours, bins or KPI formulas.

## 📒 Notebook

`final_project_v2.ipynb` documents the exploratory data analysis and the step‑by‑step path that led to the final Streamlit app – perfect if you want to reproduce or expand the analysis.

## 🗄️ Data source

The sample data is based on the **IBM HR Analytics Employee Attrition & Performance** dataset (Kaggle). All personally identifiable information was removed.

## 🤝 Contributing

1. Fork the repo and create a feature branch.
2. Follow the *Quick start* guide using a separate DB/schema to keep data safe.
3. Submit a pull request describing **what** & **why**.

## 📝 License

This project is released under the MIT License – see `LICENSE` for details.

## 👤 Author & Contact

*Maintainer:* **Your Name** · [your.email@example.com](mailto:your.email@example.com)

Have ideas or questions? Open an issue or drop me a line – happy to discuss!
