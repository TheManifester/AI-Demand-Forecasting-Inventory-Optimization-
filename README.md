# 📦 AI Demand Forecasting & Inventory Optimization

An end-to-end machine learning project that combines **demand forecasting** with **inventory optimization** to help businesses predict future product demand and make data-driven inventory decisions.

The system uses historical sales data to forecast demand and then applies inventory management techniques such as **lead-time demand, safety stock, reorder points, and inventory simulation** to generate actionable ordering recommendations.

---

## 🚀 Project Overview

Inventory management is challenging because businesses need to maintain enough stock to satisfy customer demand without holding excessive inventory.

This project addresses the problem using a complete ML and analytics pipeline:

```text
Historical Sales Data
        ↓
       EDA
        ↓
Feature Engineering
        ↓
Demand Forecasting
        ↓
Predicted Demand
        ↓
Lead-Time Demand
        ↓
Safety Stock
        ↓
Reorder Point
        ↓
Inventory Simulation
        ↓
Recommended Order Quantity
        ↓
Streamlit Dashboard
```

---

## 🎯 Objectives

* Forecast future product demand using machine learning.
* Identify demand patterns across stores and items.
* Calculate lead-time demand.
* Estimate safety stock requirements.
* Calculate reorder points.
* Simulate inventory levels over time.
* Reduce the risk of stockouts and overstocking.
* Provide business-friendly inventory recommendations through an interactive dashboard.

---

## 🧰 Tech Stack

### Programming & Data

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* XGBoost
* LightGBM
* Time-Series Forecasting
* Feature Engineering

### Explainable AI

* SHAP

### Dashboard

* Streamlit

### Development Tools

* Jupyter Notebook
* VS Code
* Git & GitHub

---

## 📊 Dataset

The project uses historical retail sales data containing:

| Column  | Description          |
| ------- | -------------------- |
| `date`  | Date of the sale     |
| `store` | Store identifier     |
| `item`  | Product identifier   |
| `sales` | Number of units sold |

The dataset is divided into:

```text
data/
├── raw/
│   ├── train.csv
│   ├── test.csv
│   └── simulation.csv
│
└── processed/
```

---

## 🔎 Exploratory Data Analysis

The EDA stage focuses on understanding:

* Sales distribution
* Daily sales trends
* Weekly patterns
* Monthly patterns
* Store-level demand
* Item-level demand
* Demand variability
* Seasonal behavior
* Outliers
* Missing values
* Relationships between stores, items, and sales

Example date features:

```python
df['date'] = pd.to_datetime(df['date'])

df['day'] = df['date'].dt.day
df['week'] = df['date'].dt.weekday
df['month'] = df['date'].dt.month
```

---

## ⚙️ Feature Engineering

Time-series and demand-related features are created to improve forecasting performance.

Examples include:

* Day
* Weekday
* Month
* Year
* Store
* Item
* Lag features
* Rolling averages
* Historical demand statistics

Example:

```text
sales_lag_1
sales_lag_7
sales_lag_14
sales_lag_30

rolling_mean_7
rolling_mean_14
rolling_mean_30
```

These features allow the model to learn both **short-term and long-term demand patterns**.

---

# 🤖 Demand Forecasting

The forecasting component uses gradient-boosting-based machine learning models.

### XGBoost

XGBoost is used as one of the primary forecasting models because it can effectively learn nonlinear relationships between:

```text
Time Features
      +
Store
      +
Item
      +
Historical Demand
      ↓
Predicted Sales
```

The model predicts future demand for each **store-item combination**.

### Evaluation Metrics

The forecasting model is evaluated using:

* MAE — Mean Absolute Error
* RMSE — Root Mean Squared Error
* R² — Coefficient of Determination
* MAPE — Mean Absolute Percentage Error

Example:

```text
MAE  : 6.25
RMSE : 8.13
R²   : 0.938
MAPE : 12.28%
```

> Model performance may change depending on the final feature set, train/test split, and model configuration.

---

# 📦 Inventory Optimization

Forecasting demand alone is not enough.

The predicted demand is converted into inventory decisions using inventory management techniques.

## 1. Lead-Time Demand

Lead-time demand estimates how much inventory will be required while waiting for a new order to arrive.

```text
Lead-Time Demand =
Average/Forecasted Daily Demand × Lead Time
```

Example:

```text
Daily Demand = 100 units
Lead Time    = 7 days

Lead-Time Demand = 100 × 7
                 = 700 units
```

---

## 2. Safety Stock

Safety stock protects against unexpected increases in demand and demand variability.

The project uses demand variability and lead time to estimate additional inventory requirements.

Conceptually:

```text
Safety Stock
      ↓
Protection against demand uncertainty
      ↓
Lower stockout risk
```

---

## 3. Reorder Point

The reorder point determines when a new inventory order should be placed.

```text
Reorder Point =
Lead-Time Demand + Safety Stock
```

When:

```text
Current Inventory ≤ Reorder Point
```

the system can trigger a replenishment recommendation.

---

# 🔄 Inventory Simulation

The project simulates inventory movement over time.

The simulation tracks variables such as:

```text
Opening Inventory
       ↓
Demand
       ↓
Inventory After Demand
       ↓
Orders / On-Order Inventory
       ↓
Incoming Inventory
       ↓
Closing Inventory
```

The simulation helps evaluate:

* Stockouts
* Overstock
* Inventory levels
* Replenishment timing
* Order quantities
* Inventory costs
* Service levels

---

# 📈 Recommended Order Quantity

When inventory reaches the reorder point, the system calculates a recommended replenishment quantity.

The recommendation considers:

* Forecasted demand
* Current inventory
* Lead time
* Safety stock
* Inventory position
* Expected future demand

This converts the ML forecast into an actionable business decision.

---

# 📊 Streamlit Dashboard

An interactive Streamlit dashboard is used to visualize the forecasting and inventory optimization results.

### Dashboard Components

```text
┌─────────────────────────────────────────────┐
│       INVENTORY OPTIMIZATION DASHBOARD      │
├─────────┬─────────┬─────────┬───────────────┤
│ Stores  │ Items   │ Demand  │ Stock Status  │
├─────────┴─────────┴─────────┴───────────────┤
│                                             │
│              INVENTORY OVERVIEW             │
│                                             │
├──────────────────────┬──────────────────────┤
│ Demand Forecast      │ Inventory Trend      │
│                      │                      │
├──────────────────────┼──────────────────────┤
│ Stockout Analysis    │ Reorder Alerts       │
│                      │                      │
└──────────────────────┴──────────────────────┘
```

The dashboard can be used to analyze individual:

* Stores
* Items
* Dates
* Forecasted demand
* Inventory levels
* Reorder points
* Safety stock
* Recommended orders

---

# 🧠 Explainable AI

SHAP is used to understand which features influence the model's demand predictions.

This helps answer questions such as:

> Why did the model predict higher demand?

Important features can include:

```text
Historical Sales
Lag Features
Rolling Demand
Month
Weekday
Store
Item
```

This improves model transparency and makes the forecasting system easier to interpret.

---

# 📁 Project Structure

```text
AI-Demand-Forecasting-Inventory-Optimization/
│
├── dashboards/
│   └── data/
│       ├── processed/
│       │   ├── inventory_by_store.csv
│       │   ├── inventory_distribution.csv
│       │   ├── inventory_simulation.csv
│       │   ├── inventory_trend.csv
│       │   ├── kpis.csv
│       │   ├── orders_by_item.csv
│       │   ├── orders_by_store.csv
│       │   ├── recommended_orders.csv
│       │   └── top_store_item.csv
│       │
│       └── raw/
│
├── models/
│   └── my_model.pkl
│
├── notebooks/
│   ├── 01_Business_Understanding.ipynb
│   ├── 02_Model_Training.ipynb
│   ├── 03_Model_Training...ipynb
│   └── 04_Inventory_Simulation.ipynb
│
├── .gitignore
├── app.py
├── README.md
├── requirements.txt
│
└── src/
---

# ▶️ How to Run

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Demand-Forecasting-Inventory-Optimization.git
```

```bash
cd AI-Demand-Forecasting-Inventory-Optimization
```

## 2. Install dependencies

Using pip:

```bash
pip install -r requirements.txt
```

Or using UV:

```bash
uv sync
```

---

## 3. Run the Streamlit Dashboard

```bash
streamlit run dashboards/app.py
```

If you are using UV:

```bash
uv run streamlit run dashboards/app.py
```

The dashboard will then open in your browser.

---

# 📌 Key Business KPIs

The dashboard focuses on important inventory KPIs such as:

| KPI               | Purpose                               |
| ----------------- | ------------------------------------- |
| Total Inventory   | Current inventory quantity            |
| Forecasted Demand | Expected future demand                |
| Safety Stock      | Protection against uncertainty        |
| Reorder Point     | Inventory threshold for replenishment |
| Stockout Rate     | Frequency of inventory shortages      |
| Overstock         | Excess inventory                      |
| Recommended Order | Suggested replenishment quantity      |

---

# 💡 Business Impact

The system is designed to help businesses:

✅ Reduce stockouts
✅ Reduce excess inventory
✅ Improve demand planning
✅ Optimize replenishment
✅ Improve inventory visibility
✅ Make data-driven ordering decisions
✅ Understand demand patterns

---

# 🔬 Future Improvements

Potential improvements include:

* [ ] LSTM-based demand forecasting
* [ ] Prophet forecasting comparison
* [ ] LightGBM model comparison
* [ ] Automated hyperparameter tuning
* [ ] Advanced inventory cost optimization
* [ ] Supplier lead-time variability
* [ ] Multi-echelon inventory optimization
* [ ] Automated reorder alerts
* [ ] Real-time sales data integration
* [ ] Cloud deployment
* [ ] User authentication
* [ ] Automated model retraining
* [ ] Forecast uncertainty / prediction intervals

---

# 🏆 Project Highlights

This project demonstrates practical experience in:

```text
Python
   ↓
Data Analysis
   ↓
Time-Series Feature Engineering
   ↓
Machine Learning
   ↓
Demand Forecasting
   ↓
Inventory Optimization
   ↓
Simulation
   ↓
Explainable AI
   ↓
Interactive Dashboard
```

Rather than stopping at **"predict sales"**, the project converts predictions into **real inventory decisions**, making it a complete end-to-end data science and business analytics solution.

---

# 👨‍💻 Author

**Saad Shaikh**

B.Tech — Computer Science & Engineering (AI & ML)

Interested in:

* Data Science
* Machine Learning
* AI/ML Engineering
* Time-Series Forecasting
* Data Engineering
* Business Analytics

---

## ⭐ If you find this project useful

Consider giving the repository a ⭐ on GitHub.
