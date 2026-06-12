 # 🚗 Indian Used Car Listed Price Prediction                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                

A machine learning project to predict the **listed price of used cars in India** using a clean, production-ready scikit-learn Pipeline. The project covers end-to-end ML — from EDA and data cleaning to model training, hyperparameter tuning, and deployment.

---



## 📌 Problem Statement

Used car prices in India vary widely based on brand, model, age, fuel type, condition, and many other factors. The goal is to build a regression model that accurately predicts the **listed price** of a used car given its features — helping buyers and sellers make informed decisions.

---

## 📂 Project Structure

```
├── car_listed_price_prediction.ipynb   # Main notebook (EDA → Model → Deployment)
├── final_random_forest_model.pkl       # Saved production-ready model pipeline
├── indian_used_car_noisy_2500.csv      # Dataset
├── images
├── LICENSE
└── README.md
```

---

## 📊 Dataset

- **Records:** 2,500 Indian used car listings
- **Target:** `listed_price` (in ₹)

| Feature | Type | Description |
|---|---|---|
| `brand` | Categorical | Car manufacturer (e.g. Maruti, Hyundai) |
| `model` | Categorical | Car model name (e.g. Swift, i20) |
| `fuel_type` | Categorical | Petrol / Diesel / CNG |
| `km_driven` | Numerical | Total kilometers driven |
| `car_age` | Numerical | Age of the car in years |
| `condition_rating` | Numerical | Rated condition of the car (1–10) |
| `premium_car` | Binary | 1 if premium segment, 0 otherwise |
| `transmission` | Categorical | Manual / Automatic |
| `owner_type` | Categorical | First / Second / Third owner |
| `city` | Categorical | City where car is listed |
| `mileage` | Numerical | Fuel efficiency (km/l) |
| `engine_cc` | Numerical | Engine displacement in CC |
| `demand_score` | Numerical | Market demand score |
| `brand_popularity_score` | Numerical | Brand popularity index |

---

## ⚙️ Feature Engineering

Two new features were derived from existing columns:

```python
# Car condition relative to its age
condition_per_year = condition_rating / (car_age + 1)

# Demand weighted by age
demand_relative_to_age = demand_score * car_age
```

> ⚠️ **Note for inference:** `condition_per_year` must be computed before passing data to the model.
> Formula: `condition_per_year = condition_rating / (car_age + 1)`
> Example: `condition_rating=5`, `car_age=1` → `5 / (1+1) = 2.5`

---

## 🧪 Models Trained

Two separate preprocessing pipelines were built — one for linear models (with scaling) and one for tree models (no scaling needed):

| Model | Preprocessor | Notes |
|---|---|---|
| Linear Regression | StandardScaler + TE + OHE | Baseline linear model |
| Lasso Regression | StandardScaler + TE + OHE | L1 regularization |
| Ridge Regression | StandardScaler + TE + OHE | L2 regularization |
| Decision Tree | passthrough + TE + OHE | No scaling needed |
| **Random Forest** | passthrough + TE + OHE | **Best model — deployed** |

**Encoding:**
- `TargetEncoder` → `brand`, `model` (high cardinality)
- `OneHotEncoder` → `fuel_type`
- `StandardScaler` → numeric columns (linear models only)

---

## 🔧 Hyperparameter Tuning

| Model | Method | Search Space |
|---|---|---|
| Lasso | GridSearchCV (5-fold) | `alpha`: 6 values |
| Ridge | GridSearchCV (5-fold) | `alpha`: 5 values |
| Decision Tree | GridSearchCV (5-fold) | `criterion`, `max_depth`, `max_features` |
| **Random Forest** | **RandomizedSearchCV** (5-fold, 15 iter) | `n_estimators`, `max_depth`, `min_samples_split`, `max_features` |

> RandomizedSearchCV was used for Random Forest to reduce ~270 fits (GridSearch) to just 75 fits — cutting tuning time by ~70% while still exploring the parameter space effectively.

---

## 🚀 Deployment

The best model (Tuned Random Forest) is saved as a full scikit-learn Pipeline — it handles all preprocessing internally, so you only need to pass raw input.

### Load and Predict

```python
import joblib
import pandas as pd

# Load model
model = joblib.load("final_random_forest_model.pkl")

# Prepare input (compute condition_per_year = condition_rating / (car_age + 1))
new_data = pd.DataFrame([{
    'brand': 'Mahindra',
    'model': 'XUV700',
    'fuel_type': 'CNG',
    'km_driven': 19600,
    'car_age': 1,
    'premium_car': 1,
    'condition_per_year': 5.0   # = condition_rating / (car_age + 1)
}])

# Predict
prediction = model.predict(new_data)
print(f"Predicted Listed Price: ₹{prediction[0]:,.0f}")
```

---

## 🛠️ Tech Stack

- **Python 3.x**
- **pandas**, **numpy** — data manipulation
- **matplotlib**, **seaborn** — visualization
- **scikit-learn** — modeling, pipelines, tuning
- **joblib** — model serialization

---

## 📈 Results

| Model | R² Score | RMSE (₹) |
|---|---|---|
| Linear Regression | 0.8522 | 466140.3475 |
| Lasso (Tuned) | 0.8522 | 466160.676 |
| Ridge (Tuned) | 0.8523 | 466063.9822 |
| Decision Tree (Tuned) | 0.944 | 286865.2159 |
| **Random Forest (Tuned)** | **0.9716** | **04381.9509** |

> ✏️ Fill in your actual scores after running the notebook.

---

## 🏃 How to Run

1. Clone the repository
```bash
git clone https://github.com/your-username/indian-used-car-price-prediction.git
cd indian-used-car-price-prediction
```

2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

3. Open the notebook
```bash
jupyter notebook car_listed_price_prediction.ipynb
```

4. Run all cells top to bottom — the model will be saved automatically as `final_random_forest_model.pkl`

---

## 👤 Author

**Your Name**
- GitHub: [@rohannn16](https://github.com/rohannn16?tab=repositories)
- LinkedIn: [Rohan Kumar](www.linkedin.com/in/rohankumards16)
- Email: diliprohankumar@gmail.com                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
