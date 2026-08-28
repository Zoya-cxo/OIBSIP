# Task 2 — Car Price Prediction with Machine Learning

## Objective
Build a regression model that predicts the selling price of a used car based on features like brand, age, mileage, fuel type, and transmission.

## Dataset
[Vehicle dataset from CarDekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho) — Kaggle, publicly available.

## Tech Stack
Python, pandas, numpy, scikit-learn, matplotlib, seaborn, Jupyter Notebook

## Approach
- **Data cleaning:** removed ~1,200 duplicate rows; extracted numeric values from `mileage`, `engine`, `max_power` (stripped embedded units like "kmpl", "CC", "bhp"); parsed `torque` with a two-tier regex (unit directly attached → fallback inference), detected and nulled 56 corrupted boilerplate entries before dropping remaining nulls (~3-4% of data)
- **Feature engineering:** extracted `brand` from `name`; derived `car_age` from `year` (dropped `year` afterward — perfectly correlated)
- **EDA:** distribution of `selling_price` (right-skewed), price by fuel type, price vs car age (non-linear depreciation curve), correlation heatmap
- **Encoding:** one-hot encoding for nominal categoricals (`fuel`, `seller_type`, `transmission`, `brand`); ordinal encoding for `owner`, ranked by wear/condition rather than raw count
- **Modeling:** Linear Regression (baseline) and Random Forest Regressor, evaluated on an 80/20 train/test split

## Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | ₹136,204.86 | ₹245,039.44 | 0.7286 |
| Random Forest | ₹70,485.52 | ₹126,208.70 | 0.9280 |

## Key Insight
`max_power` is by far the strongest predictor (~48% feature importance), followed by `car_age` and `torque` — consistent with the correlation heatmap from EDA. Random Forest substantially outperforms Linear Regression, largely because it handles the non-linear depreciation curve and right-skewed price distribution that a straight-line model struggles with.

## How to Run
```bash
git clone <repo-url>
cd DataScience-Task2-CarPricePrediction
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter lab car_price_prediction.ipynb
```
