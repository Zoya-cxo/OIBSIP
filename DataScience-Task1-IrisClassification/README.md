# Task 1 — Iris Flower Classification

Data Science Internship, Oasis Infobyte SIP

## Objective
Train and compare machine learning models to classify iris flowers into one of three species (Setosa, Versicolor, Virginica) based on their sepal and petal measurements.

## Dataset
Built directly into scikit-learn via `sklearn.datasets.load_iris()` — 150 samples, 4 numeric features, 3 balanced classes, no missing values.

## Tech Stack
Python, pandas, scikit-learn, matplotlib, seaborn, Jupyter Notebook

## Approach
- Exploratory Data Analysis (shape, types, nulls, descriptive stats, class balance)
- Visualizations: pairplot, box plots per feature, correlation heatmap
- Feature selection discussion
- 80/20 stratified train/test split with feature scaling
- Four trained classifiers: Logistic Regression, K-Nearest Neighbours, Decision Tree, Random Forest
- Evaluation: accuracy, confusion matrices, classification reports
- Best model selection with justification

## Results
| Model | Accuracy |
|---|---|
| Logistic Regression | 93.3% |
| K-Nearest Neighbours | 93.3% |
| Decision Tree | 93.3% |
| Random Forest | 90.0% |

## Key Insight
Every model correctly classified 100% of setosa flowers. All misclassifications happened between versicolor and virginica, which have some overlap in their petal measurements.

## Author
Zoya Chaudhary
