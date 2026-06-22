# Term Deposit Subscription Prediction

A machine-learning classifier that predicts whether a bank customer will subscribe to a term deposit, based on demographic, financial, and marketing-campaign data. Built with Python and Scikit-learn, with model comparison between Random Forest and XGBoost.

---

## Dataset

- **Source:** Bank Marketing dataset (`bank.csv`)
- **Size:** 11,162 records × 17 features
- **Target:** `deposit` (yes / no) — well balanced between the two classes
- **Features:** age, job, marital status, education, default, balance, housing loan, personal loan, contact type, day, month, call duration, campaign, pdays, previous contacts, and previous outcome (`poutcome`)
- **Missing values:** none

---

## Workflow

**1. Exploratory data analysis**
- Inspected distributions of all numerical features (histograms, box plots) and categorical features (count plots).
- Cross-tabulated each categorical feature against the target to understand which groups subscribe more often.

**2. Outlier handling**
- Removed extreme outliers in `campaign` (≥ 33 contacts) and `previous` (≥ 31 prior contacts) to reduce noise.

**3. Feature engineering**
- One-hot encoded categorical features (`job`, `marital`, `education`, `contact`, `month`, `poutcome`) using `drop_first` to avoid the dummy-variable trap.
- Converted binary yes/no fields (`housing`, `loan`, `deposit`) into 0/1 indicators.
- Final feature matrix: 41 engineered features.

**4. Modeling & tuning**
- Split data 80/20 into train/test sets.
- Trained and cross-validated a **Random Forest** classifier and tuned it with **GridSearchCV**.
- Trained an **XGBoost** classifier (`learning_rate=0.1`, `max_depth=10`, `n_estimators=100`).
- Compared models on the held-out test set.

---

## Results

**XGBoost was the best-performing model, achieving 85.8% test accuracy.**

| Model | Score |
|-------|-------|
| XGBoost (test accuracy) | **0.858** |
| Random Forest (5-fold cross-validated accuracy) | 0.853 |

**XGBoost confusion matrix (test set, 2,231 samples):**

|                | Predicted No | Predicted Yes |
|----------------|-------------:|--------------:|
| **Actual No**  | 993          | 186           |
| **Actual Yes** | 130          | 922           |

- **Precision (subscribers):** ~0.83 &nbsp;&nbsp; **Recall (subscribers):** ~0.88
- The model catches the large majority of customers who will actually subscribe (high recall), making it useful for targeting marketing campaigns efficiently.
- A feature-importance analysis was run on the XGBoost model to identify the strongest predictors of subscription.

---

## Tech stack

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `XGBoost` · `Matplotlib` · `Seaborn` · `Jupyter Notebook`

## How to run

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
jupyter notebook "term deposit project.ipynb"
```

Run the cells top to bottom — the notebook walks through EDA, cleaning, feature engineering, model training, tuning, and evaluation.

---

## What I learned

This project was end-to-end practice in a real classification problem: handling categorical features properly, removing outliers, and — importantly — comparing multiple models rather than settling for the first one. XGBoost edged out a tuned Random Forest, and looking at the confusion matrix (not just accuracy) showed the model was genuinely good at identifying likely subscribers.

---

*Built by Arya Khamkar.*
