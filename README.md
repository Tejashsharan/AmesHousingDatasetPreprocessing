# Ames Housing — Feature Analysis & Preprocessing

A machine-learning preprocessing and exploratory feature-analysis notebook for the **Ames Housing** dataset. The notebook focuses on understanding which numerical and categorical property attributes are associated with house sale prices, handling missing values, engineering domain-specific features, encoding categorical variables, and preparing the data for a future house-price regression model.

> **Current status:** This project currently covers exploratory analysis and feature preprocessing. A regression model, train/test split, and final evaluation metrics have not yet been implemented in the notebook.

---

## 📌 Project Overview

The goal is to prepare the Ames Housing dataset for a house-price prediction task while learning practical feature-engineering and feature-selection techniques.

The workflow includes:

- Loading the Ames Housing dataset
- Separating numerical and categorical features
- Investigating missing values
- Applying domain-aware missing-value treatment
- Creating meaningful housing features
- Studying numerical feature correlations with `SalePrice`
- Testing categorical features using ANOVA
- Converting ordinal quality features into numerical scores
- Selecting statistically significant categorical features
- One-hot encoding nominal categorical features
- Applying log transformation to selected numerical variables

---

## 📊 Dataset

The project uses the **Ames Housing Dataset**, containing residential property records from Ames, Iowa.

### Dataset characteristics

| Property | Value |
|---|---:|
| Rows | 2,930 |
| Original features | 81 predictors + `SalePrice` |
| Numerical features | 39 |
| Categorical features | 43 |
| Target | `SalePrice` |

The dataset is downloaded through `kagglehub`.

Dataset source used in the notebook:

**Ames Housing Dataset — Kaggle**

`shashanknecrothapa/ames-housing-dataset`

---

## 🧠 Workflow

```text
Ames Housing Dataset
        │
        ▼
Data Loading
        │
        ▼
Feature Type Identification
        │
        ├── Numerical Features
        │
        └── Categorical Features
        │
        ▼
Missing Value Analysis
        │
        ▼
Domain-Based Imputation
        │
        ▼
Feature Engineering
        │
        ├── Total Floor Area
        ├── Total House Area
        ├── Total Basement Finished Area
        ├── House Age
        ├── Years Since Remodeling
        ├── Total Bathrooms
        ├── Total Outdoor Area
        └── Lot Utilization
        │
        ▼
Feature Analysis
        │
        ├── Correlation Analysis
        └── ANOVA for Categorical Features
        │
        ▼
Categorical Feature Processing
        │
        ├── Ordinal Encoding
        └── One-Hot Encoding
        │
        ▼
Log Transformation
        │
        ▼
Model-Ready Dataset
```

---

## 🔍 Exploratory Feature Analysis

### Numerical Features

Numerical features are identified using their pandas data type.

The notebook calculates the absolute correlation between numerical variables and `SalePrice` to identify potentially useful predictors.

A correlation heatmap is also generated for the selected numerical features.

### House Age

A domain-specific feature is created:

```text
HouseAge = Yr Sold - Year Built
```

This captures the age of the property at the time it was sold.

The notebook also checks its relationship with `SalePrice`.

---

## 🛠️ Feature Engineering

Several housing-specific features are engineered to represent information more directly useful for prediction.

### Total Floor Area

```text
TotalFloorSF = 1st Flr SF + 2nd Flr SF
```

Represents the total above-ground floor area.

### Total House Area

```text
TotalHouseSF = Total Bsmt SF + Gr Liv Area
```

Combines basement area with above-ground living area.

### Total Finished Basement Area

```text
TotalBsmtFinSF = BsmtFin SF 1 + BsmtFin SF 2
```

Represents the total finished basement area.

### House Age

```text
HouseAge = Yr Sold - Year Built
```

### Years Since Remodeling

```text
YearsSinceRemod = Yr Sold - Year Remod/Add
```

This captures how recently the property was remodeled relative to its sale.

### Total Bathrooms

Bathrooms are weighted so that a half bathroom contributes 0.5:

```text
TotalBath =
    Full Bath
    + 0.5 × Half Bath
    + Bsmt Full Bath
    + 0.5 × Bsmt Half Bath
```

### Total Outdoor Area

```text
TotalOutdoorSF =
    Wood Deck SF
    + Open Porch SF
    + Enclosed Porch
    + 3Ssn Porch
    + Screen Porch
```

### Lot Utilization

```text
LotUtilization = Gr Liv Area / (Lot Area + 1)
```

This approximates how much of the available lot is used for above-ground living space.

---

## 🧹 Missing Value Handling

Missing values are treated according to the meaning of the corresponding housing feature rather than applying a single blanket strategy.

For example, missing garage/basement measurements can represent the **absence of that facility**, so several variables are replaced with zero:

```python
zero_imputed_features = [
    "Garage Cars",
    "Garage Area",
    "Bsmt Full Bath",
    "Bsmt Half Bath",
    "BsmtFin SF 1",
    "BsmtFin SF 2",
    "Bsmt Unf SF",
    "Total Bsmt SF"
]
```

`Garage Yr Blt` is filled using `Year Built`.

Some categorical missing values are represented explicitly as `"NA"` or `"Unf"` where the missing value indicates the absence of a feature or facility.

---

## 📈 Categorical Feature Analysis

For categorical variables, the notebook uses **one-way ANOVA** to investigate whether the mean `SalePrice` differs significantly between groups of each categorical feature.

The hypothesis being investigated is approximately:

```text
H₀: The mean SalePrice is the same across categories.

H₁: At least one category has a different mean SalePrice.
```

A significance threshold of:

```text
p < 0.01
```

is used to identify statistically significant categorical variables.

The analysis found **40 categorical features** meeting this significance threshold before the final subset used in the notebook.

The notebook then works with a selected set of 15 statistically significant categorical features:

```text
Street
Land Slope
BsmtFin Type 2
Heating
Fence
Roof Matl
Functional
Lot Config
Condition 2
Exter Cond
Bsmt Cond
Alley
Mas Vnr Type
Garage Cond
Bldg Type
```

> Note: these 15 features are the first 15 entries from the notebook's sorted significant-feature dictionary. They should not be interpreted as the 15 strongest categorical predictors.

---

## 🏠 Ordinal Feature Encoding

Several housing-quality features have a natural order.

For example:

```text
NA → 0
Po → 1
Fa → 2
TA → 3
Gd → 4
Ex → 5
```

This mapping is applied to features such as:

```text
Exter Qual
Exter Cond
Bsmt Qual
Bsmt Cond
Heating QC
Kitchen Qual
Fireplace Qu
Garage Qual
Garage Cond
Pool QC
```

This preserves the ordinal relationship between categories instead of treating them as unrelated labels.

---

## 🔤 One-Hot Encoding

Nominal categorical features do not have an inherent numerical order.

The notebook identifies the relevant nominal features and applies one-hot encoding using:

```python
pd.get_dummies(
    data=df,
    columns=nominal_feature,
    drop_first=True
)
```

`drop_first=True` removes one reference category from each encoded feature to reduce redundant columns.

After encoding, the dataframe contains:

```text
2930 rows × 130 columns
```

---

## 📐 Log Transformation

Selected numerical features are transformed using:

```python
np.log(1 + x)
```

The transformation is useful for reducing the effect of highly skewed variables and can make relationships more suitable for regression models.

The `+1` allows zero-valued observations to be transformed safely:

```text
log(1 + 0) = 0
```

The notebook applies this transformation to the selected numerical feature set.

---

## 🧰 Technologies Used

- **Python**
- **Pandas** — data manipulation and analysis
- **NumPy** — numerical operations and feature transformations
- **Matplotlib** — visualization
- **Seaborn** — correlation heatmap
- **SciPy** — statistical analysis / ANOVA
- **KaggleHub** — dataset download
- **Jupyter Notebook / Google Colab**

---

## 📁 Project Structure

```text
Ames-Housing/
│
├── AmesHousing.ipynb
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Ames-Housing
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scipy kagglehub jupyter
```

### 3. Start Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
AmesHousing.ipynb
```

Alternatively, the notebook can be run directly in Google Colab.

---

## 📌 Current Results

The preprocessing pipeline produces a dataset with:

```text
2,930 observations
130 processed columns
```

The notebook also demonstrates:

- Numerical/categorical feature identification
- Missing-value analysis
- Domain-driven feature engineering
- Numerical correlation analysis
- Categorical ANOVA
- Ordinal encoding
- One-hot encoding
- Log transformation

No final regression performance is reported yet because a predictive model has not been trained in the current notebook.

---

## 🔮 Next Steps

The natural next stage of the project is to build and evaluate regression models.

### Recommended progression

1. Create a train/test split
2. Separate `X` and `y`
3. Prevent target leakage during preprocessing
4. Build a preprocessing pipeline
5. Train baseline regression models
6. Compare multiple algorithms:
   - Linear Regression
   - Ridge Regression
   - Lasso Regression
   - Elastic Net
   - Random Forest
   - Gradient Boosting
   - XGBoost / LightGBM
7. Use cross-validation
8. Tune hyperparameters
9. Evaluate using:
   - MAE
   - RMSE
   - R²
10. Analyze residuals
11. Compare predictions vs. actual prices
12. Save the best model for inference

---

## 🎯 Learning Objectives

This project is particularly useful for practicing:

- Exploratory Data Analysis
- Feature engineering
- Missing-value treatment
- Correlation analysis
- Statistical feature selection
- ANOVA
- Ordinal encoding
- One-hot encoding
- Data transformation
- Preparing tabular data for regression
- Thinking about features from a domain perspective

---

## 📄 Notebook

The main implementation is available in:

```text
AmesHousing.ipynb
```

---

## 👤 Author

**Tejash Sharan**

Computer Science Engineering | Artificial Intelligence

---

## ⭐ Future Improvement

The project can be extended from a preprocessing notebook into a complete end-to-end **House Price Prediction System**, including model comparison, cross-validation, hyperparameter tuning, explainability with SHAP, and a deployment interface.
