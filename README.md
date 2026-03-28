# T20 Cricket Match Score Prediction using Machine Learning

## Project Overview

This project aims to predict the **final score of a T20 cricket match** using machine learning techniques. By analyzing match conditions such as runs, overs, wickets, and run rate, the system learns patterns that influence the final outcome.

The project is developed as part of the **CSE422: Artificial Intelligence Lab** course and follows the official project report template.

---

## Objectives

* Predict the final match score based on current match conditions
* Apply multiple machine learning models and compare their performance
* Perform Exploratory Data Analysis (EDA) to extract meaningful insights
* Understand the impact of features like overs, wickets, and run rate

---

## Dataset Description

* The dataset is a CSV file containing match-level information for T20 cricket matches
* Each row represents a match situation
* The **target variable** is: `total` (final score)

### Features

* **Numerical Features:**

  * Runs
  * Overs
  * Wickets
  * Run rate
  * Balls remaining (if available)

* **Categorical Features:**

  * Batting team
  * Bowling team
  * Venue (if present)

### Problem Type

* This is a **Regression Problem** because the output is a continuous numerical value

---

## Exploratory Data Analysis (EDA)

EDA was performed following standard methodology:

* **Univariate Analysis**

  * Distribution of final scores
  * Distribution of wickets

* **Bivariate Analysis**

  * Final score vs overs
  * Final score vs wickets
  * Final score vs run rate

* **Multivariate Analysis**

  * Boxplots for score distribution
  * Pairplot for feature relationships

### Key Insights

* Final score increases with overs and run rate
* Losing wickets reduces scoring potential
* Strong correlations exist between numerical features and the target

---

## ⚙️ Data Preprocessing

The following preprocessing steps were applied:

* Handling missing values (removal or imputation)
* Encoding categorical variables using **One-Hot Encoding**
* Feature scaling using **StandardScaler**

---

## Models Used

### Supervised Learning Models

* **Linear Regression**
* **Decision Tree Regressor**
* **Neural Network (MLP Regressor)**

### Unsupervised Learning

* **K-Means Clustering**

  * Used to group match situations into scoring categories

---

## 📈 Model Evaluation

The models were evaluated using:

* **R² Score** (model accuracy)
* **Mean Squared Error (MSE)** (prediction loss)

### Summary

* Neural Network performed the best
* Decision Tree captured non-linear patterns but risked overfitting
* Linear Regression had the lowest performance due to linear assumptions

---

## Project Structure

```
├── dataset/
│   └── t20_cricket_match_score_prediction.csv
│
├── eda/
│   ├── EDA_final_score_distribution.png
│   ├── EDA_score_vs_overs.png
│   ├── EDA_score_vs_wickets.png
│   ├── EDA_score_vs_runrate.png
│   └── EDA_pairplot.png
│
├── models/
│   └── model_training.py
│
├── report/
│   └── AI_Project_Report.pdf
│
├── README.md
└── requirements.txt
```

---

## How to Run the Project

1. Install required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

2. Run the EDA script:

```bash
python EDA.py
```

3. Run model training:

```bash
python main.py
```

4. Generated plots will be saved as `.png` files for report use.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## Challenges Faced

* Handling categorical variables
* Identifying correct target column (`total`)
* Avoiding overfitting in decision trees
* Tuning neural network parameters

---

## Future Improvements

* Use player-level and ball-by-ball datasets
* Apply advanced models like Random Forest or XGBoost
* Improve feature engineering

---

## Author

* Name: Shams Uz Zoha Mohammod
* Course: CSE422 - Artificial Intelligence Lab
* Institution: BRAC University

---

## Conclusion

This project demonstrates that machine learning models can effectively predict T20 cricket match scores. Among all models, Neural Networks provided the best performance due to their ability to capture complex patterns in the data.

---
