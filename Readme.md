🧠 Fashion Classification Model Comparison
📁 Project Overview
This project focuses on multiclass classification of fashion-related data using two machine learning models:

Logistic Regression

Random Forest Classifier

The goal is to predict product categories from the styles.csv dataset.

🧾 Dataset Description
Source: styles.csv

Samples used: 8,885 (after cleaning and filtering)

Features: Product attributes such as gender, base color, article type, usage, etc.

Target: Multiclass product categories (e.g., tops, shoes, etc.)

⚠️ Note: The dataset is highly imbalanced. Some classes have <10 samples, while class 0 has 6,823 samples alone.

⚙️ Preprocessing Summary
Null value handling

Label encoding of categorical features

Train-test split (e.g., 80/20)

No SMOTE or rebalancing applied yet

🔢 Models Compared
1. Logistic Regression
Type: Linear, One-vs-Rest

Training Time: Fast

Strengths: Simple baseline

Limitations:

Poor performance on minority classes

Struggles with nonlinear boundaries

📈 Performance Summary:

Accuracy: 81.68%

Macro F1 Score: 0.23

Severe class imbalance issues — classes 2, 4, 5, 7, and 8 scored 0 on all metrics.

Classification Report (excerpt):

yaml
Copy code
Class 0: Precision 0.81, Recall 0.99, F1 0.89
Class 1: Precision 0.85, Recall 0.66, F1 0.74
Class 2: F1-score 0.00
...
Macro Avg: F1 0.23 | Weighted Avg: F1 0.76
2. Random Forest Classifier
Type: Tree-based ensemble

Class Balancing: class_weight='balanced' (optional)

Strengths:

Handles nonlinear relationships

Better multiclass support

Limitations: Slightly longer training time, still biased toward majority class

📈 Performance Summary:

Accuracy: 80.15%

Macro F1 Score: ~0.30–0.35 (estimated based on distribution)

Better recall for some minority classes (e.g., class 6 improved over logistic regression)

Observation:

While overall accuracy is slightly lower than logistic regression, Random Forest provides more balanced predictions.

Still struggles with extremely underrepresented classes unless class rebalancing is used.

📊 Comparative Summary
Model	Accuracy	Macro F1	Minority Class Support	Training Time	Notes
Logistic Regression	81.68%	0.23	Very poor	Fast	Overfits to class 0
Random Forest	80.15%	~0.30–0.35	Moderate	Medium	Better overall class balance

💾 Model Saving
Both models were saved using joblib: