# CRM Sales Opportunity Forecasting

## a. Problem Statement

The objective of this project is to predict whether a closed CRM sales opportunity will be **Closed Won** or **Closed Lost** using customer/account characteristics, sales-source information, owner information, historical account activity and historical sales performance.

This is formulated as a **binary classification problem**:

- `Closed Won = 1`
- `Closed Lost = 0`

The solution compares six machine learning classification models and evaluates them using Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC).

---

## b. Dataset Description

### Dataset Source

**Source:** Kaggle  
**Dataset:** Sales CRM Data  
**URL:** https://www.kaggle.com/datasets/sushicatsan/sample-sales-crm-data/croissant/download

The dataset contains CRM entities including:

- Opportunity
- Account
- Lead
- Contact
- Order
- User

The Opportunity table contains approximately 16,000 records and the Account table contains approximately 10,000 records.

Only opportunities with a known final outcome are used for supervised classification:

- `Closed Won`
- `Closed Lost`

The final modelling dataset contains:

- **6,604 closed opportunities**
- **34 model features**
- **1,321 test instances**

This satisfies the assignment requirements of at least 12 features and at least 500 instances.

### Target Distribution

The final test dataset contains:

- Closed Won: **930**
- Closed Lost: **391**
- Test win rate: **70.40%**

### Data Leakage Prevention

The following outcome-revealing fields are intentionally excluded:

- `stage_name`
- `probability`
- `close_date`
- `days_to_close`

Including these fields for closed opportunities could directly reveal the final outcome.

Instead, the project uses forecasting-oriented historical features such as:

- Prior account opportunity count
- Prior account win rate
- Prior owner opportunity count
- Prior owner win rate
- Prior lead-source win rate
- Prior customer order count
- Prior customer order value
- Prior contact count
- Owner tenure
- Opportunity amount / account revenue ratio

Historical rates are calculated using only opportunities created before the current opportunity.

### Train / Validation / Test Strategy

A chronological split is used:

- **60% Training**
- **20% Validation**
- **20% Test**

This prevents future opportunities from being used to train the model for earlier periods and is more appropriate for a sales forecasting use case than a random split.

---

## c. GitHub Repository Link

https://github.com/mohitparnami18/ML-Assignment2-CRM_Sales_Forecasting_Assignment

## Live Streamlit App Link

**Replace this line with the final Streamlit Community Cloud URL after deployment.**

`ADD_STREAMLIT_APP_URL_HERE`

---

## d. Models Used

The assignment states that six ML models are required but explicitly lists five model names. Therefore, the project implements the five specified models plus Gradient Boosting as the sixth model.

### Models

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Gaussian Naive Bayes
5. Random Forest
6. Gradient Boosting

### Evaluation Metrics

Every model is evaluated using:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

The decision threshold is selected using validation data only. The final test data is not used for threshold selection.

---

## Model Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| **Gradient Boosting** | **0.8047** | 0.8025 | 0.8038 | **0.9559** | **0.8733** | 0.4958 |
| Random Forest | 0.8009 | 0.7803 | 0.8279 | 0.9054 | 0.8649 | 0.4965 |
| Decision Tree | 0.7979 | 0.7348 | 0.8209 | 0.9118 | 0.8640 | 0.4850 |
| Logistic Regression | 0.7812 | **0.8033** | **0.8697** | 0.8108 | 0.8392 | **0.5015** |
| KNN | 0.7358 | 0.7110 | 0.7415 | **0.9591** | 0.8364 | 0.2621 |
| Naive Bayes | 0.7350 | 0.7048 | 0.7575 | 0.9172 | 0.8298 | 0.2818 |

---

## Observations on Model Performance

### Logistic Regression

Logistic Regression achieved **78.12% accuracy** and the highest AUC of **0.8033**. It also achieved the highest precision of **86.97%** and MCC of **0.5015**. This indicates that the engineered CRM features contain meaningful linear predictive relationships. Logistic Regression is particularly useful when avoiding false positive Closed Won predictions is important.

### Decision Tree

Decision Tree achieved **79.79% accuracy** with an F1 score of **86.40%**. Its recall of **91.18%** indicates that it identifies most Closed Won opportunities. However, its AUC of **0.7348** is lower than the ensemble and Logistic Regression models, suggesting weaker overall ranking/discrimination capability.

### KNN

KNN achieved **73.58% accuracy** and an AUC of **0.7110**. Although its recall was high at **95.91%**, its MCC of **0.2621** was substantially lower than the stronger models. This indicates that KNN identifies positive opportunities well but has weaker overall class separation.

### Naive Bayes

Gaussian Naive Bayes achieved **73.50% accuracy** and an AUC of **0.7048**. Its recall was **91.72%**, but its MCC of **0.2818** was relatively low. The comparatively lower performance suggests that the independence assumptions of Naive Bayes are not ideally suited to the relationships among the CRM features.

### Random Forest

Random Forest achieved **80.09% accuracy** and an F1 score of **86.49%**. It provides a strong balance between precision, recall and classification accuracy. Its MCC of **0.4965** is also close to the best value obtained by Logistic Regression, demonstrating strong balanced classification performance.

### Gradient Boosting

Gradient Boosting achieved the highest test accuracy of **80.47%** and the highest F1 score of **87.33%**. It also achieved a high AUC of **0.8025** and recall of **95.59%**. These results indicate that Gradient Boosting provides the strongest overall performance for this dataset, particularly when correctly identifying Closed Won opportunities is important.

---

## Overall Winner

**Gradient Boosting**

Gradient Boosting is selected as the overall winner based primarily on the highest test accuracy (**80.47%**) and highest F1 score (**87.33%**). It also provides a strong AUC (**0.8025**) and recall (**95.59%**).

Random Forest is a close second with **80.09% accuracy**.

Logistic Regression achieves the highest AUC (**0.8033**), precision (**86.97%**) and MCC (**0.5015**). Therefore, model selection can depend on the business objective, but Gradient Boosting provides the best overall result when accuracy and F1 are prioritized.

---

## Majority-Class Baseline

The test dataset has a Closed Won rate of **70.40%**.

A naive classifier that always predicts Closed Won would therefore achieve approximately **70.40% accuracy**.

The winning Gradient Boosting model achieves **80.47% accuracy**, which is approximately **10.07 percentage points above the majority-class baseline**.

This demonstrates that the model provides meaningful predictive improvement beyond simply predicting the majority class.

---

## Streamlit Application

The deployed Streamlit application provides:

- CSV test-data upload
- Model-selection dropdown
- Comparison table for all six models
- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- MCC
- Confusion matrix
- Classification report
- Prediction results
- Prediction-results download

The application uses the saved trained models and does not retrain the models during deployment.

### Streamlit Main File

`app.py`

### Streamlit Deployment Configuration

The Streamlit Community Cloud **Main file path must be**:

`app.py`

Do not use:

`model/train_and_save.py`

The training script requires the original CRM source files and is not the Streamlit entry point.

---

## Repository Structure

```text
CRM_Sales_Forecasting_Assignment/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── test_data.csv
├── model_comparison.csv
│
└── model/
    ├── feature_engineering.py
    ├── train_and_save.py
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── random_forest.joblib
    ├── gradient_boosting.joblib
    └── metadata.json
```

---

## Local Execution

Install dependencies:

```bash
pip install -r requirements.txt
```

For model training, place the original CRM source CSV files in:

```text
data/
├── account.csv
├── contact.csv
├── lead.csv
├── opportunity.csv
├── order.csv
└── user.csv
```

Run training from the project root:

```bash
python model/train_and_save.py
```

This creates/updates the saved model files, metadata and test data.

Run Streamlit:

```bash
streamlit run app.py
```

---

## Important Deployment Note

Only the saved model files and `test_data.csv` are required by the Streamlit application.

The original CRM source files used for training are **not required for deployment**.

The Streamlit application loads the saved `.joblib` models from the `model/` directory.

---

## Academic Integrity

The project was developed as an original implementation for the assignment. The modelling workflow uses a chronological train/validation/test split and explicitly excludes outcome-revealing fields to reduce data leakage.

The repository should contain genuine development commits and should not include local virtual environments or IDE-specific files.
