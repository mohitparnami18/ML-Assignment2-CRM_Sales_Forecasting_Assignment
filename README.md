# CRM Sales Opportunity Forecasting

## Problem Statement

Predict whether a closed CRM sales opportunity will be **Closed Won** or **Closed Lost** using customer/account characteristics, sales-source information, owner information, historical account activity and historical sales performance.

## Dataset Source

Source: Kaggle

Dataset: Sales CRM Data

URL: https://www.kaggle.com/datasets/sushicatsan/sample-sales-crm-data/croissant/download

The dataset contains CRM entities including Opportunity,
Account, Lead, Contact, Order and User records.

## Dataset Description

The project uses a public CRM-style B2B sales dataset containing Salesforce-like entities such as:

- Opportunity
- Account
- Lead
- Contact
- Order
- User

The opportunity table contains 16,000 records and the account table contains 10,000 records. The classification target is created from the final opportunity stage:

- `Closed Won` = 1
- `Closed Lost` = 0

Only closed opportunities are used for supervised classification because their final outcome is known.

The final modelling dataset contains more than 12 features and more than 500 instances.

## Important Data-Leakage Prevention

The following outcome-revealing fields are intentionally excluded from the model:

- `stage_name`
- `probability`
- `close_date`
- `days_to_close`

Including these fields for closed opportunities would directly reveal the target.

The project instead creates forecasting-oriented historical features such as:

- prior account win rate
- prior owner win rate
- prior lead-source win rate
- prior customer order count
- prior customer order value
- prior contact count
- owner tenure
- opportunity amount / account revenue ratio

Historical rates are calculated using only opportunities created before the current opportunity.

A chronological 60% / 20% / 20% train-validation-test split is used so that the final test period represents a future period relative to the training data.

## Models Used

Six models are implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Gaussian Naive Bayes
5. Random Forest
6. Gradient Boosting

The assignment text says "6 ML models" but explicitly lists only five. Gradient Boosting is therefore included as the sixth model.

## Evaluation Metrics

Every model is evaluated using:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

The decision threshold is selected using validation data only. The final test data remains untouched during threshold selection.

## Repository Structure

```text
CRM_Sales_Forecasting_Assignment/
│
├── app.py
├── requirements.txt
├── README.md
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

## Streamlit Features

The application provides:

- Test dataset CSV upload
- Model-selection dropdown
- Accuracy, AUC, Precision, Recall, F1 and MCC
- Confusion matrix
- Classification report
- Prediction table
- Prediction-results download

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the source CSV files in a `data/` directory:

```text
data/
├── account.csv
├── contact.csv
├── lead.csv
├── opportunity.csv
├── order.csv
└── user.csv
```

Train and save the models:

```bash
cd model
python train_and_save.py
```

Start Streamlit:

```bash
streamlit run app.py
```

## Model Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Gradient Boosting | 0.8047 | 0.8025 | 0.8038 | 0.9559 | 0.8733 | 0.4958 |
| Random Forest | 0.8009 | 0.7795 | 0.8299 | 0.9022 | 0.8645 | 0.4979 |
| Decision Tree | 0.7979 | 0.7348 | 0.8209 | 0.9118 | 0.8640 | 0.4850 |
| Logistic Regression | 0.7812 | 0.8033 | 0.8697 | 0.8108 | 0.8392 | 0.5015 |
| KNN | 0.7358 | 0.7110 | 0.7415 | 0.9591 | 0.8364 | 0.2621 |
| Naive Bayes | 0.7350 | 0.7048 | 0.7575 | 0.9172 | 0.8298 | 0.2818 |

### Observations

Logistic Regression

Logistic Regression achieved an accuracy of 78.12% and the highest AUC of 0.8033 among the models. It also achieved the highest precision of 86.97% and MCC of 0.5015. This indicates that the engineered CRM features contain meaningful linear predictive relationships and that Logistic Regression is particularly effective when avoiding false positive Closed Won predictions is important.

Decision Tree

Decision Tree achieved 79.79% accuracy with an F1 score of 86.40%. Its recall of 91.18% indicates that it identifies most Closed Won opportunities. However, its AUC of 0.7348 is lower than the ensemble and Logistic Regression models, suggesting weaker overall ranking/discrimination capability.

KNN

KNN achieved 73.58% accuracy and an AUC of 0.7110. Although its recall was high at 95.91%, its MCC of 0.2621 was substantially lower than the other stronger models. This indicates that KNN identifies positive opportunities well but has weaker overall class separation.

Naive Bayes

Gaussian Naive Bayes achieved 73.50% accuracy and an AUC of 0.7048. Its recall was 91.72%, but its MCC of 0.2818 was relatively low. The comparatively lower performance suggests that the independence assumptions of Naive Bayes are not ideally suited to the relationships among the CRM features.

Random Forest

Random Forest achieved 80.09% accuracy and an F1 score of 86.49%. It provides a strong balance between precision, recall and classification accuracy. Its MCC of 0.4965 is also close to the best value obtained by Logistic Regression, demonstrating strong balanced classification performance.

Gradient Boosting

Gradient Boosting achieved the highest accuracy of 80.47% and the highest F1 score of 87.33%. It also achieved a high AUC of 0.8025 and recall of 95.59%. These results indicate that Gradient Boosting provides the strongest overall performance for this dataset, particularly when correctly identifying Closed Won opportunities is important.

- **Gradient Boosting** is the overall winner by test accuracy at **80.47%** and has a strong AUC of **0.8025**.
- **Random Forest** is a close second in accuracy and has a strong MCC (**0.4965**), indicating strong balanced classification quality.
- **Logistic Regression** has the highest precision (**0.8697**) and a strong AUC, showing that the engineered historical features provide useful linear signal.
- **Decision Tree** provides a strong balance of recall and accuracy while remaining easy to interpret.
- **KNN** and **Naive Bayes** achieve lower accuracy than the tree-based ensemble models, although both maintain high recall.
- The chronological split is intentional: the latest 20% of opportunities are treated as a future test period, which is more appropriate for a sales forecasting use case than randomly mixing future and past opportunities.

### Overall Winner

Overall Winner: Gradient Boosting

Gradient Boosting is selected as the overall winner based on the highest test accuracy of 80.47% and highest F1 score of 87.33%. It also provides a strong AUC of 0.8025 and recall of 95.59%. Random Forest is a close second with 80.09% accuracy, while Logistic Regression achieves the highest AUC, precision and MCC. Therefore, Gradient Boosting provides the best overall balance for the selected sales opportunity classification problem.

## GitHub Repository Link

https://github.com/mohitparnami18/ML-Assignment2-CRM_Sales_Forecasting_Assignment

## Live Streamlit App Link

https://ml-assignment2-crmsalesforecastingassignment.streamlit.app/


## Troubleshooting: numeric/object dtype error

If you see an error such as:

```text
ValueError: Cannot use median strategy with non-numeric data
```

make sure you are using the latest `model/feature_engineering.py` and
`model/train_and_save.py` from this repository. Boolean CRM fields are
explicitly converted to 0/1 and categorical fields are explicitly routed to
the categorical preprocessing pipeline.

Run:

```bash
python model/train_and_save.py
```

from the project root. Do not use an older copy of `train_and_save.py`.
