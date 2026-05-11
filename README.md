# Customer_Segmentation_Retention
ML Project: Customer segmentation, churn prediction, and CLV analysis on 10,000+ bank customers
# Customer Segmentation and Retention Analysis
### Predicting Churn and Identifying At-Risk Customers 
### in Retail Banking

**Author:** Success Chukwuka  
**Tools:** Python | Pandas | NumPy | Matplotlib | 
Seaborn | Scikit-learn | Google Colab  
**Dataset:** Kaggle - Bank Churners (Credit Card Customers)  
**Status:** Completed

---

## The Business Problem

Banks lose significant revenue every time a customer churns.
The challenge is that most customers do not announce they are
leaving. They quietly disengage, reduce activity and eventually
close their accounts.

This project asks three questions a bank needs answered before
it is too late:

- Which customers are at risk of leaving?
- How valuable are those customers to the business?
- What should the bank do about it right now?

---

## The Dataset

| Property | Detail |
|---|---|
| Source | Kaggle - Bank Churners Dataset |
| Rows | 10,127 customers |
| Original Columns | 23 |
| Working Columns | 21 (2 leakage columns dropped) |
| Missing Values | None confirmed via df.isnull().sum() |
| Class Balance | 83.93% retained, 16.07% churned |

**Key features used in this analysis:**

| Feature | Description |
|---|---|
| Customer_Age | Age of the customer |
| Credit_Limit | Maximum credit available |
| Total_Trans_Amt | Total transaction amount over 12 months |
| Total_Trans_Ct | Total number of transactions over 12 months |
| Months_Inactive_12_mon | Months of inactivity in last 12 months |
| Total_Revolving_Bal | Outstanding revolving balance |
| Attrition_Flag | Churn label and target variable |
| Contacts_Count_12_mon | Bank contact frequency in 12 months |
| Avg_Utilization_Ratio | Credit utilization rate |

---

## Project Structure
---

## My Approach

### Section 1: Problem Definition
Defined the business questions and identified the target
variable: Attrition_Flag indicating whether a customer
churned or was retained.

### Section 2: Data Loading and Inspection
Loaded the dataset using Pandas. Dropped the two leakage
columns flagged by Kaggle that would artificially inflate
model performance. Confirmed 10,127 rows and 21 working
columns.

### Section 3: Data Quality Check
Ran df.isnull().sum() across all columns. Confirmed zero
missing values. No imputation was required.

### Section 4: Exploratory Data Analysis
Investigated distributions, class balance and behavioral
differences between churned and retained customers using
Matplotlib and Seaborn visualizations.

### Section 5: Feature Engineering
Created an Inactivity_Score engineered feature that
combines inactivity months and contact frequency into a
single behavioral signal. This feature became one of the
top predictors of churn.

### Section 6: Customer Segmentation with K-Means
Applied K-Means clustering to segment customers into
distinct behavioral groups. Used the Elbow Method to
determine the optimal number of clusters.

### Section 7: Churn Prediction Modeling
Built three classification models: Logistic Regression
as a baseline, Random Forest and Gradient Boosting.
Evaluated all three using AUC-ROC, chosen specifically
because accuracy is misleading on imbalanced datasets.

### Section 8: Feature Importance Analysis
Extracted and visualized feature importance from the
Random Forest model to identify the strongest behavioral
drivers of churn.

### Section 9: Business Recommendations
Translated model outputs into a tiered retention strategy
based on customer lifetime value and churn probability.

---

## Key Findings

**Finding 1: The dataset is significantly imbalanced**
16.07% of customers churned versus 83.93% who stayed.
This imbalance makes accuracy an unreliable metric and
AUC-ROC the correct evaluation choice.

**Finding 2: Churned customers show clear behavioral signals**
Customers who churned had significantly fewer transactions,
higher inactivity months, lower credit utilization and more
frequent contact with the bank compared to retained customers.
The behavioral divergence is detectable before churn occurs.

**Finding 3: High contact frequency is a warning signal**
Customers who churned contacted the bank more often than
those who stayed. More contacts did not mean more loyalty.
It meant more dissatisfaction.

**Finding 4: Credit utilization drops before departure**
Churned customers stopped using their cards before they
officially closed their accounts. Declining utilization
is an early warning signal the bank can act on.

**Finding 5: The fraud detection connection**
The behavioral features that predict churn are identical
to early warning signals used in financial fraud detection.
Inactivity, declining transactions and high contact
frequency appear in both churn models and financial crime
detection systems. This is not a coincidence. Both churn
and fraudulent behavior manifest as deviation from normal
account activity patterns.

---

## Customer Segments Identified

Using K-Means clustering with 4 clusters determined by
the Elbow Method:

| Segment | Profile | Churn Risk | Action |
|---|---|---|---|
| High-Value Active | High transactions, low inactivity | Low | Loyalty rewards |
| Mid-Value At-Risk | Moderate spending, rising inactivity | High | Immediate outreach |
| Low-Value Inactive | Low transactions, high inactivity | High | Low-cost automation |
| New Low-Engagement | Short tenure, low frequency | Uncertain | Onboarding nudges |

---

## Model Performance

| Model | AUC-ROC Score |
|---|---|
| Logistic Regression | 0.914 |
| Random Forest | 0.987 |
| Gradient Boosting | 0.988 |

**Why AUC-ROC?**
A model that predicts every customer stays would achieve
84% accuracy on this dataset and still be completely
useless. AUC-ROC measures how well the model separates
churners from non-churners regardless of class imbalance.
An AUC of 0.988 means the Gradient Boosting model
correctly ranks a churner above a non-churner 98.8% of
the time.

---

## Top Churn Predictors

Based on Random Forest feature importance:

1. Total_Trans_Ct: transaction count over 12 months
2. Total_Trans_Amt: total transaction amount
3. Months_Inactive_12_mon: months of account inactivity
4. Contacts_Count_12_mon: bank contact frequency
5. Inactivity_Score: engineered behavioral signal

---

## Business Recommendations

A tiered retention strategy based on churn probability
and customer lifetime value:

**Tier 1: High CLV and High Churn Risk**
These are the customers worth spending money to keep.
Assign relationship managers, offer personalised
incentives and contact them before they disengage further.
The model identified a high-risk segment where churn
probability exceeded 70%. These customers need immediate
attention.

**Tier 2: Low CLV and High Churn Risk**
Deploy low-cost automated campaigns. Email sequences,
app notifications and targeted offers. Do not overspend
on customers whose lifetime value does not justify
high-touch retention.

**Tier 3: High CLV and Low Churn Risk**
Protect this segment with loyalty rewards and exclusive
benefits. Maintain engagement proactively rather than
reactively.

**Tier 4: Low CLV and Low Churn Risk**
Monitor passively. Standard communication cadence.
No special intervention required.

This tiered approach maximizes retention ROI instead of
treating all customers the same way.

---

## What I Would Do Next

With more time and data I would apply SMOTE or class
weighting to further address the class imbalance during
model training. I would also build a real-time scoring
pipeline that flags at-risk customers automatically as
new transaction data arrives, moving this from a static
analysis to a live retention tool.

The fraud detection connection identified in this project
also opens a research direction worth pursuing: building
a unified behavioral model that detects both churn risk
and anomalous financial activity from the same feature
set simultaneously.

---

## How to Run This Project

1. Clone or download this repository
2. Open the notebook in Google Colab
3. Upload BankChurners.csv to your Colab session or
   mount your Google Drive
4. Run all cells in order from top to bottom

---

## About the Author

I am a Data Analyst transitioning into Data Science,
based in Lagos, Nigeria. This project was
completed during the TechCrush Data Science Bootcamp,
where I was selected from over 31,000 applicants across
41 countries for a 95% fee waiver scholarship.

My long-term research interest focuses on AI-driven
fraud detection for small businesses. This project
represents the first step in building that research
foundation publicly.

📧 chukwukasuccess.cs@gmail.com  
💼 LinkedIn: https://www.linkedin.com/in/successchukwuka  
🐙 GitHub: https://github.com/YongRichy

---

*If you are a hiring manager, researcher or fellow data
practitioner, I welcome feedback, collaboration and
conversation.*
