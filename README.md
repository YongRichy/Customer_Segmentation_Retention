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
