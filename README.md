# Customer Churn Prediction

End-to-end machine learning project using the [Telco Customer Churn Dataset (Kaggle)](https://www.kaggle.com/blastchar/telco-customer-churn).  
This project builds a robust churn prediction pipeline with advanced feature engineering, customer segmentation, hyperparameter-optimized models, and interpretability tools.

---

## Project Overview

Customer churn prediction is one of the most critical applications of machine learning in the telecom and subscription-based industries. The goal is to predict whether a customer will leave the company (`Churn = Yes`) based on demographic, service usage, and financial behavior.

This project covers:

- **Exploratory Data Analysis (EDA):** Insights on churn drivers and visualization of feature patterns  
- **Feature Engineering:** Creating lifecycle, service utilization, spending efficiency, and customer value features  
- **Customer Segmentation:** Using clustering and UMAP visualization to identify customer groups  
- **Predictive Modeling:** Random Forest, XGBoost, LightGBM, and an Ensemble meta-learner  
- **Interpretability:** Explainable AI using SHAP and LIME  
- **Vector Database (Pinecone):** Customer similarity search based on embeddings  

---

## Tech Stack

- **Languages & Libraries:** Python, Pandas, NumPy, Scikit-learn, XGBoost, LightGBM  
- **Visualization:** Matplotlib, Seaborn, Plotly  
- **Clustering & Dimensionality Reduction:** KMeans, PCA, UMAP  
- **Imbalanced Data Handling:** SMOTE (Imbalanced-learn)  
- **Hyperparameter Optimization:** GridSearchCV, Optuna  
- **Interpretability:** SHAP, LIME  
- **Vector DB:** Pinecone + Sentence-Transformers  

---

## Project Structure

├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── customer_churn_prediction.ipynb
├── README.md


---

## Key Steps

### 1. Data Preprocessing & Feature Engineering
- Handled missing values (`TotalCharges`)  
- Converted categorical variables using One-Hot Encoding  
- Created advanced features:  
  - `TotalServices`, `ServiceUtilizationRate`  
  - `AvgMonthlySpend`, `ChargesPerService`  
  - `CustomerValue`, `HighValueCustomer`, `DigitalEngagement`  
  - `MonthToMonthContract`, `AutomaticPayment`  

### 2. Customer Segmentation
- Scaled customer attributes and applied **KMeans clustering**  
- Visualized segments with **UMAP (2D embeddings)**  
- Analyzed segment-wise churn rates, tenure, and spending  

### 3. Predictive Modeling
- Applied SMOTE to balance churn vs. non-churn samples  
- Trained multiple models:  
  - Random Forest  
  - XGBoost  
  - LightGBM  
  - Ensemble Meta-Learner (Logistic Regression)  

**Performance Summary (Test Set):**

| Model         | Accuracy | AUC   |
|---------------|----------|-------|
| Random Forest | 0.860    | 0.936 |
| XGBoost       | 0.852    | 0.937 |
| LightGBM      | 0.848    | 0.935 |
| **Ensemble**  | **0.861**| **0.941** |

The Ensemble model achieved the best performance with an AUC of 0.941.

### 4. Interpretability
- SHAP values used to understand key drivers of churn (e.g., contract type, tenure, monthly charges)  
- LIME applied to explain individual predictions  

### 5. Customer Similarity Search (Optional, Pinecone)
- Built customer profile embeddings using `SentenceTransformer (all-MiniLM-L6-v2)`  
- Stored vectors in Pinecone to enable fast similarity search and retrieval of "lookalike" customers  

---

## Results & Insights

- Short-term, high-spending, month-to-month contract customers are at the highest risk of churn  
- Customers with longer tenure and auto-payment methods are less likely to churn  
- Ensemble learning significantly improves performance over individual models  
- Clustering reveals distinct customer personas (loyal low spenders, high-value short-term customers, digital heavy users, etc.)  

---

## Installation & Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/customer-churn-prediction.git
   cd customer-churn-prediction
Install dependencies:

pip install -r requirements.txt


Open and run the notebook:

jupyter notebook notebooks/customer_churn_prediction.ipynb

## Acknowledgments

- Dataset: Kaggle – Telco Customer Churn
- Libraries: Scikit-learn, XGBoost, LightGBM, SHAP, LIME, UMAP, Pinecone
