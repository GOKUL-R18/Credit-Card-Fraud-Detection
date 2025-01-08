# Credit-Card-Fraud-Detection

# Credit Card Fraud Detection

This project uses machine learning techniques to detect fraudulent credit card transactions. The dataset is highly imbalanced, and various techniques like feature engineering, model optimization, and evaluation strategies are employed to achieve reliable performance, ensuring a practical approach to fraud detection.

## Objective

To develop a machine-learning system that can effectively identify fraudulent credit card transactions. The project aims to address challenges posed by the dataset's imbalance and ensure accurate detection using advanced techniques.

## Dataset

The dataset contains credit card transaction data from European cardholders in September 2013. Key characteristics:
- **Total Transactions**: 284,807
- **Fraudulent Transactions**: 492 (0.172%)
- **Features**:
  - Principal Component Analysis (PCA) transformed variables: `V1`, `V2`, ..., `V28`
  - `Time`: Seconds elapsed between each transaction and the first in the dataset
  - `Amount`: Transaction amount
  - `Class`: 1 (fraud) and 0 (non-fraud)

### Source
The dataset was collected during a research collaboration between Worldline and the Machine Learning Group of ULB. For more information, visit the [DefeatFraud project page]([https://www.researchgate.net/project/Fraud-detection-5](https://www.researchgate.net/project/Fraud-detection-5](https://www.researchgate.net/publication/319867396_Credit_Card_Fraud_Detection_A_Realistic_Modeling_and_a_Novel_Learning_Strategy)).

## Methodology

### Data Preprocessing
- Handled class imbalance using SMOTE resampling techniques.
- Performed feature scaling for `Time` and `Amount` using **RobustScaler**.
- Visualized data distribution using **Matplotlib** and **Seaborn**.

### Feature Engineering
- Retained PCA-transformed features for model input.
- Analyzed correlation to identify important features.
- Explored **feature importance** using tree-based models.

### Model Development

#### Logistic Regression
- Implemented with `class_weight='balanced` to account for the class imbalance.
- Cross-validated using Stratified K-Fold.
- Generated ROC Curve to evaluate model performance.

#### Anomaly Detection
1. **Isolation Forest**
   - Detects anomalies based on the isolation principle.
   - Detected 62 errors with an accuracy of **99.78%**.
2. **Local Outlier Factor (LOF)**
   - Uses density-based anomaly detection.
   - Detected 86 errors with an accuracy of **99.69%**.
3. **Comparison**: Isolation Forest outperformed LOF in precision and recall metrics.

- Evaluated performance using **AUPRC** (Area Under Precision-Recall Curve).

### Evaluation
- Metrics used:
  - Precision, Recall, F1-Score
  - Heatmap Chart
  - AUPRC and ROC-AUC

### Model Deployment
- Finalized the best-performing model for deployment.
- Saved and loaded models using **Joblib** for future use.

## Tools and Technologies

- **Programming Language**: Python
- **Data Manipulation**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-Learn
- **Deployment**: Joblib
- **Development Environment**: Google Colab

## Results

- Achieved a reliable performance with a **98% AUC Score**.
- Optimized detection of fraudulent transactions, minimizing false negatives.
- Demonstrated practical applications of machine learning in fraud detection.

### Logistic Regression
- Achieved AUC score of **X.XXX** (to be updated).
- Balanced precision and recall for fraud detection.
- Visualized performance using the ROC Curve.

### Anomaly Detection
1. **Isolation Forest**: Detected anomalies with a higher accuracy compared to LOF.
2. **Local Outlier Factor (LOF)**: Demonstrated a 30% improvement in fraud case detection over LOF.

### Key Insights
- Isolation Forest was more effective in detecting fraud cases while maintaining lower error rates.
- Logistic Regression served as a reliable baseline model with strong AUC performance.

## References

Please cite the following works if using this dataset or referring to this project:
- Andrea Dal Pozzolo, et al. *Calibrating Probability with Undersampling for Unbalanced Classification.* CIDM, IEEE, 2015.
- For additional references, please take a look at the dataset's [source documentation]([https://www.researchgate.net/project/Fraud-detection-5](https://www.researchgate.net/publication/319867396_Credit_Card_Fraud_Detection_A_Realistic_Modeling_and_a_Novel_Learning_Strategy)).

## Future Enhancements
- Integrate deep learning models for improved fraud detection.
- Explore advanced sampling techniques for handling class imbalance.
- Deploy the model as an API for real-time fraud detection.

