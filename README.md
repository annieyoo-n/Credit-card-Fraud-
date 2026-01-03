# Credit-card-Fraud-
**Credit Card Fraud Detection: Random Forest vs XGBoost**

**Overview**
This project presents a comparative analysis of two supervised machine learning algorithms (Random Forest and XGBoost) for credit card fraud detection. The research addresses the challenge of imbalanced datasets in financial fraud detection and evaluates the performance of these ensemble methods using the Kaggle Credit Card Fraud Detection dataset.

**Dataset**
The project uses the Credit Card Fraud Detection dataset from Kaggle, containing:
284,807 transactions (September 2013)
28 PCA-transformed features (for confidentiality)
Time feature (seconds between transactions)
Amount feature (transaction amount)
Class target variable (0: legitimate, 1: fraudulent)

**Models**

**Random Forest**
Ensemble of decision trees
Hyperparameters tuned: n_estimators, max_depth, min_samples_split
Training time: 142.3 seconds

**XGBoost**
Gradient boosting implementation
Hyperparameters tuned: n_estimators, max_depth, learning_rate, subsample, colsample_bytree
Training time: 186.7 seconds

**Key Findings**
XGBoost outperforms Random Forest with 0.06% higher ROC-AUC
XGBoost reduces false positives and false negatives by approximately 28%
XGBoost training time is 31.2% longer than Random Forest
Both algorithms effectively handle class imbalance with proper preprocessing

**References**
Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.
Breiman, L. (2001). Random Forests.
Barua, S., Islam, M. M., Yao, X., Murase, K. (2015). MWMOTE-Majority Weighted Minority Oversampling Technique.
Contact
For questions or contributions, please contact:

**Ayesha Zulfiqar**
**Email: zaishahmad12@gmail.com
GitHub: annieyoo-n**
