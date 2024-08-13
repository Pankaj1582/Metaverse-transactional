Project Overview
This project focuses on detecting anomalies in metaverse transaction data using various machine learning algorithms implemented with PySpark. The dataset includes multiple features such as transaction type, location, user demographics, and transaction details, all of which contribute to identifying unusual patterns that could indicate fraudulent or suspicious activities.

Objective
The primary goal of this project is to build and evaluate different machine learning models to predict anomalies in the dataset. Anomalies, in this context, refer to transactions that deviate significantly from the norm and may require further investigation. The models used include Decision Tree Regressor, Random Forest Regressor, Naive Bayes, and Gradient-Boosted Trees (GBT). These models are chosen to compare their performance in terms of accuracy, R-squared (R²), Root Mean Squared Error (RMSE), and Area Under the Receiver Operating Characteristic Curve (AUC-ROC).

Data Preprocessing
The dataset, loaded into a PySpark DataFrame, underwent several preprocessing steps:

Dropping Irrelevant Columns: Columns like timestamp, sending_address, receiving_address, and ip_prefix were dropped to focus on the most relevant features.
String Indexing: Categorical variables such as transaction_type, location_region, age_group, and purchase_pattern were encoded into numerical values using StringIndexer.
One-Hot Encoding: After indexing, the categorical features were one-hot encoded to convert them into a format suitable for machine learning models.
Feature Assembly: A VectorAssembler was used to combine all the features into a single vector column.
Normalization: The features were standardized using StandardScaler to ensure that each feature contributes equally to the models.
Model Training and Evaluation
The dataset was split into training (75%) and testing (25%) sets. Each model was trained on the training set and evaluated on the testing set using several performance metrics.

Decision Tree Regressor:

R²: Evaluates how well the model explains the variance in the data.
RMSE: Measures the average error between predicted and actual values.
AUC-ROC: Assesses the model’s ability to distinguish between classes.
Random Forest Regressor:

Similar metrics as the Decision Tree, but with improved accuracy due to the ensemble nature of Random Forests.
Naive Bayes Classifier:

Accuracy: Measures the proportion of correctly predicted instances.
R² and AUC-ROC: Used to further evaluate performance, although Naive Bayes is more commonly used for classification tasks.
Gradient-Boosted Trees (GBT):

R² and RMSE: GBT often outperforms other models in terms of predictive accuracy.
AUC-ROC: GBT’s performance in distinguishing anomalies from normal transactions was evaluated.
Key Findings
Random Forest and GBT models showed superior performance in terms of R² and RMSE, making them more reliable for this anomaly detection task.
Naive Bayes, while not typically used for regression, provided insights into the categorical nature of some of the features.
AUC-ROC scores across models highlighted the importance of model selection depending on the specific requirements of accuracy versus interpretability.
Conclusion
This project demonstrates the effectiveness of using PySpark for large-scale data processing and machine learning. The comparative analysis of various models provides valuable insights into anomaly detection in financial transactions within the metaverse. Future work could explore deep learning approaches or incorporate additional features for enhanced performance.

How to Run
To replicate this project, clone the repository and run the provided Jupyter notebooks in a PySpark-enabled environment. Ensure you have the necessary Python libraries installed as specified in the requirements.txt file.
