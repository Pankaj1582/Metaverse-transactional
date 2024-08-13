#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install -q findspark
pip install pyspark
import findspark
findspark.init()
findspark.find()from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.feature import StandardScaler
from pyspark.ml.regression import DecisionTreeRegressor, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator,BinaryClassificationEvaluator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.regression import GBTRegressor
spark = SparkSession.builder.master("local[*]").config("spark.driver.bindAddress", "127.0.0.1").appName("BigDataCoursework").getOrCreate()
df = spark.read.csv("/Users/pankajyadav/Downloads/metaverse_transactions_dataset.csv" , header = True, 
                    inferSchema = True)
df.show(5)
df.printSchema()
df.columns
df = df.drop("timestamp","sending_address","receiving_address","ip_prefix")
print(df.count())
len(df.columns)
df.groupby("transaction_type").count().show()
df.groupby("location_region").count().show()
df.groupby("age_group").count().show()
df.groupby("purchase_pattern").count().show()


#https://medium.com/@nutanbhogendrasharma/feature-transformer-vectorassembler-
#in-pyspark-ml-feature-part-3-b3c2c3c93ee9
transaction_indexer = StringIndexer(inputCol="transaction_type",outputCol= "transaction_type_encoded")
location_indexer = StringIndexer(inputCol="location_region", outputCol="location_region_encoded")
age_indexer = StringIndexer(inputCol="age_group", outputCol="age_group_encoded")
purchase_indexer = StringIndexer(inputCol="purchase_pattern", outputCol="purchase_pattern_encoded")
anomaly_indexer = StringIndexer(inputCol="anomaly", outputCol="anomaly_target")

# Applying the indexers, fit and transform the orb
df = transaction_indexer.fit(df).transform(df)
df = location_indexer.fit(df).transform(df)
df = age_indexer.fit(df).transform(df)
df = purchase_indexer.fit(df).transform(df)
df = anomaly_indexer.fit(df).transform(df)

ohe_transaction = OneHotEncoder(inputCol = "transaction_type_encoded", outputCol="transaction_vec")
ohe_location = OneHotEncoder(inputCol = "location_region_encoded", outputCol="location_vec")
ohe_age = OneHotEncoder(inputCol = "age_group_encoded", outputCol="age_vec")
ohe_purchase = OneHotEncoder(inputCol = "purchase_pattern_encoded", outputCol="purchase_vec")

df = ohe_transaction.fit(df).transform(df)
df = ohe_location.fit(df).transform(df)
df = ohe_age.fit(df).transform(df)
df = ohe_purchase.fit(df).transform(df)

num_col = ['hour_of_day',
 'amount','transaction_vec',
 'location_vec','login_frequency','session_duration','purchase_vec',
 'age_vec','risk_score',
 'anomaly_target']

df = df.drop(
 'transaction_type',
 'location_region',
 'purchase_pattern',
 'age_group',
 'anomaly',
 'transaction_type_encoded',
 'location_region_encoded',
 'age_group_encoded',
 'purchase_pattern_encoded',
 )

assembler = VectorAssembler(inputCols = num_col, outputCol = "features")
df1 = assembler.transform(df)
df1.show(5)

#https://stackoverflow.com/questions/51753088/standardscaler-in-spark-not-working-as-expected

#Normalize the features

scaler = StandardScaler(inputCol = "features", outputCol = "scaledFeatures", withStd=True, withMean=False)
scalerModel =scaler.fit(df1)
scaledOutput = scalerModel.transform(df1)
scaledOutput.show(5)

final_df = scaledOutput.select("scaledFeatures", "anomaly_target")
final_df.show()

train_data, test_data = final_df.randomSplit([0.75,0.25],seed = 2024)

# Train and evaluate the Decision tree Regression

dt = DecisionTreeRegressor(featuresCol="scaledFeatures", labelCol="anomaly_target")
dt_model = dt.fit(train_data)
dt_predictions = dt_model.transform(test_data)
dt_evaluator = RegressionEvaluator(labelCol="anomaly_target", predictionCol="prediction", metricName="r2")
dt_r2 = dt_evaluator.evaluate(dt_predictions)
print("Decision Tree R-Square:", dt_r2)

# Calculate the Area Under ROC for the SVM model
dt_evaluator1 = BinaryClassificationEvaluator(labelCol="anomaly_target", rawPredictionCol="prediction", 
                                              metricName="areaUnderROC")
dt_auc = dt_evaluator.evaluate(dt_predictions)
print("DT Area Under ROC:", dt_auc)

dt_evaluator = RegressionEvaluator(labelCol="anomaly_target", predictionCol="prediction", metricName="rmse")
dt_rmse = dt_evaluator.evaluate(dt_predictions)
print("Decision Tree RMSE:", dt_rmse)

# Initialize the Random Forest Regressor
rf = RandomForestRegressor(featuresCol="scaledFeatures", labelCol="anomaly_target")
# Train the model on training data
rf_model = rf.fit(train_data)
# Make predictions on test data
predictions_rf = rf_model.transform(test_data)

evaluator_rmse = RegressionEvaluator(labelCol="anomaly_target",predictionCol="prediction", metricName="rmse")
rmse = evaluator_rmse.evaluate(predictions_rf)
print("Root Mean Squared Error (RMSE):", rmse)
evaluator_r2 = RegressionEvaluator(labelCol="anomaly_target",predictionCol="prediction", metricName="r2")
r2 = evaluator_r2.evaluate(predictions_rf)
print("R2:",r2 )

# Calculate the Area Under ROC for the Random Forest model
rf_evaluator1 = BinaryClassificationEvaluator(labelCol="anomaly_target", rawPredictionCol="prediction", 
                                              metricName="areaUnderROC")
rf_auc = rf_evaluator1.evaluate(predictions_rf)
print("RF Area Under ROC:", rf_auc)

# Train the Naive Bayes model
nb = NaiveBayes(featuresCol="scaledFeatures", labelCol="anomaly_target")
nb_model = nb.fit(train_data)

# Make predictions on the test set
nb_predictions = nb_model.transform(test_data)

# Evaluate the model using MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(labelCol="anomaly_target", predictionCol="prediction", 
                                              metricName="accuracy")
accuracy = evaluator.evaluate(nb_predictions)
print("Naive Bayes Accuracy:", accuracy)

evaluator1 = RegressionEvaluator(labelCol="anomaly_target", predictionCol="prediction", metricName="r2")
r2 = evaluator1.evaluate(nb_predictions)
print("Naive Bayes r2:", r2)

# Calculate the Area Under ROC for the Naive Bayes model
nb_evaluator1 = BinaryClassificationEvaluator(labelCol="anomaly_target", rawPredictionCol="prediction", 
                                              metricName="areaUnderROC")
nb_auc = rf_evaluator1.evaluate(nb_predictions)
print("NB Area Under ROC:", nb_auc)

# Gradiant Boost Tree
gbt = GBTRegressor(featuresCol="scaledFeatures", labelCol="anomaly_target")
gbt_model = gbt.fit(train_data)
gbt_predictions = gbt_model.transform(test_data)
gbt_evaluator = RegressionEvaluator(labelCol="anomaly_target", predictionCol="prediction", metricName="r2")
gbt_r2 = gbt_evaluator.evaluate(gbt_predictions)
print("gbt R-square:", gbt_r2)

from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Calculate the Area Under ROC for the SVM model
gbt_evaluator = BinaryClassificationEvaluator(labelCol="anomaly_target", rawPredictionCol="prediction", 
                                              metricName="areaUnderROC")
gbt_auc = gbt_evaluator.evaluate(gbt_predictions)
print("GBT Area Under ROC:", gbt_auc)


gbt_evaluator = RegressionEvaluator(labelCol="anomaly_target", predictionCol="prediction", metricName="rmse")
gbt_rmse = gbt_evaluator.evaluate(gbt_predictions)
print("GBT RMSE:", gbt_rmse)

