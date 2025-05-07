#!/usr/bin/env python
# coding: utf-8

# In[1]:


spark


# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib as plt
import io
from google.cloud import storage


# In[2]:


from pyspark.sql.functions import col, isnan, isnull, when, count, udf, size, split, year, month, format_number, date_format, length
from pyspark.sql.types import IntegerType, DateType, StringType, StructType, DoubleType
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, Normalizer, MinMaxScaler, StandardScaler, HashingTF, IDF, Tokenizer, RegexTokenizer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame
from functools import reduce
import matplotlib.pyplot as plt
from pyspark.ml.stat import Correlation


# In[31]:


def read_data(): 
    ## Read Data
    # Read in all files in the cleaned folder
    sdf = spark.read.parquet("gs://my-project-bucket-flights-cl/cleaned/")

    # Take a small sample for demonstration purposes
    #sdf = sdf.sample(withReplacement=False, fraction=0.10, seed=42)

    # Check the schema
    print(sdf.printSchema())

    # Get the number of records in the dataframe
    print("# columns and rows")
    print(len(sdf.columns))
    print(sdf.count())

    return sdf


# In[32]:


def clean_data(sdf):
    print("Raw count (before dropna):", sdf.count())
    sdf.limit(5).show()
    
    sdf = sdf.drop("DivAirportLandings")
    
    # Check to see records where ArrDel15 are null
    print(sdf.select([count(when(col(c).isNull(), c)).alias(c) for c in ["ArrDel15"]] ).show())
    # drop null values in data
    #null_counts = sdf.select([
    #    count(when(col(c).isNull(), 1)).alias(c) for c in sdf.columns
    #])
    #null_counts.show()
    sdf = sdf.dropna()
    print(sdf.count())

    # Filter out rows where CRSElapsedTime is negative
    sdf = sdf.filter(col("CRSElapsedTime") >= 0)
    print("Count after filtering CRSElapsedTime >= 0:", sdf.count())
    
    # label 
    sdf = sdf.withColumn("label", when(col("ArrDelayMinutes") > 5, 1).otherwise(0))

    # Cast ArrDel15 and DepDel15 to integers
    sdf = sdf.withColumn("ArrDel15", col("ArrDel15").cast("int"))
    sdf = sdf.withColumn("DepDel15", col("DepDel15").cast("int"))

    # Show the distribution of ArrDel15
    print("ArrDel15 distribution:")
    sdf.select("ArrDel15").groupby("ArrDel15").count().show()
    
    # Show the distribution of label
    print("ArrDel greater 5 minutes distribution:")
    sdf.select("label").groupby("label").count().show()
    
    return sdf


# In[ ]:


# Define bucket and folder as constants
BUCKET_NAME = "my-project-bucket-flights-cl"  
FIGURE_FOLDER = 'figures'

def save_plot_to_gcs(sdf, column_name, file_name):
    """
    Function to plot the frequency distribution of a column and save the plot to Google Cloud Storage.
    
    """
    # Get the frequency counts of the column
    column_counts_df = sdf.groupby(column_name).count().sort(column_name).toPandas()

    # Set up a Matplotlib figure
    fig = plt.figure(facecolor='white')

    # Create a bar plot for the column frequency distribution
    plt.bar(column_counts_df[column_name], column_counts_df['count'])

    # Add a title
    plt.title(f"Count by {column_name}")

    # Add labels for clarity
    plt.xlabel(column_name)
    plt.ylabel('Count')

    plt.show()

    # Create a buffer to hold the figure
    img_data = io.BytesIO()

    # Write the figure to the img_data buffer
    fig.savefig(img_data, format='png', bbox_inches='tight')

    # Rewind the pointer to the start of the data
    img_data.seek(0)

    # Connect to Google Cloud Storage
    storage_client = storage.Client()

    # Point to the bucket on Google Cloud Storage
    bucket = storage_client.get_bucket(BUCKET_NAME)

    # Create a blob to hold the data. Use the specific folder and file name format
    blob = bucket.blob(f"{FIGURE_FOLDER}/order_count_by_{column_name}.png")

    # Upload the img_data contents to the blob
    blob.upload_from_file(img_data)

    print(f"Plot saved to Google Cloud Storage at {FIGURE_FOLDER}/order_count_by_{column_name}.png")


# In[34]:


def add_time_of_day_column(df: DataFrame, time_col: str, new_col_name: str) -> DataFrame:
    """
    Adds a new time-of-day label column based on the hour extracted from a HHMM-format time column.

    """
    
    # Extract hour from HHMM format (e.g., 1345 → 13)
    df = df.withColumn(f"{time_col}_Hour", (col(time_col) / 100).cast("int"))
    
    # Add time-of-day label column
    # 1 Morning, 2 Afternoon, 3 Evening, 4 Night
    df = df.withColumn(new_col_name, 
        when((col(f"{time_col}_Hour") >= 5) & (col(f"{time_col}_Hour") < 12), 1)
        .when((col(f"{time_col}_Hour") >= 12) & (col(f"{time_col}_Hour") < 17), 2)
        .when((col(f"{time_col}_Hour") >= 17) & (col(f"{time_col}_Hour") < 21), 3)
        .otherwise(4)
    )
    
    # Drop temporary hour column
    df = df.drop(f"{time_col}_Hour")

    return df


# In[35]:


def plot_corr_matrix(sdf):
    # Create Correlation Matrix for continuous variables
    continuous_columns = ["CRSElapsedTime", "Distance", "Year"]
    vector_column = "correlation_features"
    assembler = VectorAssembler(inputCols=continuous_columns, outputCol=vector_column)
    sdf_vector = assembler.transform(sdf).select(vector_column)

    # Create the correlation matrix, then get just the values and convert to a list
    matrix = Correlation.corr(sdf_vector, vector_column).collect()[0][0]

    # Create the correlation matrix, then get just the values and convert to a list
    matrix = Correlation.corr(sdf_vector, vector_column).collect()[0][0]
    correlation_matrix = matrix.toArray().tolist() 
    # Convert the correlation to a Pandas dataframe
    correlation_matrix_df = pd.DataFrame(data=correlation_matrix, columns=continuous_columns, index=continuous_columns) 

    heatmap_plot = plt.figure(figsize=(16,5))  
    # Set the style for Seaborn plots
    sns.set_style("white")

    sns.heatmap(correlation_matrix_df, 
                xticklabels=correlation_matrix_df.columns.values,
                yticklabels=correlation_matrix_df.columns.values,  cmap="Greens", annot=True)
    plt.savefig("correlation_matrix.png")


# In[36]:


def add_features(sdf):
    
    ## Feature Engineering 
    # Add time-of-day columns
    time_columns = [
        ("CRSArrTime", "ScheduledArrTimeOfDay"),
        ("CRSDepTime", "ScheduledDepTimeOfDay")]

    for time_col, new_col_name in time_columns:
        sdf = add_time_of_day_column(sdf, time_col, new_col_name)

    # Drop the original time columns after they've been transformed
    time_col_names = [time_col for time_col, _ in time_columns]
    sdf = sdf.drop(*time_col_names)
    
    # Add Season Column
    sdf = sdf.withColumn(
    "Season",
    when((col("Month").isin(12, 1, 2)), 1)  # Winter
    .when((col("Month").isin(3, 4, 5)), 2)  # Spring
    .when((col("Month").isin(6, 7, 8)), 3)  # Summer
    .otherwise(4)                          # Fall (Sep, Oct, Nov)
    )
    
    # Check the distribution of the derived columns 
    print("ScheduledArrTimeOfDay distribution:")
    sdf.groupBy("ScheduledArrTimeOfDay").count().show()
    save_plot_to_gcs(sdf, "ScheduledArrTimeOfDay", "distribution_schedArr.png")
    
    # Check the distribution of the derived columns 
    print("ScheduledDepTimeOfDay distribution:")
    sdf.groupBy("ScheduledDepTimeOfDay").count().show()
    save_plot_to_gcs(sdf, "ScheduledDepTimeOfDay", "distribution_schedDep.png")
        
    # Check the distribution of the derived columns
    print("Season distribution:")
    sdf.groupBy("Season").count().show()
    save_plot_to_gcs(sdf, "Season", "distribution_season.png")
    
    # drop uneeded columns
    columns_to_drop = ["CRSArrTime", "CRSDepTime", "FlightDate", "Airline", "Cancelled", "Diverted",
                       "Origin", "OriginState", "Dest", "DestState",
                       "DepartureDelayGroups", "DepTimeBlk", "ArrivalDelayGroups",
                       "ArrTimeBlk", "DepTime", "DepDelay", "ArrTime",
                       "ArrDelayMinutes", "AirTime", "ActualElapsedTime", "TaxiOut",
                       "WheelsOff", "WheelsOn", "TaxiIn", "ArrDelay", "DivAirportLandings",
                      "DepDelayMinutes", "DepDel15", "ArrDel15", "Quarter"]
    sdf = sdf.drop(*columns_to_drop)
       
    # Get some statistics on each of the columns
    # IMPORTANT: This will take a VERY long time to complete
    # sdf.summary().show()
    
    return sdf


# In[37]:


sdf = spark.read.parquet("gs://my-project-bucket-flights-cl/cleaned/")
print(sdf.printSchema())
# Get the number of records in the dataframe
print("# columns and rows")
print(len(sdf.columns))
print(sdf.count())


# In[38]:


sdf.describe("ArrDelayMinutes").show()


# In[39]:


# Clean data
sdf_cleaned = clean_data(sdf)


# In[40]:


zero_delay_count = sdf_cleaned.filter(col("ArrDelayMinutes") == 0).count()
print(f"Number of flights with exactly 0 min arrival delay: {zero_delay_count}")


# In[41]:


sdf_cleaned.select("Operating_Airline").distinct().count()


# In[42]:


# Correlation Matrix for continuous variables
plot_corr_matrix(sdf_cleaned)


# In[43]:


# Add feature columns and drop unneeded columns
sdf_features = add_features(sdf_cleaned)


# In[44]:


# Define column names for the processing
# Categorical columns
categorical_columns = ["DayOfWeek", "Month", "Season", "DayofMonth", "Operating_Airline",
                        "OriginStateName", "DestStateName", 
                        "DistanceGroup", "ScheduledArrTimeOfDay", "ScheduledDepTimeOfDay"]

string_columns = ["Operating_Airline", "OriginStateName",
                "DestStateName", "DistanceGroup"]

other_categorical_columns = ["DayOfWeek", "Month", "Season", "DayofMonth", "ScheduledArrTimeOfDay", "ScheduledDepTimeOfDay"]

# Index columns
index_output_columns = ["Operating_AirlineIndex", "OriginStateNameIndex", 
                         "DestStateNameIndex", "DistanceGroupIndex"]

# One-hot encoding input and output columns
ohe_input_columns = index_output_columns + other_categorical_columns
ohe_output_columns = [
                        "Operating_AirlineVector",  "OriginStateNameVector", 
                        "DestStateNameVector", "DistanceGroupVector",
                        "DayOfWeekVector", "MonthVector", "SeasonVector", "DayofMonthVector", 
                        "ScheduledArrTimeOfDayVector", "ScheduledDepTimeOfDayVector"]

# Continuous columns
continuous_columns = ["CRSElapsedTime", "Distance"]
scaled_columns = ["CRSElapsedTimeScaled", "DistanceScaled"]

# Create the indexer for string-based columns
indexer = StringIndexer(inputCols=string_columns, outputCols=index_output_columns, handleInvalid="keep")

# Create the encoder for the indexed string columns and other categorical columns
encoder = OneHotEncoder(inputCols=ohe_input_columns, outputCols=ohe_output_columns, dropLast=True, handleInvalid="keep")

# Create an assembler for continuous columns
assembler = [VectorAssembler(inputCols=[col], outputCol=col + "Vector") for col in continuous_columns]

# Create a scaler for continuous columns (use the corresponding Vector column)
scaler = [MinMaxScaler(inputCol=col + "Vector", outputCol=scaled_col) for col, scaled_col in zip(continuous_columns, scaled_columns)]

# Final assembler 
final_assembler = VectorAssembler(inputCols=ohe_output_columns + scaled_columns, outputCol="features")
    
# Create pipeline
stages = [indexer, encoder] + assembler + scaler + [final_assembler]
pipeline = Pipeline(stages=stages)


# In[ ]:





# In[45]:


# Split cleaned data into Training and Test Data
trainingData, testData = sdf_features.randomSplit([0.7, 0.3], seed=42)


# In[46]:


# Fit and transform the Training and Test Data
training_transformed = pipeline.fit(trainingData).transform(trainingData)
test_transformed = pipeline.fit(testData).transform(testData)

# drop previous columns
columns_to_drop = [
    "Quarter", "DayOfWeek", "Month", "Season", "DayofMonth", "Operating_Airline", 
    "OriginAirportID", "OriginCityName", "OriginStateName", 
    "DestAirportID", "DestCityName", "DestStateName", 
    "DistanceGroup", "ScheduledArrTimeOfDay", "ScheduledDepTimeOfDay",
    "Operating_AirlineIndex", "OriginAirportIDIndex",
    "OriginStateNameIndex", "DestAirportIDIndex",  
    "DestStateNameIndex", "DistanceGroupIndex", 
    "CRSElapsedTime", "Distance", "Year", 
    "CRSElapsedTimeVector", "DistanceVector"]
training_transformed = training_transformed.drop(*columns_to_drop)
test_transformed = test_transformed.drop(*columns_to_drop)


# In[47]:


# Chi Squared for categorical variables 
from pyspark.ml.stat import ChiSquareTest

chi_sq_result = ChiSquareTest.test(training_transformed, "features", "label")
result_row = chi_sq_result.head()

attrs = training_transformed.schema["features"].metadata.get("ml_attr", {}).get("attrs", {})

# Flatten the structure (PySpark stores numeric and binary attrs separately)
feature_names = []

for attr_type in ["numeric", "binary"]:
    if attr_type in attrs:
        feature_names.extend([attr["name"] for attr in attrs[attr_type]])

# Extract values
p_values = result_row.pValues
chi2_stats = result_row.statistics

# Create DataFrame
chi_df = pd.DataFrame({
    "Feature": feature_names,
    "Chi2 Statistic": chi2_stats,
    "p-Value": p_values
}).sort_values("Chi2 Statistic", ascending=False)

# Show top features
chi_df_top = chi_df.head(20)

# Plot
plt.figure(figsize=(8, 12))
sns.barplot(data=chi_df_top, y="Feature", x="Chi2 Statistic", palette="mako")
plt.title("Chi-Square Test Statistics per Feature")
plt.yticks(rotation=45) 
plt.tight_layout()
plt.show()


# In[48]:


print("Transformed Test Data Schema: ")
test_transformed.printSchema()
test_transformed.select('label','features').show(30, truncate=False)


# In[49]:


print("Transformed Training Data Schema: ")
training_transformed.printSchema()
training_transformed.select('label','features').show(30, truncate=False)


# In[50]:


# Save to /trusted folder
training_transformed.write.mode("overwrite").parquet("gs://my-project-bucket-flights-cl/trusted/training_data")
test_transformed.write.mode("overwrite").parquet("gs://my-project-bucket-flights-cl/trusted/test_data")


# Build Model
# 1. Logistic Regression
# 

# In[3]:


from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
from pyspark.ml.linalg import VectorUDT


# In[4]:


# Read saved data files
#training_transformed = spark.read.parquet("gs://my-project-bucket-flights-cl/trusted/training_data")
#test_transformed = spark.read.parquet("gs://my-project-bucket-flights-cl/trusted/test_data")


# In[53]:


# Optional: Take a small sample of the data while developing the rest of the code
#training_transformed = training_transformed.sample(False, .25)
#training_transformed = training_transformed.sample(False, .25)


# In[6]:


# cast to double for cross validator
training_transformed = training_transformed.withColumn("label", col("label").cast("double"))
test_transformed = test_transformed.withColumn("label", col("label").cast("double"))


# In[55]:


# Create a LogisticRegression Estimator
lr = LogisticRegression(featuresCol="features", labelCol="label")

# Fit the model to the training data - This can take a long time depending on the size of the data
model = lr.fit(training_transformed)

# Show model coefficients and intercept
print("Coefficients: ", model.coefficients)
print("Intercept: ", model.intercept)
# pd.DataFrame({'coefficients':model.coefficients, 'feature':list(pd.DataFrame(trainingData.schema["features"].metadata["ml_attr"]["attrs"]['numeric']).sort_values('idx')['name'])})


# In[56]:


# Test the model on the testData
test_results = model.transform(test_transformed)

# Show the test results
test_results.select('rawPrediction','probability','prediction', 'label').show(30, truncate=False)

# Show the confusion matrix
test_results.groupby('label').pivot('prediction').count().sort('label').show()


# In[7]:


from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Flight Delay Prediction") \
    .config("spark.network.timeout", "600s") \
    .config("spark.executor.heartbeatInterval", "60s") \
    .getOrCreate()


def calculate_recall_precision(confusion_matrix):
    tn = confusion_matrix[0][1]  # True Negative
    fp = confusion_matrix[0][2]  # False Positive
    fn = confusion_matrix[1][1]  # False Negative
    tp = confusion_matrix[1][2]  # True Positive
    precision = tp / ( tp + fp )            
    recall = tp / ( tp + fn )
    accuracy = ( tp + tn ) / ( tp + tn + fp + fn )
    f1_score = 2 * ( ( precision * recall ) / ( precision + recall ) )
    return accuracy, precision, recall, f1_score


# In[58]:


confusion_matrix = test_results.groupby('label').pivot('prediction').count().fillna(0).sort('label').collect()

print("Accuracy, Precision, Recall, F1 Score")
print( calculate_recall_precision(confusion_matrix) )


# In[80]:





# In[62]:


# Save to /models folder
lr.write().overwrite().save("gs://my-project-bucket-flights-cl/models/logistic_regression_model")


# Logistic Regression n-Fold Validation and Hyperparameter Grid

# In[8]:


# n-Fold Validation and Hyperparameter Grid

sc.setLogLevel("ERROR")

from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import numpy as np

# Create a LogisticRegression Estimator
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=5)

# Create a grid to hold hyperparameters 
grid = ParamGridBuilder()
grid = grid.addGrid(lr.regParam, [0.0, 0.3, 0.6, 1.0])
grid = grid.addGrid(lr.elasticNetParam, [0, 0.5, 1])

# Build the parameter grid
grid = grid.build()

# How many models to be tested
print('Number of models to be tested: ', len(grid))

# Create a BinaryClassificationEvaluator to evaluate how well the model works
BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

lr_pipe = Pipeline(stages=[lr])

evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")

# Create the CrossValidator using the hyperparameter grid
cv = CrossValidator(estimator=lr_pipe, 
                    estimatorParamMaps=grid, 
                    evaluator=evaluator, 
                    numFolds=3,
                    seed = 42)

# Train the models
cv  = cv.fit(training_transformed)


# In[12]:


# Test the predictions
predictions = cv.transform(test_transformed)

# Calculate AUC
auc = evaluator.evaluate(predictions)
print(f"AUC: {auc}")

# Create the confusion matrix
predictions.groupby('label').pivot('prediction').count().fillna(0).sort('label').show()
cm = predictions.groupby('label').pivot('prediction').count().fillna(0).collect()

def calculate_recall_precision(cm):
    tn = cm[0][1]                # True Negative
    fp = cm[0][2]                # False Positive
    fn = cm[1][1]                # False Negative
    tp = cm[1][2]                # True Positive
    precision = tp / ( tp + fp )            
    recall = tp / ( tp + fn )
    accuracy = ( tp + tn ) / ( tp + tn + fp + fn )
    f1_score = 2 * ( ( precision * recall ) / ( precision + recall ) )
    return accuracy, precision, recall, f1_score

print("Accuracy, Precision, Recall, F1 Score")
print( calculate_recall_precision(cm) )


# In[15]:


# Define bucket and folder as constants
BUCKET_NAME = "my-project-bucket-flights-cl"  
FIGURE_FOLDER = 'figures'

def save_vis_to_gcs(fig, file_name):
    """
    Function to plot the frequency distribution of a column and save the plot to Google Cloud Storage.
    given fig and file name
    """

    # Create a buffer to hold the figure
    img_data = io.BytesIO()

    # Write the figure to the img_data buffer
    fig.savefig(img_data, format='png', bbox_inches='tight')

    # Rewind the pointer to the start of the data
    img_data.seek(0)

    # Connect to Google Cloud Storage
    storage_client = storage.Client()

    # Point to the bucket on Google Cloud Storage
    bucket = storage_client.get_bucket(BUCKET_NAME)

    # Create a blob to hold the data. Use the specific folder and file name format
    blob = bucket.blob(f"{FIGURE_FOLDER}/{file_name}.png")

    # Upload the img_data contents to the blob
    blob.upload_from_file(img_data)

    print(f"Plot saved to Google Cloud Storage at {FIGURE_FOLDER}/{file_name}.png")


# In[16]:


# Confusion Matrix
# Extract TN, FP, FN, TP from cm
tn = cm[0][1]                # True Negative
fp = cm[0][2]                # False Positive
fn = cm[1][1]                # False Negative
tp = cm[1][2]                # True Positive

conf_matrix = np.array([[tn, fp], [fn, tp]])

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Predicted No Delay", "Predicted Delay"],
            yticklabels=["Actual No Delay", "Actual Delay"],
            ax=ax)
ax.set_xlabel("Prediction")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix Heatmap")
fig.tight_layout()

# Save to GCS
save_vis_to_gcs(fig, "confusion_matrix_heatmap.png")

plt.show()


# In[ ]:


# Look at the parameters for the best model that was evaluated from the grid
parammap = cv.bestModel.stages[0].extractParamMap()

for p, v in parammap.items():
    print(p, v)

# Grab the model from Stage 3 of the pipeline (also can use index -1)
mymodel = cv.bestModel.stages[0]

# Extract ROC data
fpr = [row['FPR'] for row in mymodel.summary.roc.collect()]
tpr = [row['TPR'] for row in mymodel.summary.roc.collect()]

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(fpr, tpr)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')

fig.tight_layout()

# Save plot to GCS
save_vis_to_gcs(fig, "roc_curve.png")

plt.show()


# In[71]:


hyperparams = cv.getEstimatorParamMaps()[np.argmax(cv.avgMetrics)]
# Print out the list of hyperparameters for the best model
for i in range(len(hyperparams.items())):
    print([x for x in hyperparams.items()][i])


# In[72]:


# Extract the coefficients on each of the variables
coeff = mymodel.coefficients.toArray().tolist()

# Loop through the features to extract the original column names. Store in the var_index dictionary
var_index = dict()
for variable_type in ['numeric', 'binary']:
    for variable in predictions.schema["features"].metadata["ml_attr"]["attrs"][variable_type]:
         print(f"Found variable: {variable}" )
         idx = variable['idx']
         name = variable['name']
         var_index[idx] = name      # Add the name to the dictionary

# Loop through all of the variables found and print out the associated coefficients
for i in range(len(var_index)):
    print(f"Coefficient {i} {var_index[i]}  {coeff[i]}")



# In[73]:


# Save to /models folder
cv.bestModel.write().overwrite().save("gs://my-project-bucket-flights-cl/models/best_lr_model")


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
coef_data = [(var_index[i], coeff[i]) for i in range(len(var_index))]
coef_df = pd.DataFrame(coef_data, columns=['Feature', 'Coefficient']).set_index('Feature')

# Sort coefficients by magnitude for ranking
sorted_coefs = coef_df.sort_values(by='Coefficient', ascending=False)

# Top 10 positive and negative coefficients
top_positive = sorted_coefs.head(10)
top_negative = sorted_coefs.tail(10).sort_values(by='Coefficient')

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot top positive coefficients
sns.barplot(ax=axes[0], y=top_positive.index, x=top_positive['Coefficient'], palette='Greens_d')
axes[0].set_title('Top 10 Positive Influential Features')
axes[0].set_xlabel('Coefficient')
axes[0].set_ylabel('Feature')

# Plot top negative coefficients
sns.barplot(ax=axes[1], y=top_negative.index, x=top_negative['Coefficient'], palette='Reds_d')
axes[1].set_title('Top 10 Negative Influential Features')
axes[1].set_xlabel('Coefficient')
axes[1].set_ylabel('Feature')

fig.tight_layout()

# Save to GCS
save_vis_to_gcs(fig, "top_coefficients.png")
plt.show()


# In[ ]:


# Random Forest


# In[74]:


from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Create a RandomForest Estimator
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=10)

# Fit the model to the training data - This can take a long time depending on the size of the data
model = rf.fit(training_transformed)


# In[ ]:


# Get feature importances
importances = model.featureImportances

print("Feature Importances (sparse vector):", importances)
print("Feature Importances (array):", importances.toArray())


# In[ ]:


# Test the model on the testData
test_results = model.transform(test_transformed)

# Show the test results
test_results.select('rawPrediction','probability','prediction', 'label').show(30, truncate=False)

# Show the confusion matrix
test_results.groupby('label').pivot('prediction').count().sort('label').show()


# In[78]:


confusion_matrix = test_results.groupby('label').pivot('prediction').count().fillna(0).sort('label').collect()
print("Accuracy, Precision, Recall, F1 Score")
# print( calculate_recall_precision(confusion_matrix) )
# cannot calculate because model only predicted 0


# In[12]:


# Save to /models folder
# rf.save("gs://my-project-bucket-flights-cl/models/random_forest_model")


# In[ ]:




