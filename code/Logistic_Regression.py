from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.util import MLUtils
from pyspark.ml.feature import VectorAssembler
import pyspark.sql.functions as F
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.types import FloatType
from pyspark.ml.classification import LogisticRegression

'''
The dataset contains 40 columns containing the following information:
agefirstfunding_year – quantitative
agelastfunding_year – quantitative
funding_rounds – quantitative
fundingtotalusd – quantitative
milestones – quantitative
agefirstmilestone_year – quantitative
agelastmilestone_year – quantitative
state – categorical
industry_type – categorical
has_VC – categorical
has_angel – categorical
has_roundA – categorical
has_roundB – categorical
has_roundC – categorical
has_roundD – categorical
avg_participants – quantitative
is_top500 – categorical
status(acquired/closed) – categorical (the target variable, if a startup is ‘acquired’ by some other organization, means the startup succeed) 
'''

'''
Reading the dataset into a dataframe
'''
dataset = spark.read.csv('startupdataset.csv', inferSchema=True, header=True)

'''
The dataset is further divided into two parts where:
1) The first 8 columns provide information about the company in a behaviorial format
2) The next 31 columnns provides quantitative information about the company that will be used to predict the success of the Startup
'''
f_dataset = dataset.select(dataset.columns[9:])

'''
The Actual success values of the startups a value of 1 in acq/closed implies it was a success and 0 indicating a faliure 
'''
f_dataset.groupBy('acq/closed').count().show()

f_dataset.printSchema()
f_dataset.describe().show()
f_dataset.columns

'''
Creating the features matrix for classification
'''
assembler = VectorAssembler( inputCols= ['age_first_funding_year', 'age_last_funding_year', 'age_first_milestone_year', 'age_last_milestone_year', 'funding_rounds', 'funding_total_usd', 'milestones', 'is_CA', 'is_NY', 'is_MA', 'is_TX', 'is_otherstate', 'is_software', 'is_web', 'is_mobile', 'is_enterprise', 'is_advertising', 'is_gamesvideo', 'is_ecommerce', 'is_biotech', 'is_consulting', 'is_othercategory', 'has_VC', 'has_angel', 'has_roundA', 'has_roundB', 'has_roundC', 'has_roundD', 'avg_participants', 'is_top500'], outputCol="features")

assemb_output = assembler.transform(f_dataset)

final_dataset = assemb_output.select('features', 'acq/closed')

final_dataset.show()

'''
The dataset is split into 70% of the rows for training and 30% for testing.
'''
training_df, testing_df = final_dataset.randomSplit([0.7,0.3])

training_df.count()
testing_df.count()
final_dataset.count()

'''
Applying Logistic Regression to the training data
'''
log_classifier = LogisticRegression(maxIter=10, featuresCol="features", labelCol="acq/closed")

model = log_classifier.fit(training_df)

print(model.summary.areaUnderROC)

'''
Using the classifier to predict values of testing data
'''
lr_prediction = model.transform(testing_df)

lr_prediction.show()

'''
Creating a confusion Matrix to understand the predicted values
'''
preds_and_labels = lr_prediction.select(['prediction','acq/closed']).withColumn('label', F.col('acq/closed').cast(FloatType())).orderBy('prediction')

preds_and_labels = preds_and_labels.select(['prediction','label'])

metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))

print(metrics.confusionMatrix().toArray())

print(metrics.accuracy)

print(metrics.precision(1.0))

print(metrics.precision(0.0))

print(metrics.recall(1.0))

print(metrics.recall(0.0))
