#Import Libraries
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
from pyspark.shell import spark
from pyspark import SparkContext
import csv
import timeit

start = timeit.default_timer()
#Preprocessing Dataset
dataset = spark.read.csv('startupdataset.csv', inferSchema=True, header=True)

f_dataset = dataset.select(dataset.columns[9:])

# f_dataset.groupBy('acq/closed').count().show()

# print("Schema: ",f_dataset.printSchema())
# print("Final dataset: ",f_dataset.describe().show())
# print("Final dataset columns: ",f_dataset.columns)

assembler = VectorAssembler( inputCols= ['age_first_funding_year', 'age_last_funding_year', 'age_first_milestone_year', 'age_last_milestone_year', 'funding_rounds', 'funding_total_usd', 'milestones', 'is_CA', 'is_NY', 'is_MA', 'is_TX', 'is_otherstate', 'is_software', 'is_web', 'is_mobile', 'is_enterprise', 'is_advertising', 'is_gamesvideo', 'is_ecommerce', 'is_biotech', 'is_consulting', 'is_othercategory', 'has_VC', 'has_angel', 'has_roundA', 'has_roundB', 'has_roundC', 'has_roundD', 'avg_participants', 'is_top500'], outputCol="features")

assemb_output = assembler.transform(f_dataset)

final_dataset = assemb_output.select('features', 'acq/closed')

# print("Final Final dataset: ",final_dataset.show())

#Training and Testing Split
training_df, testing_df = final_dataset.randomSplit([0.7,0.3], 99)

# print("Training dataset count: ",training_df.count())
training_count = training_df.count()
# print("Testing dataset count: ",testing_df.count())
testing_count = testing_df.count()
# print("Final dataset count: ",final_dataset.count())
finaldataset_count = final_dataset.count()

#Predicting using Decision Tree
dt_classifier = DecisionTreeClassifier(labelCol="acq/closed").fit(training_df)
dt_predictions = dt_classifier.transform(testing_df)
# dt_predictions.show()

impfeatu = dt_classifier.featureImportances

#Confusion Matrix
preds_and_labels = dt_predictions.select(['prediction','acq/closed']).withColumn('label', F.col('acq/closed').cast(FloatType())).orderBy('prediction')

preds_and_labels = preds_and_labels.select(['prediction','label'])

metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))

# print("Confusion matrix: ", metrics.confusionMatrix().toArray())

# print("Accuracy: ",metrics.accuracy)

# print("Precision 1 : ",metrics.precision(1.0))

# print("Precision 0 : ",metrics.precision(0.0))

# print("Recall 1 : ",metrics.recall(1.0))

# print("Recall 0 : ",metrics.recall(0.0))

cm = metrics.confusionMatrix().toArray()
accuracy = metrics.accuracy
precision1 = metrics.precision(1.0)
precision0 = metrics.precision(0.0)
recall1 = metrics.recall(1.0)
recall0 = metrics.recall(0.0)

end = timeit.default_timer()
time = end-start

values = { "Confusion Matrix":cm, "Accuracy" : accuracy, "Precision 1.0" : precision1,
            "Precision 0.0": precision0, "Recall 1.0" : recall1, "Recall 0.0": recall0, 
            "Important Feature": impfeatu,"Training count":training_count, "Testing count": testing_count, 
            "Final Dataset count": finaldataset_count, "Time(s)": time}

w = csv.writer(open("Result_DecisionTree.csv", "w"))

for key, val in values.items():

    # write every key and value to file
    w.writerow([key, val])