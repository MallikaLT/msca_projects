from pyspark.sql import SparkSession
import pyspark.sql.functions as func
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType, DoubleType
from math import log
from sklearn.metrics import log_loss
from random import seed
import pandas as pd


spark = SparkSession.builder.appName("Quora1_model").getOrCreate()
spark.sparkContext.setLogLevel("WARN") 
spark

# Read the features created in the first part of the project.
trainFeaturesPath = "./AML_Project2_Data/train_features.csv"           # full training data
testFeaturesPath = "./AML_Project2_Data/test_features.csv"             # full test data (for cluster job only)
outPath = "./AML_Project2_Results/predictions.csv"                       # output 

sch = StructType([StructField('id',IntegerType()), \
                  StructField('lWCount1',IntegerType()),\
		  StructField('lWCount2',IntegerType()),\
                  StructField('qWCount1',IntegerType()),\
	          StructField('qWCount2',IntegerType()),\
                  StructField('lLen1',IntegerType()),\
		  StructField('lLen2',IntegerType()),\
                  StructField('qLen1',IntegerType()),\
                  StructField('qLen2',IntegerType()),\
                  StructField('lWCount_ratio',DoubleType()),\
                  StructField('qWCount_ratio',DoubleType()),\
                  StructField('lLen_ratio',DoubleType()),\
                  StructField('qLen_ratio',DoubleType()),\
		  StructField('qNgrams_1',IntegerType()),\
		  StructField('qNgrams_2',IntegerType()),\
		  StructField('qNgrams_3',IntegerType()),\
                  StructField('lNgrams_1',IntegerType()),\
                  StructField('lNgrams_2',IntegerType()),\
                  StructField('lNgrams_3',IntegerType()),\
		  StructField('qUnigram_ratio',DoubleType()),\
                  StructField('lUnigram_ratio',DoubleType()),\
                  StructField('tfidfDistance',DoubleType()),\
                  StructField('lemma_leven',IntegerType()),\
                  StructField('question_leven',IntegerType()),\
                  StructField('is_duplicate',IntegerType())])
				  
#train_df = spark.read.csv(trainFeaturesPath, header=True, inferSchema=True)
#test_df = spark.read.csv(testFeaturesPath, header=True, inferSchema=True)

train_df = spark.read.csv(trainFeaturesPath, header=True, escape='"',quote='"',schema=sch, multiLine = True)
test_df = spark.read.csv(testFeaturesPath, header=True, escape='"',quote='"',schema=sch, multiLine = True)

print('Train DF Sample:')
train_df.show(5)
print(train_df.count())
print('Test DF Sample:')
test_df.show(5)
print(test_df.count())

# ## Create FeaturesVec for the classification models
#Prepared train features
featureNames = train_df.columns[1:-1] 
assembler = VectorAssembler(inputCols=featureNames, outputCol="features")
train_df = assembler.transform(train_df)
train_df = train_df.select("id","features", "is_duplicate")

print("train vector assembled")
train_df.show(5)

#Prepared test features
assembler = VectorAssembler(inputCols=featureNames, outputCol="features")
test_df = assembler.transform(test_df)
test_df = test_df.select("id","features")

print("test vector assembled")
test_df.show(5)

# Split `train_df` into train and test sets (30% held out for testing)
#Split train and test
seed(0)
(trainingData, testData) = train_df.randomSplit([0.7,0.3])


# ## Logistic Regression
#Fit logistic regression
glr = GeneralizedLinearRegression(family="binomial", link="logit", featuresCol="features", labelCol="is_duplicate")
trainLogitModel = glr.fit(trainingData)

#Logistic model predictions
LogitPredictions = trainLogitModel.transform(testData)
          
# Calculate AUC
evaluator = BinaryClassificationEvaluator(labelCol="is_duplicate", rawPredictionCol="prediction", metricName="areaUnderROC")
AUClogit = evaluator.evaluate(LogitPredictions)
print("Logistic Regression AUC = %g " % AUClogit)


# ## Decision trees 
#Fit decision tree model
#Train a DecisionTree model and make predictions
dt = DecisionTreeClassifier(maxDepth=15, labelCol="is_duplicate")
dtModel = dt.fit(trainingData)
dtPredictions = dtModel.transform(testData)

# Calculate AUC.
# Select (prediction, true label) and compute test AUC
evaluator = BinaryClassificationEvaluator(labelCol="is_duplicate", metricName="areaUnderROC")
AUCdt = evaluator.evaluate(dtPredictions)
print("Decision Tree AUC = %g " % AUCdt)


# ## Random Forest
#Train a Random Forest model, find optimal number of trees.
#Train random forest
#hyperparams - # of trees is bias, depth is variance 
rf = RandomForestClassifier(labelCol="is_duplicate")
RFmodel = rf.fit(trainingData)

# Make predictions.
RFpredictions = RFmodel.transform(testData)

# Calculate AUC, analyze importance of features.
#Select (prediction, true label) and compute AUC of the test
evaluator = BinaryClassificationEvaluator(labelCol="is_duplicate")
AUCrf = evaluator.evaluate(RFpredictions)
print("Random Forest AUC = %g" % AUCrf)

#Feature importance code
RFimp = pd.Series(RFmodel.featureImportances.toArray(), index = featureNames)
print(RFimp.sort_values(ascending=False))


# ## Gradient-boosted trees
# Fit GBT model and make predictions.
#iterations is main hyperparam (30 is max), eta doesn't change in Y code 
gbt = GBTClassifier(labelCol="is_duplicate", maxIter=40, stepSize=0.05, maxDepth=4) 
gbtModel = gbt.fit(trainingData)

# Make predictions
gbtPredictions = gbtModel.transform(testData)

# Calculate AUC
# Select (prediction, true label) and compute test error
evaluator = BinaryClassificationEvaluator(labelCol="is_duplicate")
AUCgbt = evaluator.evaluate(gbtPredictions)
print("Gradient Boosted Trees AUC = %g" % AUCgbt)

# Analyze importance of features.
#Feature importance code
GBTimp = pd.Series(gbtModel.featureImportances.toArray(), index = featureNames)
print(GBTimp.sort_values(ascending=False))

#find best model and assign final classifier 
if AUCrf > AUCgbt and AUCrf > AUClogit:
    final_classifer = RandomForestClassifier(labelCol="is_duplicate")
    print("rf classifier evaluated")
elif AUClogit > AUCrf and AUClogit > AUCgbt:
    final_classifer = GeneralizedLinearRegression(family="binomial", link="logit", featuresCol="features", labelCol="is_duplicate")
    print("log reg classifier evaluated")
else:
    final_classifer = GBTClassifier(labelCol="is_duplicate", maxIter=40, stepSize=0.05, maxDepth=4)
    print("gbt classifier evaluated")
	
final_model = final_classifer.fit(train_df)
final_pred = final_model.transform(test_df)

print("final predictions from final model:")
final_pred.show(5)

# The resulting file should contain two columns: id of the questions pair and a probability of identity of the questions.
# For *Decision trees*, *Random forest* and *Gradient-boosted trees* it is the second value of the vector contained in *"probability"* column. 
# Define a function to extract `value[1]` from a vector column and create a scalar column.
prob_of_one_udf = func.udf(lambda v: float(v[1]), FloatType())

if AUClogit > AUCrf and AUClogit > AUCgbt:
    outdf = final_pred.select('id','prediction')
    outdf = outdf.selectExpr('id as id','prediction as predict')
else:
    outdf = final_pred.withColumn('predict', func.round(prob_of_one_udf('probability'),6)).select('id','predict')    

outdf.cache()

print("final predictions after transform:")
outdf.show(6)

# Save the result to *csv* file using `coalesce(1)` option to get a single file.
outdf.orderBy('id').coalesce(1).write.csv(outPath,header=True,mode='overwrite',quote="")

