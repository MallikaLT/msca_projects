
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType
from pyspark.sql.functions import isnan, when, count, length, lit, udf, col, struct
import numpy as np
import time

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel("WARN") 
spark


# ## Data structure
# Look at the train and test data
start_time = time.time()

trainFileName = "./AML_Project2_Data/Quora1_data/train_sample.csv" #currently sample data - substitute for full data
testFileName = "./AML_Project2_Data/Quora1_data/test_sample.csv"


# Create schema, read train data file, remove records with missing observations and cache it.
sch = StructType([StructField('id',IntegerType()),StructField('qid1',IntegerType()),StructField('qid2',IntegerType()),StructField('question1',StringType()),StructField('question2',StringType()),StructField('is_duplicate',IntegerType())])
train = spark.read.csv(trainFileName, header=True, escape='"',quote='"',schema=sch, multiLine = True)
train = train.dropna()
train.cache()
print('Number of rows = %s' % train.count())

# Read the test file, remove records with missing observations, remove columns not found in Test, cache it.
test = spark.read.csv(testFileName, header=True, escape='"',encoding='utf8', multiLine = True)
test = test.dropna()
test.cache()
print('Number of rows = %s' % test.count())
train = train.drop('qid1', 'qid2')

#Create dataframe `test` with new column `id`
maxTrainID = train.groupBy().max('id').collect()[0][0]
test = test.withColumn("id",(test.test_id+maxTrainID+1).cast("integer")).drop('test_id')

# Add column `'is_duplicate'` containing vaues `-1` indicating that the corresponding rows did not have response.
test = test.withColumn('is_duplicate', lit(-1))

# Now join both dataframes.
data = train.union(test.select(train.columns))

print("Files Loaded and Test/Train Unioned")

# ## Python and Spark natural language processing tools

import nltk
#nltk.download("popular")

# ## Removing stopwords and punctuation
stop_words = nltk.corpus.stopwords.words('english')
stop_words = set(stop_words)

# ## POS-Tagging
nltk.download("tagsets")

# ## Lemmatization and Stemming
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

from nltk.corpus import wordnet as wn

#Create Function lemmas_nltk(s)
def lemmas_nltk(s):
    return " ".join([wordnet_lemmatizer.lemmatize(wordnet_lemmatizer.lemmatize(w,'n'),'v')
                     for w in s.lower().split() if w.isalpha() & (not w in stop_words)])

lemmas_nltk_udf = udf(lemmas_nltk, returnType=StringType())  

# Create wordsCount_udf using `.udf()` of type integer
def wordsCount(s):
    return(len(s.split()))
    #return(len(re.findall(r'\w+', lemma_s)) + 1)  

wordsCount_udf = udf(wordsCount, returnType=IntegerType())

# Create `ratio_udf` using `.udf()` of type double
def ratio(x,y): return abs(x-y)/(x+y+1e-15) 
ratio_udf = udf(ratio, DoubleType())

# ## TF-IDF
from pyspark.ml.feature import IDF, Tokenizer, CountVectorizer

# ## N-grams
# Create commonNgrams(s1,s2,n), commonNgrams_udf
import re
def commonNgrams(s1, s2, n):
    #split and lowercase both documents 
    s1_SL = [w.lower() for w in s1.split()]
    s2_SL = [w.lower() for w in s2.split()]
    #remove special chars - create regex function first
    re_match = re.compile('([^\s\w]|_)+') #if it's not whitespace or char OR an underscore?
    re_s1 = [re_match.sub('', i) for i in s1_SL] #regex works on string, not lists 
    re_s2 = [re_match.sub('', i) for i in s2_SL]
    #put words back into a string for n grams
    str_s1 = ' '.join(re_s1)
    str_s2 = ' '.join(re_s2)
    #now compare n grams from both docs
    return(len(set(nltk.ngrams(nltk.word_tokenize(str_s1),n)) & set(nltk.ngrams(nltk.word_tokenize(str_s2),n))))

commonNgrams_udf = udf(commonNgrams, IntegerType())

#unigram_ratio()
def unigram_ratio(ngrams, n1, n2): #n1 = word count from doc1, n2 = word count from doc2
    return(ngrams/(1+ max(n1,n2)))

unigram_ratio_udf = udf(unigram_ratio, DoubleType())

print("Finished Creating Functions")

# # Example: Creating Features
# ## Project description

featureNames = ['lWCount1', 'lWCount2',
                'qWCount1', 'qWCount2',
                'lLen1', 'lLen2',
                'qLen1', 'qLen2',
                'lWCount_ratio', 'qWCount_ratio',
                'lLen_ratio', 'qLen_ratio',
                'qNgrams_1', 'qNgrams_2', 'qNgrams_3', 
                'lNgrams_1', 'lNgrams_2', 'lNgrams_3', 
                'qUnigram_ratio', 'lUnigram_ratio', 
                'tfidfDistance', 'lemma_leven', 'question_leven']


# Starting DataFrame is:
data = data.select('id','question1','question2', 'is_duplicate')

# ### Lemmatization
data = data.withColumn('lemma1', lemmas_nltk_udf('question1')).withColumn('lemma2', lemmas_nltk_udf('question2'))

# Features: lWCount, qWCount, lLen, qLen
for i in ["1","2"]:
    data = data.withColumn('lWCount'+i, wordsCount_udf(data['lemma'+i])) #number of words in lemma 
    data = data.withColumn('qWCount'+i, wordsCount_udf(data['question'+i])) #number of words in original question
    data = data.withColumn('lLen'+i, length(data['lemma'+i])) #number of chars in lemma
    data = data.withColumn('qLen'+i, length(data['question'+i])) #number of chars in original question

#Lengths ratios
data = data.withColumn('lWCount_ratio', ratio_udf(data['lWCount1'], data['lWCount2']))
data = data.withColumn('qWCount_ratio', ratio_udf(data['qWCount1'],data['qWCount2']))
data = data.withColumn('lLen_ratio', ratio_udf(data['lLen1'],data['lLen2']))
data = data.withColumn('qLen_ratio', ratio_udf(data['qLen1'],data['qLen2']))

#N-grams and n-gram ratios
data = data.withColumn('qNgrams_1', commonNgrams_udf(data['question1'], data['question2'],lit(1))) 
data = data.withColumn('qNgrams_2', commonNgrams_udf(data['question1'], data['question2'],lit(2))) 
data = data.withColumn('qNgrams_3', commonNgrams_udf(data['question1'], data['question2'],lit(3))) 
data = data.withColumn('lNgrams_1', commonNgrams_udf(data['lemma1'], data['lemma2'],lit(1))) 
data = data.withColumn('lNgrams_2', commonNgrams_udf(data['lemma1'], data['lemma2'],lit(2)))
data = data.withColumn('lNgrams_3', commonNgrams_udf(data['lemma1'], data['lemma2'],lit(3))) 

#unigram ratios - number of shared unigrams over the 2 documents 
data = data.withColumn('qUnigram_ratio', unigram_ratio_udf('qNgrams_1', 'qWCount1', 'qWCount2'))
data = data.withColumn('lUnigram_ratio', unigram_ratio_udf('lNgrams_1', 'lWCount1', 'lWCount2'))

print("created features before TF-IDF")

# ### TF-IDF
#Tokenization of lemmas
tokenizer = Tokenizer(inputCol="lemma1", outputCol="words1") 
data = tokenizer.transform(data) 
tokenizer.setParams(inputCol="lemma2", outputCol="words2") #MT: why is this SetParams when lemma1 tokenizer uses Tokenizer func?
data = tokenizer.transform(data) 
#data.select('id','lemma1','words1','lemma2','words2').show(5)

# #### Creating vocabulary and calculating TF column
corpus = data.selectExpr('words1 as words').join(data.selectExpr('words2 as words'), on='words', how='full') 

# (1) Initialize class `CountVectorizer`.  
cv = CountVectorizer(inputCol="words", outputCol="tf", minDF=2.0)

# (2) Fit a `CountVectorizerModel` to *"corpus"*.  
cvModel = cv.fit(corpus)
corpus = cvModel.transform(corpus)

# (3) Apply `CountVectorizerModel.transform()` to *question1* and *question2*.
res1 = cvModel.transform(data.selectExpr('id', 'words1 as words')) #id = row#, words1 was Q1 after lemmatizing and tokenizing
res2 = cvModel.transform(data.selectExpr('id', 'words2 as words')) #words2 was Q2 after lemmatizing and then tokenizing
 
# #### Calculating IDF
idf = IDF(inputCol="tf", outputCol="idf")
idfModel = idf.fit(corpus)
res1 = idfModel.transform(res1)
res2 = idfModel.transform(res2)
res = res1.selectExpr('id','idf as idf1').join(res2.selectExpr('id','idf as idf2'), on='id', how='inner')

# Create function `tfidfDist(a,b):` calculating squared distance between vectors a and b. <br>
# Turn this function into udf.
def tfidfDist(a,b): return float(a.squared_distance(b))
dist_udf = udf(tfidfDist, DoubleType())
res = res.withColumn('dist', dist_udf(res['idf1'], res['idf2']))

# Drop unnessesary columns from `data` and join in a new feature *"tfidfDistance"*.
data = data.drop('words1', 'words2')
data = data.join(res.selectExpr('id','dist as tfidfDistance'),on='id',how='inner')
#data.select('id','tfidfDistance').show(6)

print("created feature TF-IDF")

#Add Levenshtein Distance as last feature, for both lemmas and questions 
from pyspark.sql.functions import levenshtein

data = data.withColumn('lemma_leven', levenshtein('lemma1', 'lemma2'))
data = data.withColumn('question_leven', levenshtein('question1', 'question2'))

print ('All Features Created in %d Minutes'  % (float(format(time.time() - start_time))/60))

#output features
outData = data.select(['id']+featureNames+['is_duplicate'])
outData = outData.cache()

print("Cached outData in %d Minutes" % (float(format(time.time() - start_time))/60))

outTrainFileName = "./AML_Project2_Data/train_features.csv"
outTestFileName = "./AML_Project2_Data/test_features.csv"

outData.filter(outData.id <= maxTrainID).write.option("header","true").csv(outTrainFileName,mode='overwrite',quote="")

outData.filter(outData.id > maxTrainID).withColumn('id', outData.id-maxTrainID-1).write.option("header","true").csv(outTestFileName,mode='overwrite',quote="")

