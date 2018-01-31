
# coding: utf-8

# In[1]:

import numpy as np
from pyspark.sql.types import StructType, StructField, DoubleType, LongType, StringType, IntegerType, Row
import pyspark.sql.functions as func
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
import random
import time
from datetime import datetime
import re


# In[2]:

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel("WARN") 
spark


# In[3]:

# path to files
artistdata_path = 'AdvancedML_MusicRecommenderData2/artist_data.csv'
userartist_path = 'AdvancedML_MusicRecommenderData2/user_artist_data_train.csv'  
test_path = 'AdvancedML_MusicRecommenderData2/LastFM_Test_Sample.csv'


# In[4]:

# Schemas for both files
artistdata_struct = StructType([StructField('artistId', LongType()),StructField('name', StringType())])
userartist_struct = StructType([StructField('userId', LongType()),StructField('artistId', LongType()),StructField('song_count', LongType())])


# In[5]:

# read artist_data file
artistdata_df = spark.read.csv(artistdata_path, sep = '\t', schema = artistdata_struct)
artistdata_df.cache()

# read user_artist_data file
userartist_df = spark.read.csv(userartist_path, sep = '\t', schema = userartist_struct)
userartist_df.cache()


# In[6]:

#create Artists (aggregated) DF
userData_grouped = userartist_df.groupBy("artistID").agg(func.sum("song_count").alias("total_count"),func.count("userID").alias("num_of_users"))

artists = userData_grouped.join(artistdata_df, userData_grouped.artistID==artistdata_df.artistId).drop(artistdata_df.artistId).orderBy("num_of_users", ascending=False)

#artists.cache()

PPU = artists.withColumn("plays_per_user", artists.total_count/artists.num_of_users)

PPU.cache()

#set threshold for PPU to filter out artists above threshold 
mean_ppu = PPU.select(func.mean("plays_per_user")).take(1)[0]["avg(plays_per_user)"]
std_ppu = PPU.select(func.stddev("plays_per_user")).take(1)[0]["stddev_samp(plays_per_user)"]

ppu_threshold = mean_ppu + 2*std_ppu


# In[7]:

#create ArtistsInclude DF which is the artists we want to include in the analysis
ArtistInclude = ArtistInclude = PPU.filter((PPU["name"]!="[unknown]") & (~ PPU.name.like("%.com%")) & (~ PPU.name.like("%.org%")) & (PPU.num_of_users > 5)) #(PPU.plays_per_user < ppu_threshold) & (~ PPU.name.rlike("^[^a-zA-Z]+$"))

ArtistInclude.cache()


# In[8]:

#create Users (aggregated) DF
users = userartist_df.groupBy("userID").agg(func.sum("song_count").alias("total_song_count"), func.count("artistId").alias("num_of_artists"))

users.cache()

SPA = users.withColumn("songs_per_artist", users.total_song_count/users.num_of_artists)

SPA.cache()

#set threshold for SPA to filter out users above threshold 
mean_spa = SPA.select(func.mean("songs_per_artist")).take(1)[0]["avg(songs_per_artist)"]
std_spa = SPA.select(func.stddev("songs_per_artist")).take(1)[0]["stddev_samp(songs_per_artist)"]

spa_threshold = mean_spa + 3*std_spa  #changed from mean + 2 std


# In[9]:

#create UsersInclude DF which is the users we want to include in the analysis
UserInclude = SPA.filter((SPA.songs_per_artist < spa_threshold) & (SPA.num_of_artists > 5)) #not enough artists to ascertain taste - arbitrary cutoff

UserInclude.cache()


# In[10]:

#inner join ArtistsInclude and UsersInclude to userartist_df to only see records for artists/users we want to include
userartist_df_join1 = userartist_df.join(ArtistInclude, userartist_df.artistId == ArtistInclude.artistID, "inner").select(userartist_df.userId, userartist_df.artistId, userartist_df.song_count)
    
modelData = userartist_df_join1.join(UserInclude, userartist_df_join1.userId == UserInclude.userID, "inner").select(userartist_df_join1.userId, userartist_df_join1.artistId, userartist_df_join1.song_count)

modelData.cache()


# In[11]:

# split model data:
(training, test) = modelData.randomSplit([0.9, 0.1], seed=0)
training.cache()

test = test.drop('song_count')
test.cache()


# In[ ]:

#broadcast all artist IDs
allItemIDs = training.select('artistId').distinct().rdd.map(lambda x: x[0]).collect() #get distinct artist IDs
bAllItemIDs = spark.sparkContext.broadcast(allItemIDs) #broadcast list of distinct artist IDs to each node

#broadcast top artist IDs
artists = training.groupBy('artistId').agg(func.count(func.lit(1)).alias('num_of_users')) #user counts per artist
artists.cache()
top_artists = artists.orderBy('num_of_users', ascending=False).limit(10000).rdd.map(lambda x: x['artistId']).collect() 
bTopItemIDs = spark.sparkContext.broadcast(top_artists) #broadcast list of top 10K artists to each node


# In[ ]:

# define meanAUC logic according to 'Advanced Analytics with Spark'
def areaUnderCurve(positiveData, bAllItemIDs, predictFunction):
    positivePredictions = predictFunction(positiveData.select("userId", "artistId")).withColumnRenamed("prediction", "positivePrediction")
        
    negativeData = positiveData.select("userId", "artistId").rdd.groupByKey().mapPartitions(lambda userIDAndPosItemIDs: createNegativeItemSet(userIDAndPosItemIDs,bAllItemIDs)).flatMap(lambda x: x).map(lambda x: Row(userId=x[0], artistId=x[1])).toDF()
    
    negativePredictions = predictFunction(negativeData).withColumnRenamed("prediction", "negativePrediction")

    joinedPredictions = positivePredictions.join(negativePredictions, "userId").select("userId", "positivePrediction", "negativePrediction").cache()
        
    allCounts = joinedPredictions.groupBy("userId").agg(func.count(func.lit("1")).alias("total")).select("userId", "total")
    correctCounts = joinedPredictions.where(joinedPredictions.positivePrediction > joinedPredictions.negativePrediction).groupBy("userId").agg(func.count("userId").alias("correct")).select("userId", "correct")

    joinedCounts = allCounts.join(correctCounts, "userId")
    meanAUC = joinedCounts.select("userId", (joinedCounts.correct / joinedCounts.total).alias("auc")).agg(func.mean("auc")).first()

    joinedPredictions.unpersist()

    return meanAUC[0]


def createNegativeItemSet(userIDAndPosItemIDs, bAllItemIDs):
    allItemIDs = bAllItemIDs.value
    return map(lambda x: getNegativeItemsForSingleUser(x[0], x[1], allItemIDs),userIDAndPosItemIDs)


def getNegativeItemsForSingleUser(userID, posItemIDs, allItemIDs):
    posItemIDSet = set(posItemIDs)
    negative = []
    i = 0
    # Keep about as many negative examples per user as positive.
    # Duplicates are OK
    while i < len(allItemIDs) and len(negative) < len(posItemIDSet):
        itemID = random.choice(allItemIDs) 
        if itemID not in posItemIDSet:
            negative.append(itemID)
        i += 1
    # Result is a collection of (user,negative-item) tuples
    return map(lambda itemID: (userID, itemID), negative)


# In[ ]:

#create gridsearch to find optimal hyperparameters

try_rank = [30,35,40]
try_alpha = [2,5,12]
try_reg = [2,3,3.5]
auc_res = []

for rank in try_rank:
    for alpha in try_alpha:
        for reg in try_reg:
            #fit model with params for this iteration 
            loop_model = ALS(implicitPrefs=True, userCol="userId", itemCol="artistId", ratingCol="song_count", rank=rank, alpha=alpha, regParam=reg).fit(training)
            #evaluate AUC
            loop_auc = areaUnderCurve(test, bTopItemIDs, loop_model.transform) #AUC for test data w/pred from iteration's model
            #add tuple of hyperparams and AUC to initalized results list 
            auc_res_content = (rank, alpha, reg, loop_auc)
            print(auc_res_content)
            auc_res += tuple([auc_res_content])


final_rank = max(auc_res, key=lambda item:item[3])[0]
final_alpha = max(auc_res, key=lambda item:item[3])[1]
final_reg = max(auc_res, key=lambda item:item[3])[2]

print(final_rank)
print(final_alpha)
print(final_reg)


# In[ ]:

# reading test file
test_struct = StructType([StructField('userId', IntegerType()),StructField('artistId', IntegerType())])
test_df = spark.read.csv(test_path, sep = '\t', schema = test_struct)


# In[ ]:

#run final model with optimal hyperparameters
final_model = ALS(implicitPrefs=True, userCol="userId", itemCol="artistId", ratingCol="song_count",rank=final_rank, alpha=final_alpha, regParam=final_reg).fit(modelData)

#get final predictions for test set 
final_predictions = final_model.transform(test_df)


# In[ ]:

#write results out to CSV
timestamp = datetime.today().isoformat()
timestamp=re.sub(':','',timestamp)
timestamp=timestamp.split('.')[0]
timestamp=re.sub('T','-T',timestamp)

final_predictions.coalesce(1).write.csv('./lastFM_results/test_predictions_{}.csv'.format(timestamp), sep = '\t')

