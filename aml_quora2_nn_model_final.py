
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, concatenate, BatchNormalization, Lambda, Add, add, multiply, Embedding, GaussianNoise, LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

#full data
#import embedding_matrix, data (vector representations of questions) - train and test, and nb_words
import pickle

embedding_matrix=pickle.load(open('./AML_Project3_Data/savedData/embedding_matrix.pkl', 'rb'))
nb_words=pickle.load(open('./AML_Project3_Data/savedData/nb_words.pkl', 'rb'))
data_1=pickle.load(open('./AML_Project3_Data/savedData/data_1.pkl', 'rb'))
data_2=pickle.load(open('./AML_Project3_Data/savedData/data_2.pkl', 'rb'))
test_data_1=pickle.load(open('./AML_Project3_Data/savedData/test_data_1.pkl', 'rb'))
test_data_2=pickle.load(open('./AML_Project3_Data/savedData/test_data_2.pkl', 'rb'))

#import input3 (NLP features) - train and test (re-ran and output as 1 CSV) 
train_data=pd.read_csv('./AML_Project3_Data/train_features/nlp_train_features.csv')
Y_train = train_data.is_duplicate.values
train_id = train_data.id.values
train_data = train_data.drop(["id","is_duplicate"], axis=1).values

test_data=pd.read_csv('./AML_Project3_Data/test_features/nlp_test_features.csv')
Y_test = test_data.is_duplicate.values
test_id = test_data.id.values
test_data = test_data.drop(["id","is_duplicate"], axis=1).values

print('train data shape:', train_data.shape)
print('test data shape:', test_data.shape)
print('Y train sample:')
print(Y_train[:10])
print('Y test sample:')
print(Y_test[:10])
print('test_id sample:')
print(test_id[:20])

#create function for the NN architecture

EMBEDDING_DIM=300
MAX_SEQUENCE_LENGTH=30 #this is the number of words you can replace with 

def call_nn(dropout_rate=0.1, gauss_std=.1, neurons_lstm=300):
    #input Question1 (input1), get embeddings and put through LSTM 
    input1 = Input(shape=(MAX_SEQUENCE_LENGTH,), name = "input_1") 
    embedding1 = Embedding(nb_words, EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)(input1)
    lstm1 = LSTM(neurons_lstm, recurrent_dropout=0.2)(embedding1)
    
    #input Question2 (input2), get embeddings and put through LSTM
    input2 = Input(shape=(MAX_SEQUENCE_LENGTH,), name = "input_2") 
    embedding2 = Embedding(nb_words, EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)(input2)
    lstm2 = LSTM(neurons_lstm, recurrent_activation='hard_sigmoid', recurrent_dropout=0.2)(embedding2) #add recurrent dropout rate?
    
    #take squared diff of lstm1 and lstm2
    subtract_layer = Lambda(lambda inputs: inputs[0] - inputs[1], name = "take_diff")
    diff = subtract_layer([lstm1, lstm2])  
    sqr_input = Lambda(lambda x: x*x, name = 'square_diff')(diff)
    
    #concatenate square diff vector with the sum of lstm1 and lstm2, then put in dropout layer
    add_layer = Lambda(lambda inputs: inputs[0] + inputs[1], name = "add")
    sum_lstm_output = add_layer([lstm1, lstm2])  
    concatenate_1 = concatenate([sqr_input, sum_lstm_output])
    concatenate_1 = Dropout(dropout_rate)(concatenate_1)
    
    #bring in input3, scale it, add dense layer and dropout
    input3 = Input(shape=(train_data.shape[1],), name="nlp_features")
    batch_norm1 = BatchNormalization()(input3)
    dense1 = Dense(10, activation='relu')(batch_norm1) #assumes it is fully-connected?
    nlp_dense1 = Dropout(dropout_rate)(dense1)
    
    #now concatenate the two dropout layers, then do batch norm
    concatenate_2 = concatenate([concatenate_1, nlp_dense1]) #these need to be the same dim
    batch_norm2 = BatchNormalization()(concatenate_2)
    
    #add Guassian Noise for regularization purposes - only in training 
    gaussian = GaussianNoise(gauss_std)(batch_norm2)
    
    #add one more hidden layer and a final dropout layer
    dense2 = Dense(100, activation='relu')(gaussian)
    dense2_dropout = Dropout(dropout_rate)(dense2)
    dense2_dropout = BatchNormalization()(dense2_dropout)
    output = Dense(1, activation='sigmoid')(dense2_dropout)
    
    #now create and compile the model
    model = Model(inputs=[input1, input2, input3], output=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_model():
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    lstm_layer = LSTM(75, recurrent_dropout=0.2)

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype="int32")
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype="int32")
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = lstm_layer(embedded_sequences_2)

    features_input = Input(shape=(train_data.shape[1],), dtype="float32")
    features_dense = BatchNormalization()(features_input)
    features_dense = Dense(200, activation="relu")(features_dense)
    features_dense = Dropout(0.2)(features_dense)

    addition = add([x1, y1])
    minus_y1 = Lambda(lambda x: -x)(y1)
    merged = add([x1, minus_y1])
    merged = multiply([merged, merged])
    merged = concatenate([merged, addition])
    merged = Dropout(0.4)(merged)

    merged = concatenate([merged, features_dense])
    merged = BatchNormalization()(merged)
    merged = GaussianNoise(0.1)(merged)

    merged = Dense(150, activation="relu")(merged)
    merged = Dropout(0.2)(merged)
    merged = BatchNormalization()(merged)

    out = Dense(1, activation="sigmoid")(merged)

    model = Model(inputs=[sequence_1_input, sequence_2_input, features_input], outputs=out)
    model.compile(loss="binary_crossentropy", optimizer="nadam")
    return model


#fit/tune the network and evaluate 

#set callbacks 
#checkpointer = ModelCheckpoint('./AML_Project3_Data/p3_weights.hdf5', verbose=1)
early_stopping = EarlyStopping(monitor="val_loss", patience=3)

#initialize model - tune using number of neurons, dropout rates, std of Gaussian Noise
p3_model1 = call_nn()
#p3_model1 = get_model()

#fit model - can tune epoch number and batch size 
#FIND BEST BATCH SIZE USING EARLY STOP 
p3_model_hist = p3_model1.fit([data_1, data_2, train_data], Y_train, validation_split =0.1,
              epochs = 30, batch_size = 1025, callbacks = [early_stopping], verbose = 1)  #removed checkpointer from callbacks 

print('Lowest log-loss and epoch number at early stopping:')
min(p3_model_hist.history['val_loss'])
len(p3_model_hist.history['val_loss'])

#get predictions 
pred = p3_model1.predict([test_data_1, test_data_2, test_data],verbose=0)

#prepare the data for submission  
test_id = pd.DataFrame(test_id)
pred = pd.DataFrame(pred)
outData=pd.concat([test_id, pred], axis=1)
outData.columns = ["test_id", "is_duplicate"]

print('dim outData:')
print(outData.shape)

submission = outData
#submission = pd.DataFrame({"test_id": test_id, "is_duplicate": pred})
submission.to_csv("./AML_Project3_Results/submission1.csv", index=False)

print('top 10 predictions:')
print(submission[:10])

