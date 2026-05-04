#a model to classify bad words
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer #derive root form of words
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import tensorflow as tf

#Creating the data set
dataset=pd.read_csv('TrainingData/Hate and offensive speech detection.csv',delimiter=',')

# removing the stopwords
ps=PorterStemmer()
corpus=[]

for i in range(0,dataset.last_valid_index()):
    tweet=re.sub('[^a-zA-Z]',' ',dataset['tweet'][i]) #remove dots and exlamation marks 
    tweet=tweet.lower()
    tweet=tweet.split() #split the sentence on space to get the words
    #Applying stemming
    clean_tweet=[ps.stem(word) for word in tweet if not word in set(stopwords.words('english'))]

    #concat to get the sentence
    clean_tweet=' '.join(clean_tweet)
    corpus.append(clean_tweet)

#convert the dataset into numerical format
vectorizer=TfidfVectorizer(max_features=1500,min_df=3,max_df=0.6)  
X=vectorizer.fit_transform(corpus).toarray()

#Creating a dependant variable Y
Y=dataset.iloc[:,2].values

#Split data into training and test set #80 for training 20 for testing
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#build my model these are the layers
model=tf.keras.models.Sequential([
tf.keras.layers.Dense(500,activation='relu'),
tf.keras.layers.Dense(500,activation='relu'),
tf.keras.layers.Dense(2,activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    matrics=['accuracy']
)

#train the model
model.fit(X_train,Y_train,epochs=100)

#check loss and accuracy
loss,accuracy=model.evaluate(X_test,Y_test)

#model summary
model.summary()

#test with your own data here
sample_test=["My nigga"]
sample_test=vectorizer.transform(sample_test).toarray()

#use the model to predict the sentiment
sentiment=model.predict(sample_test)[:,2]

print(sentiment)