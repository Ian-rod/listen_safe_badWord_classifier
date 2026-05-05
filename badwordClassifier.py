#a model to classify bad words
from io import StringIO

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer #derive root form of words
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle
from datetime import datetime

#Creating the data set
dataset=pd.read_csv('TrainingData/Hate and offensive speech detection.csv',delimiter=',')
dataset.drop(columns=['username'], inplace=True)
# removing the stopwords
ps=PorterStemmer()
corpus=[]
Y=[]
print(dataset.info())
print('Begining data set clean up')
for i in range(0,dataset.last_valid_index()):
    tweet=re.sub('[^a-zA-Z]',' ',dataset['tweet'][i]) #remove dots and exlamation marks 
    tweet=tweet.lower()
    tweet=tweet.split() #split the sentence on space to get the words
    #Applying stemming
    clean_tweet=[ps.stem(word) for word in tweet if not word in set(stopwords.words('english'))]
    #concat to get the sentence
    clean_tweet=' '.join(clean_tweet)
    corpus.append(clean_tweet)
    Y.append(dataset['label'][i])
    print('Procressing row '+ str(i))


print('Data set clean up complete')
#convert the dataset into numerical format
vectorizer=TfidfVectorizer(max_features=1500,min_df=3,max_df=0.6)  
X=vectorizer.fit_transform(corpus).toarray()

#Creating a dependant variable Y
Y = np.array(Y)

#Split data into training and test set #80 for training 20 for testing
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

import tensorflow as tf
#build my model these are the layers
model=tf.keras.models.Sequential([
tf.keras.layers.Dense(500,activation='relu'),
tf.keras.layers.Dense(500,activation='relu'),
tf.keras.layers.Dense(2,activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#train the model

print('training model')
model.fit(X_train,Y_train,epochs=100)
print('training model complete')

#save the model 
print("Saving the model")
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

#save the vectorizer 
print("Saving the vectorizer")
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)


#Create as markdown with the info
#model summary
loss,accuracy=model.evaluate(X_test,Y_test)
print("Creating report")
currentdate=datetime.now()

stream = StringIO()
model.summary(print_fn=lambda x: stream.write(x + "\n"))
summary_text = stream.getvalue()

with open("modelDescription.md", "w", encoding="utf-8") as f:
    f.write("# Model Report\n\n")
    f.write('- Last run: ' +currentdate.strftime("%Y-%m-%d %H:%M:%S"))
    f.write("```\n")
    f.write(summary_text+"\n")
    f.write(f"- Accuracy: {accuracy:.4f}\n")
    f.write(f"- loss: {loss:.4f}\n")
    f.write("\n```\n")

