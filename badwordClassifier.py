#a model to classify bad words
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer #derive root form of words
import re

#Creating the data set
dataset=pd.read_csv('TrainingData/Hate and offensive speech detection.csv',delimiter=',')

# removing the stopwords
ps=PorterStemmer()
corpus=[]

for i in range(0,dataset.last_valid_index()):
    tweet=re.sub('[^a-zA-Z]',' ',dataset['tweet'][i]) #remove dots and exlamation marks 
    #print(tweet+'\n')
    tweet=tweet.lower()
    tweet=tweet.split() #split the sentence on space to get the words
    #Applying stemming
    clean_tweet=[ps.stem(word) for word in tweet if not word in set(stopwords.words('english'))]

    #concat to get the sentence
    clean_tweet=' '.join(clean_tweet)
    corpus.append(clean_tweet)

print(corpus[0])
