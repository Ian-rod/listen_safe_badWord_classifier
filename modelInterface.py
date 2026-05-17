import pickle
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import PorterStemmer, SnowballStemmer
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
from io import StringIO
from inscriptis import get_text


#load the ml model
def load_model() -> tf.keras.Model:
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

#load the vectorizer
def load_vectorizer() -> TfidfVectorizer:
    with open("vectorizer.pkl", "rb") as f:
        return pickle.load(f)

#call method to load  model and vectorizer
model=load_model()
vectorizer=load_vectorizer()

def model_summary():
    print(model.summary())

#the main predictive method
def predict(rawTextInput)->int:
    ps=PorterStemmer()
    rawTextInput=re.sub('[^a-zA-Z]',' ',rawTextInput)# remove non char
    rawTextInput=rawTextInput.lower()
    rawTextInput=rawTextInput.split() 
    cleanRawTextInput=[ps.stem(word) for word in rawTextInput if not word in set(stopwords.words('english'))]
    cleanRawTextInput=' '.join(cleanRawTextInput)
    cleanRawTextInput=vectorizer.transform([cleanRawTextInput]).toarray()
    sentiment=model.predict(cleanRawTextInput)[:,1]
    return sentiment

#text to filter from HTML
def predict_Html(rawHtmlInput)->int:
    html_text=get_text(rawHtmlInput)
    return predict(html_text)

#data should be in the form of data,label
def re_train_model():
    csvpath=input("Enter the CSV path: ")
    sbs=SnowballStemmer(language='english')
    corpus=[]
    Y=[]
    dataset=pd.read_csv(csvpath,delimiter=',')
    print(dataset.info())

    #pre processing data
    print('Begining data set clean up')
    for i in range(0,dataset.last_valid_index()):
        data=str(dataset['data'][i])
        data=re.sub('[^a-zA-Z]',' ',data) #remove dots and exlamation marks 
        data=data.lower()
        data=data.split() #split the sentence on space to get the words
        #Applying stemming
        clean_data=[sbs.stem(word) for word in data if not word in set(stopwords.words('english'))]
        #concat to get the sentence
        clean_data=' '.join(clean_data)
        corpus.append(clean_data)
        Y.append(dataset['label'][i])
        print('Procressing row '+ str(i))
    print('Data set clean up complete')

    X=vectorizer.transform(corpus).toarray()

    #Creating a dependant variable Y
    Y = np.array(Y)
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

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

#main interface
#test it with data
print("\nWelcome to the bad word model\n")
while 1:
    user_command=input("\nUse 1 to enter data 2 to train model 3 to print summary: ")
    match int(user_command):
        case 1:
            test_data=input("Enter test statement: ")
            print("The chances of this being a bad word is "+str(predict(test_data))+'\n')
        case 2:
            re_train_model()
        case 3:
            model_summary()
        case 4:
            HtmlFile = open("html_test/clean_lyrics.html",'r')
            print(predict_Html(HtmlFile.read()))