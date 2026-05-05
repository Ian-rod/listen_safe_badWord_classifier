import pickle
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer

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

print("Welcome to the bad word model")
model_summary()

#test it with data
while 1:
    test_data=input("Enter test statement: ")
    test_data=vectorizer.transform([test_data]).toarray()
    #use the model to predict the sentiment
    sentiment=model.predict(test_data)[:,1]
    print("The chances of this being a bad word is "+str(sentiment)+'\n')
