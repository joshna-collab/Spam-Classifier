import streamlit as st
import pickle
import sklearn.feature_extraction.text import.TfidfVectorizer

#loading the saved vectorizer and naive bayes model
Tfidf = pickle.load(open('vectorizer.pk1','rb'))
model = pickle.load(open('model.pk1','rb'))

#transform_text function for text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.corpus import PorterStemmer
import string

nltk.dowmload('stopwords')

ps.PortStemmer()

def transform_text(text):
    text = text.lower() #converting to lower case
    text = nltk.word_tokenize(text) #tokenize

    #removing special characters and retaining alphanumeric words
    text = [word for word in text if word.isalnum()]

    #removing stopwoods and punctuation
    text = [word for word in text if word not in stopwords.words('english')]

    #applying streaming
    text = [ps.stem(word) for word in text]
    
    return"".join(text)

#saving streamlit code
st.title("Email spam classifier")
input_sms = st.text_area("enter a message:")

if st.button('predict'):
    #preprocess
transformed_sms = transforme_text(input_sms)

    #vectorize 
vector_input = tdidf.transform([transform_sms])

    #predict
result = model.predict(vector_input)[0]

    #display
if result ==1:
    st.header("spam")
else:
    st.header("not spam")