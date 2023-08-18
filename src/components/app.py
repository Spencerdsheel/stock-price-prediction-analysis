import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import pandas_datareader as data
import yfinance as yf
import datetime
from keras.models import load_model
import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from numpy import array
import seaborn as sns
from keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, GlobalMaxPooling1D, Embedding, Conv1D, LSTM
from sklearn.model_selection import train_test_split
from io import StringIO
import pickle
import time



start = '2015-01-01'
end = '2022-12-31'

st.title('Stock Market Prediction and Analysis')

user_input = st.text_input('Enter Stock Ticker', 'Tickeer')

df =yf.download("user_input", start, end)

with st.spinner('Wait for it...'):
    time.sleep(5)
st.success('Done!')

#Describing Data
st.header('Stock Market Data-set')
st.dataframe(df)


#Visualizations 
st.subheader('Closing Price vs Time Chart')
fig= plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig= plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig= plt.figure(figsize=(12,6))
plt.plot(ma200)
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)



data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))

data_training_array = scaler.fit_transform(data_training)



model = load_model('lstm_mode.h5')


past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)


x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i-100: i])
  y_test.append(input_data[i, 0]) 

x_test, y_test = np.array(x_test), np.array(y_test)


y_predicted = model.predict(x_test)

scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#Final Grapph
st.subheader('Prediction vs Originals')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


#Sentiment Analysis
st.header('Sentiment Analysis')

input = pd.read_csv("tweet_data.csv")

#encoding='windows-1252'
#tweet preprocessing
TAG_RE = re.compile(r'<[^>]+>')
def remove_tags(text):
    '''Removes HTML tags: replaces anything between opening and closing <> with empty space'''

    return TAG_RE.sub('', text)

import nltk
nltk.download('stopwords')

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    '''Removes HTML tags: replaces anything between opening and closing <> with empty space'''

    return TAG_RE.sub('', text)

import nltk
nltk.download('stopwords')

def preprocess_text(text):
    '''Cleans text data up, leaving only 2 or more char long non-stepwords composed of A-Z & a-z only
    in lowercase'''
    
    unseen_text = text.lower()

    # Remove html tags
    unseen_text = remove_tags(unseen_text)

    # Remove punctuations and numbers
    unseen_text = re.sub('[^a-zA-Z]', ' ', unseen_text)

    # Single character removal
    unseen_text = re.sub(r"\s+[a-zA-Z]\s+", ' ', unseen_text)  # When we remove apostrophe from the word "Mark's", the apostrophe is replaced by an empty space. Hence, we are left with single character "s" that we are removing here.

    # Remove multiple spaces
    unseen_text = re.sub(r'\s+', ' ', unseen_text)  # Next, we remove all the single characters and replace it by a space which creates multiple spaces in our text. Finally, we remove the multiple spaces from our text as well.

    # Remove Stopwords
    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    unseen_text = pattern.sub('', unseen_text)

    # Remove rt
    unseen_text = re.sub('RT @\w+: '," ", unseen_text)

    # Special char
    unseen_text = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ", unseen_text)

    #Remove Url
    unseen_text = re.sub(r"http\S+", '', unseen_text)

        

    return unseen_text

# Preprocess review text with earlier defined preprocess_text function

unseen_processed = []
unseen_text = list(input['text'])
for text in unseen_text:
  #text = preprocess_text(text)
  unseen_processed.append(preprocess_text(text))

#loading tokeniser
with open('tokenizer.pickle', 'rb') as handle:
    word_tokenizer = pickle.load(handle)


# Tokenising instance with earlier trained tokeniser
#word_tokenizer = Tokenizer()
unseen_tokenized = word_tokenizer.texts_to_sequences(unseen_processed)


# Pooling instance to have maxlength of 100 tokens
maxlen = 100
unseen_padded = pad_sequences(unseen_tokenized, padding='post', maxlen=maxlen)

#  Load previously trained LSTM Model

pretrained_lstm_model = load_model('lstm_model0.861.h5')

unseen_sentiments = pretrained_lstm_model.predict(unseen_padded)


st.subheader('Twitter Data Predicted Sentiments')
sent = np.round(unseen_sentiments*10,1)

input['Predicted Sentiments'] = sent
df_prediction_sentiments = pd.DataFrame(input['Predicted Sentiments'], columns = ['Predicted Sentiments'])
df_review_text           = pd.DataFrame(input['text'], columns = ['text'])



dfx=pd.concat([df_review_text, df_prediction_sentiments], axis=1)
st.dataframe(dfx)

dfy= pd.DataFrame(y_predicted, columns=['Close'])
#recommendation
mean = final_df.mean()

def recommending(df, sent,dfy,mean):


      if any(dfy.iloc[-1]['Close'] < mean):
          if any(sent > 5.0):
              idea="RISE"
              decision="BUY"
              print()
              print("##############################################################################")
              print("According to the stock Predictions and Sentiment Analysis of Tweets, a",idea,"in stock is expected => ",decision)
          elif any(sent < 5.0):
              idea="FALL"
              decision="SELL"
              print()
              print("##############################################################################")
              print("According to the stock Predictions and Sentiment Analysis of Tweets, a",idea,"in stock is expected => ",decision)
      else:
          idea="FALL"
          decision="SELL"
          print()
          print("##############################################################################")
          print("According to the stock Predictions and Sentiment Analysis of Tweets, a",idea,"in stock is expected => ",decision)
      return idea, decision

result =recommending(df, unseen_sentiments,dfy,mean)

# st.subheader('Recommendation to the User')
# st.write("According to the Stock Predictions and Sentiment Analysis of Tweets, the stock is expected to:", result, "the stock")