import string
import re
import yaml

import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize, WordPunctTokenizer

from logzero import logger

from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.models import Model, Sequential

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report



class NaturalDisasters:
    def __init__(self,config_file):
        with open(config_file, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        # Download and load required NLTK modules
        punct = nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.stop_words = set(stopwords.words("english"))
        nltk.download('omw-1.4')

        # Initialize lemmatizer, stemmer, and tokenizer from NLTK
        self.lemma=WordNetLemmatizer()
        self.stemm = PorterStemmer()
        self.tokenize = WordPunctTokenizer()

        # Load training and test data
        self.train_data = pd.read_csv(self.config["input"]["train"]["filepath"])
        self.test_data = pd.read_csv(self.config["input"]["test"]["filepath"])

        # Debugging statement to print first 5 rows of training data
        DEBUGVAL__ = self.train_data.head()
        logger.debug(f"self.train_data.head()\n {DEBUGVAL__}")

        # Debugging statement to print null values in training data
        logger.debug(f"data.isnull().sum(): \n {self.train_data.isnull().sum()}")


    def clean_text(self,text):
      text=str(text.lower())
      text=re.sub('\d+', '', text)
      text= re.sub('\[.*?\]', '', text)
      text= re.sub('https?://\S+|www\.\S+', '', text)
      text= re.sub('[%s]' % re.escape(string.punctuation),'',text)
      text = ' '.join([word for word in text if word.lower() not in self.stop_words])
      text = ' '.join(self.stemm.stem(word) for word in text.split(' '))
      text = ' '.join(self.lemma.lemmatize(word) for word in text.split(' '))
      text = self.tokenize.tokenize(text)
      return text
      
    def preprocess_data(self): 
        self.train_data.drop(['id', 'keyword', 'location'], axis=1, inplace=True)

        self.train_data.drop_duplicates(inplace=True)

        # Separate labels from text
        self.label_data = self.train_data.drop(['text'], axis=1)
        self.train_data = self.train_data['text']

        # Clean the text data
        self.train_data = self.train_data.apply(self.clean_text)
        
        # Log the preprocessed data
        logger.debug(f"Preprocessed train data:\n{self.train_data.head()}")
        logger.debug(f"Label data:\n{self.label_data.head()}")

    def build_model(self): 
        # Set the max number of features and create a tokenizer
        max_features = 3000
        tokenizer = Tokenizer(num_words=max_features, split=' ')
        
        # Fit the tokenizer on the preprocessed text data
        tokenizer.fit_on_texts(self.train_data.values) 
        self.tokenizer = tokenizer
        
        # Convert text data to sequences and pad the sequences
        self.X = self.tokenizer.texts_to_sequences(self.train_data.values)
        self.X = pad_sequences(self.X)    
        
        # Split the data into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.label_data, test_size=0.2, random_state=41)
        
        # Define the model architecture
        max_features = 30000
        embed_dim = 32
        self.model = Sequential()
        self.model.add(Embedding(max_features, embed_dim, input_length=self.X_train.shape[1]))
        self.model.add(LSTM(units=60, input_shape=(self.X_train.shape[1], 1), activation='relu', return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1, activation='sigmoid'))
            
        # Compile the model
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train_model(self): 
        # Train the model and get predictions
        self.model.fit(self.X_train, self.y_train, epochs=10, batch_size=32, validation_data=(self.X_test, self.y_test))
        self.y_pred = self.model.predict(self.X_test).round()
        
        # Evaluate the model performance
        scores = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        accuracy = scores[1]
        train_accuracy = round(metrics.accuracy_score(self.y_train, self.model.predict(self.X_train).round()) * 100)
        precision = round(precision_score(self.y_test, self.y_pred, average='weighted'), 3)
        recall = round(recall_score(self.y_test, self.y_pred, average='weighted'), 3)
        
        # Log the model performance metrics
        logger.debug(f"Test accuracy: {accuracy}")
        logger.debug(f"Train accuracy: {train_accuracy}")
        logger.debug(f"Precision: {precision}")
        logger.debug(f"Recall: {recall}")
        logger.debug(f"Classification report:\n{classification_report(self.y_test, self.y_pred)}")

    def predict(self): 
        # Drop unnecessary columns
        self.test_data.drop(columns=['keyword', 'location'], inplace=True)
                
        # Extract the 'id' column from the test data
        id = self.test_data['id']
        
        # Convert 'id' column to a dataframe and create a new dataframe with 'id' column
        id = id.to_frame()
        id = pd.DataFrame(id)
        
        # Apply the text cleaning function to the 'text' column of the test data
        self.test_data=self.test_data["text"].apply(self.clean_text)
        
        DEBUGVAL__ = self.test_data.head()
        logger.debug(f"self.test_data\n {DEBUGVAL__}")
        
        DEBUGVAL__ = self.test_data.values
        logger.debug(f"self.test_data.values\n {DEBUGVAL__}")
        
        # Convert the text to sequences and pad them to the same length
        test_token = self.tokenizer.texts_to_sequences(self.test_data.values)
        test_token = pad_sequences(test_token, maxlen =1)
        
        final_pred = self.model.predict(test_token)
        
        # Round the predictions and convert them to integers
        final_pred = np.round(final_pred).astype(int)
        
        # Convert the 'id' and 'target' columns to dataframes
        id = pd.DataFrame(id, columns=['id'])
        final_pred = pd.DataFrame(final_pred, columns=['target'])
        
        # Concatenate the 'id' and 'target' columns and save them to a csv file
        submission = pd.concat([id, final_pred], axis=1, join='inner')
        submission.to_csv(self.config["output"], index=False)


if __name__ == "__main__":
  disasters=NaturalDisasters("/Users/atnbsn/Desktop/Machine_Learning/data/natural_disasters/natural_disasters.yaml")
  disasters.preprocess_data()
  disasters.build_model()
  disasters.train_model()
  disasters.predict()
