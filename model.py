import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
import warnings
warnings.filterwarnings('ignore')
#import nltk
nltk.download('wordnet')
#from nltk.stem import PorterStemmer
#from textblob import Word
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from collections import Counter

class SentimentAnalysis():
    def __init__(self):
        print("initialised")
    #def load_preprocess(self):
        #reviews_df=pd.read_csv('sample30.csv',encoding='latin-1')
        #reviews_df['reviews_text'] = reviews_df['reviews_text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
        #reviews_df['reviews_text'] = reviews_df['reviews_text'].str.replace('[^\w\s]','')
        #reviews_df['reviews_text'] = reviews_df['reviews_text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))
        #freq_ten = pd.Series(' '.join(reviews_df['reviews_text']).split()).value_counts()[:10]
        #freq_ten = list(freq_ten.index)
        #reviews_df['reviews_text'] = reviews_df['reviews_text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq_ten))
        #freq_rare = pd.Series(' '.join(reviews_df['reviews_text']).split()).value_counts()[-10:]
        #freq_rare = list(freq_rare.index)
        #reviews_df['reviews_text'] = reviews_df['reviews_text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq_rare))
        #st = PorterStemmer()
        #reviews_df['reviews_text'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
        #reviews_df['reviews_text'] = reviews_df['reviews_text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
        #reviews_df['user_sentiment']=reviews_df['user_sentiment'].map({"Positive":1,"Negative":0})
        #reviews_df = reviews_df.dropna(subset=['user_sentiment'])
        #print(reviews_df.head(1))
        #return reviews_df
    def lr_model(self,df):
        X = df['reviews_text']
        y = df['user_sentiment']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)
        #print('X_train', X_train.shape)
        #print('y_train', y_train.shape)
        #print('X_test', X_test.shape)
        #print('y_test', y_test.shape)
        tfidf_vect = TfidfVectorizer()
        train_tfidf = tfidf_vect.fit_transform(X_train.values.astype('U'))
        test_tfidf=tfidf_vect.transform(X_test.values.astype('U'))
        LRModel=LogisticRegression(C=10,class_weight={0:0.6,1:0.4},penalty='l2',random_state=43)
        LRModel.fit(train_tfidf, y_train)
        joblib.dump(LRModel, 'LRModel.pkl')
        joblib.dump(tfidf_vect,"tfidf_vect.pkl")
        #return tfidf_vect   
    def user_recomend(self,df,user_name):
        model_load = joblib.load("LRModel.pkl")
        tfidf_vect=joblib.load("tfidf_vect.pkl")
        user_final_rating=pd.read_csv("userpredictions.csv")
        user_final_rating.set_index('reviews_username', inplace=True)
        #print(user_final_rating.head(2))
        #user_name=str(user_name)
        #print(user_name)
        final_recomended_prods=user_final_rating.loc[user_name].sort_values(ascending=False)[0:20]
        print(final_recomended_prods.head(2))
        top20_prod={}
        top20_prod['name']=[]
        top20_prod["brand"]=[]
        top20_prod["categories"]=[]
        top20_prod['manufacturer']=[]
        recom_prod=list(final_recomended_prods.index)
        pos_rated_prod=[]
        for i in range(20):
            top20_prod["brand"].append(df.loc[df.name==recom_prod[i],"brand"].values[0])
            top20_prod["categories"].append(df.loc[df.name==recom_prod[i],"categories"].values[0])
            top20_prod['manufacturer'].append(df.loc[df.name==recom_prod[i],"manufacturer"].values[0])
            top20_prod['name'].append(final_recomended_prods.index[i])
            reviews_texts=np.array(df.loc[df.name==recom_prod[i],"reviews_text"]).astype('U')
            y=Counter(model_load.predict(tfidf_vect.transform(reviews_texts)))[1]
            pos_rated_prod.append(y)                          
        top20_df=pd.DataFrame(top20_prod)
        #top20_df
        top20_df["positive_ratings"]=pos_rated_prod
        top5_df=top20_df.sort_values(by="positive_ratings",ascending=False)[:5]
        return top5_df
   
        
    
