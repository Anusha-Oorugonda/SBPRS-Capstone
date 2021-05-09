from flask import Flask, jsonify,  request, render_template
import joblib
import numpy as np
from model import SentimentAnalysis
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle

s=SentimentAnalysis()
app = Flask(__name__)

print("in app")

@app.route('/')
def home():
    #df_final=pd.read_csv("processed_txts.csv")
    #s.lr_model(df_final)
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if (request.method == 'POST'):
        user_name=request.form["reviews_username"]
        print(user_name)
        df_final=pd.read_csv("processed_txts.csv")
        #s.user_recommendation_to_csv(df_final)
        if user_name not in df_final["reviews_username"].tolist():
            return render_template('index.html', prediction_text='user name not exists')
        #print(df_final.head(1))
        top5_df=s.user_recomend(df_final,user_name)
        print(top5_df)
        top5_df=top5_df[["name","brand","manufacturer"]]
        #output=top5_df
        return render_template('index.html', columns=top5_df.columns.values, rows=list(top5_df.values.tolist()), zip=zip)
        #return render_template('index.html', prediction_text='Sentiment Output {}'.format(output))
    else :
        return render_template('index.html')
    
#@app.route("/predict_api", methods=['POST', 'GET'])
#def predict_api():
    #print(" request.method :",request.method)
    #if (request.method == 'POST'):
        #data = request.get_json()
        #return jsonify(model_load.predict([np.array(list(data.values()))]).tolist())
    #else:
        #return render_template('index.html')

        
if __name__ == '__main__':
    app.run(debug=False)
