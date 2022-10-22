from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

model=pickle.load(open('deep.pkl','rb'))
@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features= request.form.values()
    vect=CountVectorizer()
    vector=vect.fit_transform(int_features)

    
    
    # print(int_features)
    final=[np.array(vector)]
    # print(final)
    prediction=model.predict(final)[0]

    if prediction ==1:
        return int_features('index.html',pred='This website is safe.'.format('1'))
    else:
        return render_template('index.html',pred='This website is not safe.'.format('0'))


if __name__ == '__main__':
    app.run(debug=True)