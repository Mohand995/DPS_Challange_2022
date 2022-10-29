import json
from flask import Flask,jsonify,request , render_template
import pickle 
import pandas as pd
import numpy as np


def prepare_data(data,datapoint,Prediction=False):
        if Prediction:
            data=pd.DataFrame(datapoint,index=['a'])

        data.JAHR=data.JAHR.astype(int)
        data.MONAT=data.MONAT.astype(int)

        data.MONATSZAHL=data.MONATSZAHL.replace({'Alkoholunfälle':0,'Fluchtunfälle':1,'Verkehrsunfälle':2})
        data.AUSPRAEGUNG=data.AUSPRAEGUNG.replace({'insgesamt':0,
                                                   'Verletzte und Getötete':1,'mit Personenschäden':2})
        

        Features=data[['MONATSZAHL','AUSPRAEGUNG','MONAT','JAHR']]
            
        return Features

#############################################################################################




#########################################################
def predict_(model, datapoint):
    # Predict the sentiment
    preprocessed_input= prepare_data(null,datapoint,True)

    predictions = model.predict(preprocessed_input)[0]

    return {'prediction':np.floor(np.exp(predictions))}

#######################################################
def predict_pipe(datapoint):
    with open('./Model/reg_model_', 'rb') as f:
        loaded_pipe = pickle.load(f)
    return predict_(loaded_pipe,datapoint)

#####################################################

app=Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict",methods=['POST'])
def predict():
    d=[ x for x in request.form.values()]
    data={
	"MONATSZAHL": d[0],
	"AUSPRAEGUNG": d[1],
	"MONAT": d[2],
	"JAHR": d[3]
}
    print(data)
    preds=predict_pipe(data)
    try:
        result=preds
    except TypeError as e:
        result=jsonify({'error',str(e)})

    return  render_template("home.html",prediction_text="the number of accidents is {}".format(result['prediction']))


@app.route("/predict_2",methods=['POST'])
def predict_2():
    data=request.json
    
    preds=predict_pipe(data)
    return preds 






if __name__=='__main__':
    app.run(debug=True)


