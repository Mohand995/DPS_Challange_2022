import pickle 
import pandas as pd
import numpy as np
from sqlalchemy import null



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

############################################################################

with open('./Model/reg_model_', 'rb') as f:
    loaded_pipe = pickle.load(f)

#######################################################################

datapoint={
    
'MONATSZAHL': 'Alkoholunfälle',
'AUSPRAEGUNG': 'insgesamt',
'MONAT': '01',
'JAHR': '2021'
}

##########################################################
def predict(model, datapoint):
    # Predict the sentiment
    preprocessed_input= prepare_data(null,datapoint,True)

    predictions = model.predict(preprocessed_input)[0]

    return {'prediction':np.floor(np.exp(predictions))}

#######################################################
def predict_pipe(datapoint):
    return predict(loaded_pipe,datapoint)

#####################################################
if __name__=="__main__":
    
    predictions = predict_pipe( datapoint)
    print(predictions)
