import json
from flask import Flask,jsonify,request , render_template
from utility import predict_pipe


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

if __name__=='__main__':
    app.run(debug=True)


