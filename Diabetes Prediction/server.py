from requests import request
from tensorflow import keras

from flask import Flask,render_template,request

app = Flask(__name__,template_folder='src')

@app.route("/",methods=['GET','POST'])
def hello_world():
    print()
    if request.method== "POST":
        pregnancy = int(request.form["pregnancy"])
        bloodPressure = int(request.form["blood-pressure"])
        skinFoldThickness = int(request.form["skin-fold-thickness"])
        serumInsulin = int(request.form["serum-insulin"])
        bmi = int(request.form["bmi"])
        DPF = int(request.form["DPF"])
        age = int(request.form["age"])
        model = keras.models.load_model('dia_model.h5')
        ans = model.predict([[pregnancy,bloodPressure,skinFoldThickness,serumInsulin,0,bmi,age,DPF]])
        print(ans)
        return render_template('index.html',ans=ans)
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run()
