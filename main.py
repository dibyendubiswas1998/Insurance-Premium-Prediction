from src.training import training
from src.evaluation import evaluation
from src.prediction import prediction
from flask import Flask, render_template, redirect, request


app = Flask(__name__)

@app.route("/")
def Home():
    return render_template('index.html')

@app.route("/train_evaluation")
def Train():
    training("params.yaml")
    evaluation("params.yaml")
    return render_template("index.html", train_result="training completed successfully")

@app.route("/prediction", methods=['POST', 'GET'])
def Prediction():
    global predicted_value
    if request.method == 'POST':
        age = request.form['age']
        bmi = request.form['bmi']
        children = request.form['children']
        sex = request.form['sex']
        smoker = request.form['smoker']
        region = request.form['region']
        # print(age, bmi, children, sex, smoker, region)
        new_data = {
            'age': int(age), 'bmi': int(bmi), 'children': int(children), 'sex': sex.lower(),
            'smoker': smoker.lower(), 'region': region.lower()
        }
        predicted_value =  prediction(config_path="params.yaml", data=new_data)
        print(predicted_value)
    return render_template("index.html", predicted_value=predicted_value)





if __name__ == "__main__":
    # training("params.yaml")
    # evaluation("params.yaml")
    app.run(debug=True)

