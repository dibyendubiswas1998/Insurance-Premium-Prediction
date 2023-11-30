from src.insurance_premium_prediction.utils.common_utils import load_json_file
from src.insurance_premium_prediction.pipeline.prediction import PredictionPipeline
from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/")
def Home():
    """
        Renders the "index.html" template for the root URL ("/") in a Flask application.

        Returns:
            str: The rendered "index.html" template.
    """
    return render_template('index.html')


@app.route("/prediction_", methods=['POST', 'GET'])
def start_prediction():
    """
        Handles the "/prediction_" route for both GET and POST requests.
        
        This function takes user input from a form and uses it to make a prediction using a pre-trained model. 
        The predicted value is then rendered in the "index.html" template.
        
        Inputs:
        - age: the age of the individual (input from the form)
        - bmi: the body mass index of the individual (input from the form)
        - children: the number of children the individual has (input from the form)
        - sex: the sex of the individual (input from the form)
        - smoker: the smoking status of the individual (input from the form)
        
        Outputs:
        - expenses: the predicted insurance premium value, which is rendered in the "index.html" template.
    """
    try:
        prd = PredictionPipeline()
        expenses = None
        if request.method == "POST":
            age = request.form['age']
            bmi = request.form['bmi']
            children = request.form['children']
            sex = request.form['sex']
            smoker = request.form['smoker']

            sex_ = 0 if sex=="female" else 1
            smoker_ = 0 if smoker=="yes" else 1

            # age,bmi,children,sex,smoker
            data = [[age, bmi, children, sex_, smoker_]]
            expenses = prd.prediction(data=data) # predict the values
                        
        return render_template("index.html", result=f"${round(expenses[0], 2)}")

    except Exception as ex:
        raise ex
    
    finally:
        return render_template("index.html", result=f"${round(expenses[0], 2)}")






if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)