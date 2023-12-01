# Insurance Premium Prediction:

## Problem Statement:
The Goal of this Project is to give people an estimate of how much they need based on their individual health stituation. 
After that, customers can work with any health insurance carrier and its plans and perks while keeping the projected cost 
from our study in mind. This can assist a person in concentrating on the health side of an insurance policy rather than 
ineffective part.<br><br>


## Solution:
Developed and implemented a CI-CD pipeline for generating expense estimates on user information, resulting in a 20% reduction in estimation time and a 5% improvement in accuracy. Streamlined the expense estimation process, enabling customers to make informed decisions about their healthcare plans.
<br>

[DagsHub](https://dagshub.com/dibyendubiswas1998/Insurance-Premium-Prediction.mlflow)
<br>

[WebPage](http://18.210.10.175:8080/)
<br><br>



## Project Workflow:
* **Step-01:** Load the raw or custom data from **AWS S3 Bucket**, provided by user. And save the data into particular directory.

* **Step-02:** Preprocessed the raw data, like handle the missing values, duplicate values, handle-outliers, separate into train & test datasets.

* **Step-03:** Create or find better the model by comparing multiple models, and train the model and save the best model in a particular directory and also logs the model into [DagsHub](https://dagshub.com/dibyendubiswas1998/Insurance-Premium-Prediction.mlflow) by using mlflow.

* **Step-04:** Evaluate the model baed on test datasets and save the inflormation on [DagsHub](https://dagshub.com/dibyendubiswas1998/Insurance-Premium-Prediction.mlflow) by using mlflow. 

* **Step-05:** Create a Web Application for generating expencess based on user's input and host the entier application on AWS.
<br><br>



## Tech Stack:
![Tech Stack](./documents/tech%20stack.png)
<br><br>



## How to Run the Application:
```bash
    # For Windows OS:
    docker pull dibyendubiswas1998/insurance_premium_prediction
    docker run -p 8080:8080 dibyendubiswas1998/insurance_premium_prediction

    # For Ubuntu OS:
    sudo docker pull dibyendubiswas1998/insurance_premium_prediction
    sudo docker run -p 8080:8080 dibyendubiswas1998/insurance_premium_prediction

```
<br><br>



## Web Interface:
[WebPage](http://18.210.10.175:8080/)
![Web Interface](./documents/web%20interface.png)