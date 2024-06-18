from flask import Flask,render_template,request
import pandas as pd
import pickle 
import numpy as np
app=Flask(__name__)

model=pickle.load(open('LinearRegression.pkl','rb'))
car=pd.read_csv("Cleaned_Car_data.csv")

@app.route('/')

def index():
    companies=sorted(car['company'].unique())
    car_models=sorted(car['name'].unique())
    year=sorted(car['year'].unique(),reverse=True)
    fuel_type=sorted(car['fuel_type'].unique())

    # for 1 position
    companies.insert(0,"Select Company")

    return render_template('index.html',companies=companies ,car_models=car_models ,years=year, fuel_types=fuel_type)

@app.route('/predict',methods=['POST'])

def predict():
    try:
        company=request.form.get('company')
        car_model=request.form.get('car_models')
        year=int(request.form.get('year'))
        fuel_type=request.form.get('fuel_type')
        kms_driven=int(request.form.get('kilo_driven'))

        #print(company,car_model,year,fuel_type,driven)

        prediction=model.predict(pd.DataFrame([[car_model,company,year,kms_driven,fuel_type]],columns=['name','company','year','kms_driven','fuel_type']))
        print(prediction)
        return str(np.round(prediction[0],2))
        # return prediction
    except Exception as e:
        return f"Error during prediction: {e}"



if __name__=="__main__":
    app.run(host='0.0.0.0', port=8080)