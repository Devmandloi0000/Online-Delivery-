from flask import Flask,render_template,jsonify,request
from src.pipeline.prediction_pipeline import CustomeData,Predict_Pipeline

application=Flask(__name__)
app = application

@app.route("/")
def home_page():
    return render_template("index.html")

@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template("form.html")
    
    else:
        data = CustomeData(
            Delivery_person_Age=float(request.form.get('Delivery_person_Age')),
            Delivery_person_Ratings=float(request.form.get('Delivery_person_Ratings')),
            Restaurant_latitude=float(request.form.get('Restaurant_latitude')),
            Restaurant_longitude=float(request.form.get('Restaurant_longitude')),
            Delivery_location_latitude=float(request.form.get('Delivery_location_latitude')),
            Delivery_location_longitude = float(request.form.get('Delivery_location_longitude')),
            Time_Orderd=int(request.form.get('Time_Orderd')),
            Time_Order_picked=int(request.form.get('Time_Order_picked')),
            Weather_conditions=str(request.form.get('Weather_conditions')),
            Road_traffic_density=str(request.form.get('Road_traffic_density')),
            Vehicle_condition=int(request.form.get('Vehicle_condition')),
            Type_of_order=str(request.form.get('Type_of_order')),
            Type_of_vehicle=str(request.form.get('Type_of_vehicle')),
            multiple_deliveries=float(request.form.get('multiple_deliveries')),
            Festival=str(request.form.get('Festival')),
            City=str(request.form.get('City')),
            Order_day=int(request.form.get('Order_day')),
            Order_month=int(request.form.get('Order_month')),
            Order_year=int(request.form.get('Order_year'))  
            )
        
        final_new_data=data.get_data_as_dataframe()

        prediction_pipeline = Predict_Pipeline()
        pred=prediction_pipeline.New_data_Predict(final_new_data)
        results = round(pred[0],2)

        return render_template("results.html",result=results)




if __name__=="__main__":
    app.run(host="0.0.0.0")