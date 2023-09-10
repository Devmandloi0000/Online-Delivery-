import os
import sys
import pickle
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class Predict_Pipeline:
    def __init__(self):
        pass

    def New_data_Predict(self,features):
        logging.info("process of new data started")
        try:
            preprocessor_path = os.path.join("artifacts","preprocessor.pkl")
            model_path = os.path.join("artifacts","model.pkl")
            preprocessor= load_object(preprocessor_path)
            model = load_object(model_path)

            # now new data will be tranform x_test data
            data_scaled=preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred

        except Exception as e:
            logging.info("there may be error in the new data predict")
            raise CustomException(e,sys)
        

    
class CustomeData:

    def __init__(self,
                 Delivery_person_Age:float,
                 Delivery_person_Ratings:float,
                 Restaurant_latitude:float,
                 Restaurant_longitude:float,
                 Delivery_location_latitude:float,
                 Delivery_location_longitude:float,
                 Time_Orderd:int,
                 Time_Order_picked:int,
                 Weather_conditions:str	,
                 Road_traffic_density:str,
                 Vehicle_condition:int,
                 Type_of_order:str,
                 Type_of_vehicle:str,
                 multiple_deliveries:float,
                 Festival:str,
                 City:str,
                 Order_day:str,
                 Order_month:str,
                 Order_year:str                 
                 ):
        self.Delivery_person_Age=Delivery_person_Age
        self.Delivery_person_Ratings=Delivery_person_Ratings
        self.Restaurant_latitude=Restaurant_latitude
        self.Restaurant_longitude=Restaurant_longitude
        self.Delivery_location_latitude=Delivery_location_latitude
        self.Delivery_location_longitude = Delivery_location_longitude
        self.Time_Orderd=Time_Orderd
        self.Time_Order_picked=Time_Order_picked
        self.Weather_conditions=Weather_conditions
        self.Road_traffic_density=Road_traffic_density
        self.Vehicle_condition=Vehicle_condition
        self.Type_of_order=Type_of_order
        self.Type_of_vehicle=Type_of_vehicle
        self.multiple_deliveries=multiple_deliveries
        self.Festival=Festival
        self.City=City
        self.Order_day=Order_day
        self.Order_month=Order_month
        self.Order_year=Order_year             
        

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict={
                                "Delivery_person_Age":[self.Delivery_person_Age],
                                "Delivery_person_Ratings":[self.Delivery_person_Ratings],
                                "Restaurant_latitude":[self.Restaurant_latitude],
                                "Restaurant_longitude":[self.Restaurant_longitude],
                                "Delivery_location_latitude":[self.Delivery_location_latitude],
                                "Delivery_location_longitude":[self.Delivery_location_longitude],
                                "Time_Orderd":[self.Time_Orderd],
                                "Time_Order_picked":[self.Time_Order_picked],
                                "Weather_conditions":[self.Weather_conditions],
                                "Road_traffic_density":[self.Road_traffic_density],
                                "Vehicle_condition":[self.Vehicle_condition],
                                "Type_of_order":[self.Type_of_order],
                                "Type_of_vehicle":[self.Type_of_vehicle],
                                "multiple_deliveries":[self.multiple_deliveries],
                                "Festival":[self.Festival],
                                "City":[self.City],
                                "Order_day":[self.Order_day],
                                "Order_month":[self.Order_month],
                                "Order_year":[self.Order_year]}
            
            df = pd.DataFrame(custom_data_input_dict)
            logging.info("data frame of new data is complete")
            return df
        
        except Exception as e:
            logging.info("there may be some error in the get data as dataframe")
            raise CustomException(e,sys)

