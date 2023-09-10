import sys
import os
import numpy as np 
import pandas as pd

from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object




@dataclass
class DataTransformationConfig:
    preprocessor_pkl_path = os.path.join("artifacts","preprocessor.pkl")
    preprocessor_x_train_path = os.path.join("artifacts",'X_train_df.csv')
    preprocessor_x_test_path = os.path.join("artifacts",'X_test_df.csv')


class Error(Exception):
    pass



class Data_Transformation:

    def __init__(self):
        self.DataTransformationConfig_path = DataTransformationConfig()


    def Order_day_data_transformation_convertion(self,df):
        try:
            logging.info("in this method we preprocessed our column")

            # Order_day
            df['Order_day']=df['Order_Date'].str.split('-').str[0]
            df['Order_month']=df['Order_Date'].str.split('-').str[1]
            df['Order_year']=df['Order_Date'].str.split('-').str[2]
            df.drop('Order_Date',axis=1,inplace=True)
            df['Order_day']=df['Order_day'].astype(int)
            df['Order_month']=df['Order_month'].astype(int)
            df['Order_year']=df['Order_year'].astype(int)

            # dict to convert this 
            my_dict = {
                    "0.458333333" : "11:00",
                    "0.958333333" : "23:00",
                    "0.791666667" : "19:00",
                    "0.875" : "21:00",
                    "1" : "24:00",
                    "0.375" : "9:00",
                    "0.625" : "13:00",
                    "0.833333333" : "20:00",
                    "0.666666667" : "16:00",
                    "0.75" : "18:00",
                    "0.416666667" : "10:00",
                    "0.916666667" : "22:00",
                    "0.5" : "12:00",
                    "0.708333333": "17:00",
                    "0.541666667": "13:00",
                    "0.583333333": "14:00"
                    }
            df['Time_Orderd']=df['Time_Orderd'].replace(my_dict)
            #df.dropna(subset=['Time_Orderd'],inplace=True)
            # convert time into minute 
            df['Time_Orderd']=df['Time_Orderd'].str.split(":").str[0].astype(int)*60 + df['Time_Orderd'].str.split(":").str[1].astype(int)

            # Time_Order_picked
            df['Time_Order_picked']=df['Time_Order_picked'].replace(my_dict)
            df['Time_Order_picked']=df['Time_Order_picked'].str.split(":").str[0].astype(int) * 60 + df['Time_Order_picked'].str.split(":").str[1].astype(int)

            return df


        except Exception as e:
            logging.info("there are some error in secondary data transformation conversion")
            raise CustomException(e,sys)
        

    def Preprocessed_Data_Transformation(self):
        logging.info("Preprocessed has started")
        try:
            weather_map=['Sunny', 'Windy', 'Sandstorms', 'Cloudy', 'Stormy', 'Fog']
            road_traffic_map=['Low', 'Medium', 'High', 'Jam']
            type_order_map = ['Buffet', 'Drinks', 'Meal', 'Snack']
            type_vehicle_map = ['electric_scooter','scooter', 'motorcycle']
            festival_map = ['No', 'Yes']
            city_map = ['Semi-Urban', 'Urban','Metropolitian']


            # Distingues Numerical column and Categorical columns
            categorical_column = ['Weather_conditions', 'Road_traffic_density', 'Type_of_order',
                                'Type_of_vehicle', 'Festival', 'City']
            numerical_column = ['Delivery_person_Age', 'Delivery_person_Ratings', 'Restaurant_latitude',
                                'Restaurant_longitude', 'Delivery_location_latitude',
                                'Delivery_location_longitude', 'Time_Orderd', 'Time_Order_picked',
                                'Vehicle_condition', 'multiple_deliveries', 'Order_day', 'Order_month',
                                'Order_year']
            
            # numerical pipeline 

            numerical_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('Scaler',StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('OrdinalEncoder',OrdinalEncoder(categories=[weather_map,road_traffic_map,type_order_map,type_vehicle_map,festival_map,city_map])),
                    ('scaler',StandardScaler())
                ]
            )


            preprocessor = ColumnTransformer([
                ('numerical_pipeline',numerical_pipeline,numerical_column),
                ('categorical_pipeline',categorical_pipeline,categorical_column)
            ])

            return preprocessor

            
        except Exception as e:
            logging.info('there may be error in Preprocessed Data Transformation')
            raise CustomException(e,sys)


    def Intiate_Data_Transformation_Primary(self,train_path,test_path):
        logging.info("Intiate_Data Transformation Primary started")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            #raise Error "if file path is incorrect"

            logging.info("file read succefully")
            #logging.info(f"{train_df.head()}")
            #logging.info(f"{test_df.head()}")
            
            #here we drop the NaN value of the Time_Orderd beacuse this cant be changed after some times
            train_df.dropna(subset=['Time_Orderd'],inplace=True)
            test_df.dropna(subset=['Time_Orderd'],inplace=True)

            logging.info("Now preprocessing to be started")
            target_column_name = 'Time_taken (min)'
            drop_colum_name = [target_column_name,'ID','Delivery_person_ID']


            # Train data (X_train ,y_train)
            X_train_input_data=train_df.drop(labels=drop_colum_name,axis=1)
            y_train_input_data = train_df[target_column_name]

            # Test data (X_test,y_test)
            X_test_input_data = test_df.drop(labels=drop_colum_name,axis=1)
            y_test_input_data = test_df[target_column_name]

            X_train_input_data = self.Order_day_data_transformation_convertion(X_train_input_data)
            X_test_input_data = self.Order_day_data_transformation_convertion(X_test_input_data)

            logging.info(f"train data coversion \n{X_train_input_data}")
            logging.info(f"test data coversion \n{X_test_input_data}")

            #print(X_train_input_data['Time_Orderd'])

            # now call preprocessor to standardise , fixing missing value and Ordinal encoding of categorical column
            preprocessor_obj = self.Preprocessed_Data_Transformation()

            # Trnasformating using preprocessor obj
            X_train_preprocessed_obj=pd.DataFrame(preprocessor_obj.fit_transform(X_train_input_data),columns=preprocessor_obj.get_feature_names_out())
            X_test_preprocessed_obj=pd.DataFrame(preprocessor_obj.transform(X_test_input_data),columns=preprocessor_obj.get_feature_names_out())

            

            logging.info("preprocessed process sucessfully completed")
            logging.info(X_train_preprocessed_obj)
            #logging.info(X_test_preprocessed_obj)
            

            train_arr = np.c_[X_train_preprocessed_obj, np.array(y_train_input_data)]
            test_arr = np.c_[X_test_preprocessed_obj, np.array(y_test_input_data)]

            #os.makedirs(os.path.dirname(self.DataTransformationConfig_path.preprocessor_x_train_path),exist_ok=True)
            X_train_preprocessed_obj.to_csv(self.DataTransformationConfig_path.preprocessor_x_train_path,header=True,index=False)
            X_test_preprocessed_obj.to_csv(self.DataTransformationConfig_path.preprocessor_x_test_path,header=True,index=False)

            save_object(
                file_path= self.DataTransformationConfig_path.preprocessor_pkl_path,
                obj=preprocessor_obj
            )

            #this process is only for checking
            #logging.info(pd.DataFrame(train_arr))
            #logging.info(pd.DataFrame(test_arr))

            return(
                train_arr,
                test_arr,
                self.DataTransformationConfig_path.preprocessor_pkl_path
            )


         
        except Exception as e:
            logging.info("there may be some error in data transformation primary")
            raise CustomException(e,sys)