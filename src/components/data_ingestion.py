import pandas as pd
import numpy as np
import os
import sys

from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import train_test_split

from dataclasses import dataclass


@dataclass # Dataclass is decorator which is used directly without use intialiser
class DataIngestionConfig:
    # Main Aim of this file to save this as for future use and prediction purpose 
    train_data_path = os.path.join("artifacts","train.csv")
    test_data_path = os.path.join("artifacts","test.csv")
    raw_data_path = os.path.join("artifacts","raw.csv")


class Data_Ingestion:

    def __init__(self):
        self.DataIngestionConfig_path = DataIngestionConfig()

    def Intiate_Data_Ingestion_Primary(self):
        logging.info("Data ingestion process has started")
        try:

            # Read a file from source location
            df = pd.read_csv(os.path.join('notebook/data','finalTrain.csv'))
            #print(df.head())

            # saving raw file 
            os.makedirs(os.path.dirname(self.DataIngestionConfig_path.raw_data_path),exist_ok=True)
            df.to_csv(self.DataIngestionConfig_path.raw_data_path,index=False)

            # split the data into train and test
            X_train,X_test = train_test_split(df,test_size=0.33,random_state=42)

            X_train.to_csv(self.DataIngestionConfig_path.train_data_path,index=False,header=True)
            X_test.to_csv(self.DataIngestionConfig_path.test_data_path,index=False,header=True)

            logging.info("file saved sucessfully into artifacts folder")

            return(
                self.DataIngestionConfig_path.train_data_path,
                self.DataIngestionConfig_path.test_data_path
            )


        except Exception as e:
            logging.info("there may be some error in the data ingestion primary")
            raise CustomException(e,sys)


