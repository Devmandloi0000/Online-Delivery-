import pandas as pd
import numpy as np
import os

from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import Data_Ingestion
from src.components.data_transformation import Data_Transformation
from src.components.model_training import ModelTrainer



if __name__ == "__main__":
    obj = Data_Ingestion()
    X_train_path,X_test_path=obj.Intiate_Data_Ingestion_Primary()
    file_path_obj = Data_Transformation()
    X_train_preprocessed,X_test_preprocessed,_=file_path_obj.Intiate_Data_Transformation_Primary(X_train_path,X_test_path)
    model_obj = ModelTrainer() 
    model_obj.Initiate_Model_Trainer(X_train_preprocessed,X_test_preprocessed)


