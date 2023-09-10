import logging
from datetime import datetime
import os 


LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

#print(LOG_FILE)

# path of the file 
logs_path = os.path.join(os.getcwd(),"Logs",LOG_FILE)
os.makedirs(logs_path,exist_ok=True)

#print(os.path.join(os.getcwd(),"logs",LOG_FILE))

LOG_FILE_PATH = os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
    filename= LOG_FILE_PATH,
    level=logging.INFO,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"
)