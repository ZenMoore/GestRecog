import preprocessing
import os

DATA_FILE = "./data_gotten/data.xls"

while(not os.path.exists(DATA_FILE)):
    pass

with open(DATA_FILE, mode= 'r'):
    preprocessing.start.run(DATA_FILE)