import pandas as pd
import numpy as np


from sklearn.ensemble import RandomForestRegressor

import pickle

from churnexplainer import CategoricalEncoder
import cdsw, numpy
filename="./models/champion/champion.pkl"
#Load the model save earlier.
loaded_model = pickle.load(open(filename, 'rb'))
filename="./models/champion/ce.pkl"
#Load the model save earlier.
ce = pickle.load(open(filename, 'rb'))



def predict(args):
  df=pd.DataFrame(args["data"]["rows"])
  df.columns=args["data"]["colnames"]
  #df.convert_dtypes=args["data"]["coltypes"]
  df.columns=args["data"]["colnames"]
  #df.convert_dtypes=args["data"]["coltypes"]
  df=df.sort_index(axis = 1)
  df.columns= sorted(["monthlycharges", "totalcharges","tenure","gender","dependents", "onlinesecurity", "multiplelines", "internetservice","seniorcitizen", "techsupport", "contract", "streamingmovies", "deviceprotection", "paymentmethod","streamingtv", "phoneservice", "paperlessbilling","partner", "onlinebackup"],key=str.lower)
  df['monthlycharges']=pd.to_numeric(df['monthlycharges'])
  df['tenure']=pd.to_numeric(df['tenure'])
  df['totalcharges']=pd.to_numeric(df['totalcharges'])
#  cols = (('gender', True),
#        ('SeniorCitizen', True),
#        ('Partner', True),
#        ('Dependents', True),
#        ('tenure', False),
#        ('PhoneService', True),
#        ('MultipleLines', True),
#        ('InternetService', True),
#        ('OnlineSecurity', True),
#        ('OnlineBackup', True),
#        ('DeviceProtection', True),
#        ('TechSupport', True),
#        ('StreamingTV', True),
#        ('StreamingMovies', True),
#        ('Contract', True),
#        ('PaperlessBilling', True),
#        ('PaymentMethod', True),
#        ('MonthlyCharges', False),
#        ('TotalCharges', False))
  cols = (('gender', True),
          ('seniorcitizen', True),
          ('partner', True),
          ('dependents', True),
          ('tenure', False),
          ('phoneservice', True),
          ('multiplelines', True),
          ('internetservice', True),
          ('onlinesecurity', True),
          ('onlinebackup', True),
          ('deviceprotection', True),
          ('techsupport', True),
          ('streamingtv', True),
          ('streamingmovies', True),
          ('contract', True),
          ('paperlessbilling', True),
          ('paymentmethod', True),
          ('monthlycharges', False),
          ('totalcharges', False))
  data = df[[c for c, _ in cols]]
  catcols = (c for c, iscat in cols if iscat)
  for col in catcols:
    data[col] = pd.Categorical(data[col])
  
  
  X = ce.fit_transform(data)
 
  
  
  df['final']=loaded_model.predict(X)
  #df['final']=df['MonthlyCharges']*df['tenure']*df['TotalCharges']
  df['final']=df['final'].fillna(0)
  outRows = []
  for row in range(len(df['final'])):
    outRows.append([int(10*df['final'][row])])
   
  
 

  return {'data': {'colnames': ['result'],
  'coltypes': ['INT'],
  'rows': outRows}}

#{
#    "data": {
#      "colnames": ["monthlycharges", "totalcharges","tenure","gender","dependents", "onlinesecurity", "multiplelines", "internetservice","seniorcitizen", "techsupport", "contract","streamingmovies", "deviceprotection", "paymentmethod","streamingtv","phoneservice", "paperlessbilling","partner", "onlinebackup"],
#      "coltypes": ["FLOAT", "FLOAT","INT","STRING","STRING","STRING","STRING","STRING","STRING","STRING","STRING","STRING","STRING","STRING","STRING","STRING","STRING","STRING","STRING"],
#      "rows": [
#        ["70.35", "70.35","29","Male","No","No", "No", "DSL", "No", "No", "Month-to-month", "No", "No", "Bank transfer (automatic)","No",  "No", "No", "No", "No"],
#        ["70.35", "70.35","29","Female","No","No", "No", "DSL", "No", "No", "Month-to-month", "No", "No", "Bank transfer (automatic)","No", "No", "No", "No", "No"]
#      ]
#    }
#  }

#cviz_rest('{"url":"https://modelservice.ml-9bf5fe7d-8fc.se-sandb.a465-9q4k.cloudera.site/model","accessKey":"mii9nai6k8cxs3szekaf2t72b5ppk137","colnames":["monthlycharges","totalcharges","tenure","gender","dependents","onlinesecurity","multiplelines","internetservice","seniorcitizen","techsupport", "contract","streamingmovies", "deviceprotection", "paymentmethod","streamingtv","phoneservice", "paperlessbilling","partner", "onlinebackup"],"response_colname":"result"}')