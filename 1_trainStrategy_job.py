
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import average_precision_score
from sklearn.metrics import mean_squared_error

import mlflow
import mlflow.sklearn
import pickle
#from your_data_loader import load_data
from churnexplainer import CategoricalEncoder
import datetime

import time
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import xml.etree.ElementTree as ET
from cmlbootstrap import CMLBootstrap
# Set the setup variables needed by CMLBootstrap
HOST = os.getenv("CDSW_API_URL").split(
    ":")[0] + "://" + os.getenv("CDSW_DOMAIN")
USERNAME = os.getenv("CDSW_PROJECT_URL").split(
    "/")[6]  # args.username  # "vdibia"
API_KEY = os.getenv("CDSW_API_KEY") 
PROJECT_NAME = os.getenv("CDSW_PROJECT")  

# Instantiate API Wrapper
cml = CMLBootstrap(HOST, USERNAME, API_KEY, PROJECT_NAME)

uservariables=cml.get_user()
if uservariables['username'][-3] == '0':
  DATABASE = "user"+uservariables['username'][-3:]
else:
  #DATABASE = uservariables['username']
  DATABASE = 'user017'
  #DATABASE = 'acampos'

runtimes=cml.get_runtimes()
runtimes=runtimes['runtimes']
runtimesdf = pd.DataFrame.from_dict(runtimes, orient='columns')
runtimeid=runtimesdf.loc[(runtimesdf['editor'] == 'Workbench') & (runtimesdf['kernel'] == 'Python 3.7') & (runtimesdf['edition'] == 'Standard')]['id']
id_rt=runtimeid.values[0]

spark = SparkSession\
    .builder\
    .appName("PythonSQL")\
    .master("local[*]")\
    .config("spark.sql.extensions","org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")\
    .config("spark.sql.catalog.spark_catalog","org.apache.iceberg.spark.SparkSessionCatalog") \
    .config("spark.sql.catalog.local","org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.local.type","hadoop") \
    .config("spark.sql.catalog.spark_catalog.type","hive") \
    .getOrCreate()
    

# **Note:**
# Our file isn't big, so running it in Spark local mode is fine but you can add the following config
# if you want to run Spark on the kubernetes cluster
#
# > .config("spark.yarn.access.hadoopFileSystems",os.getenv['STORAGE'])\
#
# and remove `.master("local[*]")\`
#

# Since we know the data already, we can add schema upfront. This is good practice as Spark will
# read *all* the Data if you try infer the schema.
schema = StructType(
    [
        StructField("customerid", StringType(), True),
        StructField("gender", StringType(), True),
        StructField("seniorcitizen", StringType(), True),
        StructField("partner", StringType(), True),
        StructField("dependents", StringType(), True),
        StructField("tenure", DoubleType(), True),
        StructField("phoneservice", StringType(), True),
        StructField("multiplelines", StringType(), True),
        StructField("internetservice", StringType(), True),
        StructField("onlinesecurity", StringType(), True),
        StructField("onlinebackup", StringType(), True),
        StructField("deviceprotection", StringType(), True),
        StructField("techsupport", StringType(), True),
        StructField("streamingtv", StringType(), True),
        StructField("streamingmovies", StringType(), True),
        StructField("contract", StringType(), True),
        StructField("paperlessbilling", StringType(), True),
        StructField("paymentmethod", StringType(), True),
        StructField("monthlycharges", DoubleType(), True),
        StructField("totalcharges", DoubleType(), True),
        StructField("churn", StringType(), True)
    ]
)


# Now we can read in the data from Cloud Storage into Spark...
try : 
  storage=os.environ["STORAGE"]
except:
  if os.path.exists("/etc/hadoop/conf/hive-site.xml"):
    tree = ET.parse('/etc/hadoop/conf/hive-site.xml')
    root = tree.getroot()
    for prop in root.findall('property'):
      if prop.find('name').text == "hive.metastore.warehouse.dir":
        storage = prop.find('value').text.split("/")[0] + "//" + prop.find('value').text.split("/")[2]
  else:
    storage = "/user/" + os.getenv("HADOOP_USER_NAME")
  storage_environment_params = {"STORAGE":storage}
  storage_environment = cml.create_environment_variable(storage_environment_params)
  os.environ["STORAGE"] = storage


storage = os.environ['STORAGE']
hadoop_user = os.environ['HADOOP_USER_NAME']

# To get more detailed information about the hive table you can run this:
df = spark.sql("SELECT * FROM "+ DATABASE + ".telco_data_curated").toPandas()

idcol = 'customerid'
labelcol = 'churn'
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


df = df.replace(r'^\s$', np.nan, regex=True).dropna().reset_index()
df.index.name = 'id'
data, labels = df.drop(labelcol, axis=1), df[labelcol]
data = data.replace({'seniorcitizen': {1: 'Yes', 0: 'No'}})

# This is Mike's lovely short hand syntax for looping through data and doing useful things. I think if we started to pay him by the ASCII char, we'd get more readable code.
data = data[[c for c, _ in cols]]
catcols = (c for c, iscat in cols if iscat)
for col in catcols:
    data[col] = pd.Categorical(data[col])
labels = (labels == 'Yes')

ce = CategoricalEncoder()
X = ce.fit_transform(data)

y=labels.values

run_time_suffix = datetime.datetime.now()
run_time_suffix = run_time_suffix.strftime("%M%S")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
if len(sys.argv) == 2:
    try:
        a=mlflow.get_run(sys.argv[1]).data.params
        n_estimators = int(a["n_estimators"])
        rf = RandomForestRegressor(n_estimators=n_estimators)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
 
        filename = './models/champion/ce.pkl'
        pickle.dump(ce, open(filename, 'wb'))

        filename = './models/champion/champion.pkl'
        pickle.dump(rf, open(filename, 'wb'))

       
        project_id = cml.get_project()['id']
        params = {"projectId":project_id,"latestModelDeployment":True,"latestModelBuild":True}


        default_engine_details = cml.get_default_engine({})
        default_engine_image_id = default_engine_details["id"]

        example_model_input = {"streamingtv": "No", "monthlycharges": 70.35, "phoneservice": "No", "paperlessbilling": "No", "partner": "No", "onlinebackup": "No", "gender": "female", "contract": "Month-to-month", "totalcharges": 1397.475,
                       "streamingmovies": "No", "deviceprotection": "No", "paymentmethod": "Bank transfer (automatic)", "tenure": 29, "dependents": "No", "onlinesecurity": "No", "multiplelines": "No", "internetservice": "DSL", "seniorcitizen": "No", "techsupport": "No"}


        try:
                    
                      # Create the YAML file for the model lineage
            yaml_text = \
                """"ModelOpsChurn_default":
              hive_table_qualified_names:                # this is a predefined key to link to training data
                - "default.telco_data_curated@cm"               # the qualifiedName of the hive_table object representing                
              metadata:                                  # this is a predefined key for additional metadata
                query: "select * from historical_data"   # suggested use case: query used to extract training data
                training_file: "3_trainStrategy_job.py"       # suggested use case: training file used
            """

            with open('lineage.yml', 'w') as lineage:
                lineage.write(yaml_text)
            #read input file
            fin = open("lineage.yml", "rt")
            #read file contents to string
            data = fin.read()
            #replace all occurrences of the required string
            data = data.replace('default',DATABASE)
            #close the input file
            fin.close()
            #open the input file in write mode
            fin = open("lineage.yml", "wt")
            #overrite the input file with the resulting data
            fin.write(data)
            #close the file
            fin.close()

            model_id = cml.get_models(params)[0]['id']
            latest_model = cml.get_model({"id": model_id, "latestModelDeployment": True, "latestModelBuild": True})

            build_model_params = {
              "modelId": latest_model['latestModelBuild']['modelId'],
              "projectId": latest_model['latestModelBuild']['projectId'],
              "targetFilePath": "_best_model_serve.py",
              "targetFunctionName": "explain",
              "engineImageId": default_engine_image_id,
              "kernel": "python3",
              "examples": latest_model['latestModelBuild']['examples'],
              "cpuMillicores": 1000,
              "memoryMb": 2048,
              "nvidiaGPUs": 0,
              "replicationPolicy": {"type": "fixed", "numReplicas": 1},
              "environment": {},"runtimeId":int(id_rt)}

            cml.rebuild_model(build_model_params)
            sys.argv=[]
            print('rebuilding...')
            
            model_id = cml.get_models(params)[1]['id']
            latest_model = cml.get_model({"id": model_id, "latestModelDeployment": True, "latestModelBuild": True})

            build_model_params = {
              "modelId": latest_model['latestModelBuild']['modelId'],
              "projectId": latest_model['latestModelBuild']['projectId'],
              "targetFilePath": "_model_viz.py",
              "targetFunctionName": "predict",
              "engineImageId": default_engine_image_id,
              "kernel": "python3",
              "examples": latest_model['latestModelBuild']['examples'],
              "cpuMillicores": 1000,
              "memoryMb": 2048,
              "nvidiaGPUs": 0,
              "replicationPolicy": {"type": "fixed", "numReplicas": 1},
              "environment": {},"runtimeId":int(id_rt)}

            cml.rebuild_model(build_model_params)
            
            print('rebuilding...')
            # Wait for the model to deploy.

          
        except:
          
                      # Create the YAML file for the model lineage
            yaml_text = \
                """"ModelOpsChurn_default":
              hive_table_qualified_names:                # this is a predefined key to link to training data
                - "default.telco_data_curated@cm"               # the qualifiedName of the hive_table object representing                
              metadata:                                  # this is a predefined key for additional metadata
                query: "select * from historical_data"   # suggested use case: query used to extract training data
                training_file: "3_trainStrategy_job.py"       # suggested use case: training file used
            """

            with open('lineage.yml', 'w') as lineage:
                lineage.write(yaml_text)

            #read input file
            fin = open("lineage.yml", "rt")
            #read file contents to string
            data = fin.read()
            #replace all occurrences of the required string
            data = data.replace('default',DATABASE)
            #close the input file
            fin.close()
            #open the input file in write mode
            fin = open("lineage.yml", "wt")
            #overrite the input file with the resulting data
            fin.write(data)
            #close the file
            fin.close()                

            example_input_viz= {"data": {"colnames": ["monthlycharges", "totalcharges","tenure","gender","dependents", "onlinesecurity", "multiplelines", "internetservice","seniorcitizen", "techsupport", "contract","streamingmovies", "deviceprotection", "paymentmethod","streamingtv","phoneservice", "paperlessbilling","partner", "onlinebackup"],"coltypes": ["FLOAT", "FLOAT","INT","STRING","STRING","STRING","STRING","STRING","STRING","STRING","STRING","STRING","STRING","STRING","STRING","STRING","STRING","STRING","STRING"],"rows": [["70.35", "70.35","29","Male","No","No", "No", "DSL", "No", "No", "Month-to-month", "No", "No", "Bank transfer (automatic)","No",  "No", "No", "No", "No"],["70.35", "70.35","29","Female","No","No", "No", "DSL", "No", "No", "Month-to-month", "No", "No", "Bank transfer (automatic)","No", "No", "No", "No", "No"]]}}
            
            create_model_params = {
                "projectId": project_id,
                "name": "ModelViz_"+DATABASE,
                "description": "visualization a given model prediction",
                "visibility": "private",
                "enableAuth": False,
                "targetFilePath": "_model_viz.py",
                "targetFunctionName": "predict",
                "engineImageId": default_engine_image_id,
                "kernel": "python3",
                "examples": [
                    {
                        "request": example_input_viz,
                        "response": {}
                    }],
                "cpuMillicores": 1000,
                "memoryMb": 2048,
                "nvidiaGPUs": 0,
                "replicationPolicy": {"type": "fixed", "numReplicas": 1},
                "environment": {},"runtimeId":int(id_rt)}
            print("Creating new model for visualization")
            new_model_details = cml.create_model(create_model_params)
            access_key = new_model_details["accessKey"]  # todo check for bad response
            model_id = new_model_details["id"]

            print("Workspace URL: % s/model" % HOST.strip().replace("https://", "https://modelservice."))    
            print("ModelViz Access Key:", access_key)

            # Disable model_authentication
            cml.set_model_auth({"id": model_id, "enableAuth": False})
            sys.argv=[]

            # Wait for the model to deploy.
            is_deployed = False
            while is_deployed == False:
                model = cml.get_model({"id": str(
                    new_model_details["id"]), "latestModelDeployment": True, "latestModelBuild": True})
                if model["latestModelDeployment"]["status"] == 'deployed':
                    print("ModelViz is deployed")
                    break
                else:
                    print("Deploying ModelViz.....")
                    time.sleep(10)


            create_model_params = {
                "projectId": project_id,
                "name": "ModelOpsChurn_"+DATABASE,
                "description": "Explain a given model prediction",
                "visibility": "private",
                "enableAuth": False,
                "targetFilePath": "_best_model_serve.py",
                "targetFunctionName": "explain",
                "engineImageId": default_engine_image_id,
                "kernel": "python3",
                "examples": [
                    {
                        "request": example_model_input,
                        "response": {}
                    }],
                "cpuMillicores": 1000,
                "memoryMb": 2048,
                "nvidiaGPUs": 0,
                "replicationPolicy": {"type": "fixed", "numReplicas": 1},
                "environment": {},"runtimeId":int(id_rt)}
            
            print("Creating new model")
            
            new_model_details = cml.create_model(create_model_params)
            access_key = new_model_details["accessKey"]  # todo check for bad response
            model_id = new_model_details["id"]

            print("ModelOps Access Key:", access_key)

            # Disable model_authentication
            cml.set_model_auth({"id": model_id, "enableAuth": False})
            sys.argv=[]

            # Wait for the model to deploy.
            is_deployed = False
            while is_deployed == False:
                model = cml.get_model({"id": str(
                    new_model_details["id"]), "latestModelDeployment": True, "latestModelBuild": True})
                if model["latestModelDeployment"]["status"] == 'deployed':
                    print("Model is deployed")
                    break
                else:
                    print("Deploying ModelOps.....")
                    time.sleep(10)

    except:
        sys.exit("Invalid Arguments passed to Experiment")
        sys.argv=[]
else:
    try:
      experimentId=mlflow.get_experiment_by_name("expRetrain").experiment_id
      mlflow.delete_experiment(experimentId)

      time.sleep(20)
    except:
      print("First time execution")


    mlflow.set_experiment('expRetrain')
    valuesParam=[9,11,15]
    for i in range(len(valuesParam)):
      with mlflow.start_run(run_name="run_"+run_time_suffix+'_'+str(i)) as run: 

      #with mlflow.start_run() as run: 
          # tracking run parameters
          mlflow.log_param("compute", 'local')
          mlflow.log_param("dataset", 'telco-churn')
          mlflow.log_param("dataset_version", '2.0')
          mlflow.log_param("algo", 'random forest')

          # tracking any additional hyperparameters for reproducibility
          n_estimators = valuesParam[i]
          mlflow.log_param("n_estimators", n_estimators)

          # train the model
          rf = RandomForestRegressor(n_estimators=n_estimators)
          rf.fit(X_train, y_train)
          y_pred = rf.predict(X_test)

          # automatically save the model artifact to the S3 bucket for later deployment
          mlflow.sklearn.log_model(rf, "rf-baseline-model")

          # log model performance using any metric
          precision=average_precision_score(y_test, y_pred)
          #mse = mean_squared_error(y_test, y_pred)
          mlflow.log_metric("precision", precision)

          mlflow.end_run()

          
