
import sys
import mlflow
import mlflow.sklearn
#mlflow.set_experiment('Retrain_exp')
experimentId=mlflow.get_experiment_by_name("expRetrain").experiment_id
dfExperiments=mlflow.search_runs(experiment_ids=experimentId)
maxmetric=dfExperiments["metrics.precision"].max()
runId=dfExperiments[dfExperiments["metrics.precision"]==maxmetric].head(1).run_id

script_descriptor = open("2_trainStrategy_job.py")
a_script = script_descriptor.read()
sys.argv = ["2_trainStrategy_job.py", runId.item()]

exec(a_script)