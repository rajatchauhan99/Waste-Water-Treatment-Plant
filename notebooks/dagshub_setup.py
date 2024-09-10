import mlflow
import dagshub

mlflow.set_tracking_uri("https://dagshub.com/rajatchauhan99/Waste-Water-Treatment-Plant.mlflow")

dagshub.init(repo_owner='rajatchauhan99', repo_name='Waste-Water-Treatment-Plant', mlflow=True)


with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)