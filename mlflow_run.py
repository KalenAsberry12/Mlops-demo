import mlflow
import mlflow.sklearn
from train import train_model

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("demo")

with mlflow.start_run():
  model, accuracy = train_model()
  mlflow.log_metric("accuracy", accuracy)
  mlflow.log_param("model_type", "RandomForest")
  mlflow.sklearn.log_model(model,"model")
  print(f"Logged run to {mlflow.get_artifact_uri()}")
  
