import argparse
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="car_evaluation_train_encoded.csv")
parser.add_argument("--experiment_name", type=str, default="car_evaluation_experiment")
args = parser.parse_args()

mlflow.set_experiment(args.experiment_name)

data = pd.read_csv(args.data_path)

X_train, X_test, y_train, y_test = train_test_split(
    data.drop("class", axis=1),
    data["class"],
    test_size=0.2,
    random_state=42
)

with mlflow.start_run():
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=22,
        random_state=42
    )

    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)

    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 22)
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        input_example=X_train.iloc[:5]
    )