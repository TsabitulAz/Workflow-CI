import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
    ConfusionMatrixDisplay,
    confusion_matrix
)
import matplotlib.pyplot as plt
import os

# Jangan set experiment & jangan start_run
# MLflow Project yang mengatur

data = pd.read_csv("car_evaluation_train_encoded.csv")

X_train, X_test, y_train, y_test = train_test_split(
    data.drop("class", axis=1),
    data["class"],
    random_state=42,
    test_size=0.2,
    stratify=data["class"]
)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Metrics
mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
mlflow.log_metric("f1_score", f1_score(y_test, y_pred, average="weighted"))
mlflow.log_metric("log_loss", log_loss(y_test, y_proba))
mlflow.log_metric(
    "roc_auc",
    roc_auc_score(y_test, y_proba, multi_class="ovr")
)

# Log model
mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    input_example=X_train.iloc[:5]
)

# Confusion matrix artifact
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")

cm_path = "artifacts/confusion_matrix.png"
plt.savefig(cm_path)
plt.close()

mlflow.log_artifact(cm_path)
