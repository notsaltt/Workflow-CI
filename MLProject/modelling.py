import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import os

# ============================================================
# INIT DAGSHUB
# ============================================================
dagshub.init(
    repo_owner='notsaltt',
    repo_name='Eksperimen1',
    mlflow=True
)

# Load data
train = pd.read_csv('iris_preprocessing/train.csv')
test  = pd.read_csv('iris_preprocessing/test.csv')

X_train = train.drop('target', axis=1)
y_train = train['target']
X_test  = test.drop('target', axis=1)
y_test  = test['target']

# Training
params = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 2,
    'random_state': 42
}

with mlflow.start_run(run_name="CI_RandomForest"):

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall    = recall_score(y_test, y_pred, average='weighted')
    f1        = f1_score(y_test, y_pred, average='weighted')

    mlflow.log_params(params)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(model, "random_forest_model")

    # Artefak 1: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()

    # Artefak 2: Feature Importance
    plt.figure(figsize=(8, 5))
    plt.barh(X_train.columns, model.feature_importances_, color='steelblue')
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")
    plt.close()

    # Artefak 3: ROC Curve
    classes = sorted(y_test.unique())
    y_test_bin = label_binarize(y_test, classes=classes)
    y_score = model.predict_proba(X_test)
    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        plt.plot(fpr, tpr, label=f'Class {cls} (AUC={auc(fpr,tpr):.2f})')
    plt.plot([0,1],[0,1],'k--')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curve.png")
    mlflow.log_artifact("roc_curve.png")
    plt.close()

    # Artefak 4: Classification Report
    report = classification_report(y_test, y_pred)
    with open("classification_report.txt", "w") as f:
        f.write(report)
    mlflow.log_artifact("classification_report.txt")

    print(f"Accuracy: {accuracy:.4f}")
    print("Run selesai!")