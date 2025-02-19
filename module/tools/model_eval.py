import numpy as np
import warnings
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier, AdaBoostRegressor, AdaBoostClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import accuracy_score

def _regression_evaluation(x, y):
    warnings.filterwarnings("ignore")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Lasso Regression": Lasso(),
        "Ridge Regression": Ridge(),
        "ElasticNet Regression": ElasticNet(),
        "Decision Tree Regressor": DecisionTreeRegressor(),
        "Random Forest Regressor": RandomForestRegressor(),
        "Gradient Boosting Regressor": GradientBoostingRegressor(),
        "AdaBoost Regressor": AdaBoostRegressor(),
        "Support Vector Regressor": SVR(),
        "XGBoost Regressor" : XGBRegressor(),
    }

    best_model = None
    best_score = float("inf")
    best_name = ""
    best_metrics = {}
    start_time = time.time()

    for name, model in models.items():
        model.fit(x_train_scaled, y_train)
        y_pred = model.predict(x_test_scaled)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R²": r2
        }

        if rmse < best_score and mse != 0.0 and rmse != 0.0 and mae != 0.0 and r2 != 1.00:
            best_score = rmse
            best_model = model
            best_name = name
            best_metrics = metrics
    for metric, value in best_metrics.items():
        print(f"{metric: <15}: {value:.8f}")



def _binaryclass_evaluation(x, y):
    warnings.filterwarnings("ignore")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    classifiers = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'AdaBoost': AdaBoostClassifier(),
        'SVM': SVC(),
        'KNN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(),
        'XGBoost Classifier' : XGBClassifier(),
    }

    best_model = None
    best_f1_score = -1
    best_metrics = {}
    start_time = time.time()

    for model_name, model in classifiers.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        if f1 > best_f1_score and f1 != 1.00 and precision != 1.00 and recall != 1.00 and  accuracy != 1.00:
            best_f1_score = f1
            best_model = model_name
            best_metrics = {'Accuracy':accuracy,'F1 Score': f1, 'Precision': precision, 'Recall': recall}

    print(f"Best Model: {best_model}")
    print("Evaluation Metrics:")
    for metric, value in best_metrics.items():
        print(f"{metric: <15}: {value:.8f}")


def _multiclass_evaluation(x, y):
    warnings.filterwarnings("ignore")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    classifiers = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'AdaBoost': AdaBoostClassifier(),
        'SVM': SVC(),
        'KNN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(),
        'XGBoost Classifier' : XGBClassifier(),
    }

    best_model = None
    best_f1_score = -1
    best_metrics = {}
    start_time = time.time()

    for model_name, model in classifiers.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        accuracy = accuracy_score(y_test, y_pred)

        if f1 > best_f1_score and f1 != 1.00 and precision != 1.00 and recall != 1.00 and  accuracy != 1.00:
            best_f1_score = f1
            best_model = model_name
            best_metrics = {'Accuracy':accuracy,'F1 Score': f1, 'Precision': precision, 'Recall': recall}

    print(f"Best Model: {best_model}")
    print("Evaluation Metrics:")
    for metric, value in best_metrics.items():
        print(f"{metric: <15}: {value:.8f}")


def evaluation(x, y, problem_type):
    problem_categories = ['R', 'B', 'M']
    if problem_type not in problem_categories:
        raise ValueError("Enter valid problem type! Options: [R: Regression, B: Binary Classification, M: Multiclass Classification]")
    else:
        print("Best Model Evaluation:\n")
        if problem_type == 'R':
            _regression_evaluation(x, y)
        elif problem_type == 'B':
            _binaryclass_evaluation(x, y)
        else:
            _multiclass_evaluation(x, y)
