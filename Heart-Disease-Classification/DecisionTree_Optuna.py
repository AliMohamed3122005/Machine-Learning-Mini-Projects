import optuna
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from clean_prepare import prepare_data

(
    X_train,
    X_test,
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test
) = prepare_data()

def objective(trial):
    model = DecisionTreeClassifier(
        max_depth=trial.suggest_int("max_depth",3,7),
        min_samples_split=trial.suggest_int("min_samples_split",2,6),
        random_state=42
        )
    
    score = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy"
    ).mean()
    return score

study=optuna.create_study(direction="maximize")
study.optimize(objective,n_trials=20)







best_model = DecisionTreeClassifier(
    **study.best_params,
    random_state=42
)

best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)


print("Decision Tree Classifier")
print("------------------------------")

print("Accuracy:", accuracy)

print("------------------------------")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("------------------------------")
print("Classification Report:")
print(classification_report(y_test, y_pred))