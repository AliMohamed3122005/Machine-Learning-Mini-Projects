from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from clean_prepare import prepare_data


X_train_scaled, X_test_scaled, y_train, y_test = prepare_data()


parameters = {
    "n_neighbors": [3, 5, 7, 9]
}


model = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=parameters,
    cv=5,
    scoring="accuracy"
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)


print("Best Parameters:", model.best_params_)
print("Best Cross Validation Accuracy:", model.best_score_)

print("Test Accuracy:", accuracy_score(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))