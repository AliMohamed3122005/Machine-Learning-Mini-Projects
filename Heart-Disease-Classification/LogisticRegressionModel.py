from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from clean_prepare import prepare_data


X_train_scaled, X_test_scaled, y_train, y_test = prepare_data()


model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)


print("Logistic Regression")
print("------------------------------")

print("Accuracy:", accuracy)

print("------------------------------")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("------------------------------")
print("Classification Report:")
print(classification_report(y_test, y_pred))