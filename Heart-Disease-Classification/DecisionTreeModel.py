from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from clean_prepare import prepare_data


X_train, X_test, y_train, y_test = prepare_data()


model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

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