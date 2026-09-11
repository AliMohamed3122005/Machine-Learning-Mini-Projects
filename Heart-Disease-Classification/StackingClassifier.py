from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from clean_prepare import prepare_data



X_train_scaled,X_test_scaled,y_train_scaled,y_test_scaled,y_train,y_test=prepare_data()


base_models=[
    (
        "logistic",LogisticRegression(C=1,max_iter=1000,random_state=42)
    ),
    (
        "Knn",KNeighborsClassifier(n_neighbors=3)
    ),
    (
        "DecisionTree",DecisionTreeClassifier(max_depth=4,random_state=42)
    )
]
final_model=LogisticRegression(max_iter=1000,random_state=420)

stacking_model=StackingClassifier(estimators=base_models,final_estimator=final_model,cv=5,stack_method="predict_proba",n_jobs=-1)

stacking_model.fit(X_train_scaled,y_train)


y_pred=stacking_model.predict(X_test_scaled)


print("Stacking Classifier")
print("------------------------------")
print("Accuracy:",accuracy_score(y_test,y_pred))
print("------------------------------")
print("confusion matrix:",confusion_matrix(y_test,y_pred))
print("------------------------------")
print("Classification Report:")
print(classification_report(y_test, y_pred))