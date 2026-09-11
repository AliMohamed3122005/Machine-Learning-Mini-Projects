from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score,accuracy_score, classification_report
import pandas as pd
import matplotlib.pyplot as plt

def train_model(df):
    print("Reading files...")
    le = LabelEncoder()
    for col in df.columns:
        df[col]=le.fit_transform(df[col])
    x = df.drop('class', axis=1)
    y = df['class']
    X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
    train_score=[]
    test_score=[]
    k_values = range(2,151)
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train,y_train)
        y_pred_test=model.predict(X_test)
        y_pred_train=model.predict(X_train)
        train_f1=f1_score(y_train,y_pred_train)
        test_f1=f1_score(y_test,y_pred_test)
        train_score.append(train_f1)
        test_score.append(test_f1)
        print(f"K = {k},Train F1 = {train_f1},Test F1 ={test_f1} ")
    best_test=max(test_score)
    best_k_index=test_score.index(best_test)
    best_k=list(k_values)[best_k_index]
    final_model = KNeighborsClassifier(n_neighbors=best_k)
    final_model.fit(X_train,y_train)
    final_model_pred=final_model.predict(X_test)
    print("-------Evaluation------")
    print("Classification report:")
    print(classification_report(y_test,final_model_pred))
    plt.plot(k_values,train_score,label='Train F1')
    plt.plot(k_values,test_score,label='Test F1')
    plt.xlabel("K Values")
    plt.ylabel("F1")
    plt.legend()
    plt.show()