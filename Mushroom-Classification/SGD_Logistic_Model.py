import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def train_model(df):
    print("Reading files T2...")
    le = LabelEncoder()
    for col in df.columns:
        df[col]=le.fit_transform(df[col])
    x = df.drop('class', axis=1)
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42
    )

    scaler=StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    iterations_values=range(1000,10001,1000)
    train_f1_scores = []
    test_f1_scores=[]
    best_iter=None
    best_test_f1=0

    for max_iter in iterations_values:
        model = LogisticRegression(max_iter=max_iter,solver="lbfgs")
        model.fit(X_train,y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        train_f1=f1_score(y_train,y_train_pred)
        test_f1=f1_score(y_test,y_test_pred)

        test_f1_scores.append(test_f1)
        train_f1_scores.append(train_f1)
        

        if test_f1 > best_test_f1:
            best_test_f1=test_f1
            best_iter= max_iter

    print("Best iteraions:",best_iter)
    print("best test F1 score:",best_test_f1)

    plt.figure()
    plt.plot(iterations_values,train_f1_scores,label="Training F1 score")
    plt.plot(iterations_values,test_f1_scores,label="Test F1 score")

    plt.xlabel("Number of iterations")
    plt.ylabel("F1-Scores")
    plt.legend()
    plt.grid(True)
    plt.show()
