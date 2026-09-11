import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
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

    learning_rates=np.arange(0.001,1.001,0.001)
    train_f1_scores = []
    test_f1_scores=[]
    best_lr=None
    best_test_f1=0

    for lr in learning_rates:
        model = SGDClassifier(loss="log_loss",learning_rate="constant",eta0=lr,max_iter=1000,random_state=42)
        model.fit(X_train,y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        train_f1=f1_score(y_train,y_train_pred)
        test_f1=f1_score(y_test,y_test_pred)

        test_f1_scores.append(test_f1)
        train_f1_scores.append(train_f1)
        

        if test_f1 > best_test_f1:
            best_test_f1=test_f1
            best_lr= lr

    print("Best Learning ratr:",best_lr)
    print("best test F1 score:",best_test_f1)

    plt.figure()
    plt.plot(learning_rates,train_f1_scores,label="Training F1 score")
    plt.plot(learning_rates,test_f1_scores,label="Test F1 score")

    plt.xlabel("Learning Rate")
    plt.ylabel("F1-Scores")
    plt.legend()
    plt.grid(True)
    plt.show()
