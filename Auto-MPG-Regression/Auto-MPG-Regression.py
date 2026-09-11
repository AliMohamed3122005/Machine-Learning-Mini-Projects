import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.preprocessing import StandardScaler

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"

columns = [
    "mpg",
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
    "origin",
    "car_name"
]

df = pd.read_csv(
    url,
    sep=r"\s+",
    names=columns,
    na_values="?"
)

##print(df.head())
##print(df.isnull().sum())

df = df.dropna()

##print(df.head())
##print(df.isnull().sum())

##print(df.describe())
##print(df.shape)

plt.hist(df['mpg'], bins=20)
plt.xlabel('Miles per Gallon')
plt.ylabel('Frequency')
plt.title('Distribution of MPG')
plt.show()

plt.boxplot(df['mpg'], vert=False)

plt.title("Boxplot of MPG")
plt.xlabel("Miles per Gallon")
plt.show()

X = df.drop(["mpg","car_name"], axis=1)
y=df["mpg"]

results =[]
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.3,random_state=40)
linearmodel= LinearRegression()
linearmodel.fit(Xtrain,ytrain)
ypred=linearmodel.predict(Xtest)
mse = mean_squared_error(ytest,ypred)
mae = mean_absolute_error(ytest, ypred)
r2 = r2_score(ytest, ypred)
linear_mape = mean_absolute_percentage_error(ytest, ypred)
results.append({
    "Model": "Linear Regression",
    "Train RMSE": np.sqrt(mean_squared_error(ytrain, linearmodel.predict(Xtrain))),
    "MSE": mse,
    "MAE": mae,
    "RMSE": np.sqrt(mse),
    "MAPE": linear_mape,
    "R2": r2
})
## start of polynomial regression

degrees = [2, 3, 4]

for degree in degrees:
    poly_model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression()
    )

    poly_model.fit(Xtrain, ytrain)

    ypred_poly = poly_model.predict(Xtest)

    train_rmse = np.sqrt(mean_squared_error(ytrain, poly_model.predict(Xtrain)))

    test_mse = mean_squared_error(ytest, ypred_poly)
    test_mae = mean_absolute_error(ytest, ypred_poly)
    test_rmse = np.sqrt(test_mse)
    test_mape = mean_absolute_percentage_error(ytest, ypred_poly)
    test_r2 = r2_score(ytest, ypred_poly)
    test_mape = mean_absolute_percentage_error(ytest, ypred_poly)
    

    results.append({
        "Model": f"Polynomial Degree {degree}",
        "Train RMSE": train_rmse,
        "MSE": test_mse,
        "MAE": test_mae,
        "RMSE": test_rmse,
        "MAPE": test_mape,
        "R2": test_r2
    })

bestdeg = 3
alphavalues = {
    "ridge__alpha": [0.01, 0.1, 1, 10, 100]
}

ridgemodel = make_pipeline(
    PolynomialFeatures(degree=bestdeg, include_bias=False),
    StandardScaler(),
    Ridge(max_iter=10000)
)

ridge_grid = GridSearchCV(
    ridgemodel,
    param_grid=alphavalues,
    cv=5,
    scoring="neg_root_mean_squared_error"
)

ridge_grid.fit(Xtrain, ytrain)

ypred_ridge = ridge_grid.predict(Xtest)

ridge_mse = mean_squared_error(ytest, ypred_ridge)
ridge_mae = mean_absolute_error(ytest, ypred_ridge)
ridge_rmse = np.sqrt(ridge_mse)
ridge_mape = mean_absolute_percentage_error(ytest, ypred_ridge)
ridge_r2 = r2_score(ytest, ypred_ridge)

results.append({
    "Model": f"Ridge Regression (Degree){bestdeg}",
    "Train RMSE":np.sqrt(mean_absolute_error(ytrain,ridge_grid.predict(Xtrain))),
    "MSE": ridge_mse,
    "MAE": ridge_mae,
    "RMSE": ridge_rmse,
    "MAPE": ridge_mape,
    "R2": ridge_r2
})

alphavalues = {
    "lasso__alpha": [0.01, 0.1, 1, 10, 100]
}

lassomodel = make_pipeline(
    PolynomialFeatures(degree=bestdeg, include_bias=False),
    StandardScaler(),
    Lasso(max_iter=10000)
)

lasso_grid = GridSearchCV(
    lassomodel,
    param_grid=alphavalues,
    cv=5,
    scoring="neg_root_mean_squared_error"
)

lasso_grid.fit(Xtrain, ytrain)

ypred_lasso = lasso_grid.predict(Xtest)

lasso_mse = mean_squared_error(ytest, ypred_lasso)
lasso_mae = mean_absolute_error(ytest, ypred_lasso)
lasso_rmse = np.sqrt(lasso_mse)
lasso_mape = mean_absolute_percentage_error(ytest, ypred_lasso) 
lasso_r2 = r2_score(ytest, ypred_lasso)

results.append({
    "Model": f"Lasso Polynomial Degree {bestdeg}",
    "Train RMSE":np.sqrt(mean_squared_error(ytrain,lasso_grid.predict(Xtrain))),
    "MSE": lasso_mse,
    "MAE": lasso_mae,
    "RMSE":lasso_rmse,
    "MAPE": lasso_mape,
    "R2": lasso_r2
})

results_df = pd.DataFrame(results)
print(results_df)

best_model_row = results_df.sort_values(by="RMSE").iloc[0]
print("\nBest Model:")
print(best_model_row)

residuals = ypred_ridge - ytest

plt.figure(figsize=(8,5))
plt.scatter(ypred_ridge, residuals)
plt.axhline(y=0, color='red', linestyle='--')

plt.xlabel("Fitted Values")
plt.ylabel("Residuals (ŷ - y)")
plt.title("Residual Plot - Ridge Regression")

plt.show()