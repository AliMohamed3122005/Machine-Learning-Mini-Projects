import numpy as np

# x->house size
# y->price in thousands


x=np.array([50, 60, 80, 100, 120])
y=np.array([150, 180, 240, 300, 330])

size = 90                                                       #size we want to predict 
colx=x.reshape(-1,1)
coly=y.reshape(-1,1)    

rows = colx.shape[0]                                            #shape[0] count the number of rows to make the ones
colones=np.ones((rows,1))                                       #making the ones
bias = np.append(colones,colx, axis=1)                          #add the ones , axis = 1 make the rows move to the right


transx=bias.T
inverse=np.linalg.inv(transx@bias)


paravector=inverse@transx@coly

theta=np.array(paravector)                                     #changing it to array
predicted=theta[0]+size*theta[1]

print(f"The Predicted price for 90 m²: {predicted} thousand dollars")
