import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read data
path = 'C:\\ML\\Housing\\housing.csv'
data = pd.read_csv(path)
data = data.iloc[:,:-1]

# show data details 
print ('data :\n',data.head(10))
print('*' * 100)

print ('Data description : \n',data.describe())
print('*' * 100)

print ('len data: ',len(data))
# rescaling data
data = (data - data.mean()) / data.std()
print ('data after rescaling :\n',data.head(10))
print('*' * 100)

# insert new column called one has value equal 1
data.insert(0,'one',1)

# separate x values (trainning data) from y values (target variable)
cols = data.shape[1] 
x = data.iloc[:,: cols - 1]
y = data.iloc[:, cols - 1:cols]

print ('X : \n',x.head(10)) 
print('*' * 100)
print ('Y : \n',y.head(10))
print('*' * 100)

# convert data from data frame to numpy matrix
x = np.matrix(x.values) # convert x 
y = np.matrix(y.values) # convert y 
theta = np.zeros((1,x.shape[1]))

print("Are there NaN values in x?", np.isnan(x).any())
print("Are there NaN values in y?", np.isnan(y).any())

print("Are there NaN values in x?", np.isnan(x).any())
print("Number of NaN values per column:\n", np.isnan(x).sum(axis=0))

for col in range(x.shape[1]):
    if np.isnan(x[:, col]).any():
        mean_value = np.nanmean(x[:, col])  # حساب المتوسط مع تجاهل القيم NaN
        x[:, col] = np.nan_to_num(x[:, col], nan=mean_value)
# Compute Cost function
def ComputeCost(x,y,theta):
    # for i in range(x.shape[1]):
    z = np.power(((x * theta.T) - y),2) ######
    num_rows = len(x)
    return np.sum(z)  / (2 * num_rows)

for k in range (x.shape[1]):            
    error = (x[:,k] * theta[0,k]) - y
print ('error',error.shape)
print ('x',x.shape)

# # gradiant descient
def GradiantDescient(x,y,theta,alpha,iters):
    temp = np.zeros(theta.shape)
    parameters = x.shape[1]
    cost = np.zeros(iters)
    
    for i in range(iters):                
        error = (x * theta.T) - y
        for j in range(parameters):
            term = np.multiply(error, x[:,j]) 
            temp[0,j] = theta[0,j] - ((alpha / len(x)) * np.sum(term))
        theta = temp
        cost[i] = ComputeCost(x, y, theta).item()
    return theta ,cost




# print('data after insert \n',data.head(10))
# print('*' * 100)


# draw data
# data.plot (kind = 'scatter',x= '',y='')
# plt.show()
print ('Theta : ',theta)

print ('Compute Cost = ',ComputeCost(x, y, theta))
print ('*' * 100)


alpha = 0.001
iters = 1000

g , cost = GradiantDescient(x, y, theta, alpha, iters)

print ('g : ',g)
print ('cost : ',cost[-1])
# # get best fit line 
f = g[0,0] + g[0,1]
# # draw best fit line 

# # draw error graph


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# أعد تدريب النموذج على x_train و y_train
theta = np.zeros((1, x_train.shape[1]))
g, cost = GradiantDescient(x_train, y_train, theta, alpha, iters)

# التحقق من الأداء على بيانات الاختبار
y_test_pred = np.dot(x_test, g.T)
test_cost = ComputeCost(x_test, y_test, g)
print("Test Cost:", test_cost)

