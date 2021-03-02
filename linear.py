import pandas as pd
from sklearn.linear_model import LinearRegression
#from sklearn.metrics import mean_squared_error
import sklearn.metrics
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict
df=pd.read_csv(r'C:\Users\manis\PycharmProjects\5502project\5502finalproject.csv')
dfgroup=df.groupby(['Country'])
dfdeveloping=df[(df['Status']=='Developing')]
print(dfdeveloping)
dfdeveloping_life_expectancy=dfdeveloping[['Life_Expectancy']]
dfdeveloping_independent=dfdeveloping[['winsorized_Schooling','winsorized_Income_Comp_Of_Resources','winsorized_HIV','winsorized_Adult_Mortality']]


x=dfdeveloping_independent
y=dfdeveloping_life_expectancy['Life_Expectancy']
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1,test_size=0.2)

lm = LinearRegression()
model = lm.fit(x,y)
predictions = lm.predict(x)
print("Linear Regression Predicted values",predictions)

print("R square",lm.score(x,y))
print("Coefficients",lm.coef_)
print("intercept",lm.intercept_)

score = cross_val_score(model,x_train, y_train, cv=10, scoring='neg_mean_squared_error')
predictions_cross = cross_val_predict(model, x, y, cv=6)
print("Errors",score)
print("Cross validation prediction values",predictions_cross)