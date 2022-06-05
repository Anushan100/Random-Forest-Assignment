import pandas as pd
import sklearn.datasets
from sklearn.ensemble import RandomForestRegressor
import pickle
from sklearn.model_selection import train_test_split

house_price=sklearn.datasets.load_boston()

df=pd.DataFrame(house_price.data,columns=house_price.feature_names)
df['price']=house_price.target

x=df.drop(columns=['price'])
y=df['price']

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=3,test_size=0.3)

ranfro=RandomForestRegressor()
ranfro.fit(x_train,y_train)

ranfro.fit(x_train,y_train)
pickle.dump(ranfro,open('final_model.pkl','wb'))
load_model=pickle.load(open('final_model.pkl','rb'))