# Data Analysis HW2
這次作的主題是共享單車的租賃情況，
利用自己設計的特徵以及decision tree作分析並且比較兩者之間的結果差異。

## import library
```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO   
from sklearn.tree import export_graphviz
import pydotplus

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

## 利用自己設計的特徵分析
**先自行猜測那些特徵會影響租車數量**

![](https://i.imgur.com/fnLs3dO.png)

> 1.溫度
2.風速
3.天氣

溫度 小於10度 風速 大於20 天氣 屬於第 3 4 種類 都會造成租乘數量下降

meaning of weather number 1: Clear, Few clouds, Partly cloudy, Partly cloudy

2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist

3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds

4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog

```python
df_guess=pd.read_csv('./train.csv')

def badFactor(row):
    if row['temp']<=10:
        val=0
    elif row['windspeed']>=20:
        val=0
    elif row['weather']==3 or row['weather']==4:
        val=0
    else:
        val=1
        
    return val

df_guess['guess']=df_guess.apply(badFactor,axis=1)
```
**利用平均值決定當天的租車數量為高或低**

```python
mean=df_guess['count'].mean()
def amount(row):
    if row['count']>=mean:
        val=1
    else:
        val=0
    return val

df_guess['rentAmount']=df_guess.apply(amount,axis=1)
```
### result
```python
guess_result=df_guess['guess'].values
truth=df_guess['rentAmount'].values

from sklearn.metrics import accuracy_score
accuracy_score(truth,guess_result)
```
`0.5306816094065773`
以我自行預想的特徵篩選出來有53percent的準確度，只比隨機猜測好上一些

## 利用Decision Tree作分析
**mapping the weather data to 0-1**
```python
weather_map={1:0,2:33,3:66,4:100}
df['weather']=df['weather'].map(weather_map)
df.head(20)
```
**算出當日registered與casual租借用戶的比例**
```python
def ratio(row):
    if row['casual']==0:
        val=0
    else:
        val=row['registered']/row['casual']
    return val

df['registered ratio']=df.apply(ratio,axis=1)
```
**將日期分為上班日與非上班日，消去holiday的影響**
```python
def work(row):
    if row['holiday']==1:
        val=0
    elif row['holiday']==0 and row['workingday']==0:
        val=0
    else:
        val=1
    return val

df['workingdayOrNot']=df.apply(work,axis=1)
```
**drop the unnecessary feature**
```python
df=df.drop(['count','holiday','workingday','datetime',
            'registered','casual'],axis=1)
```
**decision tree**
```python
#將data 分為training set 與 valid set
train=df[:8000]
valid=df[8000:]

demand=train['rentAmount'].values
train=train.drop('rentAmount',1)

dtree=DecisionTreeClassifier(max_depth=7)
dtree.fit(train,demand)

dot_data=StringIO()
export_graphviz(dtree,dot_data,filled=True,feature_names=list(train)
                ,class_names=['low demand','high demand']
                ,special_characters=True)
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf('dtree_bike.pdf')
```


    dtree.feature_importances_
![](https://i.imgur.com/V3jocF9.png)
由上述此項結果可得知影響租車數量最多特徵的是體感溫度

**計算準確度**
```python
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_predict)
```
![](https://i.imgur.com/PQTcGBl.png)

## 結論
經過decision tree 演算法之後，得出的結果準確率可接近**70%**，比自行猜測特徵的方法準確不少

我認為兩者的結果差異可能在於沒有用到**最相關**的特徵，導致準確率只有五成多。

至於如果要改進利用decision tree的效能，我認為如果能將相關度較高的幾個**特徵權重提高**，應該能得到更好的效果。

