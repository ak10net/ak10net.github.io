---
layout: post
title: "Coal mines of India"
author: "Ankit"
tags: viz
excerpt_separator: <!--more-->
---

## In this post we are going to explore coal mines of india<!--more-->

- In India series, We am going to analyze datasets related to india from 'data is plural'
- This is first blog in that series
- [Data source](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/TDEK8O) 


```python
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.set(rc={'figure.figsize':(12,6)})
import folium
```

- Our data sets is about coal mines of india. It captures details of coal mines such as state, district, location, capacity, owner, public or private etc

```python
df = pd.read_excel("Indian Coal Mines Dataset_January 2021-1.xlsx", sheet_name='Mines Datasheet')
```


```python
# Cleaning up columns
df = df.drop(['SL No.', 'Source', 'Accuracy (exact vs approximate)', 'Unnamed: 13' ], axis=1)

df = df.rename(columns={'State/UT Name': 'State', 'Mine Name': 'Mine', 'Coal/ Lignite Production (MT) (2019-2020)': 'Production',
                       'Coal Mine Owner Name': 'Owner', 'Coal/Lignite': 'Type of coal', 'Govt Owned/Private': 'Public/Private',
                       'Type of Mine (OC/UG/Mixed)': 'Mine Type'})
```

- We are going to look at all the categorical columns and plot value counts of each category


```python
for i, col in enumerate(df[['State','Owner', 'Type of coal','Public/Private', 'Mine Type']]):
    plt.figure(i)
    sns.countplot(x=df[col], data=df, order=df[col].value_counts().index).set_title(f'{col}')
    plt.xticks(rotation=45)
```

- Jharkhand has maximum number of coal mines

![png](/assets/output_7_0.png)

- Goverment entities such as ECL, WCL, SECL owns majority of coal mines

![png](/assets/output_7_1.png)

- Most coal mines produce coal and very little Lignite

![png](/assets/output_7_2.png)

- Most of the coal mines are under government authorization

![png](/assets/output_7_3.png)

- More than half of mines are of Open type

![png](/assets/output_7_4.png)



```python
# Which mine has highest production capacity
df.loc[df['Production']==df['Production'].max()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>District Name</th>
      <th>Mine</th>
      <th>Production</th>
      <th>Owner</th>
      <th>Type of coal</th>
      <th>Public/Private</th>
      <th>Mine Type</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>311</th>
      <td>Chhattisgarh</td>
      <td>Korba</td>
      <td>GEVRA OC</td>
      <td>45.0</td>
      <td>SECL</td>
      <td>Coal</td>
      <td>G</td>
      <td>OC</td>
      <td>22.3308</td>
      <td>82.5958</td>
    </tr>
  </tbody>
</table>
</div>



- Although Jharkhand has maximum number of coal mines but chattisgarh and orissa are ahead in production capacity 


```python
df.groupby(['State'])['Production'].sum().sort_values(ascending=False).plot(kind='bar')
```
![png](/assets/output_10_1.png)


- We are going to plot all the power plant on indian map using folium package

```python
locations = df[['Latitude ', 'Longitude ']]
locationlist = locations.values.tolist()
map = folium.Map(location=[22.3308, 82.5958], zoom_start=5)
for point in range(0, len(locationlist)):
    folium.Marker(locationlist[point], popup=df['Mine'][point]).add_to(map)
map
```
![png](/assets/folium1.png)


- We are going to color the markers according to the type of coal mine

```python
def typecolors(df):
    if df['Mine Type'] == 'OC':
        return 'green'
    elif df['Mine Type'] == 'UG':
        return 'blue'
    else:
        return 'red'
df["Color"] = df.apply(typecolors, axis=1)

map = folium.Map(location=[22.3308, 82.5958], zoom_start=5)
for point in range(0, len(locationlist)):
    folium.Marker(locationlist[point], popup=df['Mine'][point],
                 icon=folium.Icon(color=df["Color"][point])).add_to(map)
map
```

![png](/assets/folium2.png)


**This was fun, watch out for more post in India series**
