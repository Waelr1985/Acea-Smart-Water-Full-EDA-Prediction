#!/usr/bin/env python
# coding: utf-8

# <h1 style='color:white; background:#50A8E3; border:0'><center>Acea Smart Water: Full EDA & Prediction</center></h1>
# 
# ![Water](https://images.immediate.co.uk/production/volatile/sites/4/2018/07/iStock_69791979_XXLARGE_1-c9eba8a.jpg?quality=90&resize=940%2C400)
# 
# **The Acea Group is one of the leading Italian multiutility operators. Listed on the Italian Stock Exchange since 1999, the company manages and develops water and electricity networks and environmental services. Acea is the foremost Italian operator in the water services sector supplying 9 million inhabitants in Lazio, Tuscany, Umbria, Molise, Campania.**
# 
# **This competition uses nine different datasets, completely independent and not linked to each other. Each dataset can represent a different kind of waterbody. As each waterbody is different from the other, the related features as well are different from each other.**
# 
# <a id="section-start"></a>
# 
# <h2 style='color:white; background:#50A8E3; border:0'><center>Here you'll find:</center></h2>
# 
# * EDA
# * Some ideas for predictions
# * Predictions
# 
# 
# 1. [**Loading and a first look at the data**](#section-one) <br>
# 2. [**EDA**](#section-two) <br>
#  [...Aquifer_Doganella](#section-three) <br>
#  [...Aquifer_Auser](#section-four) <br>
#  [...Water_Spring_Amiata](#section-five) <br>
#  [...Lake_Bilancino](#section-six) <br>
#  [...Water_Spring_Madonna_di_Canneto](#section-seven) <br>
#  [...Aquifer_Luco](#section-eight) <br>
#  [...Aquifer_Petrignano](#section-nine) <br>
#  [...Water_Spring_Lupa](#section-ten) <br>
#  [...River_Arno](#section-eleven) <br>
# 3. [**Feature overview and prediction**](#section-twelve) <br>
#  [...River Arno (features)](#section-thirteen) <br>
#  [...**River Arno (prediction)**](#section-1) <br>
#  
#  [...Lake Bilancino (features)](#section-fourteen) <br>
#  [...**Lake Bilancino (prediction)**](#section-2) <br>
#  
#  [...Aquifer Petrignano (features)](#section-fifteen) <br>
#  [...**Aquifer Petrignano (prediction)**](#section-3) <br>
#  
#  [...Aquifer Auser (features)](#section-nineteen) <br>
#  [...**Aquifer Auser (prediction)**](#section-4) <br>
#  
#  [...Aquifer Doganella (features)](#section-twenty) <br>
#  [...**Aquifer Doganella (prediction)**](#section-5) <br>
#  
#  [...Aquifer Luco (features)](#section-twentyone) <br>
#  [...**Aquifer Luco (prediction)**](#section-6) <br>
#  
#  [...Water Spring Madonna di Canneto (features)](#section-sixteen) <br>
#  [...**Water Spring Madonna di Canneto (prediction)**](#section-7) <br>
#  
#  [...Water Spring Lupa (features)](#section-seventeen) <br>
#  [...**Water Spring Lupa (prediction)**](#section-8) <br>
#  
#  [...Water Spring Amiata (features)](#section-eighteen) <br>
#  [...**Water Spring Amiata (prediction)**](#section-9) <br>

# In[1]:


# loading packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
plt.rcParams['figure.dpi'] = 300

import matplotlib.dates as mdates

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# ignoring warnings
import warnings
warnings.simplefilter("ignore")


# In[2]:


print('Datasets:')
os.listdir('c:/Users/waelr/Desktop/Python/acea-water-prediction')


# In[3]:


import glob


# In[4]:


csv_files = glob.glob('c:/Users/waelr/Desktop/python/acea-water-prediction/*.csv')


# In[5]:


csv_files


# In[6]:


# not effective way 
Aquifer_Auser = csv_files[0]
Aquifer_Doganella = csv_files[1]
Aquifer_Luco = csv_files[2]
Aquifer_Petrignano = csv_files[3]
Lake_Bilancino = csv_files[4]
River_Arno = csv_files[5]
Water_Spring_Amiata = csv_files[6]
Water_Spring_Lupa = csv_files[7]
Water_Spring_Madonna_di_Canneto = csv_files[8]


# In[7]:


Aquifer_Auser


# In[6]:


#  Iterate over csv_files
for csv in csv_files:

    #  Read csv into a DataFrame: df
    df = pd.read_csv(csv)


# In[7]:


Aquifer_Doganella = pd.read_csv('c:/Users/waelr/Desktop/python/acea-water-prediction/Aquifer_Doganella.csv', index_col = 'Date')
Aquifer_Auser = pd.read_csv('c:/Users/waelr/Desktop/python/acea-water-prediction/Aquifer_Auser.csv', index_col = 'Date')
Water_Spring_Amiata = pd.read_csv('c:/Users/waelr/Desktop/python/acea-water-prediction/Water_Spring_Amiata.csv', index_col = 'Date')
Lake_Bilancino = pd.read_csv('c:/Users/waelr/Desktop/python/acea-water-prediction/Lake_Bilancino.csv', index_col = 'Date')
Water_Spring_Madonna_di_Canneto = pd.read_csv('c:/Users/waelr/Desktop/python/acea-water-prediction/Water_Spring_Madonna_di_Canneto.csv', index_col = 'Date')
Aquifer_Luco = pd.read_csv('c:/Users/waelr/Desktop/python/acea-water-prediction/Aquifer_Luco.csv', index_col = 'Date')
Aquifer_Petrignano = pd.read_csv('c:/Users/waelr/Desktop/python/acea-water-prediction/Aquifer_Petrignano.csv', index_col = 'Date')
Water_Spring_Lupa = pd.read_csv('c:/Users/waelr/Desktop/python/acea-water-prediction/Water_Spring_Lupa.csv', index_col = 'Date')
River_Arno = pd.read_csv('c:/Users/waelr/Desktop/python/acea-water-prediction/River_Arno.csv', index_col = 'Date')


print('Datasets shape:')
print('*'*30)
print('Aquifer_Doganella: {}'.format(Aquifer_Doganella.shape))
print('Aquifer_Auser: {}'.format(Aquifer_Auser.shape))
print('Water_Spring_Amiata: {}'.format(Water_Spring_Amiata.shape))
print('Lake_Bilancino: {}'.format(Lake_Bilancino.shape))
print('Water_Spring_Madonna_di_Canneto: {}'.format(Water_Spring_Madonna_di_Canneto.shape))
print('Aquifer_Luco: {}'.format(Aquifer_Luco.shape))
print('Aquifer_Petrignano: {}'.format(Aquifer_Petrignano.shape))
print('Water_Spring_Lupa: {}'.format(Water_Spring_Lupa.shape))
print('River_Arno: {}'.format(River_Arno.shape))


# In[8]:


datasets = [Aquifer_Doganella, Aquifer_Auser, Water_Spring_Amiata,
            Lake_Bilancino, Water_Spring_Madonna_di_Canneto, Aquifer_Luco,
            Aquifer_Petrignano, Water_Spring_Lupa, River_Arno]

datasets_names = ['Aquifer_Doganella', 'Aquifer_Auser', 'Water_Spring_Amiata',
                'Lake_Bilancino', 'Water_Spring_Madonna_di_Canneto', 'Aquifer_Luco',
                'Aquifer_Petrignano', 'Water_Spring_Lupa', 'River_Arno']


# In[9]:


print('Datasets dtypes:')
print('*'*30)
for i in range(len(datasets)):
    print('{}: \n{}'.format(datasets_names[i], datasets[i].dtypes.value_counts()))
    print('-'*20)


# In[ ]:


def line_plot(data, y, title, color,
              top_visible = False, right_visible = False, 
              bottom_visible = True, left_visible = False,
              ylabel = None, figsize = (10, 4), axis_grid = 'y'):
    fig, ax = plt.subplots(figsize = figsize)
    plt.title(title, size = 15, fontweight = 'bold', fontfamily = 'serif')

    for i in ['top', 'right', 'bottom', 'left']:
        ax.spines[i].set_color('black')
    
    ax.spines['top'].set_visible(top_visible)
    ax.spines['right'].set_visible(right_visible)
    ax.spines['bottom'].set_visible(bottom_visible)
    ax.spines['left'].set_visible(left_visible)
    
    sns.lineplot(x = range(len(data[y])), y = data[y], dashes = False, 
                 color = color, linewidth = .5)
    ax.xaxis.set_major_locator(plt.MaxNLocator(20))
    
    ax.set_xticks([])
    plt.xticks(rotation = 90)
    plt.xlabel('')
    plt.ylabel(ylabel, fontfamily = 'serif')
    ax.grid(axis = axis_grid, linestyle = '--', alpha = 0.9)
    plt.show()

def corr_plot(data,
              top_visible = False, right_visible = False, 
              bottom_visible = True, left_visible = False,
              ylabel = None, figsize = (15, 11), axis_grid = 'y'):
    fig, ax = plt.subplots(figsize = figsize)
    plt.title('Correlations (Pearson)', size = 15, fontweight = 'bold', fontfamily = 'serif')
    
    mask = np.triu(np.ones_like(data.corr(), dtype = bool))
    sns.heatmap(round(data.corr(), 2), mask = mask, cmap = 'viridis', annot = True)
    plt.show()
    
def columns_viz(data, color):
    for i in range(len(data.columns)):
        line_plot(data = data, y = data.columns[i],
                  color = color,
                  title = '{} dynamics'.format(data.columns[i]),
                  bottom_visible = False, figsize = (10, 2))


# In[10]:





# In[11]:


for i in range(len(datasets)):
    NaN_values = (datasets[i].isnull().sum() / len(datasets[i]) * 100).sort_values(ascending = False)
    bar_plot(x = NaN_values, 
             y = NaN_values.index,
             title = '{}: NaN values (%)'.format(datasets_names[i]),
             palette_len = NaN_values.index, 
             xlim = (0, 100), 
             xticklabels = range(0, 101, 20),
             yticklabels = NaN_values.index,
             left_visible = True,
             figsize = (10, 8), axis_grid = 'x')


# ![competitions_Acea_Screen%20Shot%202020-12-02%20at%2012.40.17%20PM.png](attachment:competitions_Acea_Screen%20Shot%202020-12-02%20at%2012.40.17%20PM.png)

# In[13]:


datasets[0].head()


# In[14]:


print('The earliest date: \t%s' %datasets[0].index.values[[0, -1]])
print('The latest date: \t%s' %datasets[0].index.values[[0, -1]])


# In[15]:


print('The earliest date: \t%s' %datasets[0].index.values[[0, -1]][0])
print('The latest date: \t%s' %datasets[0].index.values[[0, -1]][1])


# In[16]:


corr_plot(datasets[0])


# In[17]:


columns_viz(datasets[0], color = '#FFC30F')


# In[20]:


corr_plot(datasets[1])


# In[21]:


columns_viz(datasets[1], color = '#FF5733')


# In[18]:


datasets[2].head()


# In[23]:


corr_plot(datasets[2])


# In[24]:


columns_viz(datasets[2], color = '#C70039')


# In[20]:


datasets[3].head()


# In[26]:


print('The earliest date: \t%s' %datasets[3].index.values[[0, -1]][0])
print('The latest date: \t%s' %datasets[3].index.values[[0, -1]][1])


# In[21]:


corr_plot(datasets[3])


# In[28]:


columns_viz(datasets[3], color = '#900C3F')


# In[22]:


datasets[4].head()


# In[23]:


datasets[4].index.values[[0, -1]]


# In[24]:


datasets[4].index.dropna().values[[0, -1]]


# In[25]:


datasets[4].index.values[[0, -1]]


# In[ ]:





# In[30]:


print('The earliest date: \t%s' %datasets[4].index.values[[0, -1]][0])
print('The latest date: \t%s' %datasets[4].index.dropna().values[[0, -1]][1])


# In[31]:


corr_plot(datasets[4])


# In[32]:


columns_viz(datasets[4], color = '#581845')


# In[33]:


datasets[5].head()


# In[34]:


print('The earliest date: \t%s' %datasets[5].index.values[[0, -1]][0])
print('The latest date: \t%s' %datasets[5].index.values[[0, -1]][1])


# In[35]:


corr_plot(datasets[5])


# In[36]:


columns_viz(datasets[5], color = '#547980')


# In[37]:


datasets[6].head()


# In[38]:


print('The earliest date: \t%s' %datasets[6].index.values[[0, -1]][0])
print('The latest date: \t%s' %datasets[6].index.values[[0, -1]][1])


# In[39]:


corr_plot(datasets[6])


# In[40]:


columns_viz(datasets[6], color = '#45ADA8')


# In[41]:


datasets[7].head()


# In[42]:


print('The earliest date: \t%s' %datasets[7].index.values[[0, -1]][0])
print('The latest date: \t%s' %datasets[7].index.values[[0, -1]][1])


# In[43]:


corr_plot(datasets[7])


# In[44]:


columns_viz(datasets[7], color = '#9DE0AD')


# In[45]:


datasets[8].head()


# In[46]:


print('The earliest date: \t%s' %datasets[8].index.values[[0, -1]][0])
print('The latest date: \t%s' %datasets[8].index.values[[0, -1]][1])


# In[47]:


corr_plot(datasets[8])


# In[48]:


columns_viz(datasets[8], color = '#474747')


# In[49]:


df = River_Arno[['Hydrometry_Nave_di_Rosano', 'Temperature_Firenze']].reset_index()


# In[50]:


df


# In[51]:


River_Arno 


# In[52]:


River_Arno.iloc[:, 0:-2]


# In[53]:


River_Arno.iloc[:, 0:14]


# In[54]:


River_Arno.iloc[:, 0:-2].sum(axis = 1).values 


# In[55]:


f = River_Arno.iloc[:, 0:-2].sum(axis = 1)


# In[56]:


f = np.array(f)


# In[57]:


f 


# In[58]:


pd.to_datetime(df.Date).dt.year


# pandas.Series.dt.year
# 
# Series.dt.year
# 
# The year of the datetime.

# In[59]:


pd.to_datetime(df.Date).apply(lambda x: x.strftime('%Y/%m'))


# In[60]:


df = River_Arno[['Hydrometry_Nave_di_Rosano', 'Temperature_Firenze']].reset_index()
df['rainfall'] = River_Arno.iloc[:, 0:-2].sum(axis = 1).values
df['year'] = pd.to_datetime(df.Date).dt.year
df['month'] = pd.to_datetime(df.Date).dt.month

# Monthly dynamics
df['month_year'] = pd.to_datetime(df.Date).apply(lambda x: x.strftime('%Y/%m'))


# In[61]:


df 


# In[62]:


r_means = np.log(df.groupby('month_year').Hydrometry_Nave_di_Rosano.mean() * 10).reset_index()


# In[63]:


r_means


# In[64]:


df.groupby('month_year').Hydrometry_Nave_di_Rosano.mean().reset_index()


# Why is log used?
# 
# It lets you work backwards through a calculation. It lets you undo exponential effects. Beyond just being an inverse operation, logarithms have a few specific properties that are quite useful in their own right: Logarithms are a convenient way 
# 
# # to express large numbers.

# What is the concept of log?
# 
# log 100 = 2
# 
# because
# 
# 10**2 = 100

# The natural logarithm log is the inverse of the exponential function, so that log(exp(x)) = x. The natural logarithm is logarithm in base e.

# In[65]:


df = River_Arno[['Hydrometry_Nave_di_Rosano', 'Temperature_Firenze']].reset_index()
df['rainfall'] = River_Arno.iloc[:, 0:-2].sum(axis = 1).values
df['year'] = pd.to_datetime(df.Date).dt.year
df['month'] = pd.to_datetime(df.Date).dt.month

# Monthly dynamics
df['month_year'] = pd.to_datetime(df.Date).apply(lambda x: x.strftime('%Y/%m'))

r_means = np.log(df.groupby('month_year').Hydrometry_Nave_di_Rosano.mean() * 10).reset_index()
r_means['month_year'] = pd.to_datetime(r_means['month_year'])

r_rain = np.log(df.groupby('month_year').rainfall.mean()).reset_index()
r_rain['month_year'] = pd.to_datetime(r_rain['month_year'])

r_temp = np.log(df.groupby('month_year').Temperature_Firenze.mean()).reset_index()
r_temp['month_year'] = pd.to_datetime(r_temp['month_year'])


# In[66]:


df


# In[67]:


r_temp


# In[68]:


r_means = np.log(df.groupby('month_year').Hydrometry_Nave_di_Rosano.mean() * 10).reset_index()
r_means


# In[ ]:





# In[69]:


r_means


# In[70]:


r_means.month_year[::12]


# In[71]:


r_means.month_year[::10]


# range(start, stop, step)

# In[ ]:




