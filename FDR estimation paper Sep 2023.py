#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


# In[33]:


combined_path = '/Users/ayeshaferoz/Downloads/FD output res=35k,noise=0/adder csv/masses from FD.csv'

original_path = '/Users/ayeshaferoz/Downloads/FD output res=35k,noise=0/adder csv/original masses.csv'


# In[34]:


# Read the input files into pandas dataframes
combined_df = pd.read_csv(combined_path)
original_df = pd.read_csv(original_path)


# In[35]:


print (original_df.iloc[:,21])


# In[36]:


# Define the tolerance limit in parts per million
ppmtol = 5

# Create the fdval dataframe by combining the ScanNum and MonoisotopicMass columns from the combined_df
fdval = combined_df[['ScanNum', 'MonoisotopicMass']]


# In[37]:


# Convert ScanNum column to int
#fdval['ScanNum'] = fdval['ScanNum'].astype(int)
fdval.loc[:, 'ScanNum'] = fdval['ScanNum']


# In[38]:


# Convert the ScanNum column to integers
fdval['ScanNum'] = fdval['ScanNum'].astype(int)


# In[39]:


print(fdval.iloc[:, 0])


# In[40]:


# Create the tpindex1 logical index by checking if the values in the first column of fdval are present in the first column of original_df
#tpindex1 = fdval.iloc[:,0].isin(original_df.iloc[:,0])
tpindex1 = fdval['ScanNum'].isin(original_df['Scan'].dropna().replace([np.inf, -np.inf], np.nan).astype(int))
print(tpindex1)


# In[41]:


# Define a function to calculate the ppm difference between two values
def ppm_diff(value1, value2):
    return abs(value1 - value2) / value1 * 1e6


# In[42]:


print(fdval.iloc[:,1])


# In[43]:


print(original_df.iloc[:,28])


# In[44]:


# Create the tpindex2 logical index by comparing the values in the second column of fdval and original_df within the ppm tolerance specified by ppmtol
tpindex2 = np.isclose(fdval.iloc[:,1][:, np.newaxis], original_df.iloc[:,28], rtol=ppmtol/1e6, atol=0)
print (tpindex2)


# In[45]:


# Create the tpindex logical index by combining tpindex1, tpindex2, and the DummyIndex column from combined_df with values of 0
tpindex = np.logical_and.reduce((
    tpindex1.values.flatten(),
    tpindex2.any(axis=1),
    combined_df['TargetDummyType'].values==0
    ))


# In[46]:


print(tpindex)


# In[47]:


tpindex = np.logical_and.reduce((
    tpindex1.values.flatten(),
    tpindex2.any(axis=1),
    combined_df['TargetDummyType'].values == 0
))

fpindex = np.logical_and.reduce((
    np.logical_not(tpindex),
    combined_df['TargetDummyType'].values == 0
))
# Printing the total number of fpindex
print(len(fpindex))


# In[48]:


# Create the Decoyindex logical index by checking whether the value of DummyIndex in combined_df is greater than zero
Decoyindex = combined_df['TargetDummyType'] > 0


# In[49]:


print(len(Decoyindex))


# In[55]:


# Plot the histograms of QScore column in combined_df for fpindex and Decoyindex
plt.figure(figsize=(8,6))
plt.hist(combined_df.loc[fpindex, 'Qscore'], bins=100,alpha=0.7, label='False positive masses',color='red',edgecolor='grey')
#plt.hist(combined_df.loc[Decoyindex1, 'QScore'], bins=100, alpha=0.5, label='Dummymasses1',color='green')
#plt.hist(combined_df.loc[Decoyindex2, 'QScore'], bins=100, alpha=0.9, label='Dummymasses2',color='yellow') 
#plt.hist(combined_df.loc[Decoyindex3,'QScore'], bins=100,alpha=0.6, label='Dummymasses3',color='grey')
plt.hist(combined_df.loc[tpindex, 'Qscore'], bins=100, alpha=0.9, label='True positive masses',color='green',edgecolor='grey')
plt.hist(combined_df.loc[Decoyindex, 'Qscore'], bins=100, alpha=0.4, label='Dummy masses',color='blue',edgecolor='grey')
plt.xlabel('Qscore', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.title('Res=35k,noise=0,centroid', fontsize=15)
plt.legend
plt.show()


# In[56]:


import matplotlib.pyplot as plt

# Define your data indices and colors
data_indices = [fpindex, tpindex, Decoyindex]
colors = ['red', 'green', 'blue']

# Create a figure and axis
plt.figure(figsize=(8, 6))
ax = plt.gca()

# Iterate through data indices and colors
for index, color in zip(data_indices, colors):
    plt.hist(
        combined_df.loc[index, 'Qscore'],
        bins=100,
        alpha=0.7,
        label='False positive masses' if color == 'red' else
              'True positive masses' if color == 'green' else
              'Dummy masses',
        color=color,
        edgecolor='grey'
    )

plt.xlabel('Qscore', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.title('Res=35k, noise=0, centroid', fontsize=15)
plt.legend()
plt.show()


# In[57]:


plt.figure(figsize=(8, 6))
plt.boxplot(
    [combined_df.loc[fpindex, 'Qscore'], combined_df.loc[tpindex, 'Qscore'], combined_df.loc[Decoyindex, 'Qscore']],
    labels=['False positive masses', 'True positive masses', 'Dummy masses']
)
plt.xlabel('Data Subset', fontsize=15)
plt.ylabel('Qscore', fontsize=15)
plt.title('Box Plot of Qscore', fontsize=15)
plt.show()


# In[59]:


plt.figure(figsize=(8, 6))
for index, color in zip(data_indices, colors):
    sns.kdeplot(
        combined_df.loc[index, 'Qscore'],
        label='False positive masses' if color == 'red' else
              'True positive masses' if color == 'green' else
              'Dummy masses',
        color=color,
        shade=True
    )
plt.xlabel('Qscore', fontsize=15)
plt.ylabel('Density', fontsize=15)
plt.title('Kernel Density Estimation Plot of Qscore', fontsize=15)
plt.legend()
plt.show()


# In[65]:


import numpy as np

def ecdf(data):
    x = np.sort(data)
    y = np.arange(1, len(data) + 1) / len(data)
    return x, y

plt.figure(figsize=(8, 6))
for index, color in zip(data_indices, colors):
    x, y = ecdf(combined_df.loc[index, 'Qscore'])
    plt.plot(x, y, marker='.', linestyle='none', label='False positive masses' if color == 'red' else
                                                           'True positive masses' if color == 'green' else
                                                           'Dummy masses',
             color=color)
plt.xlabel('Qscore', fontsize=15)
plt.ylabel('ECDF', fontsize=15)
plt.title('ECDF Plot of Qscore', fontsize=15)
plt.legend()
plt.show()


# In[68]:


import seaborn as sns

plt.figure(figsize=(10, 6))
sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

for index, color in zip(data_indices, colors):
    subset_data = combined_df.loc[index, 'Qscore']
    sns.kdeplot(subset_data, color=color, label='False positive masses' if color == 'red' else
                                                 'True positive masses' if color == 'green' else
                                                 'Dummy masses')

plt.xlabel('Qscore', fontsize=15)
plt.ylabel('Density', fontsize=15)
plt.title('Ridgeline Plot of Qscore', fontsize=15)
plt.legend()
plt.show()


# In[69]:


plt.figure(figsize=(10, 6))

for index, color in zip(data_indices, colors):
    plt.hist(
        combined_df.loc[index, 'Qscore'],
        bins=100,
        alpha=0.5,
        label='False positive masses' if color == 'red' else
              'True positive masses' if color == 'green' else
              'Dummy masses',
        color=color,
        edgecolor='grey',
        cumulative=True,
        density=True,
        histtype='stepfilled'
    )

plt.xlabel('Qscore', fontsize=15)
plt.ylabel('Density', fontsize=15)
plt.title('Area Plot of Qscore', fontsize=15)
plt.legend()
plt.show()


# In[72]:


plt.figure(figsize=(6, 6))
subset_sizes = [len(fpindex), len(tpindex), len(Decoyindex)]
labels = ['False positive masses', 'True positive masses', 'Dummy masses']
colors = ['red', 'green', 'blue']

plt.pie(subset_sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Pie Chart of Data Subset Composition', fontsize=15)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[75]:


plt.figure(figsize=(6, 6))
subset_sizes = [len(fpindex), len(tpindex), len(Decoyindex)]
labels = ['False positive masses', 'True positive masses', 'Dummy masses']
colors = ['red', 'green', 'blue']

plt.pie(subset_sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, wedgeprops=dict(width=0.4))
plt.title('Donut Chart of Data Subset Composition', fontsize=15)
plt.axis('equal')
plt.show()


# In[76]:


import networkx as nx

# Create a simple example network graph
G = nx.Graph()
G.add_edge('Node A', 'Node B')
G.add_edge('Node B', 'Node C')
G.add_edge('Node C', 'Node A')

plt.figure(figsize=(8, 6))
nx.draw(G, with_labels=True, node_size=1000, node_color='skyblue', font_size=12)
plt.title('Network Graph', fontsize=15)
plt.show()


# In[78]:


from matplotlib.sankey import Sankey

plt.figure(figsize=(10, 6))
sankey = Sankey()
sankey.add(flows=[len(fpindex), len(tpindex), len(Decoyindex)], labels=['False positive', 'True positive', 'Dummy'])
sankey.finish()
plt.title('Sankey Diagram of Data Subset Flow', fontsize=15)
plt.show()


# In[ ]:





# In[ ]:




