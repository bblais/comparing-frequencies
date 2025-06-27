#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from defs import *


# In[3]:


counts_data=df=pd.read_excel('data_Ilan_Gospels_Josephus_Apocrypha.xlsx')
# Example usage
df_display(df, head=10, tail=10)


# In[4]:


cols=counts_data.columns[1:]


# In[26]:


for i,c1 in enumerate(cols):

    if c1=='Ilan':
        y=array(counts_data[c1])-array(counts_data['Gospels']) 
    else:
        y=array(counts_data[c1])
    cy=np.cumsum(y)/sum(y)*100
    plot(cy,'-o',label=c1,ms=3,lw=1)

legend()
xlabel('Name Number')
ylabel('Cumulative Fraction')
#xlim([0,100])


# In[10]:


S=df.to_markdown()


# In[24]:


lines=S.split("\n")
print("\n".join(lines[:12]))
print("| ... |      ...        |    ... |       ... |        ... |         ... |")
print("\n".join(lines[-10:]))


# In[ ]:




