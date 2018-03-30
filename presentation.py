
# coding: utf-8

# In[41]:


get_ipython().run_line_magic('reset', '-f')
import matplotlib as mpl
#mpl.use('nbAgg')
mpl.rcParams['figure.figsize'] = (9,7)

from matplotlib import pyplot as plt
#%matplotlib notebook
import ipywidgets


# In[42]:


from src.beads import *
canvas = Canvas(-20,20,100)


# In[43]:


b = Beads(5,3,1,10)
print(b,len(b.components))


# In[35]:


print(len(b.components))


# In[37]:


a = Donut(-5,3,1)
b = Beads(5,3,1,10)
print(b,len(b.components))
canvas.clear()
canvas+=a
canvas+=b

fig, ax = plt.subplots(1,1)
canvas.plot(ax, nlines = 500)


# In[8]:


bs = [8, 4, 2]
sigmas = [4,1,0.1]
cs = [15, 15, 15]
colors = ['Greens', 'Blues', 'Reds']

sources = [Beads(0,b,sigma,c) for (b,sigma,c) in zip(bs,sigmas,cs)]


# In[11]:


print(len(b.components))


# In[9]:


fig, ax = plt.subplots(1,1)
for (s, color) in zip(sources, colors):
    print(s, len(s.components), color)
    s.plot(canvas, ax, color, nlines=10)


# In[10]:


print(len(sources[0].components))


# In[ ]:


plt.figure()
plt.imshow(canvas.canvas)


# In[ ]:


mix = None
for source in sources:
    mix *= source
plt.figure()
mix.plot(canvas, colors = "Greens", nlines=40)
plt.show()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)

for (s, color) in zip(sources, colors):
    s.plot(canvas, plt.gca(), color, nlines=10)

def onclick(event):
    x = event.xdata+1j*event.ydata
    post = [s.post(mix,x) for s in sources]

    ax.clear()
    for (s, color) in zip(sources, colors):
        s.plot(canvas, ax, "Greys", nlines=10, alpha=0.1)
    for (s, color) in zip(post, colors):
        s.plot(canvas, ax, color, nlines=10)
    ax.arrow(0, 0, np.real(x), np.imag(x), head_width=1, head_length=1)
    plt.draw()
    
    
cid = fig.canvas.mpl_connect('button_press_event', onclick)


# In[ ]:


sources_LGM = []
for (b,sigma,c) in zip(bs,sigmas,cs)
    
for (s, color) in zip(sources, colors):
    s.plot(canvas, plt.gca(), color, nlines=10)
plt.show()


plt.figure()
mix_LGM.plot(canvas, colors = "Greens", nlines=40)
plt.show()


# # Audio source separation with magnitude priors: the BEADS model 
# 
# ## Antoine Liutkus$^1$, Christian Rohlfing$^2$, Antoine Deleforge$^3$
# 
# $^1$ Zenith team, Inria, University of Montpellier, France<p>
# $^2$ RWTH, Aachen University, Germany<p>
# $^3$ Inria Rennes - Bretagne Atlantique, France<p>
# 
# <div class="inline-block">
#     <img src="figures/zenith.jpg" style="height:3em; margin-top:5em">
# </div>
# <div class="inline-block">
#     <img src ="figures/inria.png" style="height:3em">
# </div>
# <div class="inline-block">
#     <img src="figures/rwth.svg" style="height:3em">
# </div>
# <div class="inline-block">
#     <img src="figures/anr.png" style="height:3em">
# </div>
# </div>
# 

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="darkgrid")
iris = sns.load_dataset("iris")

# Subset the iris dataset by species
setosa = iris.query("species == 'setosa'")
virginica = iris.query("species == 'virginica'")

# Set up the figure
f, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect("equal")

# Draw the two density plots
ax = sns.kdeplot(setosa.sepal_width, setosa.sepal_length,
                 cmap="Reds", shade=True, shade_lowest=False)
ax = sns.kdeplot(virginica.sepal_width, virginica.sepal_length,
                 cmap="Blues", shade=True, shade_lowest=False)

# Add labels to the plot
red = sns.color_palette("Reds")[-2]
blue = sns.color_palette("Blues")[-2]
ax.text(2.5, 8.2, "virginica", size=16, color=blue)
ax.text(3.8, 4.5, "setosa", size=16, color=red)
plt.show()

