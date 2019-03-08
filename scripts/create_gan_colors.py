
# coding: utf-8

# In[1]:
import matplotlib as mpl
mpl.use('PDF')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import corner
from copy import deepcopy

import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
import torch.nn.functional as F


# In[2]:


# Load in training set with a k-means group taken out
sparse_train_set = pd.read_csv('../data/train_cat_sparse.dat')


# In[7]:


print(sparse_train_set.head())


# In[12]:


class discriminator(nn.Module):
    
    def __init__(self):
        super(discriminator, self).__init__()
        
        X_dim = 6
        h_dim = 24
        
        self.d = torch.nn.Sequential(
            torch.nn.Linear(X_dim, h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim, 2*h_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(2*h_dim, h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim, 1),
            torch.nn.Sigmoid()
            )
    
    def forward(self, x, c=None):
        
        y = self.d(x)
        return y


# In[13]:


class generator(nn.Module):
    
    def __init__(self):
        super(generator, self).__init__()
        
        X_dim = 6
        h_dim = 24
        
        self.g = torch.nn.Sequential(
            torch.nn.Linear(X_dim, h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim, 2*h_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(2*h_dim, h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim, X_dim)
            )
        
    def forward(self, x, c=None):
        
        y = self.g(x)
        return y


# In[14]:


d = discriminator()
g = generator()


# In[15]:


d_optimizer = optim.Adam(d.parameters(), lr=0.0002)
g_optimizer = optim.Adam(g.parameters(), lr=0.0002)


# In[16]:


good_data = pd.DataFrame()#deepcopy(sparse_train_set.values)
color_labels = ['u-g', 'g-r', 'r-i', 'i-z', 'z-y']
mag_labels = ['u', 'g', 'r', 'i', 'z', 'y']
good_data['redshift'] = sparse_train_set['redshift'].values
for idx, color in list(enumerate(color_labels)):
    good_data[color] = sparse_train_set[mag_labels[idx]].values - sparse_train_set[mag_labels[idx+1]].values
good_data = good_data.values
sparse_copy = deepcopy(good_data)
data_mean = np.mean(good_data, axis=0)
data_std = np.std(good_data, axis=0)
good_data -= data_mean
good_data /= data_std
input_data = torch.tensor(good_data, dtype=torch.float)


# In[17]:


d_params = [x for x in d.parameters()]
g_params = [x for x in g.parameters()]


# In[18]:


params = d_params + g_params
def reset_grad():
    for p in params:
        if p.grad is not None:
            data = p.grad.data
            p.grad = Variable(data.new().resize_as_(data).zero_())


# In[ ]:


mini_batch_size = 64
Z_dim = 6

ones_label = Variable(torch.ones(mini_batch_size, 1))
zeros_label = Variable(torch.zeros(mini_batch_size, 1))

epoch_samples = []

torch.manual_seed(1641)

for epoch_num in range(500):
    print(epoch_num)
    num_batches = np.ceil(len(input_data)/mini_batch_size)
    input_idx = np.arange(len(input_data))
    np.random.shuffle(input_idx)
    for it in range(int(num_batches)):
        # Sample data
        X = input_data[input_idx[it*mini_batch_size:(it+1)*mini_batch_size]]
        z = Variable(torch.randn(len(X), Z_dim))
        #X, _ = mnist.train.next_batch(mb_size)
        X = Variable(X)
        
        ones_label = Variable(torch.ones(len(X), 1))
        zeros_label = Variable(torch.zeros(len(X), 1))
        
        ### Discriminator "f-l-b" update
        
        for itd in range(1):
    
            G_sample = g(z)
            D_real = d(X)
            D_fake = d(G_sample)

            D_loss_real = F.binary_cross_entropy(D_real, ones_label)
            D_loss_fake = F.binary_cross_entropy(D_fake, zeros_label)
            D_loss = D_loss_real + D_loss_fake

            D_loss.backward()
            d_optimizer.step()
        
        reset_grad()
        
        ### Generator update
        
        z = Variable(torch.randn(mini_batch_size, Z_dim))
        G_sample = g(z)
        D_fake = d(G_sample)
        
        ones_label = Variable(torch.ones(len(z), 1))
        zeros_label = Variable(torch.zeros(len(z), 1))
        
        G_loss = F.binary_cross_entropy(D_fake, ones_label)
        
        G_loss.backward()
        g_optimizer.step()
        
        reset_grad()
        
        if it % 500 == 0:
            print('Epoch: %i, Iter: %i, D_loss: %.3f, G_loss: %.3f' % (epoch_num, it, D_loss, G_loss))
        if it == 1500:
            print(D_fake[:10], D_real[:10])
            
    z = Variable(torch.randn(25000, Z_dim))
    G_sample = g(z)
    new_sample = G_sample.detach().numpy() * data_std + data_mean
    epoch_samples.append(new_sample)

limits = [(-0.05, 3.8), (-1, 7), (-1, 4), (-1, 2), (-1, 2), (-1, 1)]
labels = ['redshift', 'u-g', 'g-r', 'r-i', 'i-z', 'z-y']

fig, axes = plt.subplots(6, 6, figsize=(18, 18))
corner.corner(new_sample, range=limits, labels=labels, fig=fig, label_kwargs={'size':20}, plot_datapoints=False)
corner.corner(sparse_copy, color='r', plot_datapoints=False,
              fig=fig, range=limits, labels=labels)
for ax in fig.get_axes():  
      ax.tick_params(axis='both', labelsize=20)
plt.tight_layout()
#plt.show()
plt.savefig('gan_test.pdf')


z = Variable(torch.randn(200000, Z_dim))
G_sample = g(z)
new_sample = G_sample.detach().numpy() * data_std + data_mean

sim_cat_df = pd.DataFrame()
sim_cat_df['redshift'] = new_sample[:,0]
for i in range(6):
    band = mag_labels[i]
    # Going to use colors anyway so just give u-band standard magnitude
    if band == 'u':
        sim_cat_df['%s' % band] = 25.#final_mags_full[keep, i]
    else:
        sim_cat_df['%s' % band] = sim_cat_df['%s' % mag_labels[i-1]] - new_sample[:, i]
sim_cat_df.to_csv('../data/sparse_gan_colors_train.csv', index=False)
