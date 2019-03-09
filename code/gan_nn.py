import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import corner
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from copy import deepcopy


class discriminator(nn.Module):

    def __init__(self, X_dim):
        super(discriminator, self).__init__()

        h_dim = 4*X_dim

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


class generator(nn.Module):

    def __init__(self, X_dim):
        super(generator, self).__init__()

        h_dim = 4*X_dim

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


class gan_nn():

    def __init__(self, seed=None, use_colors=True):

        if seed is not None:
            torch.manual_seed(seed)
        self.use_colors = use_colors

    def load_catalog(self, cat_file):

        cat_df = pd.read_csv(cat_file)

        cat_input = cat_df[['u', 'g',
                            'r', 'i', 'z', 'y']].values
        # To use colors
        if self.use_colors is True:
            cat_input = cat_input[:, :-1] - cat_input[:, 1:]

        print(np.shape(cat_input))

        cat_input = np.hstack((cat_df['redshift'].values.reshape(-1,1),
                               cat_input))

        print(np.shape(cat_input))

        print('Catalog size: %i.' % (len(cat_df)))

        return cat_input

    def reset_grad(self, params):
        for p in params:
            if p.grad is not None:
                data = p.grad.data
                p.grad = Variable(data.new().resize_as_(data).zero_())

    def train_gan(self, train_input, n_epochs,
                  mini_batch_size=64):

        Z_dim = np.shape(train_input)[1]
        print(Z_dim)

        d = discriminator(Z_dim)
        g = generator(Z_dim)

        d_optimizer = optim.Adam(d.parameters(), lr=0.0002)
        g_optimizer = optim.Adam(g.parameters(), lr=0.0002)

        good_data = deepcopy(train_input)
        data_mean = np.mean(good_data, axis=0)
        data_std = np.std(good_data, axis=0)
        good_data -= data_mean
        good_data /= data_std
        input_data = torch.tensor(good_data, dtype=torch.float)

        d_params = [x for x in d.parameters()]
        g_params = [x for x in g.parameters()]

        params = d_params + g_params

        ones_label = Variable(torch.ones(mini_batch_size, 1))
        zeros_label = Variable(torch.zeros(mini_batch_size, 1))

        epoch_samples = []

        for epoch_num in range(n_epochs):
            print(epoch_num)
            num_batches = np.ceil(len(input_data)/mini_batch_size)
            input_idx = np.arange(len(input_data))
            np.random.shuffle(input_idx)
            for it in range(int(num_batches)):
                # Sample data
                X = input_data[input_idx[it*mini_batch_size:
                                         (it+1)*mini_batch_size]]
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
                
                self.reset_grad(params)
                
                ### Generator update
                
                z = Variable(torch.randn(mini_batch_size, Z_dim))
                G_sample = g(z)
                D_fake = d(G_sample)
                
                ones_label = Variable(torch.ones(len(z), 1))
                zeros_label = Variable(torch.zeros(len(z), 1))
                
                G_loss = F.binary_cross_entropy(D_fake, ones_label)
                
                G_loss.backward()
                g_optimizer.step()
                
                self.reset_grad(params)
                
                if it % 500 == 0:
                    print('Epoch: %i, Iter: %i, D_loss: %.3f, G_loss: %.3f' %
                          (epoch_num, it, D_loss, G_loss))
                if it == 1500:
                    print(D_fake[:10], D_real[:10])
                    
            z = Variable(torch.randn(25000, Z_dim))
            G_sample = g(z)
            new_sample = G_sample.detach().numpy() * data_std + data_mean
            epoch_samples.append(new_sample)

        limits = [(-0.05, 3.8), (22, 32), (22, 30), (22, 30),
                  (22, 30), (22, 30), (22, 30)]
        labels = ['redshift', 'u', 'g', 'r', 'i', 'z', 'y']
        fig, axes = plt.subplots(7, 7, figsize=(18, 18))
        corner.corner(new_sample, range=limits, labels=labels, fig=fig,
                      label_kwargs={'size': 20}, plot_datapoints=False)
        corner.corner(sparse_train_set.values, color='r',
                      plot_datapoints=False,
                      fig=fig, range=limits, labels=labels)
        for ax in fig.get_axes():  
            ax.tick_params(axis='both', labelsize=20)
        plt.tight_layout()
        #plt.show()
        plt.savefig('gan_test.pdf')

        z = Variable(torch.randn(200000, Z_dim))
        G_sample = g(z)
        new_sample = G_sample.detach().numpy() * data_std + data_mean
        new_df = pd.DataFrame(new_sample, columns=['redshift', 'u', 'g', 'r',
                                                   'i', 'z', 'y'])
        new_df.to_csv('../data/sparse_gan_train.csv', index=False)
