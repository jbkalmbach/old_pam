import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/brycek/sd_card/photo_z_metrics/code/')
from calc_metrics import point_metrics

class data_reader():

    def __init__(self):

        return

    def get_catalog(self, filename):

        cat_df = np.genfromtxt(filename, names=['index', 'redshift',
                                                'u', 'g', 'r', 'i',
                                                'z', 'y', 'g_abs',
                                                'r_abs'])

        return cat_df

class Net(torch.nn.Module):

    # Using this as guide github.com/MorvanZhou/PyTorch-Tutorial/blob/
    # master/tutorial-contents/301_regression.py

    def __init__(self, n_input_features, n_hidden_nodes, n_output):

        super(Net, self).__init__()

        # Set up layers

        self.hidden_1 = torch.nn.Linear(n_input_features, n_hidden_nodes)
        self.hidden_2 = torch.nn.Linear(n_hidden_nodes, n_hidden_nodes)
        self.predict = torch.nn.Linear(n_hidden_nodes, n_output)

    def forward(self, x):

        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        x = self.predict(x)

        return x

if __name__ == "__main__":

    train_cat =  data_reader()
    filename = os.path.join('/home/brycek/sd_card',
                            'data_augment/scripts_bryce/cats',
                            'Euclid_trim_25p2_3p5.dat')
    cat_df = pd.DataFrame(train_cat.get_catalog(filename))
    
    cat_df['index'] = cat_df['index'].astype('int')

    train_len = 500000
    test_len = len(cat_df) - train_len
    train_input = cat_df[['u', 'g', 'r', 'i', 'z', 'y']].values[:train_len]
    test_input = cat_df[['u', 'g', 'r', 'i', 'z', 'y']].values[train_len:]
    train_true = cat_df[['redshift']].values[:train_len]
    test_true = cat_df[['redshift']].values[train_len:]
    print('Training set size: %i. Test set size: %i.' % (train_len, 
                                                         test_len))

    # Normalize data
    train_mean = np.mean(train_input, axis=0)
    train_stdev = np.std(train_input, axis=0)
    train_input -= train_mean
    train_input /= train_stdev

    # Do same for test input but with same parameters as training
    test_input -= train_mean
    test_input /= train_stdev

    net = Net(6, 20, 1)
    print(net)

    optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
    loss_func = torch.nn.MSELoss()

    for t in range(100):

        batch_size = 10000

        for batch_start in range(0, train_len, batch_size):

            nn_input = train_input[batch_start:
                                   batch_start+batch_size]

            nn_input = torch.tensor(nn_input, dtype=torch.float)

            true_output = train_true[batch_start: batch_start+batch_size]

            true_output = torch.tensor(true_output, 
                                       dtype=torch.float).reshape(batch_size,1)

            prediction = net(nn_input)

            loss = loss_func(prediction, true_output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if t % 5 == 0:
            print(nn_input[2], prediction[2], true_output[2], loss.data)
            print(nn_input[5], prediction[5], true_output[5], loss.data)

    # Run on all training data
    nn_input = torch.tensor(train_input, dtype=torch.float)
    prediction = net(nn_input)

    # Run on test data
    net_test_input = torch.tensor(test_input, dtype=torch.float)
    test_output = net(net_test_input)

    # Plot scatter plots
    fig = plt.figure(figsize=(12, 12))

    fig.add_subplot(2,2,1)

    plt.hexbin(train_true, 
               prediction.detach().numpy(), bins='log', cmap='inferno')
    plt.plot(np.arange(0, 3.5, 0.01), np.arange(0, 3.5, 0.01), ls='--', c='r')
    plt.xlabel('True Z')
    plt.ylabel('Photo Z')
    plt.title('Training Results: %i objects' % train_len)

    fig.add_subplot(2,2,2)

    plt.hexbin(test_true, test_output.detach().numpy(), bins='log',
               cmap='inferno')
    plt.plot(np.arange(0, 3.5, 0.01), np.arange(0, 3.5, 0.01), ls='--', c='r')
    plt.xlabel('True Z')
    plt.ylabel('Photo Z')
    plt.title('Test Results: %i objects' % test_len)

    fig.add_subplot(2,2,3)

    pm = point_metrics()
    print(np.shape(test_output.detach().numpy()), np.shape(test_true))
    bias = pm.photo_z_robust_bias(test_output.detach().numpy().reshape(test_len),
                                  test_true.reshape(test_len), 3.5, 15)
    plt.plot(np.linspace(0, 3.5, 15), bias)
    plt.xlabel('True Z')
    plt.ylabel('Robust Bias')

    fig.add_subplot(2,2,4)

    pm = point_metrics()
    print(np.shape(test_output.detach().numpy()), np.shape(test_true))
    stdev_iqr = pm.photo_z_robust_stdev(test_output.detach().numpy().reshape(test_len),
                                        test_true.reshape(test_len), 3.5, 15)
    plt.plot(np.linspace(0, 3.5, 15), stdev_iqr)
    plt.xlabel('True Z')
    plt.ylabel('Robust Standard Deviation')

    plt.tight_layout()
    plt.savefig('pz_results.pdf')
