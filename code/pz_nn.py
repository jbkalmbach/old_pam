import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/brycek/sd_card/photo_z_metrics/code/')
from calc_metrics import point_metrics

torch.manual_seed(1446)

def plot_color_color(mag_cat, filename):

    color_cat = mag_cat[:, :-1] - mag_cat[:, 1:]

    fig = plt.figure(figsize=(12,12))
    color_names = ['u-g', 'g-r', 'r-i', 'i-z', 'z-y']

    for color_idx in range(4):

        fig.add_subplot(2,2,color_idx+1)

        plt.hexbin(color_cat[:, color_idx], 
                   color_cat[:, color_idx+1], bins='log')

        plt.xlabel(color_names[color_idx])
        plt.ylabel(color_names[color_idx+1])

    plt.tight_layout()
    plt.savefig(filename)


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

    use_colors = True
    plot_color = True

    train_input = cat_df[['u', 'g', 'r', 'i', 'z', 'y']].values[:train_len]
    test_input = cat_df[['u', 'g', 'r', 'i', 'z', 'y']].values[train_len:]
    # To use colors
    if use_colors is True:
        train_input = train_input[:, :-1] - train_input[:, 1:]
        test_input = test_input[:, :-1] - test_input[:, 1:]

    if plot_color is True:
        plot_color_color(cat_df[['u', 'g', 'r', 'i', 'z', 'y']].values,
                         'full_data_color_color.pdf')

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

    if use_colors is True:
        net = Net(5, 20, 1)
    else:
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
               prediction.detach().numpy(), bins='log', cmap='viridis')
    plt.plot(np.arange(0, 3.5, 0.01), np.arange(0, 3.5, 0.01), ls='--', c='r')
    plt.xlabel('True Z')
    plt.ylabel('Photo Z')
    plt.title('Training Results: %i objects' % train_len)

    fig.add_subplot(2,2,2)

    plt.hexbin(test_true, test_output.detach().numpy(), bins='log',
               cmap='viridis')
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

