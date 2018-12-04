import os
import sys
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1446)


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


def train_model(train_input, train_true):

    # Normalize data
    train_mean = np.mean(train_input, axis=0)
    train_stdev = np.std(train_input, axis=0)
    train_input -= train_mean
    train_input /= train_stdev

    if use_colors is True:
        net = Net(5, 20, 1)
    else:
        net = Net(6, 20, 1)
    print(net)

    optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
    loss_func = torch.nn.MSELoss()

    for t in range(500):

        batch_size = 10000

        for batch_start in range(0, train_len, batch_size):

            nn_input = train_input[batch_start:
                                   batch_start+batch_size]

            # If last batch is not of batch_size change batch_size
            if len(nn_input) < batch_size:
                batch_size = len(nn_input)

            nn_input = torch.tensor(nn_input, dtype=torch.float)

            true_output = train_true[batch_start: batch_start+batch_size]

            true_output = torch.tensor(true_output, 
                                       dtype=torch.float).reshape(batch_size,
                                                                  1)

            prediction = net(nn_input)

            loss = loss_func(prediction, true_output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (t+1) % 5 == 0:
            print('After %i epochs' % (t+1))
            print(nn_input[2], prediction[2], true_output[2], loss.data)
            print(nn_input[5], prediction[5], true_output[5], loss.data)

    net.train_mean = train_mean
    net.train_stdev = train_stdev

    return net

if __name__ == "__main__":

    cat_suffix = sys.argv[1]

    train_filename = '../data/train_cat_%s.dat' % cat_suffix
    train_df = pd.read_csv(train_filename)
    test_filename = '../data/test_cat_%s.dat' % cat_suffix
    test_df = pd.read_csv(test_filename)

    train_len = len(train_df)
    test_len = len(test_df)

    use_colors = True

    train_input = train_df[['u', 'g', 'r', 'i', 'z', 'y']].values
    test_input = test_df[['u', 'g', 'r', 'i', 'z', 'y']].values
    # To use colors
    if use_colors is True:
        train_input = train_input[:, :-1] - train_input[:, 1:]
        test_input = test_input[:, :-1] - test_input[:, 1:]

    train_true = train_df[['redshift']].values
    test_true = test_df[['redshift']].values
    print('Training set size: %i. Test set size: %i.' % (train_len, 
                                                         test_len))

    net = train_model(train_input, train_true)

    # Normalize test input but with same parameters as training
    test_input -= net.train_mean
    test_input /= net.train_stdev

    # Run on all training data
    nn_input = torch.tensor(train_input, dtype=torch.float)
    prediction = net(nn_input)

    # Run on test data
    net_test_input = torch.tensor(test_input, dtype=torch.float)
    test_output = net(net_test_input)

    # Save outputs
    torch.save(net.state_dict(), '../data/pz_network_%s.pt' % cat_suffix)
    train_results = {'true_z':train_true.reshape(train_len),
                     'photo_z': prediction.detach().numpy().reshape(train_len)}
    train_results_df = pd.DataFrame.from_dict(data=train_results)
    test_results = {'true_z':test_true.reshape(test_len),
                    'photo_z': test_output.detach().numpy().reshape(test_len)}
    test_results_df = pd.DataFrame.from_dict(data=test_results)
    train_results_df.to_csv('../data/train_results_%s.csv' % cat_suffix, index=False)
    test_results_df.to_csv('../data/test_results_%s.csv' % cat_suffix, index=False)
    
