import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np


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


class photoz_nn():

    def __init__(self, seed=None, use_colors=True):

        if seed is not None:
            torch.manual_seed(seed)
        self.use_colors = use_colors

    def load_catalog(self, cat_file):

        cat_df = pd.read_csv(cat_file)

        cat_input = cat_df[['u', 'g', 'r', 'i', 'z', 'y']].values
        # To use colors
        if self.use_colors is True:
            cat_input = cat_input[:, :-1] - cat_input[:, 1:]

        cat_true = cat_df[['redshift']].values
        print('Catalog size: %i.' % (len(cat_df)))

        return cat_input, cat_true

    def train_model(self, train_input, train_true):

        # Normalize data
        train_mean = np.mean(train_input, axis=0)
        train_stdev = np.std(train_input, axis=0)
        train_input -= train_mean
        train_input /= train_stdev

        train_len = len(train_input)

        if self.use_colors is True:
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
                                           dtype=torch.float).reshape(
                                                batch_size, 1)

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

    def run_model(self, net, input_data):

        # Normalize test input but with same parameters as training
        input_data -= net.train_mean
        input_data /= net.train_stdev

        # Run on test data
        net_input = torch.tensor(input_data, dtype=torch.float)
        test_output = net(net_input)

        return test_output.detach().numpy()

    def save_model(self, net, filename):

        # Save outputs
        torch.save(net.state_dict(), filename)
