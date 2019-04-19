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
        self.hidden_2 = torch.nn.Linear(n_hidden_nodes, 2*n_hidden_nodes)
        self.hidden_3 = torch.nn.Linear(2*n_hidden_nodes, n_hidden_nodes)
        self.predict = torch.nn.Linear(n_hidden_nodes, n_output)
        torch.nn.init.uniform_(self.predict.weight)

        self.train_mean = None
        self.train_stdev = None

    def forward(self, x):

        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        x = F.relu(self.hidden_3(x))
        x = F.relu(self.predict(x))

        return x


class photoz_nn():

    def __init__(self, torch_seed=None,
                 numpy_seed=None, use_colors=True):

        if torch_seed is not None:
            torch.manual_seed(torch_seed)
        if numpy_seed is not None:
            np.random.seed(numpy_seed)
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

    def train_model(self, train_input, train_output, n_epochs, cv_thresh=None,
                    return_error=False):

        shuffled_idx = np.arange(len(train_input))
        np.random.shuffle(shuffled_idx)
        cv_break = int(0.8*len(train_input))  # Save 20% for cross-validation
        train_samples = train_input[shuffled_idx[:cv_break]]
        cv_samples = train_input[shuffled_idx[cv_break:]]
        train_true = train_output[shuffled_idx[:cv_break]]
        cv_true = train_output[shuffled_idx[cv_break:]]

        np.savetxt('cv_test.dat', cv_samples)

        # Normalize data
        train_mean = np.mean(train_samples, axis=0)
        train_stdev = np.std(train_samples, axis=0)
        train_samples -= train_mean
        train_samples /= train_stdev

        # CV data must use same parameters as training
        cv_samples -= train_mean
        cv_samples /= train_stdev

        train_len = len(train_samples)
        cv_len = len(cv_samples)
        print('Training Set Size: %i' % train_len)
        print('Validation Set Size: %i' % cv_len)

        loss_curve = []

        if self.use_colors is True:
            net = Net(5, 20, 1)
        else:
            net = Net(6, 20, 1)
        print(net)

        #optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        loss_func = torch.nn.MSELoss()

        for t in range(n_epochs):

            batch_size = 128
            batch_indices = torch.randperm(train_len).numpy()

            for batch_start in range(0, train_len, batch_size):

                batch_idx = batch_indices[batch_start:batch_start+batch_size]
                nn_input = train_samples[batch_idx]

                # If last batch is not of batch_size change batch_size
                if len(nn_input) < batch_size:
                    batch_size = len(nn_input)

                nn_input = torch.tensor(nn_input, dtype=torch.float)

                true_output = train_true[batch_idx]

                true_output = torch.tensor(true_output,
                                           dtype=torch.float).reshape(
                                                batch_size, 1)

                prediction = net(nn_input)

                loss = loss_func(prediction, true_output)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            nn_input = torch.tensor(cv_samples, dtype=torch.float)
            true_output = torch.tensor(cv_true,
                                       dtype=torch.float).reshape(
                                           len(cv_true), 1)
            prediction = net(nn_input)
            cv_loss = loss_func(prediction, true_output)
            loss_curve.append(cv_loss.data)

            if (t+1) % 1 == 0:
                print('After %i epochs' % (t+1))
                print(loss.data)
                print(nn_input[2], prediction[2], true_output[2], cv_loss.data)
                print(nn_input[5], prediction[5], true_output[5], cv_loss.data)

            if cv_thresh is not None:
                if cv_loss.data <= cv_thresh:
                    break

        net.train_mean = train_mean
        net.train_stdev = train_stdev

        if return_error is True:
            return net, loss_curve
        else:
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
        torch.save({'model_state_dict': net.state_dict(),
                    'model_train_mean': net.train_mean,
                    'model_train_stdev': net.train_stdev}, filename)

    def load_model(self, filename):

        if self.use_colors is True:
            net = Net(5, 20, 1)
        else:
            net = Net(6, 20, 1)

        checkpoint = torch.load(filename)
        net.load_state_dict(checkpoint['model_state_dict'])
        net.train_mean = checkpoint['model_train_mean']
        net.train_stdev = checkpoint['model_train_stdev']

        return net
