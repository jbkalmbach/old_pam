import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
                            'train_neural_net.dat')
    cat_df = pd.DataFrame(train_cat.get_catalog(filename))
    
    cat_df['index'] = cat_df['index'].astype('int')

    input_data = cat_df[['u', 'g', 'r', 'i', 'z', 'y']].values
    # Normalize data
    input_data -= np.mean(input_data, axis=0)
    input_data /= np.std(input_data, axis=0)

    net = Net(6, 20, 1)
    print(net)

    optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
    loss_func = torch.nn.MSELoss()

    for t in range(250):

        batch_size = 10000

        for batch_start in range(0, len(cat_df), batch_size):

            nn_input = input_data[batch_start:
                                  batch_start+batch_size]

            nn_input = torch.tensor(nn_input, dtype=torch.float)

            true_output = cat_df['redshift'].values[batch_start:
                                                    batch_start+batch_size]

            true_output = torch.tensor(true_output, 
                                       dtype=torch.float).reshape(batch_size,1)

            prediction = net(nn_input)

            loss = loss_func(prediction, true_output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if t % 5 == 0:
            print(input_data[2], prediction[2], true_output[2], loss.data)
            print(input_data[5], prediction[5], true_output[5], loss.data)

    fig = plt.figure()
    plt.scatter(true_output.detach().numpy(), 
                prediction.detach().numpy(), s=8, alpha=0.2)
    plt.plot(np.arange(0, 3.5, 0.01), np.arange(0, 3.5, 0.01), ls='--', c='r')
    plt.show()
