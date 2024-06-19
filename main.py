#Hyperparameters
epochs = 20
update_step = 1
learning_rate = 0.0001
beta_1, beta_2 = [0.9, 0.999]

batch = 32

import numpy as np
import time
import pickle
from ImageConverter import *
from UserInputImage import *

# Only used for dataset
from tensorflow.keras.datasets import mnist

# Used to generated images of nums (purely visualization)
import matplotlib.pyplot as plt

#np.random.seed(1234)

# Importing the dataset
(x_train_raw, y_train), (x_test_raw, y_test) = mnist.load_data()

unwrapped_len = len(x_train_raw[0][0]) * len(x_train_raw[0][1])

# 'Unwrap' the dataset
X_train = x_train_raw.reshape(len(x_train_raw), unwrapped_len)
x_test = x_test_raw.reshape(len(x_test_raw), unwrapped_len)

# Regularize the dataset
X_train = np.divide(X_train, 255)
x_test = np.divide(x_test, 255)

# Make all the pixels white or black
X_train = np.round(X_train + 0.1)
x_test = np.round(x_test + 0.1)

# Shuffle the training set
p = np.random.permutation(len(X_train))
X_train = X_train[p]
y_train = y_train[p]


class model():

    # Initialize the model
    def __init__(self, model_shape, alpha, batch_size):

        self.model_shape = model_shape
        self.batch_size = batch_size

        # For timing purposes
        self.start_time = time.time()

        W = []
        b = []

        # Initialize the weights and biases with He Normal Initialization
        for i in range(len(model_shape) - 1):
            std = np.sqrt(2/model_shape[i])
            W.append(np.random.normal(0, std, size=(model_shape[i], model_shape[i + 1])))
            b.append(np.random.normal(0, std, size=(model_shape[i + 1])))

        self.W = W
        self.b = b
        self.alpha = alpha

        # Create arrays to graph the cost and accuracy over time
        self.cost_arr = [[], []]
        self.accuracy_arr = [[], []]

    # Create a function to display images of the graphs
    def display_digits(self, *args):

        # Initialize the X_set
        X_set = args[0]

        # If you are reading this im sorry
        pred = self.forward_prop(args[0])[1][-1]

        # This rewraps the image
        X_set = X_set.reshape(len(X_set), int(np.sqrt(len(X_set[0]))), int(np.sqrt(len(X_set[0]))))

        if len(args) == 4:

            y_set, start, stop = args[1:]

            # Just create a properly sized square grid of the images
            graph_range = stop - start
            row_size = int(np.ceil(np.sqrt(graph_range + 1)))
            col_size = row_size
            fig, axs = plt.subplots(row_size, col_size)

            # Initialize the variable to start the image plots at
            k = start

            for i in range(row_size):
                for j in range(col_size):
                    axs[i, j].axis('off')
                    if k <= stop:
                        axs[i, j].imshow(X_set[k], cmap='grey')
                        title = "A: " + str(y_set[k]) + " - P: " + str(np.argmax(pred[k]))
                        axs[i, j].set_title(title)
                        k += 1
                    else:
                        break

            # I cant think of a better way to fix this
            axs[row_size - 1, col_size - 1].axis('off')

            # This just makes the numbers not overlap
            fig.tight_layout()

            plt.show()

        elif len(args) == 3:

            y_set, start = args[1:]

            # Just read it
            plt.imshow(X_set[start], cmap='gray')
            title = "P: " + str(np.argmax(pred[start]))
            plt.title(title)
            plt.show()

        elif len(args) == 1:

            if len(args[0]) == 1:

                plt.imshow(X_set[0], cmap='grey')
                title = "P: " + str(np.argmax(pred[0]))
                plt.title(title)
                plt.show()

            else:

                # Just create a properly sized square grid of the images
                graph_range = len(X_set)
                row_size = int(np.ceil(np.sqrt(graph_range)))
                col_size = row_size
                fig, axs = plt.subplots(row_size, col_size)

                # Initialize the variable to start the image plots at
                k = 0

                for i in range(row_size):
                    for j in range(col_size):
                        axs[i, j].axis('off')
                        if k < len(X_set):
                            axs[i, j].imshow(X_set[k], cmap='grey')
                            title = "P: " + str(np.argmax(pred[k]))
                            axs[i, j].set_title(title)
                            k += 1
                        else:
                            break

                # I cant think of a better way to fix this
                axs[row_size - 1, col_size - 1].axis('off')

                # This just makes the numbers not overlap
                fig.tight_layout()

                plt.show()

    # Create a function to graph data (accuracy and cost)
    def graph_data(self):

        plt.plot(self.cost_arr[0], self.cost_arr[1])
        plt.ylabel("cost")
        plt.xlabel("epochs")
        plt.show()

        plt.plot(self.accuracy_arr[0], self.accuracy_arr[1])
        plt.ylabel("test set accuracy")
        plt.xlabel("epochs")
        plt.show()

    # Create a ReLU function
    def ReLU(self, z):
       return np.maximum(0, z)

    # Create a one hot encoding function
    def one_hot(self, arr):
        t = np.zeros((len(arr), self.model_shape[-1]))

        # These lines convert the y array shape (60000, ) [5,1,3 ... 7,2,9] and converts it to a (60000, 10) [[0,0,0,1,0,0,0], ... [0,0,1,0,0,0,0,0,0]]
        rows = np.arange(0, len(t))
        t[rows, arr] = 1

        return t

    # Create softmax
    def softmax(self, z):

        # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
        shiftz = z - np.max(z)
        ez = np.exp(shiftz)

        # I dont know why you have to add keepdims=True but everything breaks if you dont have it
        #print(shiftz[0], ez[0], np.sum(ez, axis=1, keepdims=True)[0])
        a = ez / np.sum(ez, axis=1, keepdims=True)

        return a

    # Find accuracy of a dataset
    def calc_accuracy(self, X_arr, y_arr):

        pred = self.forward_prop(X_arr)[1][-1]
        predicted = np.argmax(pred, axis=1)
        return np.sum(predicted == y_arr) / y_arr.size

    # Forward Propagation function
    def forward_prop(self, X_arr):

        # Define a dense layer
        def dense(A_in, W, b, activation):
            # Just look up matrix multiplication for a more detailed look
            z = np.matmul(A_in, W) + b

            # Apply the activation function
            if activation == "relu":
                A_out = model.ReLU(self, z)
            elif activation == "linear":
                A_out = z

            return z, A_out

        # Weird name just because a_arr is used later and I dont want to confuse them
        a_arr_ret = []
        z_arr_ret = []

        # Run the first dense layer
        za_ret_first = dense(X_arr, self.W[0], self.b[0], 'relu')
        z_arr_ret.append(za_ret_first[0])
        a_arr_ret.append(za_ret_first[1])

        # Just go through each dense layer
        # Subtract 2 because: 1 for array indexing and 1 because the first layer was already done
        for i in range(1, len(self.model_shape) - 1):

            # Apply linear to the final dense layer
            if i == (len(self.model_shape) - 2):

                za_ret = dense(a_arr_ret[i - 1], self.W[i], self.b[i], 'linear')
                z_arr_ret.append(za_ret[0])

                # Apply softmax to the output
                output = model.softmax(self, za_ret[1])
                a_arr_ret.append(output)
            else:
                za_ret = dense(a_arr_ret[i - 1], self.W[i], self.b[i], 'relu')
                z_arr_ret.append(za_ret[0])
                a_arr_ret.append(za_ret[1])

        return z_arr_ret, a_arr_ret

    # Create a cost function
    def compute_cost(self, a):

        # Not much going on here, but if you get errors maybe add some small number to the log
        return -(1/self.current_batch_size) * (np.sum(self.y_one_hot_mb * np.log(a)))

    # Create Back Propogation algorithm
    def back_prop(self):

        self.t += 1

        # print(self.epoch, self.t)

        num_layers = len(self.W)

        # -- Initialize Chain rule variables --

        # Initialize the dJ_dz array with the last term in the first position, it will get reversed later
        dJ_dz = [np.array((1 / self.current_batch_size) * (self.a_arr[-1] - self.y_one_hot_mb))]

        # the dz_da and da_dz are not just inverses (ex: dz2_da1 and da1_dz1)

        # Append W[1] through W[-1] to the dz_da list
        # The asterisk before the a 'unpacks' the list (it removes the outer brackets)
        dz_da = [*self.W[1:]]
        da_dz = []

        # Append X, a[0] through a[-2] to the dz_dw list while properly modifying for the batch size
        # Upper bound is not inclusive so you can just use -1
        # The asterisk before the a 'unpacks' the list (it removes the outer brackets)
        dz_dW = [np.array(self.X_mb), *self.a_arr]

        # dz_db is always just 1 so no need for a list
        dz_db = np.ones(self.current_batch_size)

        # Create the cost function derivative arrays
        dJ_dW = []
        dJ_db = []

        # Initialize the V_dw and V_db terms for Adam
        if self.t == 1:
            self.V_dW = []
            self.V_db = []

            self.S_dW = []
            self.S_db = []

        # Initialize the corrected terms for Adam
        # Not in if statement because .append is used later for simplicity
        V_dW_cor = []
        V_db_cor = []

        S_dW_cor = []
        S_db_cor = []

        # Calculate da_dz, dJ_dz, dJ_dW and dJ_db
        for i in range(num_layers):

            # Because this loop is done in reverse this variable will be necissary for parts
            i_inv = -(i - (num_layers - 1))

            if i != 0:
                # Add terms to the da_dz in reverse order so it matches with the dJ_dz term
                # The * 1 converts the boolean into an integer (at least i think thats why it fixes itself)
                da_dz.append((self.z_arr[i_inv] > 0) * 1)

                dJ_dz_tmp = dJ_dz[i - 1]
                dJ_da_tmp = np.matmul(dJ_dz_tmp, dz_da[i_inv].T)
                dJ_dz_tmp = dJ_da_tmp * da_dz[i - 1]

                dJ_dz.append(dJ_dz_tmp)

            # Calculate the dJ_dW and dJ_db terms
            # Use i_inv for dz_dW because its not one of the reversed arrays
            dJ_dW.append(np.matmul(dz_dW[i_inv].T, dJ_dz[i]))
            dJ_db.append(np.matmul(dJ_dz[i].T, dz_db))

            if self.t == 1:
                # Initialize the V_dw_prev and V_db_prev terms for Adam
                self.V_dW.append(np.zeros_like(self.W[i]))
                self.V_db.append(np.zeros_like(self.b[i]))

                self.S_dW.append(np.zeros_like(self.W[i]))
                self.S_db.append(np.zeros_like(self.b[i]))

        # Reverse the arrays so they're in order
        dJ_dW.reverse()
        dJ_db.reverse()

        epsilon = 1e-8

        for i in range(num_layers):

            # Calculate the S and V terms
            # DO NOT USE THE CORRECTED TERMS
            self.V_dW[i] = (beta_1 * self.V_dW[i] + (1 - beta_1) * dJ_dW[i])
            self.V_db[i] = (beta_1 * self.V_db[i] + (1 - beta_1) * dJ_db[i])

            self.S_dW[i] = (beta_2 * self.S_dW[i] + (1 - beta_2) * (dJ_dW[i] ** 2))
            self.S_db[i] = (beta_2 * self.S_db[i] + (1 - beta_2) * (dJ_db[i] ** 2))

            # Correct the S and V terms
            V_dW_cor.append(self.V_dW[i] / (1 - (beta_1 ** self.t)))
            V_db_cor.append(self.V_db[i] / (1 - (beta_1 ** self.t)))
            S_dW_cor.append(self.S_dW[i] / (1 - (beta_2 ** self.t)))
            S_db_cor.append(self.S_db[i] / (1 - (beta_2 ** self.t)))

            # Update the W and b terms
            self.W[i] = self.W[i] - (self.alpha * (V_dW_cor[i] / (np.sqrt(S_dW_cor[i]) + epsilon)))
            self.b[i] = self.b[i] - (self.alpha * (V_db_cor[i] / (np.sqrt(S_db_cor[i]) + epsilon)))

    # Create a function to collect the epoch loop and variables
    def fit(self, X, y, X_dev, y_dev, epochs):

        self.X = X
        self.y = y
        self.X_dev = X_dev
        self.y_dev = y_dev

        # Initialize m and n
        self.m = len(X)
        self.n = len(X[0])

        # Initialize t for adam optimizer
        self.t = 0

        for epoch in range(1, epochs + 1):

            self.epoch = epoch

            for mini_batch in range(0, self.m, self.batch_size):

                # Handle the case of the dataset not being divisible by the batch size
                upper_bound = np.minimum((self.m - mini_batch), self.batch_size)
                self.current_batch_size = np.minimum((upper_bound), self.batch_size)

                # Prepare the minibatch arrays
                self.X_mb = self.X[mini_batch:(upper_bound + mini_batch)]
                self.y_mb = self.y[mini_batch:(upper_bound + mini_batch)]

                # Run the forward prop algorithm and save the a and z values
                self.z_arr, self.a_arr = self.forward_prop(self.X_mb)

                self.y_one_hot_mb = self.one_hot(y[mini_batch:(upper_bound + mini_batch)])

                # Specially save the final output
                output = self.a_arr[-1]

                if self.t == 0:
                    # Calculate the cost with output
                    cost = self.compute_cost(output)

                    # Save these values for graphs later
                    self.cost_arr[0].append(epoch)
                    self.cost_arr[1].append(cost)

                # Run Backprop
                self.back_prop()

            # Calculate the cost with output
            cost = self.compute_cost(output)

            # Save these values for graphs later
            self.cost_arr[0].append(epoch)
            self.cost_arr[1].append(cost)

            # Print out updates at certains milestones
            if (epoch % update_step == 0):

                print("epoch " + str(epoch) + " / " + str(epochs) + " completed -- Cost = " + str(cost) + " -- Change in cost = " + str(cost - self.cost_arr[1][-(1 + update_step)]))
                train_accuracy = self.calc_accuracy(self.X, self.y)
                print("Train set Accuracy: " + str(train_accuracy))
                test_accuracy = self.calc_accuracy(self.X_dev, self.y_dev)
                print("Test set Accuracy: " + str(test_accuracy))
                self.accuracy_arr[0].append(epoch)
                self.accuracy_arr[1].append(test_accuracy)
                print("Current time elapsed: " + str(time.time() - self.start_time))

    # Create a function to save the weights and biases
    def save(self):

        with open('weights.pkl', 'wb') as weights_file:

            pickle.dump(self.W, weights_file)

        with open('biases.pkl', 'wb') as biases_file:

            pickle.dump(self.b, biases_file)

    # Create a function to read the weights and biases
    def load(self):

        with open('weights.pkl', 'rb') as weights_file:

            W_load = pickle.load(weights_file)

            self.W = W_load

        with open('biases.pkl', 'rb') as biases_file:

            b_load = pickle.load(biases_file)

            self.b = b_load


model1 = model([unwrapped_len, 64, 32, 10], learning_rate, batch)
model1.fit(X_train, y_train, x_test, y_test, epochs)

model1.save()

#model1.load()

print(model1.calc_accuracy(x_test, y_test))

model1.graph_data()
model1.display_digits(X_train, y_train, 0, 15)


while True:
    get_user_image()

    image = np.array([(convert_input_png('num.png'))])
    image = center_image(image)

    model1.display_digits(image)





