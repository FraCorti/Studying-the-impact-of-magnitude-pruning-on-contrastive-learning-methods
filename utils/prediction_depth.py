import glob
import math
import os

import sklearn
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import joblib
import matplotlib.lines as mlines


class PredictionDepth:
    def __init__(self, method, model_sparsity, model_id, model_depth=16, k=30, serialize=False, deserialize=True,
                 root_path='{}/prediction_depth',
                 dataset_name='CIFAR10', encoder_name='WideResNet', pruning_method='GMP'):
        self.knn_classifiers = []
        self.hidden_representations = [[] for _ in range(model_depth)]
        self.model_depth = model_depth
        self.serialize = serialize
        self.method = method
        self.model_sparsity = model_sparsity
        self.model_id = model_id
        self.root_path = root_path.format(os.getcwd())
        self.dataset_name = dataset_name
        self.encoder_name = encoder_name
        self.pruning_method = pruning_method
        self.deserialized = False

        if deserialize and self.__deserialize_knns():
            return

        for i in range(model_depth):
            self.knn_classifiers.append(KNeighborsClassifier(n_neighbors=k))

    def __initialize_folders(self):
        if not os.path.exists(self.root_path + '/{}'.format(self.method)):
            print("Creating folder: {}".format(self.root_path + '/{}'.format(self.method)))
            os.mkdir(self.root_path + '/{}'.format(self.method))
        if not os.path.exists(self.root_path + '/{}/{}'.format(self.method, self.dataset_name)):
            print("Creating folder: {}".format(self.root_path + '/{}/{}'.format(self.method, self.dataset_name)))
            os.mkdir(self.root_path + '/{}/{}'.format(self.method, self.dataset_name))
        if not os.path.exists(self.root_path + '/{}/{}/{}'.format(self.method, self.dataset_name, self.encoder_name)):
            print("Creating folder: {}".format(
                self.root_path + '/{}/{}/{}'.format(self.method, self.dataset_name, self.encoder_name)))
            os.mkdir(self.root_path + '/{}/{}/{}'.format(self.method, self.dataset_name, self.encoder_name))
        if not os.path.exists(self.root_path + '/{}/{}/{}/{}'.format(self.method, self.dataset_name, self.encoder_name,
                                                                     self.pruning_method)):
            print("Creating folder: {}".format(
                self.root_path + '/{}/{}/{}/{}'.format(self.method, self.dataset_name, self.encoder_name,
                                                       self.pruning_method)))
            os.mkdir(self.root_path + '/{}/{}/{}/{}'.format(self.method, self.dataset_name, self.encoder_name,
                                                            self.pruning_method))
        if not os.path.exists(
                self.root_path + '/{}/{}/{}/{}/{}'.format(self.method, self.dataset_name, self.encoder_name,
                                                          self.pruning_method, self.model_sparsity)):
            print("Creating folder: {}".format(
                self.root_path + '/{}/{}/{}/{}/{}'.format(self.method, self.dataset_name, self.encoder_name,
                                                          self.pruning_method, self.model_sparsity)))
            os.mkdir(self.root_path + '/{}/{}/{}/{}/{}'.format(self.method, self.dataset_name, self.encoder_name,
                                                               self.pruning_method, self.model_sparsity))
        if not os.path.exists(
                self.root_path + '/{}/{}/{}/{}/{}/{}'.format(self.method, self.dataset_name, self.encoder_name,
                                                             self.pruning_method, self.model_sparsity, self.model_id)):
            print("Creating folder: {}".format(
                self.root_path + '/{}/{}/{}/{}/{}/{}'.format(self.method, self.dataset_name, self.encoder_name,
                                                             self.pruning_method, self.model_sparsity, self.model_id)))
            os.mkdir(self.root_path + '/{}/{}/{}/{}/{}/{}'.format(self.method, self.dataset_name, self.encoder_name,
                                                                  self.pruning_method, self.model_sparsity,
                                                                  self.model_id))

    def __serialize_knns(self):
        self.__initialize_folders()
        knn_classifier_number = 1
        print("Storing Knn_Classifiers! model number: {} method: {} pruning method: {} model sparsity: {}".format(
            self.model_id, self.method, self.pruning_method, self.model_sparsity))
        for knn_classifier in self.knn_classifiers:
            joblib.dump(knn_classifier, '{}/knn_classifier_{}.pkl'.format(
                self.root_path + '/{}/{}/{}/{}/{}/{}'.format(self.method, self.dataset_name, self.encoder_name,
                                                             self.pruning_method, self.model_sparsity,
                                                             self.model_id), knn_classifier_number))
            knn_classifier_number += 1
        print("Stored knn_classifiers")

    def __deserialize_knns(self):
        for knn_classifier_number in range(1, self.model_depth + 1, 1):
            for knn_path in glob.glob('{}/knn_classifier_{}.pkl'.format(
                    self.root_path + '/{}/{}/{}/{}/{}/{}'.format(self.method, self.dataset_name, self.encoder_name,
                                                                 self.pruning_method, self.model_sparsity,
                                                                 self.model_id), knn_classifier_number)):
                self.knn_classifiers.append(joblib.load(knn_path))

        if len(self.knn_classifiers) == self.model_depth:
            print("Loaded Prediction-Depth model".format(len(self.knn_classifiers)))
            self.deserialized = True
            return True
        else:
            print(
                "KNN-Classifiers training method: {} pruning method: {} sparsity: {} model number: {} need to be trained!".format(
                    self.method, self.pruning_method, self.model_sparsity, self.model_id))
            self.knn_classifiers.clear()
            return False

    def pre_trained(self):
        return self.deserialized

    def fit_knns(self, debug=False):
        print("Start fitting the KNNs")
        for i in range(len(self.knn_classifiers)):
            if debug:
                print("Dataframe created from hidden representations shape: {}".format(
                    np.stack(self.hidden_representations[i]).shape))

            df = pd.DataFrame(
                np.stack(self.hidden_representations[i]),
                dtype=float)

            labels = df.pop(df.columns[-1])
            labels = labels.astype(int)

            if debug:
                print("Dataframe to be fitted from hidden representations shape: {}".format(
                    df.shape))
                print("Labels shape: {}".format(labels.shape))
            self.knn_classifiers[i].fit(df, labels)
        print("KNNs fitted")

        if self.serialize:
            self.__serialize_knns()

    def inference_knn(self, sample_hidden_representations, sample_class, batch_size=1):
        prediction_depth = np.zeros(shape=(batch_size, 1), dtype=int)
        sample_class_labels = sample_class.cpu().detach().numpy()

        for layer_depth in range(len(self.knn_classifiers)):
            predicted_labels = self.knn_classifiers[layer_depth].predict(
                sample_hidden_representations[layer_depth].cpu().detach().numpy())

            for sample_position in range(len(sample_class_labels)):
                if predicted_labels[sample_position] != sample_class_labels[sample_position]:
                    prediction_depth[sample_position] = layer_depth

        return np.add(prediction_depth, 1)

    def add_hidden_representations(self, hidden_features_representations, batch_size, labels, debug=False):

        labels = labels.cpu().detach().numpy()

        for i in range(len(hidden_features_representations)):
            hidden_representation_layer = hidden_features_representations[i].cpu().detach().numpy()
            for sample_number in range(batch_size):
                self.hidden_representations[i].append(
                    np.append(hidden_representation_layer[sample_number], labels[sample_number]))

        if debug:
            for i in range(len(hidden_features_representations)):
                print("List number: {} Shape first item: {} Items in the list: {}".format(i,
                                                                                          self.hidden_representations[
                                                                                              i][
                                                                                              0].shape, len(
                        self.hidden_representations[i])))


def plot_prediction_depth_q_score(prediction_depth_PIEs_pruned, q_score_PIEs_pruned, prediction_depth_notPIEs_pruned,
                                  q_score_notPIEs_pruned, models_pruning, pruning_method, method,
                                  models_number_Q_score, models_number_Prediction_Depth,
                                  path='{}/pies/cifar10/{}/prediction_depth/Prediction_Depth_QScore_PIEs_{}_{}_{}sparsity.png'):
    plt.clf()
    plt.scatter(prediction_depth_notPIEs_pruned, q_score_notPIEs_pruned,
                color="green", s=5)
    plt.scatter(prediction_depth_PIEs_pruned, q_score_PIEs_pruned, color="red",
                s=5)
    not_pies_legend = mlines.Line2D([], [], color='green', marker='o',
                                    markersize=6, label='not PIEs', linestyle='None')
    pies_legend = mlines.Line2D([], [], color='red', marker='o',
                                markersize=6, label='PIEs', linestyle='None')

    plt.grid(True)
    plt.legend(handles=[not_pies_legend, pies_legend], loc='upper left')
    if method == 'supervised':
        plt.xlabel(
            'Average Prediction Depth {} {} encoders {} pruned'.format('Supervised', models_number_Prediction_Depth,
                                                                       models_pruning))
        plt.ylabel(
            'Average Q-Score {} {} encoders {} pruned'.format('Supervised', models_number_Q_score, models_pruning))
    else:
        plt.xlabel('Average Prediction Depth {} {} encoders {} pruned'.format(method, models_number_Prediction_Depth,
                                                                              models_pruning))
        plt.ylabel('Average Q-Score {} {} encoders {} pruned'.format(method, models_number_Q_score, models_pruning))

    if method == 'supervised':
        plt.title('Prediction Depth - QScore {} PIEs {} pruning {} sparsity'.format('Supervised', pruning_method,
                                                                                    models_pruning), fontsize=14)
    else:
        plt.title(
            'Prediction Depth - QScore {} PIEs {} pruning {} sparsity'.format(method, pruning_method, models_pruning),
            fontsize=14)
    plt.savefig(path.format(os.getcwd(), method, method, pruning_method, models_pruning))





def plot_loaded_prediction_depth(models_number_Prediction_Depth, models_pruning, pruning_method,
                                 method, model_depth=16,
                                 path='{}/pies/cifar10/{}/prediction_depth/Prediction_Depth_PIEs_{}_{}_{}sparsity_{}knn{}'):
    if os.path.exists(
            path.format(os.getcwd(), method, method, pruning_method, models_pruning, models_number_Prediction_Depth, '.npz')):
        with np.load(path.format(os.getcwd(), method, method, pruning_method, models_pruning, models_number_Prediction_Depth,
                                 '.npz')) as data:
            print("File {} loaded".format(
                path.format(os.getcwd(), method, method, pruning_method, models_pruning, models_number_Prediction_Depth, '.npz')))

            plt.clf()
            plt.grid(True)
            range_line = np.linspace(0, model_depth, model_depth)

            prediction_depth_notPIEs = data['prediction_depth_notPIEs']
            prediction_depth_PIEs = data['prediction_depth_PIEs']
            prediction_depth_notPIEs = np.asarray(prediction_depth_notPIEs)
            prediction_depth_PIEs = np.asarray(prediction_depth_PIEs)

            print('Prediction Depth {} {} pruning {} sparsity prediction_depth_models_number {}'.format(method,
                                                                                                        pruning_method,
                                                                                                        models_pruning,
                                                                                                        models_number_Prediction_Depth))
            #mean_PIEs_not_pruned, mean_PIEs_pruned = prediction_depth_PIEs.mean(axis=0)
            #mean_notPIEs_not_pruned, mean_notPIEs_not_pruned = prediction_depth_notPIEs.mean(axis=0)

            #            print(
            #    "Average Prediction Depth PIEs NOT-pruned: {} Average Prediction Depth PIEs pruned: {} Average Prediction Depth NOT-PIEs NOT-pruned: {} Average Prediction Depth NOT-PIEs pruned: {}".format(
            #        mean_PIEs_not_pruned, mean_PIEs_pruned, mean_notPIEs_not_pruned, mean_notPIEs_not_pruned))

            std_PIEs_not_pruned, std_PIEs_pruned = prediction_depth_PIEs.std(axis=0)
            std_notPIEs_not_pruned, std_notPIEs_not_pruned = prediction_depth_notPIEs.std(axis=0)

            print(
                "Standard deviation Prediction Depth PIEs NOT-pruned: {} Standard deviation Prediction Depth PIEs pruned: {} Standard deviation Prediction Depth NOT-PIEs NOT-pruned: {} Standard deviation Prediction Depth NOT-PIEs pruned: {}".format(
                    std_PIEs_not_pruned, std_PIEs_pruned, std_notPIEs_not_pruned, std_notPIEs_not_pruned))


            # print("Shape prediction_depth_notPIEs {}".format(prediction_depth_notPIEs.shape))
            plt.scatter(x=prediction_depth_notPIEs[:, 0] + 1, y=prediction_depth_notPIEs[:, 1] +1, color="green", s=3)


            """
            
            x_copy = prediction_depth_PIEs[:, 0].copy()
            y_copy = prediction_depth_PIEs[:, 1].copy()


            xy = np.vstack([x_copy[:,0], y_copy[:,0]])
            z = gaussian_kde(xy)(xy)
            z = preprocessing.maxabs_scale(z, axis=0, copy=True)

            # Sort the points by density, so that the densest points are plotted last
            idx = z.argsort()
            x, y, z = x_copy[:,0][idx], y_copy[:,0][idx], z[idx]
            #fig, ax = plt.subplots()
            plt.scatter(x +1, y+1, c=z, s=12)
            """
            plt.plot(range_line, range_line, 'k-', linewidth=1)  # straight line
            #plt.colorbar()

            """
            #
            #plt.scatter_density
            
            fig = plt.figure()
            #plt.grid(True)
            ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
            x_copy = np.add(x_copy[:,0], 1).tolist()
            y_copy = np.add(y_copy[:, 0], 1).tolist()
            print(x_copy)
            print(y_copy)
            density = ax.scatter_density(x=x_copy, y=y_copy, cmap=white_viridis)
            #fig.plot(range_line, range_line, 'k-', linewidth=1)
            #fig.add_subplot(range_line, range_line, 'k-', linewidth=1)
            fig.colorbar(density, label='Number of points per pixel')
            """
            plt.scatter(x=prediction_depth_PIEs[:, 0] + 1, y=prediction_depth_PIEs[:, 1] + 1, color="red", s=3)
            #ideal_ratio = mlines.Line2D([], [], color='black', marker='_',
            #                            markersize=6, label='Ideal ratio')
            #plt.legend(handles=[ideal_ratio], loc='upper left')
            """
            #fig, ax = plt.subplots()

            #fig = plt.figure()
            #ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
            #density = ax.scatter_density(prediction_depth_PIEs[:, 0], prediction_depth_PIEs[:, 1], cmap=white_viridis)
            #fig.colorbar(density, label='Number of points per pixel')
            #ax.scatter(x, y, c=z, s=50)
                        for i in range(len(prediction_depth_PIEs)):
                counter = 5
                for inner in range(len(prediction_depth_PIEs)):
                    if i != inner and prediction_depth_PIEs[inner][0] == prediction_depth_PIEs[i][0] and prediction_depth_PIEs[inner][1] == prediction_depth_PIEs[i][1]:
                        counter += 1

            """
            

            """
            #print(x_copy.shape)
            #print(y_copy.shape)
            #print(type(x_copy))
            #print(type(y_copy))
            plt.hist2d(x=x_copy[:,0],  y=y_copy[:,0],  bins=60, cmap='Reds')
            cb = plt.colorbar()
            cb.set_label('counts in bin')
            """
            
            #plt.scatter(x=prediction_depth_PIEs[:, 0], y=prediction_depth_PIEs[:, 1], cmap=, s=5)


            not_pies_legend = mlines.Line2D([], [], color='green', marker='o',
                                            markersize=6, label='not PIEs', linestyle='None')
            pies_legend = mlines.Line2D([], [], color='red', marker='o',
                                        markersize=6, label='PIEs', linestyle='None')
            ideal_ratio = mlines.Line2D([], [], color='black', marker='_',
                                        markersize=6, label='Ideal ratio')
            

            plt.legend(handles=[not_pies_legend, pies_legend, ideal_ratio], loc='upper left')  #
            
            
            
            if method == 'supervised':
                plt.xlabel(
                    'Prediction Depth {} {} encoders NOT pruned'.format('Supervised', models_number_Prediction_Depth))
                plt.ylabel(
                    'Prediction Depth {} {} encoders {} pruned'.format('Supervised', models_number_Prediction_Depth,
                                                                       models_pruning))
            else:
                plt.xlabel('Prediction Depth {} {} encoders NOT pruned'.format(method, models_number_Prediction_Depth))
                plt.ylabel('Prediction Depth {} {} encoders {} pruned'.format(method, models_number_Prediction_Depth,
                                                                              models_pruning))
            
            """
            if method == 'supervised':
                plt.title('Prediction Depth {} {} pruning {} sparsity'.format('Supervised', pruning_method,
                                                                              models_pruning), fontsize=14)
            else:
                plt.title(
                    'Prediction Depth {} {} pruning {} sparsity'.format(method, pruning_method, models_pruning),
                    fontsize=14)
            """
            plt.savefig(
                path.format(os.getcwd(), method, method, pruning_method, models_pruning, models_number_Prediction_Depth, '.png'))

    else:
        print("File {} not found!".format(
            path.format(os.getcwd(), method, method, pruning_method, models_pruning, models_number_Prediction_Depth, '.npz')))


def plot_prediction_depth_pruned_against_not_pruned(prediction_depth_PIEs, prediction_depth_notPIEs,
                                                    models_number_Prediction_Depth, models_pruning, pruning_method,
                                                    method, model_depth=16, serialize=True,
                                                    path='{}/pies/cifar10/{}/prediction_depth/Prediction_Depth_PIEs_{}_{}_{}sparsity_{}knn{}'):
    plt.clf()
    plt.grid(True)
    x = np.linspace(0, model_depth + 1, model_depth + 1)
    prediction_depth_notPIEs = np.array(prediction_depth_notPIEs)
    # print("Shape prediction_depth_notPIEs {}".format(prediction_depth_notPIEs.shape))
    plt.scatter(x=prediction_depth_notPIEs[:, 0], y=prediction_depth_notPIEs[:, 1], color="green", s=5)
    prediction_depth_PIEs = np.array(prediction_depth_PIEs)
    # print("Shape prediction_depth_PIEs {}".format(prediction_depth_PIEs.shape))
    plt.scatter(x=prediction_depth_PIEs[:, 0], y=prediction_depth_PIEs[:, 1], color="red", s=5)
    not_pies_legend = mlines.Line2D([], [], color='green', marker='o',
                                    markersize=6, label='not PIEs', linestyle='None')
    pies_legend = mlines.Line2D([], [], color='red', marker='o',
                                markersize=6, label='PIEs', linestyle='None')
    ideal_ratio = mlines.Line2D([], [], color='black', marker='_',
                                markersize=6, label='Ideal ratio')
    plt.plot(x, x, 'k-', linewidth=1)  # straight line

    plt.legend(handles=[not_pies_legend, pies_legend, ideal_ratio], loc='upper left')
    if method == 'supervised':
        plt.xlabel(
            'Prediction Depth {} {} encoders NOT pruned'.format('Supervised', models_number_Prediction_Depth))
        plt.ylabel(
            'Prediction Depth {} {} encoders {} pruned'.format('Supervised', models_number_Prediction_Depth,
                                                               models_pruning))
    else:
        plt.xlabel('Prediction Depth {} {} encoders NOT pruned'.format(method, models_number_Prediction_Depth))
        plt.ylabel('Prediction Depth {} {} encoders {} pruned'.format(method, models_number_Prediction_Depth,
                                                                      models_pruning))

    if method == 'supervised':
        plt.title('Prediction Depth {} {} pruning {} sparsity'.format('Supervised', pruning_method,
                                                                      models_pruning), fontsize=14)
    else:
        plt.title(
            'Prediction Depth {} {} pruning {} sparsity'.format(method, pruning_method, models_pruning),
            fontsize=14)
    if serialize or not os.path.exists(
            path.format(os.getcwd(), method, method, pruning_method, models_pruning, models_number_Prediction_Depth, '.npz')):
        np.savez(path.format(os.getcwd(), method, method, pruning_method, models_pruning, models_number_Prediction_Depth, '.npz'),
                 prediction_depth_notPIEs=prediction_depth_notPIEs,
                 prediction_depth_PIEs=prediction_depth_PIEs)
    plt.savefig(path.format(os.getcwd(), method, method, pruning_method, models_pruning, models_number_Prediction_Depth, '.png'))
