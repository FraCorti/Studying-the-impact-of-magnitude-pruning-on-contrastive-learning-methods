import os

import numpy as np
# from sklearn.datasets import load_digits
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
import umap.umap_ as umap
import matplotlib.lines as mlines


# method to call from outside
def plot_loaded_umap(method, models_pruning, pruning_method,
                     path='{}/pies/cifar10/{}/umap/UMAP_{}_{}_sparsity{}_{}_pruning{}'):
    if os.path.exists(path.format(os.getcwd(), method, 'PIEs', method, models_pruning, pruning_method, '.npz')):

        with np.load(path.format(os.getcwd(), method, 'PIEs', method, models_pruning, pruning_method, '.npz')) as data:
            print("File {} loaded".format(path.format(os.getcwd(), method, 'PIEs', method, models_pruning, pruning_method, '.npz')))

            embedding = data['embedding']
            #pies_position = data['pies_position'][0]
            labels = data['labels_predicted']

            """
            embedded_latent_representations_PIEs = embedding[:pies_position - 1, :]
            embedded_latent_representations_notPIEs = embedding[pies_position + 1:, :]
            plt.clf()
            plt.scatter(embedded_latent_representations_notPIEs[:, 0], embedded_latent_representations_notPIEs[:, 1],
                        color="green", s=5)
            plt.scatter(embedded_latent_representations_PIEs[:, 0], embedded_latent_representations_PIEs[:, 1],
                        color="red",
                        s=5)
            not_pies_legend = mlines.Line2D([], [], color='green', marker='o',
                                            markersize=6, label='not PIEs', linestyle='None')
            pies_legend = mlines.Line2D([], [], color='red', marker='o',
                                        markersize=6, label='PIEs', linestyle='None')

            plt.legend(handles=[not_pies_legend, pies_legend], loc='upper left')
            if method == 'supervised':
                plt.title('UMAP {} PIEs projection {} sparsity {} pruning'.format('Supervised', models_pruning,
                                                                                  pruning_method),
                          fontsize=14)
            else:
                plt.title(
                    'UMAP {} PIEs projection {} sparsity {} pruning'.format(method, models_pruning, pruning_method),
                    fontsize=14)
            plt.savefig(path.format(method, 'PIEs', method, models_pruning, pruning_method, '.png'))
            print(
                "New plot path: {}".format(path.format(method, 'PIEs', method, models_pruning, pruning_method, '.png')))
            """

            # plot classes clusterized
            plt.clf()
            plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral', s=5)
            plt.gca().set_aspect('equal', 'datalim')
            plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
            """
                        if method == 'supervised':
                plt.title('{} {} sparsity {}'.format('Supervised', models_pruning,
                                                                  pruning_method),
                          fontsize=14)
            else:
                plt.title(
                    '{} {} sparsity {}'.format(method, models_pruning, pruning_method),
                    fontsize=14)
            """

            plt.savefig(path.format(os.getcwd(), method, 'Classes', method, models_pruning, pruning_method, '.png'))
            print(
                "New plot path: {}".format(
                    path.format(os.getcwd(), method, 'Classes', method, models_pruning, pruning_method, '.png')))
    else:
        print("File {} not found!".format(path.format(os.getcwd(), method, 'PIEs', method, models_pruning, pruning_method, '.npz')))


def plot_umap(latent_representations, models_pruning, labels_predicted, method,
              pruning_method='GMP',
              path='{}/pies/cifar10/{}/umap/UMAP_{}_{}_sparsity{}_{}_pruning{}',
              serialize=False, plot_no_sparsity=False):
    latent_representations = latent_representations[1:, :]
    pies_labels = latent_representations[:, -1]
    latent_representations = latent_representations[:, :-1]
    labels_predicted = np.array(labels_predicted)
    reducer = umap.UMAP()

    print("computing UMAP for {} {} sparsity {} pruning".format(method, models_pruning,
                                                                pruning_method))
    embedding = reducer.fit_transform(latent_representations)

    plt.clf()
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels_predicted, cmap='Spectral', s=5)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))

    if plot_no_sparsity:
        if method == 'supervised':
            plt.title('{} {} sparsity'.format('Supervised', models_pruning),
                      fontsize=14)
        else:
            plt.title('{} {} sparsity'.format(method, models_pruning),
                      fontsize=14)
    elif method == 'supervised':
        plt.title('UMAP {} {} sparsity {} pruning'.format('Supervised', models_pruning,
                                                          pruning_method),
                  fontsize=14)
    else:
        plt.title(
            'UMAP {} {} sparsity {} pruning'.format(method, models_pruning, pruning_method),
            fontsize=14)
    plt.savefig(path.format(os.getcwd(), method, 'Classes', method, models_pruning, pruning_method, '.png'))

    if serialize or plot_no_sparsity:
        if os.path.exists(path.format(os.getcwd(), method, 'PIEs', method, models_pruning, pruning_method, '.npz')):
            print("Removing old {} file".format(
                path.format(os.getcwd(), method, 'PIEs', method, models_pruning, pruning_method, '.npz')))
            os.remove(path.format(os.getcwd(), method, 'PIEs', method, models_pruning, pruning_method, '.npz'))
        print("Storing {}".format(os.getcwd(), path.format(method, 'PIEs', method, models_pruning, pruning_method, '.npz')))
        np.savez(path.format(method, 'PIEs', method, models_pruning, pruning_method, '.npz'), embedding=embedding,
                 labels_predicted=labels_predicted, pies_labels=pies_labels)
