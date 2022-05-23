import math
import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchshow as ts
import matplotlib
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import torchvision.transforms.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


def print_z_score_l1_norm_mean_std(z_score_PIEs, z_score_notPIEs, l1_norm_PIEs,
                                   l1_norm_notPIEs, sparsity, method, serialize=False,
                                   pruning_method='GMP'):
    print("------------ L1 norm, Z-Score, mean, standard_deviation Analysis {} pruning: {} ------------".format(method,
                                                                                                                pruning_method))

    mean_l1_norm_PIEs_notPruned, mean_l1_norm_PIEs_Pruned = np.array(l1_norm_PIEs).mean(axis=0)
    mean_l1_norm_notPIEs_notPruned, mean_l1_norm_notPIEs_Pruned = np.array(l1_norm_notPIEs).mean(axis=0)
    mean_l1_norm_not_pruned, mean_l1_norm_pruned = np.vstack((np.array(l1_norm_PIEs), np.array(l1_norm_notPIEs))).mean(
        axis=0)

    std_l1_norm_PIEs_notPruned, std_l1_norm_PIEs_Pruned = np.array(l1_norm_PIEs).std(axis=0)
    std_l1_norm_notPIEs_notPruned, std_l1_norm_notPIEs_Pruned = np.array(l1_norm_notPIEs).std(axis=0)
    std_l1_norm_not_pruned, std_l1_norm_pruned = np.vstack((np.array(l1_norm_PIEs), np.array(l1_norm_notPIEs))).std(
        axis=0)

    print(
        "Sparsity: {} Method: {} mean_l1_norm_not_pruned: {} mean_l1_norm_pruned: {}".format(
            sparsity, method, mean_l1_norm_not_pruned, mean_l1_norm_pruned))
    print(
        "Sparsity: {} Method: {} sstd_l1_norm_not_pruned: {} std_l1_norm_pruned: {}".format(
            sparsity, method, std_l1_norm_not_pruned, std_l1_norm_pruned))

    mean_z_score_PIEs_notPruned, mean_z_score_PIEs_Pruned = np.array(z_score_PIEs).mean(axis=0)
    mean_z_score_notPIEs_notPruned, mean_z_score_notPIEs_Pruned = np.array(z_score_notPIEs).mean(axis=0)
    mean_z_score_not_pruned, mean_z_score_pruned = np.vstack((np.array(z_score_PIEs), np.array(z_score_notPIEs))).mean(
        axis=0)

    std_z_score_PIEs_notPruned, std_z_score_PIEs_Pruned = np.array(z_score_PIEs).std(axis=0)
    std_z_score_notPIEs_notPruned, std_z_score_notPIEs_Pruned = np.array(z_score_notPIEs).std(axis=0)
    std_z_score_not_pruned, std_z_score_pruned = np.vstack((np.array(z_score_PIEs), np.array(z_score_notPIEs))).std(
        axis=0)

    print(
        "Sparsity: {} Method: {} mean_z_score_not_pruned: {} mean_z_score_pruned: {}".format(
            sparsity, method, mean_z_score_not_pruned, mean_z_score_pruned))
    print(
        "Sparsity: {} Method: {} std_z_score_not_pruned: {} std_z_score_pruned: {}".format(
            sparsity, method, std_z_score_not_pruned, std_z_score_pruned))

    plot_z_scores(z_score_notPIEs=z_score_notPIEs, z_score_PIEs=z_score_PIEs, sparsity=sparsity, method=method,
                  pruning_method=pruning_method, serialize=serialize)
    plot_l1_norm(l1_norm_notPIEs=l1_norm_notPIEs, l1_norm_PIEs=l1_norm_PIEs, sparsity=sparsity, method=method,
                 pruning_method=pruning_method, serialize=serialize)


def plot_z_scores(z_score_notPIEs, z_score_PIEs, sparsity, method,
                  path='{}/pies/cifar10/{}/q_score/Z-SCORE_{}_sparsity{}_{}{}{}',
                  pruning_method='GMP', serialize=True):
    plt.clf()
    for i in range(len(z_score_notPIEs)):
        plt.scatter(z_score_notPIEs[i][0], z_score_notPIEs[i][1], color="green", s=5)
    for i in range(len(z_score_PIEs)):
        plt.scatter(z_score_PIEs[i][0], z_score_PIEs[i][1], color="red", s=5)

    not_pies_legend = mlines.Line2D([], [], color='green', marker='o',
                                    markersize=6, label='not PIEs', linestyle='None')
    pies_legend = mlines.Line2D([], [], color='red', marker='o',
                                markersize=6, label='PIEs', linestyle='None')

    plt.legend(handles=[not_pies_legend, pies_legend], loc='upper left')
    if method == 'supervised':
        plt.title('{} Z-Score {} {} pruning'.format(pruning_method, 'Supervised', sparsity), fontsize=14)
    else:
        plt.title('{} Z-Score {} {} pruning'.format(pruning_method, method, sparsity), fontsize=14)

    plt.grid(True)
    plt.xlabel('Average Z-Score encoders NOT pruned')
    plt.ylabel('Average Z-Score encoders {} pruned'.format(sparsity))
    plt.savefig(path.format(os.getcwd(), method, pruning_method, sparsity, method, ""))

    # plot with best ratio
    plt.clf()

    if method == 'supervised':
        x = np.linspace(0, 9, 3)
    else:
        x = np.linspace(0, 8, 3)
    not_pies_legend = mlines.Line2D([], [], color='green', marker='o',
                                    markersize=6, label='not PIEs', linestyle='None')
    pies_legend = mlines.Line2D([], [], color='red', marker='o',
                                markersize=6, label='PIEs', linestyle='None')
    ideal_ratio = mlines.Line2D([], [], color='black', marker='_',
                                markersize=6, label='Ideal ratio')
    plt.legend(handles=[not_pies_legend, pies_legend, ideal_ratio], loc='upper left')
    plt.plot(x, x, 'k-', linewidth=1)  # straight line

    for i in range(len(z_score_notPIEs)):
        plt.scatter(z_score_notPIEs[i][0], z_score_notPIEs[i][1], color="green", s=5)
    for i in range(len(z_score_PIEs)):
        plt.scatter(z_score_PIEs[i][0], z_score_PIEs[i][1], color="red", s=5)

    if method == 'supervised':
        plt.title('{} Z-Score {} {} pruning'.format(pruning_method, 'Supervised', sparsity), fontsize=14)
    else:
        plt.title('{} Z-Score {} {} pruning'.format(pruning_method, method, sparsity), fontsize=14)

    plt.grid(True)
    # plt.legend(["not PIEs", "PIEs"], loc="upper left")
    plt.xlabel('Average Z-Score encoders NOT pruned')
    plt.ylabel('Average Z-Score encoders {} pruned'.format(sparsity))
    plt.savefig(path.format(method, pruning_method, sparsity, method, "_wb", '.png'))

    if serialize:
        if os.path.exists(path.format(method, pruning_method, sparsity, method, "", '.npz')):
            print("Removing old{} file".format(path.format(method, pruning_method, sparsity, method, "", '.npz')))
            os.remove(path.format(os.getcwd(), method, pruning_method, sparsity, method, "", '.npz'))
        print("Storing {}".format(path.format(method, pruning_method, sparsity, method, "", '.npz')))
        np.savez(path.format(os.getcwd(), method, pruning_method, sparsity, method, "", '.npz'),
                 z_score_PIEs=np.array(z_score_PIEs),
                 z_score_notPIEs=np.array(z_score_notPIEs))


def plot_l1_norm(l1_norm_notPIEs, l1_norm_PIEs, sparsity, method,
                 path='{}/pies/cifar10/{}/q_score/L1_NORM_{}_sparsity{}_{}{}{}',
                 pruning_method='GMP', serialize=True):
    plt.clf()

    for i in range(len(l1_norm_notPIEs)):
        plt.scatter(l1_norm_notPIEs[i][0], l1_norm_notPIEs[i][1], color="green", s=5)
    for i in range(len(l1_norm_PIEs)):
        plt.scatter(l1_norm_PIEs[i][0], l1_norm_PIEs[i][1], color="red", s=5)

    not_pies_legend = mlines.Line2D([], [], color='green', marker='o',
                                    markersize=6, label='not PIEs', linestyle='None')
    pies_legend = mlines.Line2D([], [], color='red', marker='o',
                                markersize=6, label='PIEs', linestyle='None')

    plt.legend(handles=[not_pies_legend, pies_legend], loc='upper left')
    if method == 'supervised':
        plt.title('{} L1-norm {} {} pruning'.format(pruning_method, 'Supervised', sparsity), fontsize=14)
    else:
        plt.title('L1-norm {} {} pruning'.format(pruning_method, method, sparsity), fontsize=14)

    plt.grid(True)
    plt.xlabel('Average L1-norm encoders NOT pruned')
    plt.ylabel('Average L1-norm encoders {} pruned'.format(sparsity))
    plt.savefig(path.format(os.getcwd(), method, pruning_method, sparsity, method, ""))

    # plot with best ratio
    plt.clf()
    if method == 'supervised':
        x = np.linspace(0, 6, 3)
    else:
        x = np.linspace(0, 11, 3)

    not_pies_legend = mlines.Line2D([], [], color='green', marker='o',
                                    markersize=6, label='not PIEs', linestyle='None')
    pies_legend = mlines.Line2D([], [], color='red', marker='o',
                                markersize=6, label='PIEs', linestyle='None')
    ideal_ratio = mlines.Line2D([], [], color='black', marker='_',
                                markersize=6, label='Ideal ratio')
    plt.legend(handles=[not_pies_legend, pies_legend, ideal_ratio], loc='upper left')
    plt.plot(x, x, 'k-', linewidth=1)  # straight line

    for i in range(len(l1_norm_notPIEs)):
        plt.scatter(l1_norm_notPIEs[i][0], l1_norm_notPIEs[i][1], color="green", s=5)
    for i in range(len(l1_norm_PIEs)):
        plt.scatter(l1_norm_PIEs[i][0], l1_norm_PIEs[i][1], color="red", s=5)

    if method == 'supervised':
        plt.title('{} L1-norm {} {} pruning'.format(pruning_method, 'Supervised', sparsity), fontsize=14)
    else:
        plt.title('L1-norm {} {} pruning'.format(pruning_method, method, sparsity), fontsize=14)

    plt.grid(True)
    # plt.legend(["not PIEs", "PIEs"], loc="upper left")
    plt.xlabel('Average L1-norm encoders NOT pruned')
    plt.ylabel('Average L1-norm encoders {} pruned'.format(sparsity))
    plt.savefig(path.format(os.getcwd(), method, pruning_method, sparsity, method, "_wb", '.png'))

    if serialize:
        if os.path.exists(path.format(method, pruning_method, sparsity, method, "", '.npz')):
            print("Removing old {} file".format(path.format(method, pruning_method, sparsity, method, "", '.npz')))
            os.remove(path.format(os.getcwd(), method, pruning_method, sparsity, method, "", '.npz'))
        print("Storing {}".format(path.format(method, pruning_method, sparsity, method, "", '.npz')))
        np.savez(path.format(os.getcwd(), method, pruning_method, sparsity, method, "", '.npz'),
                 l1_norm_PIEs=np.array(l1_norm_PIEs),
                 l1_norm_notPIEs=np.array(l1_norm_notPIEs))


def plot_q_score_from_loaded(sparsity, method,
                             path='{}/pies/cifar10/{}/q_score/Q-SCORE_{}_sparsity{}_{}{}{}',
                             pruning_method='GMP'):
    if os.path.exists(path.format(os.getcwd(), method, pruning_method, sparsity, method, "", '.npz')):
        with np.load(path.format(os.getcwd(), method, pruning_method, sparsity, method, "", '.npz')) as data:

            z_score_loaded = np.load("{}/pies/cifar10/{}/q_score/{}".format(os.getcwd(), method,
                                                                          "Z-SCORE_{}_sparsity{}_{}{}.npz".format(
                                                                              pruning_method, sparsity,
                                                                              method, '')))
            print("{}/pies/cifar10/{}/q_score/{}".format(os.getcwd(), method,
                                                                          "L1_NORM_{}_sparsity{}_{}{}.npz".format(
                                                                              pruning_method, sparsity,
                                                                              method, '')))
            l1_norm_loaded = np.load("{}/pies/cifar10/{}/q_score/{}".format(os.getcwd(), method,
                                                                          "L1_NORM_{}_sparsity{}_{}{}.npz".format(
                                                                              pruning_method, sparsity,
                                                                              method, '')))
            
            l1_score_PIEs = l1_norm_loaded['l1_norm_PIEs']
            l1_score_notPIEs = l1_norm_loaded['l1_norm_notPIEs']

            z_score_PIEs = z_score_loaded['z_score_PIEs']
            z_score_notPIEs = z_score_loaded['z_score_notPIEs']
            

            std_l1_score_PIEs_notPruned, std_l1_score_PIEs_Pruned = np.array(l1_score_PIEs).std(axis=0)
            std_l1_score_NOT_PIEs_notPruned, std_l1_score_NOT_PIEs_Pruned = np.array(l1_score_notPIEs).std(axis=0)
            std_l1_score_not_pruned, std_l1_score_pruned = np.vstack(
                (np.array(l1_score_PIEs), np.array(l1_score_notPIEs))).std(axis=0)

            print(
                "Sparsity: {} Std L1-Norm PIEs not pruned: {} Std PIEs pruned: {} Std not PIEs not pruned: {} Std not PIEs pruned: {} Std not pruned: {} Std pruned: {}".format(
                    sparsity, std_l1_score_PIEs_notPruned, std_l1_score_PIEs_Pruned, std_l1_score_NOT_PIEs_notPruned,
                    std_l1_score_NOT_PIEs_Pruned,
                    std_l1_score_not_pruned,
                    std_l1_score_pruned))

            std_z_score_PIEs_notPruned, std_z_score_PIEs_Pruned = np.array(z_score_PIEs).std(axis=0)
            std_z_score_NOT_PIEs_notPruned, std_z_score_NOT_PIEs_Pruned = np.array(z_score_notPIEs).std(axis=0)
            std_z_score_not_pruned, std_z_score_pruned = np.vstack(
                (np.array(z_score_PIEs), np.array(z_score_notPIEs))).std(axis=0)
            print(
                "Sparsity: {} Std Z-Score PIEs not pruned: {} Std PIEs pruned: {} Std not PIEs not pruned: {} Std not PIEs pruned: {} Std not pruned: {} Std pruned: {}".format(
                    sparsity, std_z_score_PIEs_notPruned, std_z_score_PIEs_Pruned, std_z_score_NOT_PIEs_notPruned,
                    std_z_score_NOT_PIEs_Pruned,
                    std_z_score_not_pruned,
                    std_z_score_pruned))

            max = data['max']
            q_score_notPIEs = data['q_score_notPIEs']
            q_score_PIEs = data['q_score_PIEs']

            plt.clf()
            x = np.linspace(0, max, max)

            not_pies_legend = mlines.Line2D([], [], color='green', marker='o',
                                            markersize=6, label='not PIEs', linestyle='None')
            pies_legend = mlines.Line2D([], [], color='red', marker='o',
                                        markersize=6, label='PIEs', linestyle='None')
            ideal_ratio = mlines.Line2D([], [], color='black', marker='_',
                                        markersize=6, label='Ideal ratio')
            plt.legend(handles=[not_pies_legend, pies_legend, ideal_ratio], loc='upper left')
            plt.plot(x, x, 'k-', linewidth=1)  # straight line
            plt.scatter(x=q_score_notPIEs[:, 0], y=q_score_notPIEs[:, 1], color="green", s=3)
            plt.scatter(x=q_score_PIEs[:, 0], y=q_score_PIEs[:, 1], color="red", s=3)

            std_q_score_PIEs_notPruned, std_q_score_PIEs_Pruned = np.array(q_score_PIEs).std(axis=0)
            std_q_score_NOT_PIEs_notPruned, std_q_score_NOT_PIEs_Pruned = np.array(q_score_notPIEs).std(axis=0)
            std_q_score_not_pruned, std_q_score_pruned = np.vstack(
                (np.array(q_score_PIEs), np.array(q_score_notPIEs))).std(axis=0)


            if method == 'supervised':
                plt.title('{} {} sparsity {}'.format('Supervised', sparsity, pruning_method), fontsize=14)
            else:
                plt.title('{} {} sparsity {}'.format(method, sparsity, pruning_method), fontsize=14)

            plt.grid(True)
            # plt.legend(["not PIEs", "PIEs"], loc="upper left")
            plt.xlabel('Average Q-Score encoders NOT pruned')
            plt.ylabel('Average Q-Score encoders {} pruned'.format(sparsity))

            plt.savefig(path.format(os.getcwd(), method, pruning_method, sparsity, method, "", '.png'))


def plot_q_scores(q_score_notPIEs, q_score_PIEs, sparsity, method,
                  path='{}/pies/cifar10/{}/q_score/Q-SCORE_{}_sparsity{}_{}{}{}',
                  pruning_method='GMP', serialize=False):
    print("------------ Q-Score Analysis {} pruning: {} ------------".format(method, pruning_method))
    mean_PIEs_notPruned, mean_PIEs_Pruned = np.array(q_score_PIEs).mean(axis=0)
    mean_NOT_PIEs_notPruned, mean_NOT_PIEs_Pruned = np.array(q_score_notPIEs).mean(axis=0)
    mean_not_pruned, mean_pruned = np.vstack((np.array(q_score_PIEs), np.array(q_score_notPIEs))).mean(axis=0)

    std_PIEs_notPruned, std_PIEs_Pruned = np.array(q_score_PIEs).std(axis=0)
    std_NOT_PIEs_notPruned, std_NOT_PIEs_Pruned = np.array(q_score_notPIEs).std(axis=0)
    std_not_pruned, std_pruned = np.vstack((np.array(q_score_PIEs), np.array(q_score_notPIEs))).std(axis=0)

    var_PIEs_notPruned, var_PIEs_Pruned = np.array(q_score_PIEs).var(axis=0)
    var_NOT_PIEs_notPruned, var_NOT_PIEs_Pruned = np.array(q_score_notPIEs).var(axis=0)
    var_not_pruned, var_pruned = np.vstack((np.array(q_score_PIEs), np.array(q_score_notPIEs))).var(axis=0)

    print(
        "Sparsity: {} Q-SCORE Std not pruned: {} Std pruned: {}".format(
            sparsity, std_not_pruned,
            std_pruned))

    print("Sparsity: {} Method: {} Average Q-Score pruned: {} Average Q-Score not pruned: {}".format(sparsity, method,
                                                                                                     mean_pruned,
                                                                                                     mean_not_pruned))
    plt.clf()
     
    max = (-1, -1)

    for i in range(len(q_score_notPIEs)):
        if q_score_notPIEs[i][0] > max[0] and q_score_notPIEs[i][1] > max[1]:
            max = (q_score_notPIEs[i][0], q_score_notPIEs[i][1])
        plt.scatter(q_score_notPIEs[i][0], q_score_notPIEs[i][1], color="green", s=5)
    for i in range(len(q_score_PIEs)):
        if q_score_PIEs[i][0] > max[0] and q_score_PIEs[i][1] > max[1]:
            max = (q_score_PIEs[i][0], q_score_PIEs[i][1])
        plt.scatter(q_score_PIEs[i][0], q_score_PIEs[i][1], color="red", s=5)

    max = (math.ceil(max[0]), math.ceil(max[1]))
    if max[0] > max[1]:
        max = (max[0], max[0])
    else:
        max = (max[1], max[1])

    not_pies_legend = mlines.Line2D([], [], color='green', marker='o',
                                    markersize=6, label='not PIEs', linestyle='None')
    pies_legend = mlines.Line2D([], [], color='red', marker='o',
                                markersize=6, label='PIEs', linestyle='None')

    plt.legend(handles=[not_pies_legend, pies_legend], loc='upper left')
    if method == 'supervised':
        plt.title('{} Q-Score {} {} pruning'.format(pruning_method, 'Supervised', sparsity), fontsize=14)
    else:
        plt.title('{} Q-Score {} {} pruning'.format(pruning_method, method, sparsity), fontsize=14)
    plt.grid(True)
    plt.xlabel('Average Q-Score encoders NOT pruned')
    plt.ylabel('Average Q-Score encoders {} pruned'.format(sparsity))
    # plt.savefig(path.format(method, pruning_method, sparsity, method, "",'.png'))

    # plot with best ratio
    plt.clf()

    x = np.linspace(0, max[0], max[1])

    not_pies_legend = mlines.Line2D([], [], color='green', marker='o',
                                    markersize=6, label='not PIEs', linestyle='None')
    pies_legend = mlines.Line2D([], [], color='red', marker='o',
                                markersize=6, label='PIEs', linestyle='None')
    ideal_ratio = mlines.Line2D([], [], color='black', marker='_',
                                markersize=6, label='Ideal ratio')
    plt.legend(handles=[not_pies_legend, pies_legend, ideal_ratio], loc='upper left')
    plt.plot(x, x, 'k-', linewidth=1)  # straight line
    for i in range(len(q_score_notPIEs)):
        plt.scatter(q_score_notPIEs[i][0], q_score_notPIEs[i][1], color="green", s=5)
    for i in range(len(q_score_PIEs)):
        plt.scatter(q_score_PIEs[i][0], q_score_PIEs[i][1], color="red", s=5)

    if method == 'supervised':
        plt.title('{} Q-Score {} {} pruning'.format(pruning_method, 'Supervised', sparsity), fontsize=14)
    else:
        plt.title('{} Q-Score {} {} pruning'.format(pruning_method, method, sparsity), fontsize=14)
    plt.grid(True)
    # plt.legend(["not PIEs", "PIEs"], loc="upper left")
    plt.xlabel('Average Q-Score encoders NOT pruned')
    plt.ylabel('Average Q-Score encoders {} pruned'.format(sparsity))

    plt.savefig(path.format(os.getcwd(), method, pruning_method, sparsity, method, "_wb", '.png'))

    if serialize:
        if os.path.exists(path.format(method, pruning_method, sparsity, method, "", '.npz')):
            print("Removing olf {} file".format(path.format(method, pruning_method, sparsity, method, "", '.npz')))
            os.remove(path.format(method, pruning_method, sparsity, method, "", '.npz'))
        print("Storing {}".format(path.format(method, pruning_method, sparsity, method, "", '.npz')))
        np.savez(path.format(os.getcwd(), method, pruning_method, sparsity, method, "", '.npz'),
                 q_score_notPIEs=np.array(q_score_notPIEs),
                 q_score_PIEs=np.array(q_score_PIEs), max=np.array(max[0]))

def inverse_transform(tensor):
    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1 / 0.2047, 1 / 0.2435, 1 / 0.2616]),
                                   transforms.Normalize(mean=[-0.4915, -0.4823, -0.4468],
                                                        std=[1., 1., 1.]),
                                   ])

    return invTrans(tensor)


# return q_score, z_score, L1 norm, mean, standard_deviation of the latent representation
def q_score(latent_representation):
    return (((np.max(latent_representation) - np.mean(latent_representation)) / np.std(
        latent_representation)) / np.linalg.norm(latent_representation, ord=1)), (
                   (np.max(latent_representation) - np.mean(latent_representation)) / np.std(
               latent_representation)), np.linalg.norm(latent_representation, ord=1), np.mean(
        latent_representation), np.std(
        latent_representation)


cifar_10_classes = {'0': 'airplane', '1': 'automobile', '2': 'bird', '3': 'cat', '4': 'deer', '5': 'dog', '6': 'frog',
                    '7': 'horse', '8': 'ship', '9': 'truck', }


def pies_class_distribution(pies_class_distribution, sparsity, method, pruning_method,
                            path='{}/pies/cifar10/{}/pies_distribution/PIEs_distribution_{}_{}sparsity_{}_pruning.png'):
    labels = []
    pies_for_classes_amount = []

    max_score = max(pies_class_distribution.keys(), key=(lambda k: pies_class_distribution[k]))
    min_score = min(pies_class_distribution.keys(), key=(lambda k: pies_class_distribution[k]))

    sum_pies = sum(pies_class_distribution.values())

    for label in pies_class_distribution:
        labels.append(cifar_10_classes[label])
        # pies_for_classes_amount.append((pies_class_distribution[label] - pies_class_distribution[min_score]) / (
        #        pies_class_distribution[max_score] - pies_class_distribution[min_score]))
        pies_for_classes_amount.append(pies_class_distribution[label])  # /sum_pies

    plt.rcdefaults()
    y_pos = np.arange(len(pies_class_distribution))
    plt.clf()
    fig, ax = plt.subplots()
    ax.barh(y_pos, pies_for_classes_amount,
            color=['green', 'blue', 'purple', 'brown', 'teal', 'indigo', 'orange', 'black', 'saddlebrown', 'cadetblue'])
    if method == 'supervised':
        ax.set_title('{} PIEs {} sparsity {} pruning'.format('Supervised', sparsity, pruning_method), fontsize=14)
    else:
        ax.set_title('{} PIEs {} sparsity {} pruning'.format(method, sparsity, pruning_method),
                     fontsize=14)
    print(labels)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    # ax.set_yticks(y_pos, labels=labels)
    ax.set_xlabel('PIEs number')
    # plt.grid(True)
    plt.savefig(path.format(os.getcwd(), method, method, sparsity, pruning_method))


def pload_loaded_pies_distribution(sparsity, method,
                                   pruning_method, serialize=True,
                                   path='{}/pies/cifar10/{}/pies_distribution/PIEs_distribution_{}_{}sparsity_{}_pruning{}'):
    if os.path.exists(
            path.format(os.getcwd(), method, method, sparsity, pruning_method, '.npz')):
        with np.load(
                path.format(os.getcwd(), method, method, sparsity, pruning_method, '.npz')) as data:
            print("Loaded {} ".format(path.format(os.getcwd(), method, method, sparsity, pruning_method, '.npz')))

            labels = []
            for label in range(10):
                labels.append(cifar_10_classes[str(label)])

            pies_for_classes_common = data['pies_common'].tolist()
            pies_for_classes_uniques = data['pies_unique'].tolist()
            print(pies_for_classes_common)
            print(pies_for_classes_uniques)

            plt.rcdefaults()
            y_pos = np.arange(10)
            plt.clf()
            fig, ax = plt.subplots()
            #ax.ticklabel_format(useOffset=False, style='plain')
            #ax.ticklabel_format(style='plain', axis='x')
            ax.xaxis.get_major_locator().set_params(integer=True)
            ax.barh(y_pos, pies_for_classes_common,
                    color=['green', 'blue', 'purple', 'brown', 'teal', 'indigo', 'orange', 'black', 'saddlebrown', 'cadetblue'],
                    alpha=1)
            ax.barh(y_pos, pies_for_classes_uniques,
                    color=['green', 'blue', 'purple', 'brown', 'teal', 'indigo', 'orange', 'black', 'saddlebrown', 'cadetblue'],
                    alpha=0.2)
            """
            
            if method == 'supervised':
                ax.set_title('{} PIEs {} sparsity {} pruning'.format('Supervised', sparsity, pruning_method), fontsize=14)
            else:
                ax.set_title('{} PIEs {} sparsity {} pruning'.format(method, sparsity, pruning_method),
                             fontsize=14)
            """
            #print(labels)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=20)
            # ax.set_yticks(y_pos, labels=labels)
            ax.set_xlabel('Number of PIEs', fontsize=25)
            plt.xticks(size=20)
            plt.tight_layout()
            #ax.set_xticks(size=11)
            # plt.grid(True)
            path_last = '{}/pies/cifar10/{}/pies_distribution/PIEs_distribution_{}_{}sparsity_comparison_{}_pruning{}'
            plt.savefig(path_last.format(os.getcwd(), method, method, sparsity, pruning_method, '.png'))
    else:
        print("File {} not found!".format(
            path.format(os.getcwd(), method, method, sparsity, pruning_method, '.npz')))


def pies_class_distribution_common(pies_class_distribution, pies_class_distribution_common, sparsity, method,
                                   pruning_method, serialize=True,
                                   path='{}/pies/cifar10/{}/pies_distribution/PIEs_distribution_{}_{}sparsity_{}_pruning{}'):
    labels = []
    pies_for_classes_uniques = []
    pies_for_classes_common = []

    for label in pies_class_distribution:
        labels.append(cifar_10_classes[label])
        pies_for_classes_uniques.append(pies_class_distribution[label] + pies_class_distribution_common[label])
        pies_for_classes_common.append(pies_class_distribution_common[label])

    plt.rcdefaults()
    y_pos = np.arange(len(pies_class_distribution))
    plt.clf()
    fig, ax = plt.subplots()
    ax.barh(y_pos, pies_for_classes_common,
            color=['green', 'blue', 'purple', 'brown', 'teal', 'indigo', 'orange', 'black', 'saddlebrown', 'cadetblue'],
            alpha=1)
    ax.barh(y_pos, pies_for_classes_uniques,
            color=['green', 'blue', 'purple', 'brown', 'teal', 'indigo', 'orange', 'black', 'saddlebrown', 'cadetblue'],
            alpha=0.2)
    """
    
    if method == 'supervised':
        ax.set_title('{} PIEs {} sparsity {} pruning'.format('Supervised', sparsity, pruning_method), fontsize=14)
    else:
        ax.set_title('{} PIEs {} sparsity {} pruning'.format(method, sparsity, pruning_method),
                     fontsize=14)
    """
    #print(labels)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    # ax.set_yticks(y_pos, labels=labels)
    ax.set_xlabel('Number of PIEs')
    # plt.grid(True)
    plt.savefig(path.format(os.getcwd(), method, method, sparsity, pruning_method, '.png'))

    if serialize or not os.path.exists(
            path.format(path.format(os.getcwd(), method, method, sparsity, pruning_method, '.npz'))):

        np.savez(path.format(os.getcwd(), method, method, sparsity, pruning_method, '.npz'),
                 pies_common=np.array(pies_for_classes_common),
                 pies_unique=np.array(pies_for_classes_uniques))


def plot_prediction_misalignment(q_score_PredMis, q_score_notPredMis, sparsity,
                                 path='{}/pies/cifar10/prediction_misalignments/Q-SCORE_SupCon_vs_Supervised_{}sparsity{}_.png'):

    plt.clf()
    for i in range(len(q_score_notPredMis)):
        plt.scatter(q_score_notPredMis[i][0], q_score_notPredMis[i][1], color="green", s=5)
    for i in range(len(q_score_PredMis)):
        plt.scatter(q_score_PredMis[i][0], q_score_PredMis[i][1], color="red", s=5)

    not_pred_legend = mlines.Line2D([], [], color='green', marker='o',
                                    markersize=6, label='not Prediction Misalignment', linestyle='None')
    pred_legend = mlines.Line2D([], [], color='red', marker='o',
                                markersize=6, label='Prediction Misalignment', linestyle='None')

    plt.legend(handles=[not_pred_legend, pred_legend], loc='upper left')

    plt.title('Predictions Misalignment Q-Score {} {} pruning'.format('Supervised vs SupCon', sparsity), fontsize=14)

    plt.grid(True)
    plt.xlabel('Average Q-Score supervised encoders {} pruned'.format(sparsity))
    plt.ylabel('Average Q-Score SupCon encoders {} pruned'.format(sparsity))
    plt.savefig(path.format(os.getcwd(), sparsity, ""))

    # plot with best ratio
    plt.clf()
    # if method == "supervised":
    #    x = np.linspace(0, 0.5, 3)
    # else:
    x = np.linspace(0, 2.5, 3)
    not_pred_legend = mlines.Line2D([], [], color='green', marker='o',
                                    markersize=6, label='not Prediction Misalignment', linestyle='None')
    pred_legend = mlines.Line2D([], [], color='red', marker='o',
                                markersize=6, label='Prediction Misalignment', linestyle='None')
    ideal_ratio = mlines.Line2D([], [], color='black', marker='_',
                                markersize=6, label='Ideal ratio')
    plt.legend(handles=[not_pred_legend, pred_legend, ideal_ratio], loc='upper left')
    plt.plot(x, x, 'k-', linewidth=1)  # straight line
    for i in range(len(q_score_notPredMis)):
        plt.scatter(q_score_notPredMis[i][0], q_score_notPredMis[i][1], color="green", s=5)
    for i in range(len(q_score_PredMis)):
        plt.scatter(q_score_PredMis[i][0], q_score_PredMis[i][1], color="red", s=5)

    plt.title('Predictions Misalignment Q-Score {} {} pruning'.format('Supervised vs SupCon', sparsity), fontsize=14)

    plt.grid(True)
    # plt.legend(["not PIEs", "PIEs"], loc="upper left")
    plt.xlabel('Average Q-Score supervised encoders {} pruned'.format(sparsity))
    plt.ylabel('Average Q-Score SupCon encoders {} pruned'.format(sparsity))
    plt.savefig(path.format(os.getcwd(), sparsity, "_wb"))


def PIEs_detection_supervised(models_pruned, models_not_pruned, data):
    most_frequent_labels_not_pruned = []
    most_frequent_labels_pruned = []

    results_model_notpruned = []
    results_model_pruned = []

    pies_result = np.zeros(len(data), dtype=int)

    for model in models_not_pruned:
        model.eval()

        with torch.no_grad():
            output_model = model(data)
            # q_score_notpruned.append(utils.pies.q_score(latent_representation=latent_representation))

        _, preds_output_model = torch.max(output_model, 1)
        results_model_notpruned.append(preds_output_model.cpu().detach().numpy())

    for i in range(len(data)):
        models_not_pruned_results = []

        for model_output in range(len(results_model_notpruned)):
            models_not_pruned_results.append(results_model_notpruned[model_output][i])

        most_frequent_labels_not_pruned.append(
            np.bincount(np.array(models_not_pruned_results)).argmax())

    for model in models_pruned:
        model.eval()

        with torch.no_grad():
            output_model_pruned = model(data)

        _, preds_output_model_pruned = torch.max(output_model_pruned, 1)
        results_model_pruned.append(preds_output_model_pruned.cpu().detach().numpy())

    # iterate over the batch length and take model output
    for i in range(len(data)):
        models_pruned_results = []

        for model_output in range(len(results_model_pruned)):
            models_pruned_results.append(results_model_pruned[model_output][i])

        most_frequent_labels_pruned.append(np.bincount(np.array(models_pruned_results)).argmax())

    for i in range(len(data)):
        if most_frequent_labels_not_pruned[i] != most_frequent_labels_pruned[i]:
            pies_result[i] = 1

    return pies_result


def PIEs_detection_clr(encoders_pruned, heads_pruned, encoders_not_pruned, heads_not_pruned, data):
    most_frequent_labels_not_pruned = []
    most_frequent_labels_pruned = []

    results_model_notpruned = []
    results_model_pruned = []

    pies_result = np.zeros(len(data), dtype=int)

    for model_not_pruned_number in range(len(encoders_not_pruned)):
        encoders_not_pruned[model_not_pruned_number].eval()
        heads_not_pruned[model_not_pruned_number].eval()

        with torch.no_grad():
            output_model = heads_not_pruned[model_not_pruned_number](
                encoders_not_pruned[model_not_pruned_number].encoder(
                    data))
        _, preds_output_model = torch.max(output_model, 1)
        results_model_notpruned.append(preds_output_model.cpu().detach().numpy())

    for i in range(len(data)):
        models_not_pruned_results = []

        for model_output in range(len(results_model_notpruned)):
            models_not_pruned_results.append(results_model_notpruned[model_output][i])

        most_frequent_labels_not_pruned.append(
            np.bincount(np.array(models_not_pruned_results)).argmax())

    for model_pruned_number in range(len(encoders_pruned)):
        encoders_pruned[model_pruned_number].eval()
        heads_pruned[model_pruned_number].eval()

        with torch.no_grad():
            output_model_pruned = heads_pruned[model_pruned_number](
                encoders_pruned[model_pruned_number].encoder(
                    data))

        _, preds_output_model_pruned = torch.max(output_model_pruned, 1)
        results_model_pruned.append(preds_output_model_pruned.cpu().detach().numpy())

    # iterate over the batch length and take model output
    for i in range(len(data)):
        models_pruned_results = []

        for model_output in range(len(results_model_pruned)):
            models_pruned_results.append(results_model_pruned[model_output][i])

        most_frequent_labels_pruned.append(np.bincount(np.array(models_pruned_results)).argmax())

    for i in range(len(data)):
        if most_frequent_labels_not_pruned[i] != most_frequent_labels_pruned[i]:
            pies_result[i] = 1

    return pies_result
