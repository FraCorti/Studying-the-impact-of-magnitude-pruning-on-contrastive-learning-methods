

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import glob
from collections import OrderedDict
import os


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def get_cifar10_contrastive(batch_size=128, workers=0):
    transform_train = transforms.Compose([

        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2047, 0.2435, 0.2616)),
    ])


    training_set = torchvision.datasets.CIFAR10(root='../data', train=True,
                                                download=True, transform=TwoCropTransform(transform_train))

    train_sampler = None
    loader_train = torch.utils.data.DataLoader(training_set, batch_size=batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=workers, pin_memory=True)
    return loader_train


def get_cifar10(batch_size=128, workers=0, shuffle=True):
    transform_train = transforms.Compose([
        transforms.RandomCrop(size=32, padding=4),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2047, 0.2435, 0.2616)),
    ])

    transform_validation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2047, 0.2435, 0.2616)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                            download=True, transform=transform_train)

    loader_train = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=shuffle, num_workers=workers, pin_memory=True)

    validationset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                                 download=True, transform=transform_validation)
    loader_valid = torch.utils.data.DataLoader(validationset, batch_size=batch_size,
                                               shuffle=False, num_workers=workers, pin_memory=True)
    loaders = {"train": loader_train, "valid": loader_valid}

    return loaders


def get_ordered_cifar10_validation(batch_size=1, workers=0, seed=0):
    def _init_fn(worker_id):
        np.random.seed(seed + worker_id)

    g = torch.Generator()
    g.manual_seed(seed)

    transform_validation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2047, 0.2435, 0.2616)),
    ])
    print("Loading CIFAR10, batch size {}".format(batch_size))
    validationset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                                 download=True, transform=transform_validation)
    loader_valid = torch.utils.data.DataLoader(validationset, batch_size=batch_size,
                                               shuffle=False, worker_init_fn=_init_fn,
                                               generator=g)
    loaders = OrderedDict()
    loaders["valid"] = loader_valid
    return loaders


def get_dataloader_cifar10_validation(batch_size=1, workers=0, seed=0):
    def _init_fn(worker_id):
        np.random.seed(seed + worker_id)

    g = torch.Generator()
    g.manual_seed(seed)

    transform_validation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2047, 0.2435, 0.2616)),
    ])

    validationset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                                 download=True, transform=transform_validation)

    loader_valid = torch.utils.data.DataLoader(validationset, batch_size=batch_size,
                                               shuffle=False, worker_init_fn=_init_fn,
                                               generator=g)
    return loader_valid


def get_ordered_cifar10_unnormalized(batch_size=1, workers=0, seed=0):
    def _init_fn(worker_id):
        np.random.seed(seed + worker_id)

    g = torch.Generator()
    g.manual_seed(seed)

    transform_validation = transforms.Compose([
        transforms.ToTensor()
    ])
    validationset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                                 download=True, transform=transform_validation)
    loader_valid = torch.utils.data.DataLoader(validationset, batch_size=batch_size,
                                               shuffle=False, worker_init_fn=_init_fn,
                                               generator=g)
    loaders = OrderedDict()
    loaders["valid"] = loader_valid
    return loaders


def plot_loss_curves(loss_train, loss_valid, final_sparsity, epochs, depth, dropout, model_id, clr_method="",
                     temperature=0.5, batch_size=1024):
    iterations = np.arange(0, len(loss_train), 1)

    plt.clf()
    plt.plot(iterations, loss_train, 'b-')

    if len(loss_valid) > 0:
        plt.plot(iterations, loss_valid, 'r-')
        plt.legend(["Loss training", "Loss validation"], loc="upper right")
    else:
        plt.legend(["Loss training"], loc="upper right")

    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    if len(clr_method) > 0:
        plt.savefig(
            '/home/f/fraco1997/compressed_model_v2/result/{}_loss_curves_{}pruning_{}epochs_{}temperature_{}batch_{}depth_{}dropout_id{}.png'.format(
                clr_method, final_sparsity, epochs, temperature, batch_size, depth, dropout, model_id))
    else:
        plt.savefig(
            '/home/f/fraco1997/compressed_model_v2/result/loss_curves_{}pruning_{}epochs_{}depth_{}dropout_id{}.png'.format(
                final_sparsity, epochs, depth, dropout, model_id))

def plot_accuracy_curves(accuracy_train, accuracy_valid, final_sparsity, epochs, depth, dropout, model_id,
                         clr_method=""):
    iterations = np.arange(0, len(accuracy_train), 1)

    plt.clf()
    plt.plot(iterations, accuracy_train, 'b-')
    plt.plot(iterations, accuracy_valid, 'r-')
    plt.grid(True)
    plt.legend(["Accuracy training", "Accuracy validation"], loc="lower right")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    if len(clr_method) > 0:
        plt.savefig(
            '{}/result/{}_accuracy_curves_{}pruning_{}epochs_{}depth_{}dropout_id{}.png'.format(os.getcwd(),
                clr_method, final_sparsity, epochs, depth, dropout, model_id))
    else:
        plt.savefig(
            '{}/result/accuracy_curves_{}pruning_{}epochs_{}depth_{}dropout_id{}.png'.format(os.getcwd(),
                final_sparsity, epochs, depth, dropout, model_id))


def plot_clr_distribution_shift_robustness(custom_dataset_name, accuracy_custom_dataset, accuracy_cifar10,
                                           best_accuracy_ratio, clr=""):
    plt.clf()
    if best_accuracy_ratio:
        x = np.linspace(0, 100, 100)
        plt.plot(x, x, 'k-')  # straight line

    plt.plot(accuracy_cifar10[0], accuracy_custom_dataset[0], color='c', marker='o', linestyle='None')  # pruning 0.0
    plt.plot(accuracy_cifar10[1], accuracy_custom_dataset[1], color='b', marker='o', linestyle='None')  # pruning 0.3
    plt.plot(accuracy_cifar10[2], accuracy_custom_dataset[2], color='r', marker='o', linestyle='None')  # pruning 0.5
    plt.plot(accuracy_cifar10[3], accuracy_custom_dataset[3], color='g', marker='o', linestyle='None')  # pruning 0.7
    # plt.plot(accuracy_cifar10[4], accuracy_custom_dataset[4], color='y', marker='o', linestyle='None')  # pruning 0.9
    plt.plot(accuracy_cifar10[4], accuracy_custom_dataset[4], color='tab:gray', marker='o',
             linestyle='None')  # pruning 0.95
    plt.plot(accuracy_cifar10[5], accuracy_custom_dataset[5], color='tab:brown', marker='o',
             linestyle='None')  # pruning 0.99
    plt.grid(True)
    if best_accuracy_ratio:
        plt.legend(["Ideal accuracy", "0.0 pruning", "0.3 pruning", "0.5 pruning", "0.7 pruning",
                    "0.95 pruning", "0.99 pruning"],
                   loc="upper left")
    else:
        # change loc to "upper left" if the points are plotted on it
        plt.legend(
            ["0.0 pruning", "0.3 pruning", "0.5 pruning", "0.7 pruning", "0.95 pruning", "0.99 pruning"],
            loc="lower right")
    plt.xlabel('Accuracy {} CIFAR10'.format(clr))
    plt.ylabel('Accuracy {} {}'.format(clr, custom_dataset_name))

    print("Custom dataset name {} accuracies {}: accuracies CIFAR10: {} Best accuracy ratio: {}".format(
        custom_dataset_name,
        accuracy_custom_dataset,
        accuracy_cifar10, best_accuracy_ratio))

    if best_accuracy_ratio:
        plt.savefig(
            '{}/result/{}_{}_distribution_shift_accuracy_wb.png'.format(os.getcwd(),
                custom_dataset_name, clr))
    else:
        plt.savefig(
            '{}/result/{}_{}_distribution_shift_accuracy.png'.format(os.getcwd(),
                custom_dataset_name, clr))


def plot_distribution_shift_robustness(custom_dataset_name, accuracy_custom_dataset, accuracy_cifar10,
                                       best_accuracy_ratio, clr=""):
    plt.clf()
    if best_accuracy_ratio:
        x = np.linspace(0, 100, 100)
        plt.plot(x, x, 'k-')  # straight line

    plt.plot(accuracy_cifar10[0], accuracy_custom_dataset[0], color='c', marker='o', linestyle='None')  # pruning 0.0
    plt.plot(accuracy_cifar10[1], accuracy_custom_dataset[1], color='b', marker='o', linestyle='None')  # pruning 0.3
    plt.plot(accuracy_cifar10[2], accuracy_custom_dataset[2], color='r', marker='o', linestyle='None')  # pruning 0.5
    plt.plot(accuracy_cifar10[3], accuracy_custom_dataset[3], color='g', marker='o', linestyle='None')  # pruning 0.7
    plt.plot(accuracy_cifar10[4], accuracy_custom_dataset[4], color='y', marker='o', linestyle='None')  # pruning 0.9
    plt.plot(accuracy_cifar10[5], accuracy_custom_dataset[5], color='tab:gray', marker='o',
             linestyle='None')  # pruning 0.95
    plt.plot(accuracy_cifar10[6], accuracy_custom_dataset[6], color='tab:brown', marker='o',
             linestyle='None')  # pruning 0.99
    plt.grid(True)
    if best_accuracy_ratio:
        plt.legend(["Ideal accuracy", "0.0 pruning", "0.3 pruning", "0.5 pruning", "0.7 pruning", "0.9 pruning",
                    "0.95 pruning", "0.99 pruning"],
                   loc="upper left")
    else:
        # change loc to "upper left" if the points are plotted on it
        plt.legend(
            ["0.0 pruning", "0.3 pruning", "0.5 pruning", "0.7 pruning", "0.9 pruning", "0.95 pruning", "0.99 pruning"],
            loc="lower right")
    plt.xlabel('Accuracy {} CIFAR10'.format(clr))
    plt.ylabel('Accuracy {} {}'.format(clr, custom_dataset_name))

    print("Custom dataset name {} accuracies {}: accuracies CIFAR10: {} Best accuracy ratio: {}".format(
        custom_dataset_name,
        accuracy_custom_dataset,
        accuracy_cifar10, best_accuracy_ratio))

    if best_accuracy_ratio:
        plt.savefig(
            '{}/result/{}_{}_distribution_shift_accuracy_wb.png'.format(os.getcwd(),
                custom_dataset_name, clr))
    else:
        plt.savefig(
            '{}/result/{}_{}_distribution_shift_accuracy.png'.format(os.getcwd(),
                custom_dataset_name, clr))


def load_dataset(dataset_name, batch_size=128, seed=1, debug=False):
    if dataset_name is None:
        raise Exception(
            "No distribution datasets name passed, name accepted: cifar-10.1 (with dataset_version v4 or v6) or cifar-10.2")
    dataset_name = ("".join(str(dataset_name).rstrip().lstrip())).lower()

    print("Dataset name: {}".format(dataset_name))

    if dataset_name == 'cifar10':
        return get_dataloader_cifar10_validation(batch_size=batch_size, seed=seed)

    if dataset_name == 'cifar10.1_v4' or dataset_name == 'cifar10.1_v6':
        return load_cifar10_1(dataset_name, batch_size=batch_size, seed=seed, debug=debug)

    if dataset_name == 'cifar10.2':
        return load_cifar10_2(batch_size=batch_size, seed=seed, debug=debug)


def normalize_4d_tensor(data, debug=False):
    # normalize the array of images by subtracting the mean and divide by the standard deviation
    if debug:
        print("Data means: {} type: {} shape: {} ".format(data.mean(axis=(0, 1, 2), keepdims=True),
                                                          type(data.mean(axis=(0, 1, 2), keepdims=True)),
                                                          data.mean(axis=(0, 1, 2), keepdims=True).shape))
        print("Data standard deviations: {} type: {} shape: {} ".format(data.std(axis=(0, 1, 2), keepdims=True),
                                                                        type(data.std(axis=(0, 1, 2), keepdims=True)),
                                                                        data.std(axis=(0, 1, 2), keepdims=True).shape))

    return (data - data.mean(axis=(0, 1, 2), keepdims=True)) / data.std(axis=(0, 1, 2), keepdims=True)

def convert_to_HWC(images_data, debug=False):
    return np.transpose(images_data, (0, 3, 1, 2))


def create_tensor_dataset(images, labels):
    return TensorDataset(torch.from_numpy(images).type(torch.cuda.FloatTensor),
                         torch.from_numpy(labels))


def load_cifar10_1(dataset_name, batch_size=128, seed=0, debug=False):
    def _init_fn(worker_id):
        np.random.seed(seed + worker_id)

    g = torch.Generator()
    g.manual_seed(seed)

    directory = "cifar-10.1/{}".format(dataset_name)
    current_directory = os.path.dirname(os.path.abspath(__file__))
    current_directory = os.path.dirname(os.path.normpath(current_directory))
    cifar_dir = os.path.join(current_directory, "CIFARs", directory)

    if debug:
        print("Complete path cifar: {}".format(cifar_dir))

    label_file = glob.glob("{}/*labels.npy".format(cifar_dir))
    data_file = glob.glob("{}/*test.npy".format(cifar_dir))

    if debug:
        print("Labels path: {}".format(label_file))
        print("Dataset path: {}".format(data_file))

    images_data = np.load(data_file[0])
    if debug:
        print("First sample loaded")
        print(images_data[0])
    labels = np.load(label_file[0]).astype(np.int64)

    images_data = convert_to_HWC(images_data=images_data, debug=debug)
    images_data = normalize_4d_tensor(data=images_data, debug=debug)

    if debug:
        print("Images dataset shape: {}".format(images_data.shape))
        print("Labels dataset shape: {}".format(labels.shape))

    tensor_dataset = create_tensor_dataset(images=images_data, labels=labels)
    dataset_loader = torch.utils.data.DataLoader(tensor_dataset,
                                                 batch_size=batch_size, shuffle=True,
                                                 num_workers=0, worker_init_fn=_init_fn,
                                                 generator=g)
    return dataset_loader


def load_cifar10_2(seed=0, batch_size=128, debug=False):
    def _init_fn(worker_id):
        np.random.seed(seed + worker_id)

    g = torch.Generator()
    g.manual_seed(seed)

    directory = "cifar-10.2"

    current_directory = os.path.dirname(os.path.abspath(__file__))
    current_directory = os.path.dirname(os.path.normpath(current_directory))
    dataset_path = os.path.join(current_directory, "CIFARs", directory)

    if debug:
        print("Complete path cifar: {}".format(dataset_path))
    data_file = glob.glob("{}/*test.npz".format(dataset_path))

    data = np.load(data_file[0])
    images = data['images']
    if debug:
        print("First sample loaded")
        print(images[0])
    labels = data['labels'].astype(np.int64)

    images = convert_to_HWC(images)
    images = normalize_4d_tensor(images)

    if debug:
        print("Images shape: {}".format(images.shape))
        print("Labels shape: {}".format(labels.shape))

    tensor_dataset = create_tensor_dataset(images=images, labels=labels)
    dataset_loader = torch.utils.data.DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False,
                                                 worker_init_fn=_init_fn,
                                                 generator=g)
    return dataset_loader
