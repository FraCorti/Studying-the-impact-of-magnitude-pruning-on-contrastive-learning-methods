import numpy as np
import argparse
from datetime import datetime, date
import logging
import os


import utils
from utils import pruning, data, models

import random

import torch
from torch import nn, optim

parser = argparse.ArgumentParser()
parser.add_argument('--models_number', default=30, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--model_depth', default=16, type=int)
parser.add_argument('--debug', default=False, type=bool)
parser.add_argument('--cuda_device', default=0, type=int)
parser.add_argument('--dropout', default=0.0, type=float)
parser.add_argument('--workers', default=0, type=int)
parser.add_argument('--widen_factor', default=2, type=int)
parser.add_argument('--cuda_deterministic', default=True, type=bool)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--total_seed', default=1, type=int)
parser.add_argument('--epochs', default=30, type=int)
parser.add_argument('--pruning_percentages', default='0.3,0.5,0.7,0.9', type=str)
parser.add_argument('--save_path',
                    default='{}/models/wideResNet{}_{}pruning_{}epochs_{}depth_{}dropout_id{}.pt',
                    type=str)
args = parser.parse_args()
save_path = args.save_path
epochs = args.epochs
depth = args.model_depth
seed = args.seed
cuda_deterministic = args.cuda_deterministic
batch_size = args.batch_size
widen_factor = args.widen_factor
dropout = args.dropout
cuda_device = args.cuda_device
debug = args.debug
models_number = args.models_number
workers = args.workers
total_seed = args.total_seed
torch.set_printoptions(profile="full")

pruning_percentages_str = args.pruning_percentages.split(',')
pruning_percentages = list([])
for it in pruning_percentages_str:
    pruning_percentages.append(float(it))

device = torch.device("cuda:{}".format(cuda_device) if torch.cuda.is_available() else "cpu")

if cuda_deterministic:
    # see https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)

    today = date.today()
    day = today.strftime("%d_%m_%Y")
    now = datetime.now()
    time = now.strftime("%H_%M_%S")
    logging.basicConfig(
        filename="{}/logs/finetune_supervised_DEBUG_wideResnet_day:{}_time:{}.log".format(os.getcwd(),
            day, time),
        level=logging.INFO)

    print(torch.cuda.memory_stats(device=device))

for seed in range(0, total_seed, 1):

    print("Current seed: {}".format(seed))

    if cuda_deterministic:
        print("Setting Pytorch and CUBLAS to deterministic behaviour with seed: {}".format(seed))
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    loaders = utils.data.get_cifar10(batch_size=batch_size, workers=workers)

    print(
        "Models pruning percentages: {}, Models to finetune: {} Batch Size: {}, Deterministic Setup: {}, Models depth: {}, Models widen factor: {}".format(
            pruning_percentages, models_number, batch_size, cuda_deterministic, depth, widen_factor))

    for models_pruning in pruning_percentages:

        for pruning_post_training in ['global']:

            pruned_models = utils.models.get_supervised_models(models_number=models_number, debug=debug, sparsity=0.0,
                                                               device=device, attach_gpu=False)

            for pruned_model in pruned_models:
                if pruning_post_training == 'global':
                    utils.pruning.global_pruning(model=pruned_model, pruning_percentage=models_pruning)
                if pruning_post_training == 'layers':
                    utils.pruning.layer_wise_pruning(model=pruned_model, pruning_percentage=models_pruning)

            for model in pruned_models:
                model_sparsity = utils.pruning.get_model_sparsity(model)
                # print("Model sparsity: {}".format(model_sparsity))
                if debug:
                    logging.info("Loaded encoder with sparsity of {}".format(round(model_sparsity / 100, 1)))
                if models_pruning != round(model_sparsity / 100, 1):
                    print("Model loaded with a different sparsity of {}!".format(model_sparsity))

            if not debug:
                print("Loaded {} models {} pruned  pruned with: {}".format(len(pruned_models), models_pruning,
                                                                           pruning_post_training))
            else:
                logging.info("Loaded {} pruned models: {} pruned with: {}".format(models_pruning, len(pruned_models),
                                                                                  pruning_post_training))

            loss_train = []
            loss_valid = []
            accuracy_train = []
            accuracy_valid = []

            step = 0
            model_number = 0

            for pruned_model in pruned_models:

                if utils.models.check_supervised_model_pretrained(save_path=save_path, model_number=model_number,
                                                                  pruning_technique='global', epochs=epochs,
                                                                  depth=depth, dropout=dropout, model_name='wideResNet',
                                                                  final_sparsity=models_pruning):
                    continue

                model_sparsity = utils.pruning.get_model_sparsity(pruned_model)
                if models_pruning != round(model_sparsity / 100, 1):
                    print("Model sparsity: {}!".format(model_sparsity))

                pruned_model.to(device=device)

                criterion = nn.CrossEntropyLoss().to(device=device)
                optimizer = optim.SGD(pruned_model.parameters(), lr=1e-3, nesterov=True, momentum=0.9)
                print("Model number {} initial sparsity: {}".format(
                    model_number, utils.pruning.get_model_sparsity(pruned_model)))

                for epoch in range(epochs):

                    step_train = 0
                    step_validation = 0
                    loss_training = 0
                    loss_validation = 0
                    correct_validation = 0
                    correct_training = 0
                    total_training = 0
                    total_validation = 0

                    # alternate training and validation phase
                    for phase in ["train", "valid"]:
                        if phase == "train":
                            pruned_model.train()
                        else:
                            pruned_model.eval()

                        # cycle on the batches of the train and validation dataset
                        for i, data in enumerate(loaders[phase]):

                            if phase == "train":
                                step += 1
                                step_train += 1

                            if phase == "valid":
                                step_validation += 1

                            images, labels = data
                            images, labels = images.to(device), labels.to(device)

                            optimizer.zero_grad()

                            with torch.set_grad_enabled(phase == "train"):

                                outputs = pruned_model(images)

                                loss = criterion(outputs, labels)

                                _, predicted = torch.max(outputs.data, 1)

                                if phase == "train":
                                    loss_training += loss.item()
                                    total_training += labels.size(0)
                                    correct_training += (predicted == labels).sum().item()
                                    loss.backward()
                                    optimizer.step()

                                if phase == "valid":
                                    loss_validation += loss.item()
                                    total_validation += labels.size(0)
                                    correct_validation += (predicted == labels).sum().item()

                    current_train_loss = loss_training / step_train
                    current_validation_loss = loss_validation / step_validation
                    current_train_accuracy = 100 * correct_training / total_training
                    current_validation_accuracy = 100 * correct_validation / total_validation

                    logging.info("epoch {} train accuracy {} validation accuracy {} sparsity: {}".format(epoch,
                                                                                                         current_train_accuracy,
                                                                                                         current_validation_accuracy,
                                                                                                         utils.pruning.get_model_sparsity(
                                                                                                             pruned_model.base)))

                    loss_train.append(current_train_loss)
                    accuracy_train.append(current_train_accuracy)

                    loss_valid.append(current_validation_loss)
                    accuracy_valid.append(current_validation_accuracy)

                    loss_training = 0
                    loss_validation = 0
                    correct_training = 0
                    correct_validation = 0
                    step_train = 0
                    step_validation = 0
                    total_training = 0
                    total_validation = 0

                utils.pruning.remove_pruning_masks(pruned_model)
                model_sparsity = utils.pruning.get_model_sparsity(pruned_model)
                print("Model number {} final sparsity without masks: {}".format(model_number, model_sparsity))

                if not debug:
                    if pruning_post_training == 'global':
                        torch.save(pruned_model.state_dict(),
                                   save_path.format(os.getcwd(),
                                       '_global', models_pruning, epochs, depth, dropout, model_number))
                    if pruning_post_training == 'layers':
                        torch.save(pruned_model.state_dict(),
                                   save_path.format(os.getcwd(),
                                       '_layers', models_pruning, epochs, depth, dropout, model_number))

                model_number += 1
