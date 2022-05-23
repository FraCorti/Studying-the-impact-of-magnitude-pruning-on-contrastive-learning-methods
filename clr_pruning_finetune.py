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
parser.add_argument('--clr_method', default="SupCon", type=str)
parser.add_argument('--projection_head', default='mlp', type=str)
parser.add_argument('--epochs', default=30, type=int)
parser.add_argument('--temperature', default=0.5, type=float)
parser.add_argument('--encoder_name', default='WideResNet', type=str)
parser.add_argument('--pruning_percentages', default='0.3,0.5,0.7,0.9', type=str)
parser.add_argument('--encoder_save_path',
                    default='{}/models/{}_encoder_{}_{}pruning{}_{}batch_{}temperature_{}epochs_id{}.pt',
                    type=str)
parser.add_argument('--head_save_path',
                    default='{}/models/{}_head_{}_{}pruning{}_{}batch_{}temperature_{}epochs_id{}.pt',
                    type=str)

args = parser.parse_args()
depth = args.model_depth
seed = args.seed
encoder_save_path = args.encoder_save_path
head_save_path = args.head_save_path
encoder_name = args.encoder_name
cuda_deterministic = args.cuda_deterministic
batch_size = args.batch_size
widen_factor = args.widen_factor
dropout = args.dropout
epochs = args.epochs
cuda_device = args.cuda_device
projection_head = args.projection_head
debug = args.debug
models_number = args.models_number
workers = args.workers
temperature = args.temperature
method = args.clr_method
total_seed = args.total_seed

pruning_percentages_str = args.pruning_percentages.split(',')
pruning_percentages = list([])
for it in pruning_percentages_str:
    pruning_percentages.append(float(it))

device = torch.device("cuda:{}".format(cuda_device) if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(cuda_device)

if cuda_deterministic:
    # see https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)

if debug:
    today = date.today()
    day = today.strftime("%d_%m_%Y")
    now = datetime.now()
    time = now.strftime("%H_%M_%S")
    logging.basicConfig(
        filename="/home/f/fraco1997/compressed_model_v2/logs/CLR_PIEs_DEBUG_wideResnet_day:{}_time:{}.log".format(
            day, time),
        level=logging.INFO)

    print(torch.cuda.memory_stats(device=device))

for seed in range(0, total_seed, 1):

    print("Current seed: {}".format(seed))

    if cuda_deterministic:
        print("Setting Pytorch and CUBLAS to deterministic behaviour with seed: {}".format(seed))
        # torch.set_printoptions(profile='full')
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    print(
        "Models pruning percentages: {}, Models to finetune: {} Batch Size: {}, Deterministic Setup: {}, Models depth: {}, Models widen factor: {}".format(
            pruning_percentages, models_number, batch_size, cuda_deterministic, depth, widen_factor))

    loaders = utils.data.get_cifar10(batch_size=batch_size, workers=workers)

    for models_pruning in pruning_percentages:

        for pruning_post_training in ['global']:

            clr_pruned_models_encoders, clr_pruned_models_heads = utils.models.get_clr_models(
                models_number=models_number,
                encoder_name=encoder_name,
                method=method,
                sparsity=0.0,
                temperature=temperature,
                device=device,
                projection_head=projection_head, attach_gpu=False)

            pruned_models = []
            for model_number in range(len(clr_pruned_models_encoders)):
                pruned_models.append(
                    utils.models.get_wideResNet_contrastive(encoder=clr_pruned_models_encoders[model_number].encoder,
                                                            classification_head=clr_pruned_models_heads[
                                                                model_number]))
            for model in pruned_models:
                if pruning_post_training == 'global':
                    utils.pruning.global_pruning(model=model, pruning_percentage=models_pruning)
                if pruning_post_training == 'layers':
                    utils.pruning.layer_wise_pruning(model=model, pruning_percentage=models_pruning)

                model_sparsity = utils.pruning.get_model_sparsity(model)
                if debug:
                    logging.info("Loaded encoder with sparsity of {}".format(round(model_sparsity / 100, 1)))
                if models_pruning != round(model_sparsity / 100, 1):
                    print("Encoder loaded with a different sparsity of {}!".format(model_sparsity))

            if not debug:
                print("Loaded {} {} encoders {} pruned with {}".format(len(clr_pruned_models_encoders), method,
                                                                       models_pruning, pruning_post_training))
                print(
                    "Loaded {} {} heads {} pruned with {}".format(len(clr_pruned_models_heads), method, models_pruning,
                                                                  pruning_post_training))
            else:
                logging.info("Loaded {} pruned models: {}".format(models_pruning, len(clr_pruned_models_encoders)))

            total_pies = 0
            batch_index = 0

            total_samples = 0
            correct_clr = 0
            correct_clr_notpruned = 0

            model_number = 0

            # reverse iteration
            for model in reversed(pruned_models):

                if utils.models.check_clr_model_already_trained(encoder_save_path=encoder_save_path, encoder_name=encoder_name, method=method, final_sparsity=models_pruning, batch_size=batch_size,
                                    temperature=temperature, epochs=epochs, model_number=model_number, pruning_method=pruning_post_training):
                    continue

                model.to(device=device)

                criterion = nn.CrossEntropyLoss().to(device=device)
                optimizer = optim.SGD(model.parameters(), lr=1e-3, nesterov=True, momentum=0.9)

                print("Model number {} initial sparsity: {} pruning post-training method: {}".format(
                    model_number, utils.pruning.get_model_sparsity(model), pruning_post_training))

                for epoch in range(epochs):

                    step_train = 0
                    step_validation = 0
                    loss_training = 0
                    loss_validation = 0
                    correct_validation = 0
                    correct_training = 0
                    total_training = 0
                    total_validation = 0

                    for phase in ["train", "valid"]:
                        if phase == "train":
                            model.train()
                        else:
                            model.eval()

                        for i, data in enumerate(loaders[phase]):

                            if phase == "train":
                                step_train += 1

                            if phase == "valid":
                                step_validation += 1

                            images, labels = data
                            images, labels = images.to(device), labels.to(device)

                            optimizer.zero_grad()

                            with torch.set_grad_enabled(phase == "train"):

                                outputs = model(images)

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
                                                                                                             model)))
                    loss_training = 0
                    loss_validation = 0
                    correct_training = 0
                    correct_validation = 0
                    step_train = 0
                    step_validation = 0
                    total_training = 0
                    total_validation = 0

                utils.pruning.remove_pruning_masks(model)

                model_sparsity = utils.pruning.get_model_sparsity(model)
                print("Model number {} final sparsity without masks: {}".format(model_number, model_sparsity))

                if not debug:
                    state_encoder = {
                        'model': model.encoder.state_dict(),
                    }

                    state_head = {
                        'head': model.classification_head.state_dict(),
                    }
                    torch.save(state_head,
                               head_save_path.format(os.getcwd(),
                                   encoder_name, method, models_pruning, '_global', batch_size, temperature, epochs,
                                   model_number))
                    torch.save(state_encoder,
                               encoder_save_path.format(os.getcwd(),
                                   encoder_name, method, models_pruning, '_global', batch_size, temperature, epochs,
                                   model_number))

                model_number += 1
