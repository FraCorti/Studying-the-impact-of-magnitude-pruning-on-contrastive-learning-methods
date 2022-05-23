import torch
from torch.optim import lr_scheduler
import utils
import torch.optim as optim
import random
import argparse
import math
import logging
from datetime import datetime, date
import os
import numpy as np
from utils import models, data, pruning, losses, clr_debug, optimizers

parser = argparse.ArgumentParser()
parser.add_argument('--model_depth', default=16, type=int)
parser.add_argument('--begin_step', default=2440, type=int , help='pruning start step')
parser.add_argument('--end_step', default=4760, type=int, help='pruning end step')
parser.add_argument('--frequency', default=62, type=int, help='pruning number of steps frequency')
parser.add_argument('--begin_step_head', default=500, type=int)
parser.add_argument('--end_step_head', default=10000, type=int)
parser.add_argument('--frequency_head', default=250, type=int)
parser.add_argument('--batch_size', default=1024, type=int)
parser.add_argument('--cuda_device', default=10, type=int, help='id of the GPU to use')
parser.add_argument('--debug', default=False, type=bool)
parser.add_argument('--shuffle', default=True, type=bool)
parser.add_argument('--dropout', default=0.0, type=float)
parser.add_argument('--workers', default=0, type=int)
parser.add_argument('--learning_rate', default=0.05, type=float)
parser.add_argument('--weight_decay', default=0.0004, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--index_start', default=16, type=int)
parser.add_argument('--index_end', default=17, type=int)
parser.add_argument('--widen_factor', default=2, type=int)
parser.add_argument('--warmup_epochs', default=0, type=int)
parser.add_argument('--num_classes', default=10, type=int)
parser.add_argument('--head_epochs', default=100, type=int)
parser.add_argument('--batch_size_head', default=256, type=int)
parser.add_argument('--epochs', default=500, type=int)
parser.add_argument('--temperature', default=0.5, type=float, help='temperature, contrastive loss function hyperparameter')
parser.add_argument('--base_temperature', default=0.07, type=float)
parser.add_argument('--projection_head', default='mlp', type=str)
parser.add_argument('--method', default='SupCon', type=str)
parser.add_argument('--lr_decay_epochs', default='1500', type=str)
parser.add_argument('--lars', default=False, action='store_true', help='lars optimizer')
parser.add_argument('--cosine_annealing', default=False, action='store_true', help='cosine annealing')
parser.add_argument('--debug_classification_head', default=False, action='store_true')
parser.add_argument('--debug_classification_head_epochs', default=25, type=int)
parser.add_argument('--encoder_name', default='WideResNet', type=str)
parser.add_argument('--pruning_percentages', default='0.0,0.3,0.5,0.7,0.9', type=str, help='final sparsity percentages')
parser.add_argument('--clipping', default=False, action='store_true')
parser.add_argument('--encoder_save_path',
                    default='{}/models/{}_encoder_{}_{}pruning{}_{}batch_{}temperature_{}epochs_id{}.pt',
                    type=str)
parser.add_argument('--head_save_path',
                    default='{}/models/{}_head_{}_{}pruning{}_{}batch_{}temperature_{}epochs_id{}.pt',
                    type=str)

args = parser.parse_args()
clipping = args.clipping
begin_step_head = args.begin_step_head
end_step_head = args.end_step_head
frequency_head = args.frequency_head
lars = args.lars
base_temperature = args.base_temperature
method = args.method
encoder_name = args.encoder_name
epochs = args.epochs
batch_size = args.batch_size
debug_classification_head_epochs = args.debug_classification_head_epochs
debug_classification_head = args.debug_classification_head
cosine = args.cosine_annealing
head_epochs = args.head_epochs
cuda_device = args.cuda_device
begin_step = args.begin_step
end_step = args.end_step
num_classes = args.num_classes
widen_factor = args.widen_factor
frequency = args.frequency
encoder_save_path = args.encoder_save_path
head_save_path = args.head_save_path
batch_size_head = args.batch_size_head
warmup_epochs = args.warmup_epochs
dropout = args.dropout
depth = args.model_depth
shuffle = args.shuffle
debug = args.debug
workers = args.workers
weight_decay = args.weight_decay
learning_rate = args.learning_rate
momentum = args.momentum
index_start = args.index_start
index_end = args.index_end
projection_head = args.projection_head
temperature = args.temperature

pruning_percentages_str = args.pruning_percentages.split(',')
pruning_percentages = list([])

for it in pruning_percentages_str:
    pruning_percentages.append(float(it))


initial_sparsity = 0.0

torch.cuda.empty_cache()
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

device = torch.device("cuda:{}".format(cuda_device) if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(cuda_device)

if not debug:
    now = datetime.now()
    today = date.today()
    day = today.strftime("%d_%m_%Y")
    time = now.strftime("%H_%M_%S")
    logging.basicConfig(
        filename="{}/logs/{}_{}_indexStart{}_indexEnd{}_temp{}_batch{}_day:{}_time:{}.log".format(os.getcwd(),
            method, encoder_name, index_start, index_end, temperature, batch_size, day,
            time),
        level=logging.INFO)

    logging.info("------------------- Setup Configuration -------------------")
    logging.info(
        "Models to train: {} Method: {} Encoder: {} Final sparsities: {} Initial learning rate: {} Cosine annealing: {} Lars: {} Warmup epochs: {} Temperature: {} Base temperature: {} Debug classification head: {} Debug head epochs: {} Pruning begin step: {} Pruning end step: {} Pruning frequency: {} Start seed index: {} End seed index: {} Models depth: {} Widen factor: {} Epochs: {} Batch size: {} Weight Decay: {} Dropout: {} Momentum: {}".format(
            index_end - index_start + 1, method, encoder_name, pruning_percentages, learning_rate, cosine, lars,
            warmup_epochs,
            temperature,
            base_temperature,
            debug_classification_head, debug_classification_head_epochs, begin_step,
            end_step,
            frequency,
            index_start, index_end,
            depth, widen_factor,
            epochs, batch_size, weight_decay, dropout, momentum))

print(
    "Models to train: {} Method: {} Final sparsities: {} Encoder: {} Final sparsities: {} Initial learning rate: {} Cosine annealing: {} Lars: {} Warmup epochs: {} Temperature: {} Base temperature: {} Debug classification head: {} Debug head epochs: {} Pruning begin step: {} Pruning end step: {} Pruning frequency: {} Start seed index: {} End seed index: {} Models depth: {} Widen factor: {} Epochs: {} Batch size: {} Weight Decay: {} Dropout: {} Momentum: {}".format(
        index_end - index_start + 1, method, pruning_percentages, encoder_name, pruning_percentages, learning_rate, cosine, lars, warmup_epochs,
        temperature,
        base_temperature,
        debug_classification_head, debug_classification_head_epochs, begin_step,
        end_step,
        frequency,
        index_start, index_end,
        depth, widen_factor,
        epochs, batch_size, weight_decay, dropout, momentum))

loader_encoder = utils.data.get_cifar10_contrastive(batch_size=batch_size, workers=workers)
loaders_head = utils.data.get_cifar10(batch_size=batch_size_head, workers=workers)

for final_sparsity in pruning_percentages:

    for model_number in range(index_start, index_end + 1):

        if utils.models.check_clr_model_already_trained(encoder_save_path=encoder_save_path, encoder_name=encoder_name,
                                                        method=method, begin_step=begin_step,
                                                        final_sparsity=final_sparsity,
                                                        batch_size=batch_size,
                                                        temperature=temperature, epochs=epochs,
                                                        model_number=model_number):
            continue

        print("Training model with sparsity {} number: {}".format(final_sparsity, model_number))
        logging.info("Training model with sparsity {} number: {}".format(final_sparsity, model_number))

        # os.environ['PYTHONHASHSEED'] = str(model_number)
        torch.cuda.manual_seed(model_number)
        torch.manual_seed(model_number)
        random.seed(model_number)
        np.random.seed(model_number)

        is_pruning = False
        gamma_parameters = utils.pruning.get_gammas(initial_sparsity=initial_sparsity, final_sparsity=final_sparsity,
                                                    begin_step=begin_step, end_step=end_step, frequency=frequency)
        pruning_iterations = 0
        frequency_counter = 0

        if final_sparsity > 0.0:
            logging.info("Gamma parameters encoder: {}".format(gamma_parameters))

        model = utils.models.get_clr_encoder(encoder=encoder_name, depth=depth, dropout=dropout,
                                             widen_factor=widen_factor,
                                             head=projection_head)
        model.to(device)

        criterion = utils.losses.get_clr_loss(loss_name=method, device=device, temperature=temperature,
                                              base_temperature=base_temperature)
        criterion.to(device)

        optimizer = utils.optimizers.get_optimizer(model=model, encoder_name=encoder_name, learning_rate=learning_rate,
                                                   weight_decay=weight_decay, momentum=momentum, clipping=clipping,
                                                   lars=lars)

        if cosine:
            eta_min = optimizer.param_groups[0]['lr'] * (0.1 ** 3)
            scheduler_cos = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)

        warmup_from = 0.01

        # setup warm-up
        if batch_size > 256 and warmup_epochs > 0:
            if cosine:
                eta_min = learning_rate * (0.1 ** 3)
                warmup_to = eta_min + (learning_rate - eta_min) * (
                        1 + math.cos(math.pi * warmup_epochs / epochs)) / 2
            else:
                warmup_to = learning_rate

        loss_train = []
        loss_valid = []

        step = 0

        for epoch in range(epochs):

            step_train = 0
            loss_training = 0
            total_training = 0

            for phase in ["train"]:

                if phase == "train":
                    model.train()

                for idx, (images, labels) in enumerate(loader_encoder):

                    images = torch.cat([images[0], images[1]], dim=0)
                    bsz = labels.shape[0]

                    if batch_size > 256 and warmup_epochs > 0 and epoch <= warmup_epochs:
                        p = (idx + (epoch - 1) * len(loader_encoder)) / \
                            (warmup_epochs * len(loader_encoder))
                        lr = warmup_from + p * (warmup_to - warmup_from)

                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                    # pruning setup
                    if phase == "train":
                        step += 1
                        step_train += 1

                        if step == begin_step and final_sparsity > 0.0:
                            if debug:
                                logging.info("-- Start encoder pruning --")
                            is_pruning = True

                        if is_pruning and final_sparsity > 0.0:
                            if step % frequency == 0:
                                utils.pruning.magnitude_pruning_encoder_clr(model=model,
                                                                            theta=gamma_parameters[pruning_iterations])
                                if debug:
                                    logging.info("Current encoder sparsity: {}".format(utils.pruning.get_model_sparsity_clr(model=model.encoder)))

                                pruning_iterations += 1

                        if step == end_step and final_sparsity > 0.0:
                            if debug:
                                logging.info("-- End encoder pruning --")
                            is_pruning = False

                    images, labels = images.to(device), labels.to(device)

                    with torch.set_grad_enabled(phase == "train"):

                        features = model(images)
                        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

                        # compute loss
                        if method == 'SupCon':
                            loss = criterion(features=features, labels=labels)

                        elif method == 'SimCLR':
                            loss = criterion(features=features)
                        else:
                            raise ValueError('contrastive method not supported: {}'.
                                             format(method))

                        if phase == "train":
                            loss_training += loss.item()
                            total_training += labels.size(0)

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

            current_train_loss = loss_training / step_train

            if not debug:
                logging.info(
                    "encoder epoch: {} loss contrastive: {} current_steps: {} model_sparsity: {:.2f}% learning rate: {}".format(
                        epoch, current_train_loss,
                        step, utils.pruning.get_model_sparsity_clr(model), optimizer.param_groups[0]['lr']))
                print(
                    "encoder epoch: {} loss contrastive: {} current_steps: {} model_sparsity: {:.2f}% learning rate: {}".format(
                        epoch, current_train_loss,
                        step, utils.pruning.get_model_sparsity_clr(model), optimizer.param_groups[0]['lr']))

            if debug_classification_head and epoch % debug_classification_head_epochs == 0 and epoch != 0:
                utils.clr_debug.classification_head_accuracy(model=model, encoder_name=encoder_name, device=device,
                                                             args=args)

            if cosine:
                scheduler_cos.step()

            loss_train.append(current_train_loss)

        if final_sparsity > 0.0:
            utils.pruning.remove_pruning_masks(model=model.encoder)

        if not debug:
            utils.data.plot_loss_curves(loss_train=loss_train, loss_valid=loss_valid, final_sparsity=final_sparsity,
                                        epochs=epochs, depth=depth, dropout=dropout, model_id=model_number,
                                        clr_method="encoder_{}".format(method))

        """
        Classification head training 
        """

        criterion_head = torch.nn.CrossEntropyLoss()
        classifier = utils.models.LinearClassifier(encoder_name=encoder_name, num_classes=num_classes)

        gamma_parameters_head = utils.pruning.get_gammas(initial_sparsity=initial_sparsity,
                                                         final_sparsity=final_sparsity,
                                                         begin_step=begin_step_head, end_step=end_step_head,
                                                         frequency=frequency_head)

        if final_sparsity > 0.0:
            logging.info("Gamma parameters head: {}".format(gamma_parameters_head))

        classifier.to(device)
        criterion_head.to(device)

        optimizer_head = optim.SGD(classifier.parameters(), lr=0.1, weight_decay=0.0, momentum=0.9)
        scheduler_decay_head = lr_scheduler.MultiStepLR(optimizer_head, milestones=[60, 75, 90], gamma=0.2)
        eta_min_head = optimizer_head.param_groups[0]['lr'] * (0.1 ** 3)
        scheduler_cos_head = lr_scheduler.CosineAnnealingLR(optimizer_head, T_max=head_epochs, eta_min=eta_min_head)

        accuracy_train_head = []
        accuracy_valid_head = []

        loss_train_head = []
        loss_valid_head = []

        pruning_iterations_head = 0
        step_head = 0

        logging.info(
            "-- Fine-tune classification head of sparsity {} number {} --".format(final_sparsity, model_number))

        print("Fine-tune classification head of sparsity {} number {}".format(final_sparsity, model_number))

        for epoch in range(head_epochs):

            step_train_head = 0
            loss_training_head = 0
            total_training_head = 0
            correct_training_head = 0

            step_validation_head = 0
            loss_validation_head = 0
            total_validation_head = 0
            correct_validation_head = 0

            for phase in ["train", "valid"]:

                if phase == "train":
                    model.eval()
                    classifier.train()

                if phase == "valid":
                    model.eval()
                    classifier.eval()

                for i, data in enumerate(loaders_head[phase]):

                    # pruning setup
                    if phase == "train":
                        step_head += 1
                        step_train_head += 1

                        if step_head == begin_step_head and final_sparsity > 0.0:
                            if debug:
                                logging.info("-- Start head pruning --")
                            is_pruning = True

                        if is_pruning:
                            if step_head % frequency_head == 0:
                                utils.pruning.magnitude_pruning_head_clr(head=classifier,
                                                                         theta=gamma_parameters_head[
                                                                             pruning_iterations_head])
                                if debug:
                                    if debug:
                                        logging.info("Current head sparsity: {}".format(
                                            utils.pruning.get_model_sparsity_clr_head(head=classifier)))

                                pruning_iterations_head += 1

                        if step_head == end_step_head and final_sparsity > 0.0:
                            if debug:
                                logging.info("-- End head pruning --")
                            is_pruning = False

                    images_head, labels_head = data
                    images_head, labels_head = images_head.to(device), labels_head.to(device)

                    with torch.no_grad():
                        features_head = model.encoder(images_head)

                    output_head = classifier(features_head.detach())

                    loss_head = criterion_head(output_head, labels_head)

                    _, predicted_head = torch.max(output_head.data, 1)

                    if phase == "train":
                        loss_training_head += loss_head.item()
                        step_train_head += 1
                        total_training_head += labels_head.size(0)
                        correct_training_head += (predicted_head == labels_head).sum().item()

                        optimizer_head.zero_grad()
                        loss_head.backward()
                        optimizer_head.step()

                    if phase == "valid":
                        loss_validation_head += loss.item()
                        step_validation_head += 1
                        total_validation_head += labels_head.size(0)
                        correct_validation_head += (predicted_head == labels_head).sum().item()

            current_train_loss_head = loss_training_head / step_train_head
            current_validation_loss_head = loss_validation_head / step_validation_head
            current_train_accuracy_head = 100 * correct_training_head / total_training_head
            current_validation_accuracy_head = 100 * correct_validation_head / total_validation_head

            logging.info(
                "Epoch: {} Learning rate: {} Loss training: {} Loss validation: {} Accuracy Training: {} Accuracy Validation: {} Head sparsity: {:.2f}%".format(
                    epoch, scheduler_decay_head.get_last_lr(), current_train_loss_head,
                    current_validation_loss_head, current_train_accuracy_head, current_validation_accuracy_head,
                    utils.pruning.get_model_sparsity_clr_head(classifier)))

            print(
                "Head epoch: {} Learning rate: {} Loss training: {} Loss validation: {} Accuracy Training: {} Accuracy Validation: {} Head sparsity: {:.2f}%".format(
                    epoch, scheduler_decay_head.get_last_lr(), current_train_loss_head,
                    current_validation_loss_head, current_train_accuracy_head, current_validation_accuracy_head,
                    utils.pruning.get_model_sparsity_clr_head(classifier)))

            scheduler_decay_head.step()
            scheduler_cos_head.step()

            loss_train_head.append(current_train_loss_head)
            accuracy_train_head.append(current_train_accuracy_head)

            loss_valid_head.append(current_validation_loss_head)
            accuracy_valid_head.append(current_validation_accuracy_head)

        if not debug:
            utils.data.plot_loss_curves(loss_train=loss_train_head, loss_valid=loss_valid_head,
                                        final_sparsity=final_sparsity,
                                        epochs=head_epochs, depth=depth, dropout=dropout, model_id=model_number,
                                        clr_method="head_finetune_{}".format(final_sparsity,
                                                                             model_number, method))

            utils.data.plot_accuracy_curves(accuracy_train=accuracy_train_head, accuracy_valid=accuracy_valid_head,
                                            epochs=head_epochs,
                                            depth=depth, dropout=dropout, final_sparsity=final_sparsity,
                                            model_id=model_number,
                                            clr_method="head_finetune_{}".format(method))

        if final_sparsity > 0.0:
            utils.pruning.remove_pruning_masks(model=classifier)

        if not debug:
            state_encoder = {
                'model': model.state_dict(),
            }

            state_head = {
                'head': classifier.state_dict(),
            }

            if begin_step > 2000:
                torch.save(state_head,
                           head_save_path.format(os.getcwd(),
                               encoder_name, method, final_sparsity, '_later', batch_size, temperature, head_epochs,
                               model_number))
                torch.save(state_encoder,
                           encoder_save_path.format(os.getcwd(),
                               encoder_name, method, final_sparsity, '_later', batch_size, temperature, epochs,
                               model_number))

            else:
                torch.save(state_head,
                           head_save_path.format(os.getcwd(),
                               encoder_name, method, final_sparsity, '', batch_size, temperature, head_epochs,
                               model_number))
                torch.save(state_encoder,
                           encoder_save_path.format(os.getcwd(),
                               encoder_name, method, final_sparsity, '', batch_size, temperature, epochs, model_number))

            del state_head
            del state_encoder