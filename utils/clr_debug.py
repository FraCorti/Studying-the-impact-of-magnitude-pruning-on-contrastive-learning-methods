import torch
from torch import optim
from torch.optim import lr_scheduler

import utils
from utils import models, data, losses
import logging


def classification_head_accuracy(model, device, encoder_name, args):
    logging.info("--- debug fine-tune classification head -- ")
    loaders_head_debug = utils.data.get_cifar10(batch_size=256, workers=0)
    criterion_head_debug = torch.nn.CrossEntropyLoss()
    classifier_head_debug = utils.models.LinearClassifier(encoder_name=encoder_name, num_classes=args.num_classes)

    classifier_head_debug.to(device)
    criterion_head_debug.to(device)

    optimizer_head_debug = optim.SGD(classifier_head_debug.parameters(), lr=0.1, weight_decay=0.0,
                                     momentum=args.momentum)
    scheduler_decay_head_debug = lr_scheduler.MultiStepLR(optimizer_head_debug, milestones=[60, 75, 90], gamma=0.2)
    eta_min_head_debug = optimizer_head_debug.param_groups[0]['lr'] * (0.1 ** 3)
    scheduler_cos = lr_scheduler.CosineAnnealingLR(optimizer_head_debug, T_max=100, eta_min=eta_min_head_debug)

    accuracy_train_head_debug = []
    accuracy_valid_head_debug = []

    loss_train_head_debug = []
    loss_valid_head_debug = []

    for epoch in range(2):

        step_train = 0
        loss_training = 0
        total_training = 0
        correct_training = 0

        step_validation = 0
        loss_validation = 0
        total_validation = 0
        correct_validation = 0

        # alternate training and validation phase
        for phase in ["train", "valid"]:

            if phase == "train":
                model.eval()
                classifier_head_debug.train()

            if phase == "valid":
                model.eval()
                classifier_head_debug.eval()

            # cycle on the batches of the train and validation dataset
            for i, data in enumerate(loaders_head_debug[phase]):

                images_head_debug, labels_head_debug = data
                images_head_debug, labels_head_debug = images_head_debug.to(device), labels_head_debug.to(device)

                with torch.no_grad():
                    features = model.encoder(images_head_debug)

                output = classifier_head_debug(features.detach())

                # compute loss
                loss = criterion_head_debug(output, labels_head_debug)

                # compute accuracy
                _, predicted_head_debug = torch.max(output.data, 1)

                if phase == "train":
                    loss_training += loss.item()
                    step_train += 1
                    total_training += labels_head_debug.size(0)
                    correct_training += (predicted_head_debug == labels_head_debug).sum().item()

                    optimizer_head_debug.zero_grad()
                    loss.backward()
                    optimizer_head_debug.step()

                if phase == "valid":
                    loss_validation += loss.item()
                    step_validation += 1
                    total_validation += labels_head_debug.size(0)
                    correct_validation += (predicted_head_debug == labels_head_debug).sum().item()

        current_train_loss = loss_training / step_train
        current_validation_loss = loss_validation / step_validation
        current_train_accuracy = 100 * correct_training / total_training
        current_validation_accuracy = 100 * correct_validation / total_validation

        logging.info(
            "head Epoch: {} loss training: {} loss validation: {} accuracy Training: {} accuracy Validation: {}".format(
                epoch, current_train_loss,
                current_validation_loss, current_train_accuracy, current_validation_accuracy))

        print("head Epoch: {} loss training: {} loss validation: {} accuracy Training: {} accuracy Validation: {}".format(
                epoch, current_train_loss,
                current_validation_loss, current_train_accuracy, current_validation_accuracy))

        # schedulers update
        scheduler_decay_head_debug.step()
        scheduler_cos.step()

        loss_train_head_debug.append(current_train_loss)
        accuracy_train_head_debug.append(current_train_accuracy)

        loss_valid_head_debug.append(current_validation_loss)
        accuracy_valid_head_debug.append(current_validation_accuracy)
