import numpy as np
import argparse
from datetime import datetime, date
import logging
import os
import utils
from utils import data, models, prediction_depth
import random

import torch

from utils.prediction_depth import plot_prediction_depth_pruned_against_not_pruned

parser = argparse.ArgumentParser()
parser.add_argument('--models_number', default=30, type=int)
parser.add_argument('--batch_size', default=1024, type=int)
parser.add_argument('--model_depth', default=16, type=int)
parser.add_argument('--debug', default=False, action='store_true')
parser.add_argument('--cuda_device', default=0, type=int)
parser.add_argument('--dropout', default=0.0, type=float)
parser.add_argument('--workers', default=0, type=int)
parser.add_argument('--widen_factor', default=2, type=int)
parser.add_argument('--cuda_deterministic', default=True, type=bool)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--total_seed', default=1, type=int)
parser.add_argument('--clr_method', default="SupCon", type=str)
parser.add_argument('--projection_head', default='mlp', type=str)
parser.add_argument('--figures_number', default=0, type=int)
parser.add_argument('--temperature', default=0.5, type=float)
parser.add_argument('--pies_to_store', default=1, type=int)
parser.add_argument('--plot_umap', default=False, action='store_true')
parser.add_argument('--prediction_depth_models', default=5, type=int,
                    help='number of network used to compute the prediction score for each sample')
parser.add_argument('--encoder_name', default='WideResNet', type=str)
parser.add_argument('--pruning_methods', default='GMP,global,later GMP', type=str)
parser.add_argument('--pruning_percentages', default='0.3,0.5,0.7,0.9', type=str)

args = parser.parse_args()
depth = args.model_depth
seed = args.seed
encoder_name = args.encoder_name
prediction_depth_models = args.prediction_depth_models
umap = args.plot_umap
cuda_deterministic = args.cuda_deterministic
batch_size = args.batch_size
widen_factor = args.widen_factor
dropout = args.dropout
cuda_device = args.cuda_device
projection_head = args.projection_head
debug = args.debug
models_number = args.models_number
workers = args.workers
temperature = args.temperature
method = args.clr_method
total_seed = args.total_seed
pies_to_store = args.pies_to_store
pruning_methods = args.pruning_methods.split(',')

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
    torch.use_deterministic_algorithms(True)

if debug:
    today = date.today()
    day = today.strftime("%d_%m_%Y")
    now = datetime.now()
    time = now.strftime("%H_%M_%S")
    logging.basicConfig(
        filename="{}/logs/CLR_PIEs_DEBUG_wideResnet_day:{}_time:{}.log".format(os.getcwd(),
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

    clr_models_notpruned_encoders, clr_models_notpruned_heads = utils.models.get_clr_models(models_number=models_number,
                                                                                            encoder_name=encoder_name,
                                                                                            method=method,
                                                                                            sparsity=0.0,
                                                                                            temperature=temperature,
                                                                                            device=device,
                                                                                            projection_head=projection_head,
                                                                                            prediction_depth=True)
    prediction_depth_not_pruned_objects = []

    for model_id in range(prediction_depth_models):
        prediction_depth_not_pruned_objects.append(
            utils.prediction_depth.PredictionDepth(model_depth=16, k=30, method=method, model_id=model_id,
                                                   serialize=True, deserialize=True,
                                                   model_sparsity=0.0, pruning_method='GMP'))

    loaders = utils.data.get_cifar10(batch_size=batch_size, workers=workers, shuffle=False)
    print(
        "Models loaded: {} Pruning percentages: {} Pruning methods: {} Batch Size: {}, Deterministic Setup: {}, Models depth: {}, Models widen factor: {}".format(
            models_number, pruning_percentages, pruning_methods, batch_size, cuda_deterministic, depth, widen_factor))

    for pruning_method in pruning_methods:

        for models_pruning in pruning_percentages:

            clr_pruned_models_encoders, clr_pruned_models_heads = utils.models.get_clr_models(
                models_number=models_number,
                encoder_name=encoder_name,
                method=method,
                sparsity=models_pruning,
                temperature=temperature,
                device=device,
                pruning_method=pruning_method,
                projection_head=projection_head,
                prediction_depth=True)

            prediction_depth_pruned_objects = []

            for model_id in range(prediction_depth_models):
                prediction_depth_pruned_objects.append(
                    utils.prediction_depth.PredictionDepth(model_depth=16, k=30, method=method, model_id=model_id,
                                                           serialize=True, deserialize=True,
                                                           model_sparsity=models_pruning,
                                                           pruning_method=pruning_method))

            total_pies = 0
            batch_index = 0

            total_samples = 0
            correct_clr = 0
            correct_clr_notpruned = 0

            saved_pies = 0

            train_KNNs_classifiers = False
            for prediction_depth_number in range(len(prediction_depth_pruned_objects)):

                if prediction_depth_pruned_objects[prediction_depth_number].pre_trained() is False:
                    train_KNNs_classifiers = True

                if prediction_depth_not_pruned_objects[prediction_depth_number].pre_trained() is False:
                    train_KNNs_classifiers = True

            with torch.no_grad():
                if train_KNNs_classifiers:
                    print("Training KNNs classifiers")
                    for input_, label in loaders["train"]:

                        input_ = input_.to(device)
                        label = label.to(device)

                        for clr_model_pruned_number in range(len(clr_pruned_models_encoders)):
                            clr_pruned_models_encoders[clr_model_pruned_number].eval()
                            clr_pruned_models_heads[clr_model_pruned_number].eval()

                            with torch.no_grad():
                                if clr_model_pruned_number < prediction_depth_models and \
                                        prediction_depth_pruned_objects[
                                            clr_model_pruned_number].pre_trained() is False:
                                    latent_representation_pruned = clr_pruned_models_encoders[
                                        clr_model_pruned_number].encoder(
                                        input_)
                                    prediction_depth_pruned_objects[clr_model_pruned_number].add_hidden_representations(
                                        hidden_features_representations=clr_pruned_models_encoders[
                                            clr_model_pruned_number].get_hidden_representations(),
                                        labels=label, batch_size=len(input_))

                        for clr_model_not_pruned_number in range(len(clr_models_notpruned_encoders)):
                            clr_models_notpruned_encoders[clr_model_not_pruned_number].eval()
                            clr_models_notpruned_heads[clr_model_not_pruned_number].eval()

                            with torch.no_grad():
                                if clr_model_not_pruned_number < prediction_depth_models and \
                                        prediction_depth_not_pruned_objects[
                                            clr_model_not_pruned_number].pre_trained() is False:
                                    latent_representation_pruned = clr_models_notpruned_encoders[
                                        clr_model_not_pruned_number].encoder(
                                        input_)
                                    prediction_depth_not_pruned_objects[
                                        clr_model_not_pruned_number].add_hidden_representations(
                                        hidden_features_representations=clr_models_notpruned_encoders[
                                            clr_model_not_pruned_number].get_hidden_representations(),
                                        labels=label, batch_size=len(input_))

                for model_number in range(len(prediction_depth_pruned_objects)):

                    if prediction_depth_pruned_objects[model_number].pre_trained() is False:
                        prediction_depth_pruned_objects[model_number].fit_knns()

                    if prediction_depth_not_pruned_objects[model_number].pre_trained() is False:
                        prediction_depth_not_pruned_objects[model_number].fit_knns()

                q_score_PIEs_pruned = []
                q_score_notPIEs_pruned = []

                prediction_depth_PIEs_pruned = []
                prediction_depth_notPIEs_pruned = []

                prediction_depth_PIEs_notpruned = []
                prediction_depth_notPIEs_notpruned = []

                prediction_depth_PIEs_comparison = []
                prediction_depth_notPIEs_comparison = []

                print("Starting to analyze test set")
                for input_, label in loaders["valid"]:

                    batch_pies = 0
                    total_samples += label.size(0)

                    if debug:
                        logging.info("Input shape {}".format(input_.shape))

                    input_ = input_.to(device)
                    label = label.to(device)

                    results_model_clr_notpruned = []
                    results_model_clr = []

                    most_frequent_labels_clr_notpruned = []
                    most_frequent_labels_clr = []

                    prediction_depth_pruned = np.zeros(shape=(len(input_), 1), dtype=float)
                    prediction_depth_not_pruned = np.zeros(shape=(len(input_), 1), dtype=float)

                    q_score_pruned = []

                    for clr_model_notpruned_number in range(len(clr_models_notpruned_encoders)):
                        clr_models_notpruned_encoders[clr_model_notpruned_number].eval()
                        clr_models_notpruned_heads[clr_model_notpruned_number].eval()

                        with torch.no_grad():
                            latent_representation = clr_models_notpruned_encoders[clr_model_notpruned_number].encoder(
                                input_)

                            if clr_model_notpruned_number < prediction_depth_models:
                                prediction_depth_not_pruned = np.add(prediction_depth_not_pruned,
                                                                     prediction_depth_not_pruned_objects[
                                                                         clr_model_notpruned_number].inference_knn(
                                                                         sample_hidden_representations=
                                                                         clr_models_notpruned_encoders[
                                                                             clr_model_notpruned_number].get_hidden_representations(),
                                                                         batch_size=len(input_),
                                                                         sample_class=label))

                            output_model = clr_models_notpruned_heads[clr_model_notpruned_number](latent_representation)

                        _, preds_output_model = torch.max(output_model, 1)
                        correct_clr_notpruned += (preds_output_model == label).sum().item()
                        results_model_clr_notpruned.append(preds_output_model.cpu().detach().numpy())

                    for i in range(len(input_)):
                        models_not_pruned_results = []

                        for model_output in range(len(results_model_clr_notpruned)):
                            models_not_pruned_results.append(results_model_clr_notpruned[model_output][i])

                        most_frequent_labels_clr_notpruned.append(
                            np.bincount(np.array(models_not_pruned_results)).argmax())

                    for clr_model_pruned_number in range(len(clr_pruned_models_encoders)):
                        clr_pruned_models_encoders[clr_model_pruned_number].eval()
                        clr_pruned_models_heads[clr_model_pruned_number].eval()

                        with torch.no_grad():
                            latent_representation_pruned = clr_pruned_models_encoders[clr_model_pruned_number].encoder(
                                input_)

                            if clr_model_pruned_number < prediction_depth_models:
                                prediction_depth_pruned = np.add(prediction_depth_pruned,
                                                                 prediction_depth_pruned_objects[
                                                                     clr_model_pruned_number].inference_knn(
                                                                     sample_hidden_representations=
                                                                     clr_pruned_models_encoders[
                                                                         clr_model_pruned_number].get_hidden_representations(),
                                                                     batch_size=len(input_),
                                                                     sample_class=label))

                            output_clr_model_pruned = clr_pruned_models_heads[clr_model_pruned_number](
                                latent_representation_pruned)

                        _, preds_output_model_clr = torch.max(output_clr_model_pruned, 1)
                        correct_clr += (preds_output_model_clr == label).sum().item()
                        results_model_clr.append(preds_output_model_clr.cpu().detach().numpy())

                    for i in range(len(input_)):
                        models_pruned_results = []

                        for model_output in range(len(results_model_clr)):
                            models_pruned_results.append(results_model_clr[model_output][i])

                        most_frequent_labels_clr.append(np.bincount(np.array(models_pruned_results)).argmax())

                    if prediction_depth_models:
                        prediction_depth_pruned = np.divide(prediction_depth_pruned, prediction_depth_models)
                        prediction_depth_not_pruned = np.divide(prediction_depth_not_pruned, prediction_depth_models)

                    for i in range(len(input_)):

                        if most_frequent_labels_clr_notpruned[i] != most_frequent_labels_clr[i]:

                            if prediction_depth_models:
                                prediction_depth_PIEs_pruned.append(prediction_depth_pruned[i])
                                prediction_depth_PIEs_notpruned.append(prediction_depth_not_pruned[i])

                                prediction_depth_PIEs_comparison.append((prediction_depth_not_pruned[i],
                                                                         prediction_depth_pruned[i]))
                            batch_pies += 1
                        else:
                            if prediction_depth_models:
                                prediction_depth_notPIEs_pruned.append(prediction_depth_pruned[i])
                                prediction_depth_notPIEs_notpruned.append(prediction_depth_not_pruned[i])

                                prediction_depth_notPIEs_comparison.append((prediction_depth_not_pruned[i],
                                                                            prediction_depth_pruned[i]))

                            if debug:
                                logging.info(
                                    "PIEs found inside this batch: {} , total PIEs currently found: {}".format(
                                        batch_pies,
                                        total_pies))
                    batch_index += 1
                    total_pies += batch_pies
                    if prediction_depth_models:
                        print(
                            "Number of samples analyzed: {} Prediction Depth mean pruned: {} Prediction Depth mean not-pruned: {}".format(
                                total_samples, prediction_depth_pruned.mean(), prediction_depth_not_pruned.mean()))

            if prediction_depth_models:
                print(
                    "Method: {} Pruning method: {} Sparsity: {} Prediction Depth models: {} Average Prediction depth PIEs: {} Average Prediction depth notPIEs: {} Std Prediction depth PIEs: {} Std Prediction depth notPIEs: {}".format(
                        method, pruning_method, models_pruning, prediction_depth_models,
                        np.array(prediction_depth_PIEs_pruned).mean(),
                        np.array(prediction_depth_notPIEs_pruned).mean(),
                        np.array(prediction_depth_PIEs_pruned).std(),
                        np.array(prediction_depth_notPIEs_pruned).std()))
                print(
                    "Method: {} Pruning method: {} Sparsity: {} Prediction Depth models: {} Average Prediction depth PIEs: {} Average Prediction depth notPIEs: {} Std Prediction depth PIEs: {} Std Prediction depth notPIEs: {}".format(
                        method, 'NOT-pruned', 0.0, prediction_depth_models,
                        np.array(prediction_depth_PIEs_notpruned).mean(),
                        np.array(prediction_depth_notPIEs_notpruned).mean(),
                        np.array(prediction_depth_notPIEs_notpruned).std(),
                        np.array(prediction_depth_notPIEs_notpruned).std()))

                plot_prediction_depth_pruned_against_not_pruned(prediction_depth_PIEs=prediction_depth_PIEs_comparison,
                                                                prediction_depth_notPIEs=prediction_depth_notPIEs_comparison,
                                                                models_number_Prediction_Depth=prediction_depth_models,
                                                                models_pruning=models_pruning,
                                                                pruning_method=pruning_method,
                                                                method=method)

            print("Method: {} Pruning method: {} Pruned: {} Models used: {} Found PIEs: {}".format(
                method, pruning_method,
                models_pruning,
                models_number,
                total_pies))