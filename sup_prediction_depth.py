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
parser.add_argument('--projection_head', default='mlp', type=str)
parser.add_argument('--figures_number', default=0, type=int)
parser.add_argument('--temperature', default=0.5, type=float)
parser.add_argument('--pies_to_store', default=1, type=int)
parser.add_argument('--plot_umap', default=False, action='store_true')
parser.add_argument('--prediction_depth_models', default=5, type=int, help='number of network used to compute the prediction score for each sample')
parser.add_argument('--encoder_name', default='WideResNet', type=str)
parser.add_argument('--pruning_methods', default='GMP,global', type=str)
parser.add_argument('--pruning_percentages', default='0.3,0.5,0.7,0.9', type=str)

args = parser.parse_args()
depth = args.model_depth
seed = args.seed
encoder_name = args.encoder_name
umap = args.plot_umap
prediction_depth_models = args.prediction_depth_models
cuda_deterministic = args.cuda_deterministic
batch_size = args.batch_size
figures_number = args.figures_number
widen_factor = args.widen_factor
dropout = args.dropout
cuda_device = args.cuda_device
projection_head = args.projection_head
debug = args.debug
models_number = args.models_number
workers = args.workers
temperature = args.temperature
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

    today = date.today()
    day = today.strftime("%d_%m_%Y")
    now = datetime.now()
    time = now.strftime("%H_%M_%S")
    logging.basicConfig(
        filename="{}/logs/Prediction_Depth_Supervised_wideResnet_day:{}_time:{}.log".format(os.getcwd(),
            day, time),
        level=logging.INFO)

for seed in range(0, total_seed, 1):

    print("Current seed: {}".format(seed))


    if cuda_deterministic:
        print("Setting Pytorch and CUBLAS to deterministic behaviour with seed: {}".format(seed))
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    not_pruned_models = utils.models.get_supervised_models(models_number=models_number, debug=debug, sparsity=0.0,
                                                           device=device,
                                                           prediction_depth=True)
    prediction_depth_not_pruned_objects = []
    for model_id in range(prediction_depth_models):
        prediction_depth_not_pruned_objects.append(
            utils.prediction_depth.PredictionDepth(model_depth=16, k=30, method='Supervised', model_id=model_id,
                                                       serialize=True, deserialize=True,
                                                       model_sparsity=0.0))

    loaders = utils.data.get_cifar10(batch_size=batch_size, workers=workers, shuffle=False)
    print("Models loaded: {} Batch Size: {}, Deterministic Setup: {}, Models depth: {}, Models widen factor: {}".format(
        models_number, batch_size, cuda_deterministic, depth, widen_factor))

    for pruning_method in pruning_methods:
        for models_pruning in pruning_percentages:


            pruned_models = utils.models.get_supervised_models(models_number=models_number, debug=debug,
                                                               sparsity=models_pruning, device=device,
                                                               prediction_depth=True, pruning_method=pruning_method)

            prediction_depth_pruned_objects = []
            for model_id in range(prediction_depth_models):
                prediction_depth_pruned_objects.append(
                    utils.prediction_depth.PredictionDepth(model_depth=16, k=30, method='Supervised', model_id=model_id,
                                                            serialize=True, deserialize=True,
                                                            model_sparsity=models_pruning, pruning_method=pruning_method))

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

                        for pruned_model in range(len(pruned_models)):
                            pruned_models[pruned_model].eval()

                            with torch.no_grad():
                                if pruned_model < prediction_depth_models and \
                                        prediction_depth_pruned_objects[pruned_model].pre_trained() is False:
                                    output_model = pruned_models[pruned_model](input_)
                                    latent_representation = pruned_models[pruned_model].get_latent_representation()

                                    prediction_depth_pruned_objects[pruned_model].add_hidden_representations(
                                        hidden_features_representations=pruned_models[
                                            pruned_model].get_hidden_representations(),
                                        labels=label, batch_size=len(input_))

                        # train classifier for 0.0 sparsity
                        for model in range(len(not_pruned_models)):
                            not_pruned_models[model].eval()

                            with torch.no_grad():
                                if model < prediction_depth_models and \
                                        prediction_depth_not_pruned_objects[model].pre_trained() is False:
                                    output_model = not_pruned_models[model](input_)
                                    latent_representation = not_pruned_models[model].get_latent_representation()

                                    prediction_depth_not_pruned_objects[model].add_hidden_representations(
                                        hidden_features_representations=not_pruned_models[
                                            model].get_hidden_representations(),
                                        labels=label, batch_size=len(input_))

            # train the KNNs classifiers
            for model_number in range(len(prediction_depth_pruned_objects)):

                if prediction_depth_pruned_objects[model_number].pre_trained() is False:
                    prediction_depth_pruned_objects[model_number].fit_knns()

                if prediction_depth_not_pruned_objects[model_number].pre_trained() is False:
                    prediction_depth_not_pruned_objects[model_number].fit_knns()

            print("Starting to analyze test set")
            prediction_depth_PIEs_pruned = []
            prediction_depth_notPIEs_pruned = []

            prediction_depth_PIEs_notpruned = []
            prediction_depth_notPIEs_notpruned = []

            q_score_PIEs_pruned = []
            q_score_notPIEs_pruned = []

            prediction_depth_PIEs_comparison = []
            prediction_depth_notPIEs_comparison = []

            for input_, label in loaders["valid"]:

                batch_pies = 0
                total_samples += label.size(0)

                if debug:
                    logging.info("Input shape {}".format(input_.shape))

                input_ = input_.to(device)
                label = label.to(device)

                results_model_notpruned = []
                results_model_pruned = []

                most_frequent_labels_not_pruned = []
                most_frequent_labels_pruned = []

                q_score_pruned = []

                prediction_depth_pruned = np.zeros(shape=(len(input_), 1), dtype=float)
                prediction_depth_not_pruned = np.zeros(shape=(len(input_), 1), dtype=float)

                for not_pruned_model in range(len(not_pruned_models)):
                    not_pruned_models[not_pruned_model].eval()

                    with torch.no_grad():
                        output_model = not_pruned_models[not_pruned_model](input_)

                        if not_pruned_model < prediction_depth_models:
                            prediction_depth_not_pruned = np.add(prediction_depth_not_pruned,
                                prediction_depth_not_pruned_objects[not_pruned_model].inference_knn(
                                    sample_hidden_representations=not_pruned_models[
                                        not_pruned_model].get_hidden_representations(),
                                    sample_class=label,
                                    batch_size=len(input_)))

                    _, preds_output_model = torch.max(output_model, 1)
                    results_model_notpruned.append(preds_output_model.cpu().detach().numpy())

                for i in range(len(input_)):
                    models_not_pruned_results = []

                    for model_output in range(len(results_model_notpruned)):
                        models_not_pruned_results.append(results_model_notpruned[model_output][i])

                    most_frequent_labels_not_pruned.append(np.bincount(np.array(models_not_pruned_results)).argmax())

                for pruned_model in range(len(pruned_models)):

                    pruned_models[pruned_model].eval()

                    with torch.no_grad():
                        output_model_pruned = pruned_models[pruned_model](input_)

                        if pruned_model < prediction_depth_models:
                            prediction_depth_pruned = np.add(prediction_depth_pruned,
                                prediction_depth_pruned_objects[pruned_model].inference_knn(
                                    sample_hidden_representations=pruned_models[
                                        pruned_model].get_hidden_representations(),
                                    batch_size=len(input_),
                                    sample_class=label))

                    _, preds_output_model_pruned = torch.max(output_model_pruned, 1)

                    results_model_pruned.append(preds_output_model_pruned.cpu().detach().numpy())

                if prediction_depth_models:
                    prediction_depth_pruned = np.divide(prediction_depth_pruned, prediction_depth_models)
                    prediction_depth_not_pruned = np.divide(prediction_depth_not_pruned, prediction_depth_models)

                for i in range(len(input_)):
                    models_pruned_results = []

                    for model_output in range(len(results_model_pruned)):
                        models_pruned_results.append(results_model_pruned[model_output][i])

                    most_frequent_labels_pruned.append(np.bincount(np.array(models_pruned_results)).argmax())

                for i in range(len(input_)):

                    if most_frequent_labels_not_pruned[i] != most_frequent_labels_pruned[i]:

                        if prediction_depth_models:
                            prediction_depth_PIEs_pruned.append(prediction_depth_pruned[i])
                            prediction_depth_PIEs_notpruned.append(prediction_depth_not_pruned[i])

                            prediction_depth_PIEs_comparison.append((prediction_depth_not_pruned[i],
                                                                     prediction_depth_pruned[i]))
                            saved_pies += 1
                        batch_pies += 1
                    else:
                        if prediction_depth_models:

                            prediction_depth_notPIEs_pruned.append(prediction_depth_pruned[i])
                            prediction_depth_notPIEs_notpruned.append(prediction_depth_not_pruned[i])

                            prediction_depth_notPIEs_comparison.append((prediction_depth_not_pruned[i],
                                                                        prediction_depth_pruned[i]))


                batch_index += 1
                total_pies += batch_pies
                if prediction_depth_models:
                    print("Number of samples analyzed: {} Prediction Depth mean pruned: {} Prediction Depth mean not-pruned: {}".format(total_samples, prediction_depth_pruned.mean(), prediction_depth_not_pruned.mean()))

            if prediction_depth_models:
                plot_prediction_depth_pruned_against_not_pruned(prediction_depth_PIEs=prediction_depth_PIEs_comparison,
                                                                prediction_depth_notPIEs=prediction_depth_notPIEs_comparison,
                                                                models_number_Prediction_Depth=prediction_depth_models,
                                                                models_pruning=models_pruning, pruning_method=pruning_method,
                                                                method='supervised')

                print(
                    "Method: {} Sparsity: {} Pruning method: {} Prediction Depth models: {} Average Prediction depth PIEs: {} Average Prediction depth notPIEs: {} Std Prediction depth PIEs: {} Std Prediction depth notPIEs: {}".format(
                        'Supervised', models_pruning, pruning_method, prediction_depth_models,
                        np.array(prediction_depth_PIEs_pruned).mean(),
                        np.array(prediction_depth_notPIEs_pruned).mean(),
                        np.array(prediction_depth_PIEs_pruned).std(),
                        np.array(prediction_depth_notPIEs_pruned).std()))
                print(
                    "Method: {} Pruning method: {} Sparsity: {} Prediction Depth models: {} Average Prediction depth PIEs: {} Average Prediction depth notPIEs: {} Std Prediction depth PIEs: {} Std Prediction depth notPIEs: {}".format(
                        'Supervised', 'NOT-pruned', 0.0, prediction_depth_models,
                        np.array(prediction_depth_PIEs_notpruned).mean(),
                        np.array(prediction_depth_notPIEs_notpruned).mean(),
                        np.array(prediction_depth_notPIEs_notpruned).std(),
                        np.array(prediction_depth_notPIEs_notpruned).std()))

            print("Method: {} Pruning method: {} Pruned: {} Models used: {} Found PIEs: {}".format(
                'Supervised', pruning_method,
                models_pruning,
                models_number,
                total_pies))
