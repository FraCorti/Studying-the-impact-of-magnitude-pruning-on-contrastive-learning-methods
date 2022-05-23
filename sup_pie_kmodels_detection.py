import numpy as np
import argparse
from datetime import datetime, date
import logging
import os
import utils
from utils import data, pies, models
import random
import torch
from utils.umap_application import plot_umap

parser = argparse.ArgumentParser()
parser.add_argument('--models_number', default=30, type=int)
parser.add_argument('--batch_size', default=1024, type=int)
parser.add_argument('--model_depth', default=16, type=int)
parser.add_argument('--debug', default=False, type=bool)
parser.add_argument('--cuda_device', default=0, type=int)
parser.add_argument('--dropout', default=0.0, type=float)
parser.add_argument('--workers', default=0, type=int)
parser.add_argument('--widen_factor', default=2, type=int)
parser.add_argument('--cuda_deterministic', default=True, type=bool)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--total_seed', default=1, type=int)
parser.add_argument('--plot_umap', default=False, action='store_true')
parser.add_argument('--plot_q_score', default=False, action='store_true')
parser.add_argument('--not_serialize_plots', default=True, action='store_false')

args = parser.parse_args()
depth = args.model_depth
not_serialize_plots = args.not_serialize_plots
umap = args.plot_umap
plot_q_score = args.plot_q_score
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

device = torch.device("cuda:{}".format(cuda_device) if torch.cuda.is_available() else "cpu")

if cuda_deterministic:
    # see https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

if debug:
    today = date.today()
    day = today.strftime("%d_%m_%Y")
    now = datetime.now()
    time = now.strftime("%H_%M_%S")
    logging.basicConfig(
        filename="{}/logs/PIEs_DEBUG_wideResnet_day:{}_time:{}.log".format(os.getcwd(),
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

    loaders = utils.data.get_ordered_cifar10_validation(batch_size=batch_size, workers=workers, seed=seed)

    print("Models loaded: {} Batch Size: {}, Deterministic Setup: {}, Models depth: {}, Models widen factor: {}".format(
        models_number, batch_size, cuda_deterministic, depth, widen_factor))

    models = utils.models.get_supervised_models(models_number=models_number, debug=debug, sparsity=0.0,
                                                device=device)

    for pruning_method in ['global', 'GMP']:

        plot_no_sparsity_umap = False

        for models_pruning in [0.3, 0.5, 0.7, 0.9, 0.95, 0.99]:

            pruned_models = utils.models.get_supervised_models(models_number=models_number, debug=debug,
                                                               sparsity=models_pruning, device=device,
                                                               pruning_method=pruning_method)

            total_pies = 0
            batch_index = 0

            saved_pies = 0
            current_image = 0

            q_score_PIEs = []
            q_score_notPIEs = []

            l1_norm_PIEs = []
            l1_norm_notPIEs = []

            mean_features_PIEs = []
            mean_features_notPIEs = []

            standard_deviation_features_PIEs = []
            standard_deviation_features_notPIEs = []

            latent_representations_pruned = np.empty(shape=[1, 129], dtype=float)
            latent_representations_not_pruned = np.empty(shape=[1, 129], dtype=float)

            z_score_PIEs = []
            z_score_notPIEs = []

            modal_labels_pruned = []
            modal_labels_not_pruned = []

            with torch.no_grad():
                for input_, label in loaders["valid"]:

                    batch_pies = 0

                    if debug:
                        logging.info("Input shape {}".format(input_.shape))

                    input_ = input_.to(device)
                    label = label.to(device)

                    results_model_notpruned = []
                    results_model_pruned = []

                    most_frequent_labels_not_pruned = []
                    most_frequent_labels_pruned = []

                    q_score_pruned = np.zeros((len(input_)), dtype=float)
                    q_score_notpruned = np.zeros((len(input_)), dtype=float)

                    mean_score_features_pruned = []
                    mean_score_features_notpruned = []

                    standard_deviation_features_pruned = []
                    standard_deviation_features_notpruned = []

                    l1_norm_pruned = np.zeros((len(input_)), dtype=float)
                    l1_norm_not_pruned = np.zeros((len(input_)), dtype=float)

                    z_score_pruned = np.zeros((len(input_)), dtype=float)
                    z_score_not_pruned = np.zeros((len(input_)), dtype=float)

                    average_latent_representation_pruned = np.zeros((len(input_), 128), dtype=float)
                    average_latent_representation_not_pruned = np.zeros((len(input_), 128), dtype=float)

                    for model in models:
                        model.eval()

                        with torch.no_grad():
                            output_model = model(input_)
                            latent_representation = model.get_latent_representation()

                            if plot_q_score:
                                latent_representation = latent_representation.cpu().detach().numpy()
                                for i in range(len(latent_representation)):
                                    single_latent_representation = latent_representation[i]
                                    q_score, z_score, l1_norm, mean, standard_deviation = utils.pies.q_score(
                                        latent_representation=single_latent_representation / np.linalg.norm(
                                            single_latent_representation, ord=2))
                                    q_score_notpruned[i] += q_score
                                    z_score_not_pruned[i] += z_score
                                    l1_norm_not_pruned[i] += l1_norm

                            if umap and not plot_no_sparsity_umap:
                                average_latent_representation_not_pruned = np.add(
                                    average_latent_representation_not_pruned,
                                    latent_representation.cpu().detach().numpy())

                        _, preds_output_model = torch.max(output_model, 1)
                        results_model_notpruned.append(preds_output_model.cpu().detach().numpy())

                    for i in range(len(input_)):
                        models_not_pruned_results = []

                        for model_output in range(len(results_model_notpruned)):
                            models_not_pruned_results.append(results_model_notpruned[model_output][i])

                        most_frequent_labels_not_pruned.append(
                            np.bincount(np.array(models_not_pruned_results)).argmax())

                    if debug:
                        logging.info("NOT PRUNED modal label: {}".format(most_frequent_labels_not_pruned))
                        logging.info("NOT PRUNED models output: {}".format(models_not_pruned_results))

                    for model in pruned_models:
                        model.eval()

                        with torch.no_grad():
                            output_model_pruned = model(input_)
                            latent_representation_pruned = model.get_latent_representation()

                            if plot_q_score:
                                latent_representation_pruned = latent_representation_pruned.cpu().detach().numpy()
                                for i in range(len(latent_representation_pruned)):
                                    single_latent_representation = latent_representation_pruned[i]
                                    q_score_p, z_score_p, l1_norm_p, mean_p, standard_deviation_p = utils.pies.q_score(
                                        latent_representation=single_latent_representation / np.linalg.norm(
                                            single_latent_representation, ord=2))
                                    q_score_pruned[i] += q_score_p
                                    z_score_pruned[i] += z_score_p
                                    l1_norm_pruned[i] += l1_norm_p

                            if umap:
                                average_latent_representation_pruned = np.add(average_latent_representation_pruned,
                                                                              latent_representation_pruned.cpu().detach().numpy())

                        _, preds_output_model_pruned = torch.max(output_model_pruned, 1)
                        results_model_pruned.append(preds_output_model_pruned.cpu().detach().numpy())

                    for i in range(len(input_)):
                        models_pruned_results = []

                        for model_output in range(len(results_model_pruned)):
                            models_pruned_results.append(results_model_pruned[model_output][i])

                        most_frequent_labels_pruned.append(np.bincount(np.array(models_pruned_results)).argmax())

                    if debug:
                        logging.info("PRUNED modal label: {}".format(most_frequent_labels_pruned))
                        logging.info("PRUNED models output: {}".format(models_pruned_results))

                    if plot_q_score:
                        q_score_pruned = np.divide(q_score_pruned, models_number)
                        z_score_pruned = np.divide(z_score_pruned, models_number)
                        l1_norm_pruned = np.divide(l1_norm_pruned, models_number)

                        q_score_notpruned = np.divide(q_score_notpruned, models_number)
                        z_score_not_pruned = np.divide(z_score_not_pruned, models_number)
                        l1_norm_not_pruned = np.divide(l1_norm_not_pruned, models_number)

                    for i in range(len(input_)):

                        modal_labels_pruned.append(most_frequent_labels_pruned[i])
                        modal_labels_not_pruned.append(most_frequent_labels_not_pruned[i])

                        if most_frequent_labels_not_pruned[i] != most_frequent_labels_pruned[i]:

                            if plot_q_score:
                                q_score_PIEs.append(
                                    (q_score_notpruned[i], q_score_pruned[i]))
                                z_score_PIEs.append(
                                    (z_score_not_pruned[i], z_score_pruned[i]))
                                l1_norm_PIEs.append(
                                    (l1_norm_not_pruned[i], l1_norm_pruned[i]))

                            if umap:
                                latent_representations_pruned = np.vstack((latent_representations_pruned,
                                                                           np.append(np.divide(
                                                                               average_latent_representation_pruned[i,
                                                                               :],
                                                                               models_number), 1)))
                                if not plot_no_sparsity_umap:
                                    latent_representations_not_pruned = np.vstack((latent_representations_not_pruned,
                                                                                   np.append(np.divide(
                                                                                       average_latent_representation_not_pruned[
                                                                                       i, :],
                                                                                       models_number), 1)))

                                saved_pies += 1

                            logging.info(
                                "not_pruned label: {} pruned label: {}".format(most_frequent_labels_not_pruned,
                                                                               most_frequent_labels_pruned[i]))
                            logging.info("Current PIEs counter value: {}".format(batch_pies))
                            batch_pies += 1
                        else:

                            if plot_q_score:
                                q_score_notPIEs.append(
                                    (q_score_notpruned[i], q_score_pruned[i]))
                                z_score_notPIEs.append(
                                    (z_score_not_pruned[i], z_score_pruned[i]))
                                l1_norm_notPIEs.append(
                                    (l1_norm_not_pruned[i], l1_norm_pruned[i]))

                            if umap:
                                latent_representations_pruned = np.vstack((latent_representations_pruned,
                                                                           np.append(np.divide(
                                                                               average_latent_representation_pruned[i,
                                                                               :],
                                                                               models_number), 0)))
                                if not plot_no_sparsity_umap:
                                    latent_representations_not_pruned = np.vstack(
                                        (latent_representations_not_pruned,
                                         np.append(np.divide(
                                             average_latent_representation_not_pruned[
                                             i, :],
                                             models_number), 0)))

                    batch_index += 1
                    total_pies += batch_pies
                    current_image += 1
                    if debug:
                        logging.info(
                            "PIEs found inside this batch: {} , total PIEs currently found: {}".format(batch_pies,
                                                                                                       total_pies))

            if plot_q_score:
                utils.pies.plot_q_scores(q_score_notPIEs=q_score_notPIEs, q_score_PIEs=q_score_PIEs,
                                         sparsity=models_pruning, method='supervised', pruning_method=pruning_method,
                                         serialize=True)

                utils.pies.print_z_score_l1_norm_mean_std(l1_norm_PIEs=l1_norm_PIEs,
                                                          l1_norm_notPIEs=l1_norm_notPIEs,
                                                          z_score_notPIEs=z_score_notPIEs,
                                                          z_score_PIEs=z_score_PIEs, sparsity=models_pruning,
                                                          pruning_method=pruning_method,
                                                          serialize=False,
                                                          method='supervised')
            if umap:

                if not plot_no_sparsity_umap:
                    utils.umap_application.plot_umap(latent_representations=latent_representations_not_pruned,
                                                     models_pruning=0.0,
                                                     method='supervised', pruning_method=pruning_method,
                                                     labels_predicted=modal_labels_not_pruned,
                                                     serialize=not_serialize_plots,
                                                     plot_no_sparsity=True)
                    plot_no_sparsity_umap = True

                utils.umap_application.plot_umap(latent_representations=latent_representations_pruned,
                                                 models_pruning=models_pruning,
                                                 method='supervised', pruning_method=pruning_method,
                                                 labels_predicted=modal_labels_pruned, serialize=True)

            print(
                "Method: {} Pruning method: {} Pruned: {} Models pruned: {} Models not pruned: {} Found PIEs: {}".format(
                    'Supervised', pruning_method,
                    models_pruning,
                    len(pruned_models),
                    len(models),
                    total_pies))
