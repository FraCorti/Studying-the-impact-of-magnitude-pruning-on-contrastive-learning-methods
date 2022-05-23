import numpy as np
import argparse
from datetime import datetime, date
import logging
import os
import utils
from utils import data, models, pies
import random
import torch
from utils.umap_application import plot_umap

parser = argparse.ArgumentParser()
parser.add_argument('--models_number', default=30, type=int, help='number of models to load')
parser.add_argument('--batch_size', default=1024, type=int)
parser.add_argument('--model_depth', default=16, type=int)
parser.add_argument('--debug', default=False, type=bool)
parser.add_argument('--cuda_device', default=0, type=int, help='id of the GPU to use')
parser.add_argument('--dropout', default=0.0, type=float)
parser.add_argument('--workers', default=0, type=int)
parser.add_argument('--widen_factor', default=2, type=int)
parser.add_argument('--cuda_deterministic', default=True, type=bool, help='setup a deterministic computation')
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--total_seed', default=1, type=int)
parser.add_argument('--clr_method', default="SupCon", type=str, help='contrastive learning training method')
parser.add_argument('--projection_head', default='mlp', type=str)
parser.add_argument('--temperature', default=0.5, type=float, help='temperature loss function hyperparameter')
parser.add_argument('--plot_umap', default=False, action='store_true', help='plot feature vectors with predicted class')
parser.add_argument('--plot_q_score', default=False, action='store_true')
parser.add_argument('--not_serialize_plots', default=True, action='store_false')
parser.add_argument('--encoder_name', default='WideResNet', type=str, help='encoder network to use')

args = parser.parse_args()
depth = args.model_depth
seed = args.seed
encoder_name = args.encoder_name
plot_q_score = args.plot_q_score
not_serialize_plots = args.not_serialize_plots
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
        # torch.set_printoptions(profile='full')
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    loaders = utils.data.get_ordered_cifar10_validation(batch_size=batch_size, workers=workers, seed=seed)
    print(
        "Models loaded: {} Batch Size: {}, Deterministic Setup: {}, Models depth: {}, Models widen factor: {} Serialize plots: {}".format(
            models_number, batch_size, cuda_deterministic, depth, widen_factor, not_serialize_plots))

    clr_models_notpruned_encoders, clr_models_notpruned_heads = utils.models.get_clr_models(
        models_number=30,
        encoder_name=encoder_name,
        method=method,
        sparsity=0.0,
        temperature=temperature,
        device=device,
        pruning_method='',
        attach_gpu=True,
        projection_head=projection_head,
        prediction_depth=False)

    for pruning_method in ['global', 'GMP', 'later_GMP']:

        plot_no_sparsity_umap = False

        for models_pruning in [0.3, 0.5, 0.7, 0.9]:

            clr_pruned_models_encoders, clr_pruned_models_heads = utils.models.get_clr_models(
                models_number=models_number,
                encoder_name=encoder_name,
                method=method,
                sparsity=models_pruning,
                temperature=temperature,
                pruning_method=pruning_method,
                device=device,
                attach_gpu=True,
                projection_head=projection_head)

            if len(clr_pruned_models_encoders) == 0 or len(clr_pruned_models_heads) == 0:
                continue

            total_pies = 0
            batch_index = 0

            q_score_PIEs = []
            q_score_notPIEs = []

            mean_features_PIEs = []
            mean_features_notPIEs = []

            standard_deviation_features_PIEs = []
            standard_deviation_features_notPIEs = []

            latent_representations_pruned = np.empty(shape=[1, 129], dtype=float)
            latent_representations_not_pruned = np.empty(shape=[1, 129], dtype=float)

            l1_norm_PIEs = []
            l1_norm_notPIEs = []

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

                    results_model_clr_notpruned = []
                    results_model_clr = []

                    most_frequent_labels_clr_notpruned = []
                    most_frequent_labels_clr = []

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

                    for clr_model_notpruned_number in range(len(clr_models_notpruned_encoders)):
                        clr_models_notpruned_encoders[clr_model_notpruned_number].eval()
                        clr_models_notpruned_heads[clr_model_notpruned_number].eval()

                        with torch.no_grad():
                            latent_representation = clr_models_notpruned_encoders[clr_model_notpruned_number].encoder(
                                input_)

                            if plot_q_score:
                                latent_representation_q_score = latent_representation.cpu().detach().numpy()
                                for i in range(len(latent_representation_q_score)):
                                    single_latent_representation = latent_representation_q_score[i]
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

                            output_model = clr_models_notpruned_heads[clr_model_notpruned_number](latent_representation)

                        _, preds_output_model = torch.max(output_model, 1)
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

                            if plot_q_score:
                                latent_representation_pruned_q_score = latent_representation_pruned.cpu().detach().numpy()
                                for i in range(len(latent_representation_pruned_q_score)):
                                    single_latent_representation = latent_representation_pruned_q_score[i]
                                    q_score_p, z_score_p, l1_norm_p, mean_p, standard_deviation_p = utils.pies.q_score(
                                        latent_representation=single_latent_representation / np.linalg.norm(
                                            single_latent_representation, ord=2))
                                    q_score_pruned[i] += q_score_p
                                    z_score_pruned[i] += z_score_p
                                    l1_norm_pruned[i] += l1_norm_p

                            if umap:
                                average_latent_representation_pruned = np.add(average_latent_representation_pruned,
                                                                              latent_representation_pruned.cpu().detach().numpy())

                            output_clr_model_pruned = clr_pruned_models_heads[clr_model_pruned_number](
                                latent_representation_pruned)

                        _, preds_output_model_clr = torch.max(output_clr_model_pruned, 1)
                        results_model_clr.append(preds_output_model_clr.cpu().detach().numpy())

                    for i in range(len(input_)):
                        models_pruned_results = []

                        for model_output in range(len(results_model_clr)):
                            models_pruned_results.append(results_model_clr[model_output][i])

                        most_frequent_labels_clr.append(np.bincount(np.array(models_pruned_results)).argmax())

                    if plot_q_score:
                        q_score_pruned = np.divide(q_score_pruned, models_number)
                        z_score_pruned = np.divide(z_score_pruned, models_number)
                        l1_norm_pruned = np.divide(l1_norm_pruned, models_number)

                        q_score_notpruned = np.divide(q_score_notpruned, models_number)
                        z_score_not_pruned = np.divide(z_score_not_pruned, models_number)
                        l1_norm_not_pruned = np.divide(l1_norm_not_pruned, models_number)

                    for i in range(len(input_)):
                        modal_labels_pruned.append(most_frequent_labels_clr[i])
                        modal_labels_not_pruned.append(most_frequent_labels_clr_notpruned[i])

                        if most_frequent_labels_clr_notpruned[i] != most_frequent_labels_clr[i]:


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
                                                                               len(clr_pruned_models_encoders)), 0)))
                                if not plot_no_sparsity_umap:
                                    latent_representations_not_pruned = np.vstack((latent_representations_not_pruned,
                                                                                   np.append(np.divide(
                                                                                       average_latent_representation_not_pruned[
                                                                                       i, :],
                                                                                       models_number), 0)))
                    batch_index += 1
                    total_pies += batch_pies

                    if debug:
                        logging.info(
                            "PIEs found inside this batch: {} , total PIEs currently found: {}".format(batch_pies,
                                                                                                       total_pies))

                if plot_q_score:
                    utils.pies.plot_q_scores(q_score_notPIEs=q_score_notPIEs, q_score_PIEs=q_score_PIEs,
                                             sparsity=models_pruning, method=method, pruning_method=pruning_method,
                                             serialize=not_serialize_plots)

                    utils.pies.print_z_score_l1_norm_mean_std(l1_norm_PIEs=l1_norm_PIEs,
                                                              l1_norm_notPIEs=l1_norm_notPIEs,
                                                              z_score_notPIEs=z_score_notPIEs,
                                                              z_score_PIEs=z_score_PIEs,
                                                              sparsity=models_pruning,
                                                              pruning_method=pruning_method,
                                                              method=method)

                if umap:
                    if not plot_no_sparsity_umap:
                        utils.umap_application.plot_umap(latent_representations=latent_representations_not_pruned,
                                                         models_pruning=0.0,
                                                         method=method, pruning_method=pruning_method,
                                                         labels_predicted=modal_labels_not_pruned,
                                                         serialize=not_serialize_plots,
                                                         plot_no_sparsity=True)
                        plot_no_sparsity_umap = True

                    utils.umap_application.plot_umap(latent_representations=latent_representations_pruned,
                                                     models_pruning=models_pruning,
                                                     method=method, pruning_method=pruning_method,
                                                     labels_predicted=modal_labels_pruned,
                                                     serialize=not_serialize_plots)

                print(
                    "Method: {} Pruning method: {} Pruned: {} Models pruned: {} Models not pruned: {} Found PIEs: {}".format(
                        method, pruning_method,
                        models_pruning,
                        len(clr_pruned_models_encoders),
                        len(clr_models_notpruned_encoders),
                        total_pies,

                    ))