import torch
import torch.nn.utils.prune as prune


def magnitude_pruning(model, theta):
    for name, module in model.named_modules():

        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=theta)

        elif isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=theta)
            prune.l1_unstructured(module, name='bias', amount=theta)


def global_pruning(model, pruning_percentage):
    module_tups = []
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            module_tups.append((module, 'weight'))
        if isinstance(module, torch.nn.Linear):
            module_tups.append((module, 'weight'))
            module_tups.append((module, 'bias'))

    prune.global_unstructured(
        parameters=module_tups, pruning_method=prune.L1Unstructured,
        amount=pruning_percentage
    )

def layer_wise_pruning(model, pruning_percentage):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=pruning_percentage)
        elif isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=pruning_percentage)
            prune.l1_unstructured(module, name='bias', amount=pruning_percentage)


def magnitude_pruning_encoder_clr(model, theta):
    for name, module in model.encoder.named_modules():

        # prune connections in 2D-Conv modules
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=theta)


def magnitude_pruning_head_clr(head, theta):
    for name, module in head.named_modules():

        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=theta)
            prune.l1_unstructured(module, name='bias', amount=theta)


# remove the weight_mask and bias_mask to make pruning permanent
def remove_pruning_masks(model):
    for name, module in model.named_modules():

        if isinstance(module, torch.nn.Conv2d):
            try:
                prune.remove(module, 'weight')
            except ValueError:
                print("The module has not been pruned")

        if isinstance(module, torch.nn.Linear):
            try:
                prune.remove(module, 'bias')
                prune.remove(module, 'weight')
            except ValueError:
                print("The module has not been pruned")


def check_pruning(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            print("Unpruned parameters of the module:")
            print(list(module.named_parameters()))
            print("Pruning masks of the module:")
            print(list(module.named_buffers()))
            print("Module weights by merging the pruning mask:")
            print(module.weight)
            print("Module biases by merging the pruning mask:")
            print(module.bias)


# Returns the gamma parameters
def get_gammas(initial_sparsity, final_sparsity, begin_step, end_step, frequency):
    polynomial_pruning_percentage = []
    for step in range(begin_step, end_step + 1, frequency):
        current_pruning_percentage = final_sparsity + (initial_sparsity - final_sparsity) * pow(
            1 - ((step - begin_step) / (end_step - begin_step)), 3)
        polynomial_pruning_percentage.append(current_pruning_percentage)

    gamma_parameters = []

    first = False

    for pruning_percentage in polynomial_pruning_percentage:
        if first:
            denominator = 1
            for gamma in gamma_parameters:
                denominator *= (1 - gamma)
            gamma_parameters.append(1 - ((1 - pruning_percentage) / denominator))
        else:
            gamma_parameters.append(pruning_percentage)
            if pruning_percentage != 0.0:
                first = True
    return gamma_parameters


def get_model_sparsity_clr(model):
    zeros_weight = 0
    total_weight = 0

    for name, module in model.encoder.named_modules():

        if isinstance(module, torch.nn.Conv2d):
            zeros_weight += float(torch.sum(module.weight == 0))
            total_weight += float(module.weight.nelement())

        elif isinstance(module, torch.nn.Linear):
            zeros_weight += float(torch.sum(module.weight == 0))
            total_weight += float(module.weight.nelement())
    if zeros_weight == 0 and total_weight == 0:
        return 0
    return 100. * zeros_weight / total_weight


def get_model_sparsity_clr_head(head):
    zeros_weight = 0
    total_weight = 0

    for name, module in head.named_modules():
        if isinstance(module, torch.nn.Linear):
            zeros_weight += float(torch.sum(module.weight == 0))
            total_weight += float(module.weight.nelement())

    if zeros_weight == 0 and total_weight == 0:
        return 0.0
    return 100. * zeros_weight / total_weight


def get_model_sparsity(model):
    zeros_weight = 0
    total_weight = 0

    for name, module in model.named_modules():

        if isinstance(module, torch.nn.Conv2d):
            zeros_weight += float(torch.sum(module.weight == 0))
            total_weight += float(module.weight.nelement())

        elif isinstance(module, torch.nn.Linear):
            zeros_weight += float(torch.sum(module.weight == 0))
            total_weight += float(module.weight.nelement())

    if zeros_weight == 0 and total_weight == 0:
        return 0
    return 100. * zeros_weight / total_weight
