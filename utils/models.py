import glob
import logging
import os

import torch
from torch import nn
import torch.nn.functional as F

import utils
from utils import pruning

def check_clr_model_already_trained(encoder_save_path, encoder_name, method, final_sparsity, batch_size,
                                    temperature, epochs, model_number, begin_step=0, pruning_method=''):
    if begin_step > 2000:
        print("Encoder path: {} ".format(encoder_save_path.format(os.getcwd(),
            encoder_name, method, final_sparsity, '_later', batch_size, temperature, epochs, model_number)))
        logging.info("Encoder path: {} ".format(encoder_save_path.format(os.getcwd(),
            encoder_name, method, final_sparsity, '_later', batch_size, temperature, epochs, model_number)))
        return os.path.exists(encoder_save_path.format(os.getcwd(),
            encoder_name, method, final_sparsity, '_later', batch_size, temperature, epochs, model_number))
    if pruning_method == 'global':
        return os.path.exists(encoder_save_path.format(os.getcwd(),
                encoder_name, method, final_sparsity, '_global', batch_size, temperature, epochs,
                model_number))
    if pruning_method == 'clr_global':
        print("Encoder path: {} ".format(encoder_save_path.format(os.getcwd(),
            encoder_name, method, final_sparsity, '_clr_global', batch_size, temperature, epochs, model_number)))
        return os.path.exists(encoder_save_path.format(os.getcwd(),
            encoder_name, method, final_sparsity, '_clr_global', batch_size, temperature, epochs,
            model_number))
    else:
        print("Encoder path: {} ".format(encoder_save_path.format(os.getcwd(),
            encoder_name, method, final_sparsity, '', batch_size, temperature, epochs, model_number)))
        logging.info("Encoder path: {} ".format(encoder_save_path.format(os.getcwd(),
            encoder_name, method, final_sparsity, '', batch_size, temperature, epochs, model_number)))
        return os.path.exists(encoder_save_path.format(os.getcwd(),
            encoder_name, method, final_sparsity, '', batch_size, temperature, epochs, model_number))


def check_supervised_model_pretrained(save_path, model_number, pruning_technique, final_sparsity,
                                      model_name='wideResNet',
                                      epochs=30, depth=16,
                                      dropout=0.0):
    print("Model path: {} ".format(save_path.format(
        '_global', final_sparsity, epochs, depth, dropout, model_number)))
    return os.path.exists(save_path.format(
        model_name, pruning_technique, final_sparsity, epochs, depth, dropout, model_number))


class WideResNet(nn.Module):
    def __init__(self, depth=16, widen_factor=2, dropout=0.0, num_classes=10, prediction_depth=False
                 ):
        super().__init__()
        self.base = WideResNetBase(depth=depth, num_classes=num_classes, widen_factor=widen_factor, dropRate=dropout,
                                   prediction_depth=prediction_depth)

    def forward(self, x):
        return self.base(x)

    def get_latent_representation(self):
        return self.base.get_latent_representation()

    def get_hidden_representations(self):
        return self.base.get_hidden_representations()


def get_wideResNet_contrastive(encoder, classification_head):
    return WideResNetContrastive(encoder=encoder, classification_head=classification_head)


class WideResNetContrastive(nn.Module):
    def __init__(self, encoder, classification_head):
        super().__init__()
        self.encoder = encoder
        self.classification_head = classification_head

    def forward(self, x):
        return self.classification_head(self.encoder(x))

class WideResNetBase(nn.Module):
    def __init__(self, depth=16, num_classes=10, widen_factor=1, dropRate=0.0, prediction_depth=False):
        super(WideResNetBase, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, prediction_depth=prediction_depth)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, prediction_depth=prediction_depth)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, prediction_depth=prediction_depth)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        self.prediction_depth = prediction_depth
        self.latent_representation = None
        self.hidden_representations = []
        self.first_hidden_representation = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        if self.prediction_depth:
            self.first_hidden_representation = F.avg_pool2d(out, out.shape[3]).view(-1, 16)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        self.latent_representation = out
        return self.fc(out)

    def get_latent_representation(self):
        return self.latent_representation

    def get_hidden_representations(self):
        self.hidden_representations.clear()
        self.hidden_representations.append(self.first_hidden_representation)

        for hidden_representation in self.block1.get_hidden_representations():
            self.hidden_representations.append(hidden_representation)
        for hidden_representation in self.block2.get_hidden_representations():
            self.hidden_representations.append(hidden_representation)
        for hidden_representation in self.block3.get_hidden_representations():
            self.hidden_representations.append(hidden_representation)

        return self.hidden_representations


def get_wide_resnet(depth=16, widen_factor=2, dropout=0.0, num_classes=10):
    return WideResNet(depth=depth, widen_factor=widen_factor, dropout=dropout, num_classes=num_classes)


def get_clr_encoder(encoder='WideResNet', depth=16, widen_factor=2, dropout=0.0, head='mlp', feat_dim=128,
                    prediction_depth=False
                    ):
    return EncoderNetwork(encoder_name=encoder, depth=depth, widen_factor=widen_factor, dropout=dropout, head=head,
                          feat_dim=feat_dim, prediction_depth=prediction_depth)


def get_clr_linear_classifier(widen_factor=2, num_classes=10):
    return LinearClassifier(num_classes=num_classes)


def get_clr_wide_resnet_classification(depth=16, widen_factor=2, dropout=0.0, head='mlp', feat_dim=128):
    return SupConWideResNet(depth=depth, widen_factor=widen_factor, dropout=dropout, head=head,
                            feat_dim=feat_dim)


class SupConWideResNet(nn.Module):
    """backbone + projection head"""

    def __init__(self, depth=16, widen_factor=2, dropout=0.0, head='mlp', feat_dim=128):
        super().__init__()
        self.encoder = WideResNetEncoder(depth=depth, widen_factor=widen_factor, dropout=dropout)
        dim_in = 64 * widen_factor
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat


class LinearClassifier(nn.Module):
    """Linear classifier"""

    def __init__(self, num_classes=10, encoder_name='WideResNet'):
        super(LinearClassifier, self).__init__()
        _, feat_dim = get_clr_encoder_network(encoder_name=encoder_name)
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)


def get_clr_encoder_network(encoder_name='WideResNet', depth=16, widen_factor=2, dropout=0.0, prediction_depth=False):
    if encoder_name == 'WideResNet':
        return WideResNetEncoder(depth=depth, widen_factor=widen_factor, dropout=dropout,
                                 prediction_depth=prediction_depth), 64 * widen_factor


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


def get_pruning_method_setup_clr(pruning_method):
    if pruning_method == 'GMP':
        return 1024, '', 500, 100
    if pruning_method == 'global':
        return 128, '_global', 30, 30
    if pruning_method == 'later GMP' or pruning_method == 'GMP_later' or pruning_method == 'later_GMP':
        return 1024, '_later', 500, 100
    if pruning_method == '':
        return 1024, '', 500, 100


def check_head_sparsity(head, sparsity):
    head_sparsity = utils.pruning.get_model_sparsity(head)
    if sparsity != round(head_sparsity / 100, 1):
        print("Head loaded with a different sparsity of {}!".format(head_sparsity))

def check_encoder_sparsity(encoder, sparsity):
    encoder_sparsity = utils.pruning.get_model_sparsity(encoder.encoder)
    if sparsity != round(encoder_sparsity / 100, 1):
        print("Encoder loaded with a different sparsity of {}!".format(encoder_sparsity))

def get_clr_models(models_number, encoder_name, method, sparsity, temperature, device, projection_head='mlp',
                   debug=False, dropout=0.0, depth=16, pruning_method='', widen_factor=2,
                   attach_gpu=True, prediction_depth=False):
    models_clr_encoders_paths = []
    models_clr_head_paths = []
    encoder_models_loaded = 0
    head_models_loaded = 0

    batch_size, pruning_method_path, epochs_encoder, epochs_head = get_pruning_method_setup_clr(
        pruning_method=pruning_method)

    for model_id in range(models_number):

        if os.path.exists(
                    "{}/models/{}_head_{}_{}pruning{}_{}batch_{}temperature_{}epochs_id{}.pt".format(
                        os.getcwd(), encoder_name, method, sparsity, pruning_method_path, batch_size, temperature, epochs_head,
                        model_id)) and os.path.exists(
                    "{}/models/{}_encoder_{}_{}pruning{}_{}batch_{}temperature_{}epochs_id{}.pt".format(os.getcwd(),
                        encoder_name, method, sparsity, pruning_method_path, batch_size, temperature, epochs_encoder,
                        model_id)):

            # load CLR encoder
            for file in glob.glob(
                    "{}/models/{}_encoder_{}_{}pruning{}_{}batch_{}temperature_{}epochs_id{}.pt".format(os.getcwd(),
                        encoder_name, method, sparsity, pruning_method_path, batch_size, temperature, epochs_encoder,
                        model_id)):
                if encoder_models_loaded < models_number:
                    encoder_models_loaded += 1
                    models_clr_encoders_paths.append(file)

            # load CLR head
            for file in glob.glob(
                    "{}/models/{}_head_{}_{}pruning{}_{}batch_{}temperature_{}epochs_id{}.pt".format(
                        os.getcwd(), encoder_name, method, sparsity, pruning_method_path, batch_size, temperature, epochs_head,
                        model_id)):
                if head_models_loaded < models_number:
                    head_models_loaded += 1
                    models_clr_head_paths.append(file)

    clr_models_encoder = []
    clr_models_heads = []

    for model_number in range(len(models_clr_encoders_paths)):

        encoder = utils.models.get_clr_encoder(depth=depth, dropout=dropout,
                                               widen_factor=widen_factor,
                                               head=projection_head, prediction_depth=prediction_depth)
        head = utils.models.get_clr_linear_classifier(widen_factor=widen_factor, num_classes=10)

        if pruning_method == 'global':
            encoder.encoder.load_state_dict(
                torch.load(models_clr_encoders_paths[model_number],
                           map_location=device)['model'])
            head.load_state_dict(
                torch.load(models_clr_head_paths[model_number],
                           map_location=device)['head'])
            if attach_gpu:
                encoder.to(device)
                head.to(device)
            clr_models_encoder.append(encoder)
            clr_models_heads.append(head)
        else:
            encoder.load_state_dict(
                    torch.load(models_clr_encoders_paths[model_number],
                               map_location=device)['model'])
            head.load_state_dict(
                    torch.load(models_clr_head_paths[model_number],
                               map_location=device)['head'])
            if attach_gpu:
                encoder.to(device)
                head.to(device)
            clr_models_encoder.append(encoder)
            clr_models_heads.append(head)

        if sparsity > 0.0 and pruning_method != 'global':
            check_encoder_sparsity(encoder=encoder, sparsity=sparsity)
            check_head_sparsity(head=head, sparsity=sparsity)

    if not debug:
        print("Loaded {} {} pruned {} models".format(len(clr_models_encoder), sparsity, pruning_method))
    else:
        logging.info("Loaded {} {} pruned {} models".format(len(clr_models_encoder), sparsity, pruning_method))

    return clr_models_encoder, clr_models_heads


def get_pruning_method_setup_supervised(pruning_method):
    if pruning_method == 'GMP':
        return '', 205
    if pruning_method == 'global':
        return '_global', 30
    if pruning_method == '':
        return '', 205


def get_supervised_models(models_number, sparsity, device, pruning_method='', encoder_name='wideResNet',
                          depth=16,
                          debug=False, dropout=0.0,
                          widen_factor=2, attach_gpu=True, prediction_depth=False):
    models_pruned_paths = []
    pruned_models_loaded = 0

    pruning_method_path, epochs = get_pruning_method_setup_supervised(pruning_method=pruning_method)

    for file in glob.glob(
            "{}/models/{}{}_{}pruning_{}epochs_{}depth_{}dropout*.pt".format(os.getcwd(),
                encoder_name, pruning_method_path, sparsity, epochs, depth, dropout)):
        if pruned_models_loaded < models_number:
            pruned_models_loaded += 1
            models_pruned_paths.append(file)

    pruned_models = []

    for path in models_pruned_paths:
        model = WideResNet(depth=depth, widen_factor=widen_factor, dropout=dropout, prediction_depth=prediction_depth)
        model.load_state_dict(
            torch.load(path,
                       map_location=device))
        model_sparsity = utils.pruning.get_model_sparsity(model)
        if debug:
            logging.info("Loaded model with sparsity of {}".format(round(model_sparsity / 100, 1)))
            logging.info("Path model loaded: {}".format(path))
        if sparsity != round(model_sparsity / 100, 1):
            print("Model loaded with a different sparsity of {}!".format(model_sparsity))
        if attach_gpu:
            model.to(device)
        pruned_models.append(model)

    if not debug:
        print("Loaded {} {} pruned {} models".format(len(pruned_models), sparsity, pruning_method))
    else:
        logging.info("Loaded {} {} pruned models".format(len(pruned_models), sparsity))

    return pruned_models


class EncoderNetwork(nn.Module):
    """backbone + projection head"""

    def __init__(self, depth=16, widen_factor=2, dropout=0.0, head='mlp', feat_dim=128, encoder_name='WideResNet',
                 prediction_depth=False):
        super().__init__()
        self.encoder, dim_in = get_clr_encoder_network(encoder_name=encoder_name, depth=depth,
                                                       widen_factor=widen_factor, dropout=dropout,
                                                       prediction_depth=prediction_depth)
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat

    def get_hidden_representations(self):
        return self.encoder.get_hidden_representations()


class WideResNetEncoder(nn.Module):
    def __init__(self, depth=16, widen_factor=2, dropout=0.0, prediction_depth=False):
        super().__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropout, prediction_depth=prediction_depth)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropout, prediction_depth=prediction_depth)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropout, prediction_depth=prediction_depth)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.prediction_depth = prediction_depth
        self.nChannels = nChannels[3]
        self.hidden_representations = []
        self.first_hidden_representation = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        if self.prediction_depth:
            self.first_hidden_representation = F.avg_pool2d(out, out.shape[3]).view(-1, 16)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))

        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return out

    def get_hidden_representations(self):
        self.hidden_representations.clear()
        self.hidden_representations.append(self.first_hidden_representation)

        for hidden_representation in self.block1.get_hidden_representations():
            self.hidden_representations.append(hidden_representation)
        for hidden_representation in self.block2.get_hidden_representations():
            self.hidden_representations.append(hidden_representation)
        for hidden_representation in self.block3.get_hidden_representations():
            self.hidden_representations.append(hidden_representation)

        return self.hidden_representations


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, prediction_depth=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_planes = out_planes
        self.hidden_representations = []
        self.prediction_depth = prediction_depth

    def forward(self, x):

        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))

        if self.prediction_depth:
            self.hidden_representations.clear()
            self.hidden_representations.append(F.avg_pool2d(out, out.shape[3]).view(-1, self.out_planes))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)

        if self.prediction_depth:
            self.hidden_representations.append(F.avg_pool2d(out, out.shape[3]).view(-1, self.out_planes))
            if self.equalInOut:
                self.hidden_representations.append(
                    F.avg_pool2d(torch.add(x if self.equalInOut else self.convShortcut(x), out), out.shape[3]).view(
                        -1, self.out_planes))
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

    def get_hidden_representations(self):
        return self.hidden_representations


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, prediction_depth=False):
        super(NetworkBlock, self).__init__()
        self.prediction_depth = prediction_depth
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
        self.hidden_representations = []

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate,
                                prediction_depth=self.prediction_depth))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

    def get_hidden_representations(self):
        self.hidden_representations.clear()
        for block in self.layer:
            for hidden_representation in block.get_hidden_representations():
                self.hidden_representations.append(hidden_representation)
        return self.hidden_representations