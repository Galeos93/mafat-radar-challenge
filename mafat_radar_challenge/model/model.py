from io import BytesIO
import numpy as np
import requests
import timm

import torch
import torch.nn as nn
import torch.nn.functional as F

from efficientnet_pytorch import EfficientNet

from mafat_radar_challenge.base import ModelBase
from mafat_radar_challenge.utils import setup_logger
from mafat_radar_challenge.model.pann import PANNsCNN14Att, AttBlock
from mafat_radar_challenge.model import bit


log = setup_logger(__name__)


class MnistModel(ModelBase):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

        log.info(f"<init>: \n{self}")

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class BaselineModel(ModelBase):
    def __init__(
        self, num_classes=5,
    ):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(3, 16, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(16, 32, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(5760, 128), nn.Linear(128, 32), nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


class EfficientNetBase(ModelBase):
    def __init__(
        self,
        num_classes=5,
        pretrained=False,
        checkpoint_path=None,
        net_name="efficientnet-b0",
    ):
        super().__init__()
        if pretrained:
            self.model = EfficientNet.from_pretrained(net_name)
        else:
            self.model = EfficientNet.from_name(net_name)
        in_features = self.model._fc.in_features
        self.model._fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class EfficientNetMetadata(EfficientNetBase):
    def __init__(
        self,
        num_classes=5,
        pretrained=False,
        checkpoint_path=None,
        net_name="efficientnet-b0",
    ):
        super().__init__(num_classes, pretrained, checkpoint_path, net_name)
        if pretrained:
            self.model = EfficientNet.from_pretrained(net_name)
        else:
            self.model = EfficientNet.from_name(net_name)
        in_features = self.model._fc.in_features
        out_features = 25
        self.model._fc = nn.Linear(in_features, out_features)
        self.meta = nn.Sequential(
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(16, 8),  # metadata FC layer output will have 8 features
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        self.output = nn.Linear(out_features + 8, 1)

    def forward(self, inputs):
        x, meta = inputs
        meta_features = self.meta(meta)
        cnn_features = self.model(x)
        features = torch.cat((cnn_features, meta_features), dim=1)
        return self.output(features)


class SimpleNet(ModelBase):
    def __init__(
        self, num_classes=5,
    ):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(3, 16, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(16, 32, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(5760, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


class CustomNet(ModelBase):
    def __init__(
        self, num_classes=5,
    ):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(3, 16, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=(3, 3), stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.linear_layers = nn.Sequential(nn.Linear(96, 1))

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


class ResNest(ModelBase):
    def __init__(
        self,
        num_classes=5,
        pretrained=False,
        checkpoint_path=None,
        net_name="resnest50",
    ):
        super().__init__()

        self.model = torch.hub.load(
            "zhanghang1989/ResNeSt", net_name, pretrained=pretrained
        )
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class VGGish(ModelBase):
    def __init__(
        self, num_classes=5, pretrained=False, checkpoint_path=None, net_name="vggish",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.model = torch.hub.load("harritaylor/torchvggish", "vggish")

    def forward(self, x):
        embedding = self.model(x)
        output = nn.Linear(embedding, self.num_classes)(embedding)
        return output


class PANNsCNN(ModelBase):
    def __init__(
        self,
        num_classes=5,
        pretrained=False,
        checkpoint_path=None,
        net_name="pannscnn14",
    ):
        super().__init__()
        if net_name == "pannscnn14":
            self.model = PANNsCNN14Att(64, classes_num=527)  # Default values
            weights = torch.load(
                "/home/agarcia/repos/mafat-radar-challenge/mafat_radar_challenge/model/pann/Cnn14_DecisionLevelAtt_mAP0.425.pth"
            )
            self.model.load_state_dict(weights["model"], strict=False)
            self.model.att_block = AttBlock(2048, num_classes, activation="sigmoid")
            self.model.att_block.init_weights()
        else:
            raise ValueError("Net name is not valid.")
        self.linear_layers = nn.Sequential(
            nn.Linear(4096, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )

    def freeze_model(self, model):
        freeze_layers = [
            "conv_block1",
            "conv_block2",
            "conv_block3",
            "conv_block4",
            "conv_block5",
            "conv_block6",
        ]
        for name, child in model.named_children():
            if name not in freeze_layers:
                continue
            for param in child.parameters():
                param.requires_grad = False
            self.freeze_model(child)

    def forward(self, x):
        self.freeze_model(self.model)
        features = self.model.cnn_feature_extractor(x.mean([1]).unsqueeze_(1))
        features = features.view(features.size(0), -1)
        return self.linear_layers(features)
        # return self.model(x.mean([1]).unsqueeze_(1))


class ResNext(ModelBase):
    def __init__(
        self,
        num_classes=5,
        pretrained=False,
        checkpoint_path=None,
        net_name="resnext50_32x4d",
    ):
        super().__init__()

        self.model = torch.hub.load(
            "pytorch/vision:v0.6.0", net_name, pretrained=pretrained
        )
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class DenseNet(ModelBase):
    def __init__(
        self,
        num_classes=5,
        pretrained=False,
        checkpoint_path=None,
        net_name="densenet121",
    ):
        super().__init__()

        self.model = torch.hub.load(
            "pytorch/vision:v0.6.0", net_name, pretrained=pretrained
        )
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class TimmEfficientNet(ModelBase):
    def __init__(
        self,
        num_classes=5,
        pretrained=False,
        checkpoint_path=None,
        net_name="tf_efficientnet_b2_ns",
    ):
        super().__init__()

        self.model = timm.create_model(
            net_name, pretrained=pretrained, num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)


class CNNLSTM(ModelBase):
    def __init__(self):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=126, out_channels=63, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=63, out_channels=32, kernel_size=3)
        self.lstm1 = nn.GRU(input_size=126, hidden_size=5, num_layers=1)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.squeeze(x)
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        x = torch.transpose(x, 0, 2)
        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm1(x)
        x = x[-1, :, :]
        x = self.fc2(x)
        return x


class RCNN(ModelBase):
    def __init__(self):
        super(RCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 1))
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3))
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 1))
        self.gru1 = nn.GRU(input_size=64, hidden_size=32, num_layers=1)
        self.fc1 = nn.Linear(7, 1)

    def forward(self, x):
        print(x.shape)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        print(x.shape)
        x = torch.squeeze(x)
        print(x.shape)
        x = torch.transpose(x, 0, 2)
        x = torch.transpose(x, 1, 2)
        x, _ = self.gru1(x)
        x = x[-1, :, :]
        x = self.fc1(x)
        return x


class BiT(ModelBase):
    def __init__(
        self,
        num_classes=5,
        pretrained=False,
        checkpoint_path=None,
        net_name="BiT-M-R50x1",
    ):
        super().__init__()
        self.model = bit.KNOWN_MODELS[net_name](head_size=num_classes, zero_head=True)
        r = requests.get(
            "https://storage.googleapis.com/bit_models/{}.npz".format(net_name),
            stream=True,
        )
        checkpoint = np.load(BytesIO(r.raw.read()))
        self.model.load_from(checkpoint)

    def forward(self, x):
        return self.model(x)
