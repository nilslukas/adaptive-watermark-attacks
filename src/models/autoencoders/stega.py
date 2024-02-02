import math

import torch
from torch import nn
from torch.nn.functional import relu
from torchvision import models, transforms


class StegaStampEncoder(nn.Module):
    def __init__(
            self,
            resolution,
            IMAGE_CHANNELS=3,  # Channels of the input image
            fingerprint_size=100,  # Number of bits
            fingerprint_resolution=16,  # Resolution of the fingerprint *must be a power of 2*.
            # Will be scaled up to image resolution
    ):
        super(StegaStampEncoder, self).__init__()
        self.fingerprint_size = fingerprint_size
        self.fingerprint_resolution = fingerprint_resolution
        self.IMAGE_CHANNELS = IMAGE_CHANNELS

        log_fingerprint_resolution = int(math.log(fingerprint_resolution, 2))
        assert fingerprint_resolution == 2 ** log_fingerprint_resolution, f"Fingerprint resolution must be 16 or greater and " \
                                                                          f"a power of 2, got {fingerprint_resolution}"

        secret_layers = [nn.Linear(self.fingerprint_size, 16 * 16 * IMAGE_CHANNELS)]
        for i in range(log_fingerprint_resolution - 4):
            in_dim = (16 * (2 ** i)) ** 2 * IMAGE_CHANNELS
            out_dim = (16 * (2 ** (i + 1))) ** 2 * IMAGE_CHANNELS
            secret_layers.append(nn.ReLU())
            secret_layers.append(nn.Linear(in_dim, out_dim))
        self.secret_dense = nn.Sequential(*secret_layers)

        log_resolution = int(math.log(resolution, 2))
        assert resolution == 2 ** log_resolution, f"Image resolution must be a power of 2, got {resolution}."

        scale_factor = resolution // fingerprint_resolution

        self.fingerprint_upsample = nn.Upsample(scale_factor=(scale_factor, scale_factor))
        self.conv1 = nn.Conv2d(2 * IMAGE_CHANNELS, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(32, 32, 3, 2, 1)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.conv4 = nn.Conv2d(64, 128, 3, 2, 1)
        self.bn4 = nn.BatchNorm2d(num_features=128)
        self.conv5 = nn.Conv2d(128, 256, 3, 2, 1)
        self.bn5 = nn.BatchNorm2d(num_features=256)
        self.pad6 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up6 = nn.Conv2d(256, 128, 2, 1)
        self.upsample6 = nn.Upsample(scale_factor=(2, 2))
        self.conv6 = nn.Conv2d(128 + 128, 128, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(num_features=128)
        self.pad7 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up7 = nn.Conv2d(128, 64, 2, 1)
        self.upsample7 = nn.Upsample(scale_factor=(2, 2))
        self.conv7 = nn.Conv2d(64 + 64, 64, 3, 1, 1)
        self.bn7 = nn.BatchNorm2d(num_features=64)
        self.pad8 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up8 = nn.Conv2d(64, 32, 2, 1)
        self.upsample8 = nn.Upsample(scale_factor=(2, 2))
        self.conv8 = nn.Conv2d(32 + 32, 32, 3, 1, 1)
        self.bn8 = nn.BatchNorm2d(num_features=32)
        self.pad9 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up9 = nn.Conv2d(32, 32, 2, 1)
        self.upsample9 = nn.Upsample(scale_factor=(2, 2))
        self.conv9 = nn.Conv2d(32 + 32 + 2 * IMAGE_CHANNELS, 32, 3, 1, 1)
        self.bn9 = nn.BatchNorm2d(num_features=32)
        self.conv10 = nn.Conv2d(32, 32, 3, 1, 1)
        self.residual = nn.Conv2d(32, IMAGE_CHANNELS, 1)

    def forward(self, image, message=None):
        if len(message.shape) == 1:  # unsqueeze and repeat
            message = message.unsqueeze(0)
        if message.shape[0] != len(image):
            message = message.repeat([len(image), 1])

        message = relu(self.secret_dense(message))
        message = message.view(
            (image.shape[0], self.IMAGE_CHANNELS, self.fingerprint_resolution, self.fingerprint_resolution))
        fingerprint_enlarged = self.fingerprint_upsample(message)

        inputs = torch.cat([fingerprint_enlarged, image], dim=1)
        conv1 = relu(self.bn1(self.conv1(inputs)))
        conv2 = relu(self.bn2(self.conv2(conv1)))
        conv3 = relu(self.bn3(self.conv3(conv2)))
        conv4 = relu(self.bn4(self.conv4(conv3)))
        conv5 = relu(self.bn5(self.conv5(conv4)))
        up6 = relu(self.up6(self.pad6(self.upsample6(conv5))))
        merge6 = torch.cat([conv4, up6], dim=1)
        conv6 = relu(self.bn6(self.conv6(merge6)))
        up7 = relu(self.up7(self.pad7(self.upsample7(conv6))))
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = relu(self.bn7(self.conv7(merge7)))
        up8 = relu(self.up8(self.pad8(self.upsample8(conv7))))
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = relu(self.bn8(self.conv8(merge8)))
        up9 = relu(self.up9(self.pad9(self.upsample9(conv8))))
        merge9 = torch.cat([conv1, up9, inputs], dim=1)
        conv9 = relu(self.bn9(self.conv9(merge9)))
        conv10 = relu(self.conv10(conv9))
        residual = torch.tanh(self.residual(conv10))
        return (residual + 1) / 2  # ensure image is scaled to [0,1]


class StegaStampDecoder(nn.Module):
    def __init__(self, fingerprint_size=1, model_type="resnet18"):
        super(StegaStampDecoder, self).__init__()
        self.model_type = model_type
        self.preprocess = transforms.Compose([
            transforms.Resize(224, antialias=True),
            transforms.Lambda(lambda x: (x + 1) / 2),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        if model_type == "resnet18":
            base_model1 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.hidden_size = 512
            self.dense = nn.Linear(self.hidden_size, fingerprint_size)
        elif model_type == "resnet50":
            base_model1 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.hidden_size = 2048
            self.dense = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Linear(512, fingerprint_size)
            )
        elif model_type == "resnet101":
            base_model1 = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
            self.hidden_size = 2048
            self.dense = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Linear(512, fingerprint_size)
            )
        else:
            raise ValueError(model_type)
        self.base_model1 = torch.nn.Sequential(*list(base_model1.children())[:-1])

    def forward(self, image1):
        f1 = self.base_model1(self.preprocess(image1))
        return self.dense(f1.view(-1, self.hidden_size))


class StegaStampDecoder50(nn.Module):
    def __init__(self, fingerprint_size=1):
        super(StegaStampDecoder50, self).__init__()
        self.preprocess = transforms.Compose([transforms.Resize(224)])

        base_model = models.resnet50(pretrained=True)
        self.decoder = torch.nn.Sequential(*list(base_model.children())[:-1])
        self.dense = nn.Sequential(
            nn.Linear(2048, fingerprint_size),
        )

    def forward(self, image):
        x = self.decoder(self.preprocess(image))
        return self.dense(x.view(-1, 2048))


class StegaStampDecoder101(nn.Module):
    def __init__(self, resolution=32, IMAGE_CHANNELS=3, fingerprint_size=1):
        super(StegaStampDecoder101, self).__init__()
        self.resolution = resolution
        self.IMAGE_CHANNELS = IMAGE_CHANNELS
        self.preprocess = transforms.Compose([transforms.Resize(224)])

        base_model = models.wide_resnet101_2(pretrained=True)
        self.decoder = torch.nn.Sequential(*list(base_model.children())[:-1])
        self.dense = nn.Sequential(
            nn.Linear(2048, fingerprint_size),
        )

    def forward(self, image):
        x = self.decoder(self.preprocess(image))
        return self.dense(x.view(-1, 2048))
