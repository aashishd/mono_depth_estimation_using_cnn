from torch import nn
import torchvision
from model.upsampling import FastUpConvolution, FastUpProjection, UpConvolution


def get_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Resnet50BasedModel(nn.Module):
    """"
    Resnet 50 Based model which accepts input of size 304 X 228 X 3
    And generates a prediction of 160 X 128 X 1
    Architecture :
        Resnet Base Model ( without FC layers ) , followed by UnConvolutionLayers
    """

    def __init__(self, device='cpu', freeze_decoder_weights=False):
        super(Resnet50BasedModel, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.freeze_decoder_weights = freeze_decoder_weights
        self.device = device

        if self.freeze_decoder_weights:
            for param in resnet50.parameters():
                param.requires_grad = False

        # remove until the resnet-50 AvgPool Layer
        self.model = nn.Sequential(*(list(resnet50.children())[:-2]),
                                   nn.Conv2d(2048, 1024, (1, 1)),
                                   nn.BatchNorm2d(1024), UpConvolution(1024, 512),
                                   UpConvolution(512, 256),
                                   UpConvolution(256, 128),
                                   UpConvolution(128, 64),
                                   nn.Dropout2d(),
                                   nn.Conv2d(64, 1, 3, padding=1, padding_mode='zeros'),
                                   nn.ReLU(),
                                   nn.Upsample(size=(228, 304), mode='bilinear'))
        print(f'Trainable parameters for {self._get_name()} => {get_trainable_parameters(self.model)}')

    def forward(self, input_tensor):
        # input_tensor.to(self.device)
        """Runs a forward pass over the Model"""
        output_tensor = self.model(input_tensor)
        return output_tensor


class AlexNetBasedModel(nn.Module):
    """"
    Alexnet Based model which accepts input of size 304 X 228 X 3
    And generates a prediction of 160 X 128 X 1
    Architecture :
        AlexNet Base Model ( without FC layers ) , followed by UnConvolutionLayers
    """

    def __init__(self, device='cpu', freeze_decoder_weights=False):
        super(AlexNetBasedModel, self).__init__()
        alexnet = torchvision.models.alexnet(pretrained=True)
        self.freeze_decoder_weights = freeze_decoder_weights
        self.device = device

        if self.freeze_decoder_weights:
            for param in alexnet.parameters():
                param.requires_grad = False
        print("--------AlexNet Based--------")
        print(alexnet)
        self.model = nn.Sequential(*(list(alexnet.children())[:-2]),
                                   nn.Conv2d(256, 256, 3, padding=1, padding_mode='zeros'),
                                   nn.BatchNorm2d(256),
                                   FastUpConvolution(256, 128),
                                   FastUpConvolution(128, 64),
                                   nn.Dropout2d(),
                                   nn.Conv2d(64, 1, 3, padding=1, padding_mode='zeros'),
                                   nn.ReLU(),
                                   nn.Upsample(size=(228, 304), mode='bilinear'))
        print(f'Trainable parameters for {self._get_name()} => {get_trainable_parameters(self.model)}')

    def forward(self, input_tensor):
        # input_tensor.to(self.device)
        """Runs a forward pass over the Model"""
        output_tensor = self.model(input_tensor)
        return output_tensor


class Resnet50BasedUpProjModel(nn.Module):
    """"
    Resnet 50 Based model which accepts input of size 304 X 228 X 3
    And generates a prediction of 160 X 128 X 1
    Architecture :
        Resnet Base Model ( without FC layers ) , followed by UnConvolutionLayers
    """

    def __init__(self, device='cpu', freeze_decoder_weights=False):
        super(Resnet50BasedUpProjModel, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.freeze_decoder_weights = freeze_decoder_weights
        self.device = device

        if self.freeze_decoder_weights:
            for param in resnet50.parameters():
                param.requires_grad = False

        # remove until the resnet-50 AvgPool Layer
        self.model = nn.Sequential(*(list(resnet50.children())[:-2]),
                                   nn.Conv2d(2048, 1024, (1, 1), bias=False),
                                   nn.BatchNorm2d(1024), FastUpProjection(1024, 512),
                                   FastUpProjection(512, 256),
                                   FastUpProjection(256, 128),
                                   FastUpProjection(128, 64),
                                   nn.Dropout2d(),
                                   nn.Conv2d(64, 1, 3, padding=1, padding_mode='zeros'),
                                   nn.ReLU(),
                                   nn.Upsample(size=(228, 304), mode='bilinear'))
        print(f'Trainable parameters for {self._get_name()} => {get_trainable_parameters(self.model)}')

    def forward(self, input_tensor):
        # input_tensor.to(self.device)
        """Runs a forward pass over the Model"""
        output_tensor = self.model(input_tensor)
        return output_tensor


class EfficientNet(nn.Module):
    """"
    EfficientNet Based model which accepts input of size 304 X 228 X 3
    And generates a prediction of 160 X 128 X 1
    Architecture :
        EfficientNet Base Model ( without FC layers ) , followed by UnConvolutionLayers
    """

    def __init__(self, device='cpu', freeze_decoder_weights=False):
        super(EfficientNet, self).__init__()
        efficientNet = torchvision.models.efficientnet_b0(pretrained=True)
        # print("___________________________________")
        # print(list(efficientNet.children())[:-2])
        self.freeze_decoder_weights = freeze_decoder_weights
        self.device = device

        if self.freeze_decoder_weights:
            for param in efficientNet.parameters():
                param.requires_grad = False
        print("--------efficientnet Based--------")
        self.model = nn.Sequential(*(list(efficientNet.children())[:-2]),
                                   nn.Conv2d(1280, 512, 3, padding=1, padding_mode='zeros'),
                                   nn.BatchNorm2d(512),
                                   FastUpConvolution(512, 256),
                                   FastUpConvolution(256, 128),
                                   FastUpConvolution(128, 64),
                                   nn.Dropout2d(),
                                   nn.Conv2d(64, 1, 3, padding=1, padding_mode='zeros'),
                                   nn.ReLU(),
                                   nn.Upsample(size=(228, 304), mode='bilinear'))
        print(f'Trainable parameters for {self._get_name()} => {get_trainable_parameters(self.model)}')

    def forward(self, input_tensor):
        # input_tensor.to(self.device)
        """Runs a forward pass over the Model"""
        output_tensor = self.model(input_tensor)
        return output_tensor


class VGG16(nn.Module):
    """"
    VGG16 Based model which accepts input of size 304 X 228 X 3
    And generates a prediction of 160 X 128 X 1
    Architecture :
        VGG16 Base Model ( without FC layers ) , followed by UnConvolutionLayers
    """

    def __init__(self, device='cpu', freeze_decoder_weights=False):
        super(VGG16, self).__init__()
        vgg16 = torchvision.models.vgg16(pretrained=True)
        # print("___________________________________")
        # print(list(efficientNet.children())[:-2])
        self.freeze_decoder_weights = freeze_decoder_weights
        self.device = device

        if self.freeze_decoder_weights:
            for param in vgg16.parameters():
                param.requires_grad = False
        print("--------VGG-16 Based--------")
        self.model = nn.Sequential(*(list(vgg16.children())[:-2]),
                                   nn.Conv2d(512, 512, 3, padding=1, padding_mode='zeros'),
                                   nn.BatchNorm2d(512),
                                   FastUpConvolution(512, 256),
                                   FastUpConvolution(256, 128),
                                   FastUpConvolution(128, 64),
                                   nn.Dropout2d(),
                                   nn.Conv2d(64, 1, 3, padding=1, padding_mode='zeros'),
                                   nn.ReLU(),
                                   nn.Upsample(size=(228, 304), mode='bilinear'))
        print(f'Trainable parameters for {self._get_name()} => {get_trainable_parameters(self.model)}')

    def forward(self, input_tensor):
        # input_tensor.to(self.device)
        """Runs a forward pass over the Model"""
        output_tensor = self.model(input_tensor)
        return output_tensor


class VGG19(nn.Module):
    """"
    VGG19 Based model which accepts input of size 304 X 228 X 3
    And generates a prediction of 160 X 128 X 1
    Architecture :
        VGG19 Base Model ( without FC layers ) , followed by UnConvolutionLayers
    """

    def __init__(self, device='cpu', freeze_decoder_weights=False):
        super(VGG19, self).__init__()
        vgg19 = torchvision.models.vgg19(pretrained=True)
        # print("___________________________________")
        # print(list(efficientNet.children())[:-2])
        self.freeze_decoder_weights = freeze_decoder_weights
        self.device = device

        if self.freeze_decoder_weights:
            for param in vgg19.parameters():
                param.requires_grad = False
        print("--------VGG-19 Based--------")
        self.model = nn.Sequential(*(list(vgg19.children())[:-2]),
                                   nn.Conv2d(512, 512, 3, padding=1, padding_mode='zeros', bias=False),
                                   nn.BatchNorm2d(512),
                                   FastUpProjection(512, 256),
                                   FastUpProjection(256, 128),
                                   FastUpProjection(128, 64),
                                   nn.Dropout2d(),
                                   nn.Conv2d(64, 1, 3, padding=1, padding_mode='zeros'),
                                   nn.ReLU(),
                                   nn.Upsample(size=(228, 304), mode='bilinear'))
        print(f'Trainable parameters for {self._get_name()} => {get_trainable_parameters(self.model)}')

    def forward(self, input_tensor):
        # input_tensor.to(self.device)
        """Runs a forward pass over the Model"""
        output_tensor = self.model(input_tensor)
        return output_tensor
