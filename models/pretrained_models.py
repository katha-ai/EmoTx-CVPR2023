from torchvision import models
from models.vgg_m_face_bn_fer_dag import Vgg_m_face_bn_fer_dag
from models.resnet50_face_sfew_dag import Resnet50_face_sfew_dag

import torch
import torch.nn as nn


class MViT_v1(nn.Module):
    """
    Load the pretrained MViT model pretrained on Kinetics400 dataset from "MViT: Video Vision Transformer (CVPR), 2021"
    """
    def __init__(self, weights_path):
        """
        Args:
            weights_path (str): path to the pretrained weights
        """
        super(MViT_v1, self).__init__()
        self.model = torch.hub.load("facebookresearch/pytorchvideo", model="mvit_base_32x3", pretrained=False, verbose=False)
        ckpt = torch.load(weights_path)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.head = self.model.head.sequence_pool

    def forward(self, frames):
        """
        Args:
            frames (Tensor): Tensor of shape (batch_size, num_frames, 3, 224, 224)

        Returns:
            out (Tensor): Tensor of shape (batch_size, 768)
        """
        out = self.model(frames)
        return out


class Places365_ResNet50(nn.Module):
    def __init__(self, weights_path, device):
        super(Places365_ResNet50, self).__init__()
        self.model = models.__dict__["resnet50"](num_classes=365)
        ckpt = torch.load(weights_path, map_location=device)
        state_dict = {str.replace(k,'module.',''): v for k,v in ckpt['state_dict'].items()}
        self.model.load_state_dict(state_dict)
        self.model = nn.Sequential(*(list(self.model.children())[:-1]))

    def forward(self, imgs):
        batch_size = imgs.shape[0]
        out = self.model(imgs)
        out = out.reshape(batch_size, -1)
        return out


class ImageNet_ResNet152(nn.Module):
    def __init__(self):
        super(ImageNet_ResNet152, self).__init__()
        self.model = models.resnet152(pretrained=True)
        self.model = nn.Sequential(*(list(self.model.children())[:-1]))

    def forward(self, frames):
        batch_size = frames.shape[0]
        out = self.model(frames)
        out = out.reshape(batch_size, -1)
        return out


class VGGF2_face(nn.Module):
    def __init__(self, weights_path):
        super(VGGF2_face, self).__init__()
        self.model = Vgg_m_face_bn_fer_dag()
        self.model.load_state_dict(torch.load(weights_path))
        self.model = nn.Sequential(*(list(self.model.children())[:-4]))

    def forward(self, imgs):
        batch_size = imgs.shape[0]
        out = self.model(imgs)
        out = out.reshape(batch_size, -1)
        return out


class Resnet50_FER(nn.Module):
    def __init__(self, weights_path):
        super(Resnet50_FER, self).__init__()
        self.model = Resnet50_face_sfew_dag()
        self.model.load_state_dict(torch.load(weights_path))

    def forward(self, imgs):
        out = self.model(imgs)
        return out
