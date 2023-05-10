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
