import torch
import torch.nn as nn


class mlp_char(nn.Module):
    """
    MLP for character-level emotion classification
    """
    def __init__(self, num_labels, feat_dim, device):
        """
        Args:
            num_labels (int): Number of emotion labels
            feat_dim (int): Dimension of input features
            device (torch.device): Device to store model parameters
        """
        super(mlp_char, self).__init__()
        self.lin1 = nn.Linear(feat_dim, 1024)
        self.lin2 = nn.Linear(1024, num_labels)
        self.blank_scene_out = torch.empty((0, num_labels)).to(device)

    def forward(self, feats, masks):
        """
        Args:
            feats (list): List of features. feats[0] is a tensor of shape
                (batch_size, 1, max_seq_len, feat_dim). Character feature.
            masks (list): List of masks. masks[0] is a tensor of shape
                (batch_size, 1, max_seq_len). Character feature mask.

        Returns:
            list: List of output tensors. The first tensor is a tensor of shape
                (batch_size, num_labels) containing the character-level
                predictions. The second tensor is a tensor of shape
                (0, num_labels) containing the scene-level predictions.
        """
        char_feats = feats[0]
        x = char_feats[:,0,:,:]
        max_pooled_x = torch.max(x, dim=1)
        out = self.lin1(max_pooled_x.values)
        out = self.lin2(out)
        return [out, self.blank_scene_out]


class mlp_scene(nn.Module):
    """
    MLP for scene-level emotion classification
    """
    def __init__(self, num_labels, feat_dim, device):
        """
        Args:
            num_labels (int): Number of emotion labels
            feat_dim (int): Dimension of input features
            device (torch.device): Device to store model parameters
        """
        super(mlp_scene, self).__init__()
        self.lin1 = nn.Linear(feat_dim, 1024)
        self.lin2 = nn.Linear(1024, num_labels)
        self.blank_char_out = torch.empty((0, num_labels)).to(device)

    def forward(self, feats, masks):
        """
        Args:
            feats (list): List of features. feats[1] is a tensor of shape
                (batch_size, max_seq_len, feat_dim). Scene feature.
            masks (list): List of masks. masks[1] is a tensor of shape
                (batch_size, max_seq_len). Scene feature mask.

        Returns:
            list: List of output tensors. The first tensor is a tensor of shape
                (0, num_labels) containing the character-level predictions.
                The second tensor is a tensor of shape
                (batch_size, num_labels) containing the scene-level
                predictions.
        """
        scene_feat = feats[1]
        max_pooled_x = torch.max(scene_feat, dim=1)
        out = self.lin1(max_pooled_x.values)
        out = self.lin2(out)
        return [self.blank_char_out, out]
