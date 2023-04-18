import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Positional encoding from "Attention is all you need"
    """
    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, feat, bins):
        pos_emb = self.pe[:, bins, :]
        out = feat + pos_emb.squeeze(0)
        return self.dropout(out)



class tx_char(nn.Module):
    """
    A Transformer Encoder for character-level emotion recognition.
    """
    def __init__(self, num_labels, feat_dim, device):
        """
        Args:
            num_labels (int): Number of labels.
            feat_dim (int): Dimension of the input features.
            device (torch.device): Device on which the model will be run.
        """
        super(tx_char, self).__init__()
        self.cls = nn.Parameter(torch.randn(1,1,feat_dim))
        self.pos_encoder = PositionalEncoding(feat_dim, 0.5)
        encoder_layer = nn.TransformerEncoderLayer(d_model=feat_dim, nhead=8)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.logits = nn.Linear(feat_dim, num_labels)
        self.cls_mask = torch.zeros((1,1)).to(device)
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
        x = feats[0][:,0,:,:]
        mask = masks[0][:,0,:]
        bins = feats[3][:,0,:]
        batches = x.shape[0]
        x = self.pos_encoder(x, bins)
        cls_emb = torch.cat([self.cls for i in range(batches)])
        cls_mask = self.cls_mask.repeat(batches, 1)
        x = torch.cat([cls_emb, x], dim=1)
        mask = torch.cat([cls_mask, mask], dim=1)
        x = x.permute(1,0,2)
        enc = self.encoder(x, src_key_padding_mask=mask)
        out = self.logits(enc[0,:,:])
        return [out, self.blank_scene_out]


class tx_scene(nn.Module):
    """
    A Transformer Encoder for scene-level emotion recognition.
    """
    def __init__(self, num_labels, feat_dim, device):
        """
        Args:
            num_labels (int): Number of labels.
            feat_dim (int): Dimension of the input features.
            device (torch.device): Device on which the model will be run.
        """
        super(tx_scene, self).__init__()
        self.cls = nn.Parameter(torch.randn(1,1,feat_dim))
        self.pos_encoder = PositionalEncoding(feat_dim, 0.5)
        encoder_layer = nn.TransformerEncoderLayer(d_model=feat_dim, nhead=8)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.logits = nn.Linear(feat_dim, num_labels)
        self.cls_mask = torch.zeros((1,1)).to(device)
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
        x = feats[1]
        mask = masks[1]
        bins = feats[4]
        batches = x.shape[0]
        x = self.pos_encoder(x, bins)
        cls_emb = torch.cat([self.cls for i in range(batches)])
        cls_mask = self.cls_mask.repeat(batches, 1)
        x = torch.cat([cls_emb, x], dim=1)
        mask = torch.cat([cls_mask, mask], dim=1)
        x = x.permute(1,0,2)
        enc = self.encoder(x, src_key_padding_mask=mask)
        out = self.logits(enc[0,:,:])
        return [self.blank_char_out, out]
