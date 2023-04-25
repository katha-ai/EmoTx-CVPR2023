from transformers import RobertaModel

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F


class finetune_RoBERTa(nn.Module):
    """
    This is a RoBERTa model that is finetuned for the task of multi-label emotion recognition
    on MovieGraphs dataset. The model is initialized with the weights of the RoBERTa-base model
    """
    def __init__(self, num_labels, hf_cache_dir):
        """
        Args:
            num_labels (int): Number of labels in the dataset
            hf_cache_dir (str): Path to the cache directory for the huggingface transformers
        """
        super(finetune_RoBERTa, self).__init__()
        self.RoBERTa = RobertaModel.from_pretrained("roberta-base", cache_dir=hf_cache_dir)
        self.logits = nn.Linear(768, num_labels)

    def forward(self, feats, masks, logits=True):
        """
        Args:
            feats (list): List of tensors with token-ids of shape (batch_size, seq_len, feat_dim)
            masks (list): List of tensors of shape (batch_size, seq_len)
            logits (bool): If True, returns the logits, else returns the pooler output

        Returns:
            x (list): List of Tensor of shape (batch_size, num_labels) if logits=True, else (batch_size, 768)
        """
        x = feats[0]; mask = masks[0]
        x = self.RoBERTa(x, mask)
        x = self.logits(x.pooler_output) if logits else x.pooler_output
        return [x]


class featExtract_finetuned_RoBERTa(nn.Module):
    """
    This is a wrapper model used to extract features from the finetuned RoBERTa model.
    """
    def __init__(self, num_labels, model_path):
        """
        Args:
            num_labels (int): Number of labels in the dataset
            model_path (str): Path to the finetuned RoBERTa model
        """
        super(featExtract_finetuned_RoBERTa, self).__init__()
        self.model = self.get_model(model_path)

    def get_model(self, model_path):
        """
        Loads the finetuned RoBERTa model from the given path.
        Args:
            model_path (str): Path to the finetuned RoBERTa model

        Returns:
            model (torch.nn.Module): Finetuned RoBERTa model
        """
        model = torch.load(model_path, map_location="cpu")
        return model

    def forward(self, feats, masks):
        """
        Args:
            feats (list): List of tensors with token-ids of shape (batch_size, seq_len, feat_dim)
            masks (list): List of tensors of shape (batch_size, seq_len)

        Returns:
            feat (tensor): Tensor of shape (batch_size, 768)
        """
        feat = self.model([feats], [masks], logits=False)
        return feat[0]


class featExtract_pretrained_RoBERTa(nn.Module):
    """
    This is a wrapper model used to extract features from the pretrained RoBERTa model.
    """
    def __init__(self, hf_cache_dir):
        """
        Args:
            hf_cache_dir (str): Path to the cache directory for the huggingface transformers
        """
        super(featExtract_pretrained_RoBERTa, self).__init__()
        self.RoBERTa = RobertaModel.from_pretrained("roberta-base", cache_dir=hf_cache_dir)

    def forward(self, feat, mask):
        """
        Args:
            feat (list): List of tensors with token-ids of shape (batch_size, seq_len, feat_dim)
            mask (list): List of tensors of shape (batch_size, seq_len)

        Returns:
            pooler_output (tensor): Tensor of shape (batch_size, 768)
        """
        x = self.RoBERTa(feat, mask)
        return x.pooler_output
