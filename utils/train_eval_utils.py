from pathlib import Path
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from utils.AverageMeter import AverageMeter

import numpy as np
import os
import pickle
import random
import torch
import torch.nn as nn
import time
import wandb
import yaml


def set_seed(seed_value):
    """
    Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def save_config(config, path, filename):
    """
    Save config to yaml file.

    Args:
        config (dict): Dictionary containing config.
        path (str): Path to save config.
    """
    if not os.path.exists(path):
        os.mkdir(path)
    f = open(path/filename, 'w')
    yaml.dump(config, f)
    f.close()


def select_feat_ndxs(num_of_feats, skip_steps, max_feats, random_feat_selection=True):
    """
    Selects the feature indices to be used for selecting scene and character features.
    Operates in two modes:
        1. Randomly select a subset of features. (Augmentation during training)
        2. Select a subset of features by skipping a fixed number of features. (Inference mode)

    Args:
        num_of_feats (int): Number of features in the input.
        skip_steps (int): Number of features to skip.
        max_feats (int): Maximum number of features to select.
        random_feat_selection (bool, optional): Whether to select features randomly or uniformaly. Defaults to True.

    Returns:
        sampled_ndxs (torch.tensor): Indices of features to be selected.
    """
    sampled_ndxs = None
    if random_feat_selection:
        total_samples = max(1, num_of_feats//skip_steps)
        sampled_ndxs = random.sample(range(0, num_of_feats), min(total_samples, max_feats))
        sampled_ndxs = torch.tensor(sampled_ndxs).sort().values
    else:
        sampled_ndxs = torch.tensor(range(0, num_of_feats, skip_steps))
        sampled_ndxs = sampled_ndxs[:max_feats]
    return sampled_ndxs


def calculate_metrics(outputs, predictions, targets, num_labels):
    """
    Calculates the metrics (class AP, P, R and F1) for given targets and predictions.

    Args:
        outputs (list): List of output tensors from the model. (Post Sigmoid)
        predictions (list): List of prediction tensors from the model. (Post Sigmoid and Thresholding)
        targets (list): List of target tensors. (One-hot vectors)
        num_labels (int): Number of labels in the dataset.

    Returns:
        metrics (dict): Dictionary containing the metrics.
    """
    stacked_outputs = [torch.empty((0, num_labels))]*len(outputs[0])
    stacked_preds = [torch.empty((0, num_labels))]*len(predictions[0])
    stacked_targets = [torch.empty((0, num_labels))]*len(targets[0])
    for ndx, (out, pred, target) in enumerate(zip(outputs, predictions, targets)):
        stacked_outputs = [np.vstack([stacked_outputs[i], out[i]]) for i in range(len(out))]
        stacked_preds = [np.vstack([stacked_preds[i], pred[i]]) for i in range(len(pred))]
        stacked_targets = [np.vstack([stacked_targets[i], target[i]]) for i in range(len(target))]
    class_AP_scores = [average_precision_score(stacked_targets[i], stacked_outputs[i], average=None) if stacked_targets[i].shape[0] else 0.0 for i in range(len(targets[0]))]
    class_F1_scores = [f1_score(stacked_targets[i], stacked_preds[i], average=None, zero_division=0) if stacked_targets[i].shape[0] else 0.0 for i in range(len(targets[0]))]
    class_P_scores = [precision_score(stacked_targets[i], stacked_preds[i], average=None, zero_division=0) if stacked_targets[i].shape[0] else 0.0 for i in range(len(targets[0]))]
    class_R_scores = [recall_score(stacked_targets[i], stacked_preds[i], average=None, zero_division=0) if stacked_targets[i].shape[0] else 0.0 for i in range(len(targets[0]))]
    metrics = {
        "AP": class_AP_scores,
        "F1": class_F1_scores,
        "P": class_P_scores,
        "R": class_R_scores
    }
    return metrics


def save_checkpoint_metadata(save_path, model_name, suffix, emo2id, train_metrics, eval_metrics):
    """
    Save the checkpoint metadata to pickle file.

    Args:
        save_path (Path): Path to save the metadata.
        model_name (str): Name of the model.
        suffix (str): Suffix to add to the filename.
        emo2id (dict): Dictionary mapping emotion to id.
        train_metrics (dict): Dictionary containing the training metrics.
        eval_metrics (dict): Dictionary containing the evaluation metrics.
    """
    f = open(save_path/(model_name + suffix), "wb")
    pickle.dump(emo2id, f)
    pickle.dump(train_metrics, f)
    pickle.dump(eval_metrics, f)
    f.close()


def save_model(device, save_path, model, model_name, suffix):
    """
    Save the model to the given path.

    Args:
        save_path (Path): Path to save the model.
        model (nn.Module): Model to save.
        model_name (str): Name of the model.
        suffix (str): Suffix to add to the filename.
    """
    save_dict = {
        "num_labels": int(model.num_labels),
        "num_pos_embeddings": int(model.positional_encoder.pe.shape[1]),
        "scene_feat_dim": int(model.scene_dim_reduction.weight.shape[1]),
        "char_feat_dim": int(model.char_dim_reduction.weight.shape[1]),
        "srt_feat_dim": int(model.srt_dim_reduction.weight.shape[1]),
        "num_chars": int(model.num_chars),
        "num_enc_layers": len(model.encoder.layers),
        "max_individual_tokens": int(model.max_individual_tokens),
        "hidden_dim": int(model.hidden_dim),
        "state_dict": model.cpu().state_dict(),
    }
    torch.save(save_dict, str(save_path/(model_name+suffix)))
    model = model.to(device)


def wandb_log_summary(wandb_logging, eval_mAPs, epoch, summary_str):
    """
    Log the summary metrics to wandb.

    Args:
        wandb_logging (bool): Whether to log to wandb.
        eval_mAPs (list): List of evaluation mAPs.
        epoch (int): Current epoch.
        summary_str (str): Summary string.
    """
    if wandb_logging:
        for ndx, eval_mAP in enumerate(eval_mAPs):
            wandb.run.summary[summary_str.format(ndx)] = eval_mAP
        wandb.run.summary[summary_str.format("epoch")] = epoch


def create_checkpoint(eval_mAPs, model, save_path, model_name, emo2id, train_metrics, eval_metrics, wandb_logging,
                      device, epoch, best_ev_score_scene, best_ev_score_char, best_ev_score_avg, best_ev_score_gm):
    """
    Create checkpoint if the current evaluation mAP is better than the best evaluation mAP.
    Four different types of checkpoints are created:
        1. Best evaluation mAP for scene level.
        2. Best evaluation mAP for character level.
        3. Average of all the evaluation mAPs.
        4. Geometric mean of all the evaluation mAPs.

    Args:
        eval_mAPs (list): List of evaluation mAPs.
        model (nn.Module): Model to save.
        save_path (Path): Path to save the model.
        model_name (str): Name of the model.
        emo2id (dict): Dictionary mapping emotion to id.
        train_metrics (dict): Dictionary containing the training metrics.
        eval_metrics (dict): Dictionary containing the evaluation metrics.
        wandb_logging (bool): Whether to log to wandb.
        device (torch.device): Device to use.
        epoch (int): Current epoch.
        best_ev_score_scene (float): Best evaluation mAP for scene level.
        best_ev_score_char (float): Best evaluation mAP for character level.
        best_ev_score_avg (float): Best evaluation mAP for average.
        best_ev_score_gm (float): Best evaluation mAP for geometric mean.
    """
    if eval_mAPs[-1] >= best_ev_score_scene:
        save_model(device=device, save_path=save_path, model=model, model_name=model_name, suffix="_scene.pt")
        save_checkpoint_metadata(save_path, model_name, "_scene.pkl", emo2id, train_metrics, eval_metrics)
        wandb_log_summary(wandb_logging, eval_mAPs, epoch, "best_eval_mAP_{}_sceneCkpt")
        best_ev_score_scene = eval_mAPs[-1]
        print("New best eval_mAP_scene: {}".format(best_ev_score_scene))
    if eval_mAPs[0] >= best_ev_score_char:
        save_model(device=device, save_path=save_path, model=model, model_name=model_name, suffix="_char.pt")
        save_checkpoint_metadata(save_path, model_name, "_char.pkl", emo2id, train_metrics, eval_metrics)
        wandb_log_summary(wandb_logging, eval_mAPs, epoch, "best_eval_mAP_{}_charCkpt")
        best_ev_score_char = eval_mAPs[0]
        print("New best eval_mAP_char: {}".format(best_ev_score_char))
    if np.sum(eval_mAPs)/len(eval_mAPs) >= best_ev_score_avg:
        save_model(device=device, save_path=save_path, model=model, model_name=model_name, suffix="_avg.pt")
        save_checkpoint_metadata(save_path, model_name, "_avg.pkl", emo2id, train_metrics, eval_metrics)
        wandb_log_summary(wandb_logging, eval_mAPs, epoch, "best_eval_mAP_{}_avgCkpt")
        best_ev_score_avg = np.sum(eval_mAPs)/len(eval_mAPs)
        print("New best eval_mAP_avg: {}".format(best_ev_score_avg))
    if np.sqrt(np.prod(eval_mAPs)) >= best_ev_score_gm:
        save_model(device=device, save_path=save_path, model=model, model_name=model_name, suffix="_gm.pt")
        save_checkpoint_metadata(save_path, model_name, "_gm.pkl", emo2id, train_metrics, eval_metrics)
        wandb_log_summary(wandb_logging, eval_mAPs, epoch, "best_eval_mAP_{}_gmCkpt")
        best_ev_score_gm = np.sqrt(np.prod(eval_mAPs))
        print("New best eval_mAP_gm: {}".format(best_ev_score_gm))
    return best_ev_score_scene, best_ev_score_char, best_ev_score_avg, best_ev_score_gm


def train(epochs, num_labels, train_dataloader, val_dataloader, device, emo2id,
          model, optimizer, scheduler, criterion, pred_thresh=0.5,
          masking=True, wandb_logging=False, model_name="NA", save_path=Path("./")):
    """
    Main train loop for training the models.

    Args:
        epochs (int): Number of epochs to train for.
        num_labels (int): Number of labels.
        train_dataloader (DataLoader): Dataloader for the training set.
        val_dataloader (DataLoader): Dataloader for the validation set.
        device (torch.device): Device to use.
        emo2id (dict): Dictionary mapping emotion to id.
        model (nn.Module): Model to train.
        optimizer (torch.optim.Optimizer): Optimizer to use.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler to use.
        criterion (nn.Module): Loss function to use.
        pred_thresh (float): Prediction threshold.
        masking (bool): Whether to use masking.
        wandb_logging (bool): Whether to log to wandb.
        model_name (str): Name of the model.
        save_path (Path): Path to save the model.
    """
    train_metrics = dict()
    eval_metrics = dict()
    data_to_log = dict()
    log_line = "Epoch: {} | Train_Loss: {} | Eval_Loss: {} | Train_mAP: {} | Eval_mAP: {} | Total time taken: {}"
    eval_loss, eval_metrics = evaluate(num_labels, val_dataloader, device, model, criterion, pred_thresh, masking)
    best_ev_score_scene, best_ev_score_char, best_ev_score_avg, best_ev_score_gm = -1, -1, -1, -1
    if wandb_logging:
        wandb.run.summary["first_eval_loss"] = eval_loss.avg
    print("Before train:\n eval_loss: {} | eval_mAPs: {}".format(eval_loss.avg, [np.mean(AP_scores) for AP_scores in eval_metrics["AP"]]))
    print("Starting model training")
    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        train_loss = AverageMeter("train_loss", ":.5f")
        train_mAPs = list()
        epoch_targets, epoch_outputs, epoch_preds = list(), list(), list()
        for data in tqdm(train_dataloader, disable=wandb_logging):
            for key, tensors in data.items():
                data[key] = [tensor.to(device) for tensor in tensors]
            optimizer.zero_grad()
            model_outputs = model(data["feats"], data["masks"]) if masking else model(data["feats"])
            stacked_targets = torch.cat([target for target in data["targets"]], dim=0)
            stacked_outputs = torch.cat([out for out in model_outputs], dim=0)
            net_loss = criterion(stacked_outputs, stacked_targets)
            train_loss.update(net_loss.item(), sum([target.size(0) for target in data["targets"]]))
            preds = [(torch.sigmoid(out) >= pred_thresh).to(torch.float32) for out in model_outputs]
            epoch_targets.append([target.detach().cpu().numpy() for target in data["targets"]])
            epoch_outputs.append([torch.sigmoid(out).detach().cpu().numpy() for out in model_outputs])
            epoch_preds.append([pred.detach().cpu().numpy() for pred in preds])
            net_loss.backward()
            optimizer.step()
        train_metrics = calculate_metrics(epoch_outputs, epoch_preds, epoch_targets, num_labels)
        eval_loss, eval_metrics = evaluate(num_labels, val_dataloader, device, model, criterion, pred_thresh, masking)
        train_mAPs = [np.mean(AP_scores) for AP_scores in train_metrics["AP"]]
        eval_mAPs = [np.mean(AP_scores) for AP_scores in eval_metrics["AP"]]
        train_F1 = [np.mean(F1_scores) for F1_scores in train_metrics["F1"]]
        eval_F1 = [np.mean(F1_scores) for F1_scores in eval_metrics["F1"]]
        train_P = [np.mean(P_scores) for P_scores in train_metrics["P"]]
        eval_P = [np.mean(P_scores) for P_scores in eval_metrics["P"]]
        train_R = [np.mean(R_scores) for R_scores in train_metrics["R"]]
        eval_R = [np.mean(R_scores) for R_scores in eval_metrics["R"]]
        if scheduler:
            scheduler.step(eval_loss.avg)
        data_to_log.update({
            "train_loss": train_loss.avg,
            "eval_loss": eval_loss.avg,
            "train_mAP": sum(train_mAPs)/len(train_mAPs),
            "eval_mAP": sum(eval_mAPs)/len(eval_mAPs),
            "mean_F1": sum(train_F1)/len(train_F1),
            "mean_P": sum(train_P)/len(train_P),
            "mean_R": sum(train_R)/len(train_R),
            "lr": optimizer.state_dict()["param_groups"][0]["lr"],
        })
        for ndx, (train_mAP, eval_mAP) in enumerate(zip(train_mAPs, eval_mAPs)):
            data_to_log["train_mAP_{}".format(ndx)] = train_mAP
            data_to_log["eval_mAP_{}".format(ndx)] = eval_mAP
        for ndx, (train_f1, eval_f1) in enumerate(zip(train_F1, eval_F1)):
            data_to_log["train_F1_{}".format(ndx)] = train_f1
            data_to_log["eval_F1_{}".format(ndx)] = eval_f1
        for ndx, (train_r, eval_r) in enumerate(zip(train_R, eval_R)):
            data_to_log["train_R_{}".format(ndx)] = train_r
            data_to_log["eval_R_{}".format(ndx)] = eval_r
        for ndx, (train_p, eval_p) in enumerate(zip(train_P, eval_P)):
            data_to_log["train_P_{}".format(ndx)] = train_p
            data_to_log["eval_P_{}".format(ndx)] = eval_p
        if wandb_logging:
            wandb.log(data_to_log)
        print(log_line.format(epoch+1,
                              data_to_log["train_loss"],
                              data_to_log["eval_loss"],
                              data_to_log["train_mAP"],
                              [eval_mAPs, data_to_log["eval_mAP"]],
                              time.time() - start_time))
        # Checkpoint selection (Scene, char, average, geometric-mean)
        new_ckpt_threshs = create_checkpoint(eval_mAPs, model, save_path, model_name, emo2id, train_metrics, eval_metrics, wandb_logging,
                                             device, epoch, best_ev_score_scene, best_ev_score_char, best_ev_score_avg, best_ev_score_gm)
        best_ev_score_scene, best_ev_score_char, best_ev_score_avg, best_ev_score_gm = new_ckpt_threshs


@torch.no_grad()
def evaluate(num_labels, val_dataloader, device, model, criterion, pred_thresh=0.5, masking=True):
    """
    Evaluate the model on the validation set.

    Args:
        num_labels (int): Number of labels.
        val_dataloader (DataLoader): Dataloader for the validation set.
        device (str): Device to use for training.
        model (torch.nn.Module): Model to evaluate.
        criterion (torch.nn.Module): Loss function.
        pred_thresh (float): Threshold for predictions.
        masking (bool): Whether to use masking.

    Returns:
        eval_loss (AverageMeter): Loss on the validation set.
        eval_metrics (dict): Metrics on the validation set.
    """
    model.eval()
    eval_loss = AverageMeter("eval_loss", ":.5f")
    epoch_targets, epoch_outputs, epoch_preds = list(), list(), list()
    for data in val_dataloader:
        for key, tensors in data.items():
            data[key] = [tensor.to(device) for tensor in tensors]
        model_outputs = model(data["feats"], data["masks"]) if masking else model(data["feats"])
        stacked_targets = torch.cat([target for target in data["targets"]], dim=0)
        stacked_outputs = torch.cat([out for out in model_outputs], dim=0)
        net_loss = criterion(stacked_outputs, stacked_targets)
        eval_loss.update(net_loss.item(), sum([target.size(0) for target in data["targets"]]))
        preds = [(torch.sigmoid(out) >= pred_thresh).to(torch.float32) for out in model_outputs]
        epoch_targets.append([target.detach().cpu().numpy() for target in data["targets"]])
        epoch_outputs.append([torch.sigmoid(out).detach().cpu().numpy() for out in model_outputs])
        epoch_preds.append([pred.detach().cpu().numpy() for pred in preds])
    eval_metrics = calculate_metrics(epoch_outputs, epoch_preds, epoch_targets, num_labels)
    return eval_loss, eval_metrics
