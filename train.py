import wandb
import json
from typing import List, Dict, Tuple
import multiprocessing
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import os
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
from PIL import Image
from torchvision import transforms 
from transformers import XLMRobertaTokenizer, get_cosine_schedule_with_warmup
import torch.nn.functional as F
from argparse import Namespace
from timm.utils import ModelEma
from optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner, get_num_layer_for_vit

from combiner import Combiner
from my_dataset import ComposedRetrievalDataset, TargetPad, CorpusDataset,ValidationQueriesDataset,QueryImageDataset
from my_utils import update_train_running_results, set_train_bar_description
from metric_loss import TripletLoss, CircleLoss, SoftSimilarityLoss

from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data.transforms import RandomResizedCropAndInterpolation
# from randaug import RandomAugment 

import utils
import modeling_finetune
from modeling_utils import _get_large_config, _get_base_config
from beit3_config import Config

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

import torch
import torch.nn.functional as F

def compute_hnm_loss(logits, ground_truth, topk, margin, temperature=1.0):

    logits = logits.float()
    batch_size = logits.size(0)
    
    # 1. Get positive-pair scores (N, 1)
    pos_scores = logits.gather(1, ground_truth.view(-1, 1))

    # 2. Mask positive pairs and mine negative pairs
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(1, ground_truth.view(-1, 1), True)
    neg_logits = logits.clone()
    neg_logits.masked_fill_(mask, float('-inf'))

    # 3. Select Top-K hard negatives
    actual_k = min(topk, batch_size - 1)
    if actual_k <= 0:
        return torch.tensor(0.0, device=logits.device), 0.0, 0.0
        
    # hard_neg_scores: (N, K)
    hard_neg_scores, _ = torch.topk(neg_logits, k=actual_k, dim=1)

    # 4. Compute the loss
    # delta = s_neg - s_pos + m
    delta = (hard_neg_scores - pos_scores + margin) / temperature
    loss_matrix = F.softplus(delta) 
    loss = loss_matrix.mean()
    
    # Mean positive-pair score
    mean_pos_score = pos_scores.detach().mean().item()
    # Mean hard-negative score
    mean_neg_score = hard_neg_scores.detach().mean().item()
    
    return loss, mean_pos_score, mean_neg_score

def save_combiner_only_checkpoint(
    name: str,
    cur_epoch: int,
    combiner: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    best_metric: float,
    training_path: Path,
    scheduler,
    hyper_params: dict,
    combiner_ema: ModelEma = None,
):
    models_path = training_path / "saved_models"
    models_path.mkdir(exist_ok=True, parents=True)

    checkpoint = {
        'epoch': cur_epoch,
        'training_mode': 'combiner_only',
        'combiner_state_dict': combiner.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_recall_at_10': float(best_metric),
        'meta': {
            'beit3_checkpoint_path': hyper_params['beit3_checkpoint_path'],
            'projection_dim': hyper_params['projection_dim'],
            'hidden_dim': hyper_params['hidden_dim'],
            'fusion_strategy': hyper_params['fusion_strategy'],
            'input_size': hyper_params['input_size'],
        }
    }

    if combiner_ema is not None:
        checkpoint['combiner_ema_state_dict'] = combiner_ema.ema.state_dict()

    torch.save(checkpoint, str(models_path / f'{name}.pt'))

def build_beit3_transform(is_train: bool, config: dict):

    if is_train:
        # Training mode: include data augmentation
        t = [
            RandomResizedCropAndInterpolation(config['input_size'], scale=(0.5, 1.0), interpolation=config['train_interpolation']), 
            transforms.RandomHorizontalFlip(),
        ]
        if config.get('randaug', False): # Use .get() to safely read the configuration
            t.append(transforms.RandAugment(num_ops=2, magnitude=9))
        t += [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD), 
        ]
        return transforms.Compose(t)
    else:
        return transforms.Compose([
            transforms.Resize((config['input_size'], config['input_size']), interpolation=transforms.InterpolationMode.BICUBIC), 
            # transforms.CenterCrop(config['input_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
        ])

def save_checkpoint(name: str, cur_epoch: int, model: nn.Module, optimizer: optim.Optimizer,
                    scaler: torch.cuda.amp.GradScaler, best_metric: float, training_path: Path,
                    scheduler,combiner: Combiner = None, model_ema: ModelEma = None,combiner_ema: ModelEma = None):

    models_path = training_path / "saved_models"
    models_path.mkdir(exist_ok=True, parents=True)
    
    checkpoint = {
        'epoch': cur_epoch,
        # 'model_state_dict': model.state_dict(),
        'model': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_recall_at_10': float(best_metric)
    }
    if combiner:
        checkpoint['combiner_state_dict'] = combiner.state_dict()
    if model_ema is not None:
        checkpoint['model_ema_state_dict'] = model_ema.ema.state_dict()
    if combiner_ema is not None:
        checkpoint['combiner_ema_state_dict'] = combiner_ema.ema.state_dict()
        
    torch.save(checkpoint, str(models_path / f'{name}.pt'))

def export_merged_release_checkpoint(
    name: str,
    training_path: Path,
    model: nn.Module,
    combiner: nn.Module,
    hyper_params: dict,
):
    models_path = training_path / "saved_models"
    models_path.mkdir(exist_ok=True, parents=True)

    release_ckpt = {
        'model': model.state_dict(),
        'combiner_state_dict': combiner.state_dict(),
        'training_mode': 'merged_release',
        'meta': {
            'source_training_mode': hyper_params['training_mode'],
            'beit3_checkpoint_path': hyper_params['beit3_checkpoint_path'],
            'projection_dim': hyper_params['projection_dim'],
            'hidden_dim': hyper_params['hidden_dim'],
            'fusion_strategy': hyper_params['fusion_strategy'],
            'input_size': hyper_params['input_size'],
        }
    }

    torch.save(release_ckpt, str(models_path / f'{name}.pt'))

def extract_corpus_features(corpus_dataset: CorpusDataset, beit3_model: nn.Module, batch_size: int, 
                            device: torch.device):
    
    def corpus_collate_fn(batch):
        image_paths, images_tensors = zip(*batch)
        pixel_values = torch.stack(images_tensors)
        ids = [corpus_dataset.path_to_id_map[p] for p in image_paths]
        return torch.tensor(ids), pixel_values

    corpus_loader = DataLoader(
        dataset=corpus_dataset, 
        batch_size=batch_size,
        # num_workers=multiprocessing.cpu_count(),
        num_workers=4,
        collate_fn=corpus_collate_fn,
        pin_memory=True
    )
    
    corpus_vectors_list = []
    corpus_ids_list = []
    
    with torch.no_grad():
        for batch_ids, pixel_values in tqdm(corpus_loader, desc="Extracting features for the large-scale corpus (BEiT-3)"):
            pixel_values = pixel_values.to(device)
            batch_vectors, _ = beit3_model(image=pixel_values, only_infer=True)
            batch_vectors = F.normalize(batch_vectors, dim=-1)
            
            corpus_vectors_list.append(batch_vectors.cpu())
            corpus_ids_list.append(batch_ids.cpu())
            
    corpus_vectors = torch.cat(corpus_vectors_list)
    corpus_ids = torch.cat(corpus_ids_list)
    
    # Sort by ID (from eval4.py)
    arg_ids = torch.argsort(corpus_ids)
    corpus_vectors = corpus_vectors[arg_ids]
    corpus_ids = corpus_ids[arg_ids]   
    corpus_data = (corpus_ids, corpus_vectors)
       
    return (corpus_data[0].to(device), corpus_data[1].to(device))

def _create_or_load_generated_cache_beit3(model: nn.Module, preprocessor: callable, batch_size: int,
                                          queries_path: str, gen_image_dir: str, 
                                          num_eval_rounds: int, device: torch.device):

    with open(queries_path, 'r', encoding='utf-8') as f:
        queries = json.load(f)

    query_dataset = QueryImageDataset(
        queries=queries, 
        gen_image_dir=gen_image_dir, 
        num_rounds=num_eval_rounds, 
        transform=preprocessor
    )

    query_dataset_loader = DataLoader(
        dataset=query_dataset, 
        batch_size=batch_size,
        # num_workers=multiprocessing.cpu_count(),
        num_workers=4,
        pin_memory=True
    )
    cache_data = {}
    model.eval()


    with torch.no_grad():
        for filenames, images in tqdm(query_dataset_loader, desc="Caching generated image features"):
            images = images.to(device, non_blocking=True)
            
            # Extract features in batches
            gen_feats, _ = model(image=images, only_infer=True)
            gen_feats = F.normalize(gen_feats, dim=-1)

            gen_feats = gen_feats.detach().cpu()

            # Store features in a dictionary for later computation
            for i, filename in enumerate(filenames):
                cache_data[filename] = {
                    "gen_feat": gen_feats[i]
                }

    return cache_data

def _calculate_fused_scores_beit3(method: str, text_features: torch.Tensor, gen_features: torch.Tensor, 
                                  corpus_features: torch.Tensor, dialog_length: int,
                                  fusion_strategy: str, combiner_model: nn.Module):

    corpus_features_T = corpus_features.T

    if method == 'text':
        return text_features @ corpus_features_T

    if method == 'image':
        return gen_features @ corpus_features_T

    if method == 'dar':
        text_scores = text_features @ corpus_features_T
        gen_scores = gen_features @ corpus_features_T

        # DAR weights (from eval4.py)
        if dialog_length < 2:
            w_text, w_img = 0.8, 0.2
        else:
            w_text, w_img = 0.5, 0.5
        return w_text * text_scores + w_img * gen_scores
    if method == 'fused_feature':
        fused_features = None
        if fusion_strategy == 'combiner':
            with torch.no_grad(): # Ensure no gradients are tracked during evaluation
                fused_features = combiner_model.combine_features(gen_features, text_features)
        elif fusion_strategy == 'add':
            fused_features = F.normalize(gen_features + text_features, dim=-1)
        return fused_features @ corpus_features_T
    
    raise ValueError(f"Unknown fusion method: {method}")

def _calculate_ranks(ranked_indices, target_ids):
    ranks = []
    for i in range(ranked_indices.shape[0]):
        rank_tensor = (ranked_indices[i] == target_ids[i]).nonzero(as_tuple=True)[0]
        if rank_tensor.numel() > 0:
            ranks.append(rank_tensor.squeeze())
        else:
            ranks.append(torch.tensor(float('inf'), device=ranked_indices.device))
    return torch.stack(ranks)

def get_first_hitting_time(target_recall, num_rounds, hitting_recall=10):

    if len(target_recall) == 0:
        return torch.tensor([])
    target_recalls = target_recall.view(num_rounds, -1).T
    hits = (target_recalls < hitting_recall)
    final_hits = torch.inf * torch.ones(target_recalls.shape[0])
    hitting_times = []
    for ro_i in range(num_rounds):
        rh = hits[:, ro_i]
        final_hits[rh] = torch.min(final_hits[rh], torch.ones(final_hits[rh].shape) * ro_i)
        hitting_times.append(final_hits.clone())
    return torch.stack(hitting_times)

def cumulative_hits_per_round(target_recall, num_rounds, hitting_recall=10):


    if len(target_recall) == 0:
        return [0.0] * num_rounds
    ht_times = get_first_hitting_time(target_recall, num_rounds, hitting_recall)
    if ht_times.numel() == 0:
        return [0.0] * num_rounds
    return ((ht_times < torch.inf).sum(dim=-1) * 100 / ht_times.shape[1])

def run_eval4_validation(
    beit3_model: nn.Module, 
    combiner: nn.Module,
    tokenizer: XLMRobertaTokenizer, 
    preprocessor: callable, 
    hyper_params: dict, 
    # experiment: Experiment, 
    epoch: int, 
    device: torch.device,
    is_ema: bool = False
    ):

    print(f"\n--- Starting validation (EMA={is_ema}) (R@10 only) ---")

    beit3_model.eval()
    if combiner:
        combiner.eval()
    # 1. Prepare datasets (unchanged)
    val_queries_dataset = ValidationQueriesDataset(
        queries_path=hyper_params['val_queries_path'],
        generated_image_dir=hyper_params['val_generated_image_dir'],
        dialogue_format=hyper_params['dialogue_format']
    )
    corpus_val_dataset = CorpusDataset(
        json_file_path=hyper_params['val_corpus_json_path'], 
        pil_transform=preprocessor
    )
    path_to_id_map = corpus_val_dataset.path_to_id_map
    
    # 2. Extract features without caching (unchanged)
    corpus_ids, corpus_vectors = extract_corpus_features(
        corpus_val_dataset, beit3_model, hyper_params['batch_size'], device
    )
    num_eval_rounds = 11 
    gen_features_cache = _create_or_load_generated_cache_beit3(
        beit3_model, preprocessor, hyper_params['batch_size'],
        hyper_params['val_queries_path'], 
        hyper_params['val_generated_image_dir'],
        num_eval_rounds, 
        device
    )

    # 3. Define experiments (unchanged)
    experiments = {
        "BEiT3_Text_Only": "text", 
        "BEiT3_Image_Only": "image", 
        "BEiT3_DAR": "dar",
        "BEiT3_Fused_Feature": "fused_feature"
    }
    experiments_names = list(experiments.keys())
    all_rounds_recalls = {name: [] for name in experiments_names}

    feature_dim = beit3_model.language_head.out_features
    zero_feature = torch.zeros((feature_dim,), device="cpu")

    # 4. Evaluate round by round (unchanged)
    for dl in range(num_eval_rounds):
        val_queries_dataset.set_dialog_length(dl)
        val_loader = DataLoader(
            val_queries_dataset, batch_size=hyper_params['batch_size'], shuffle=False,
            # num_workers=multiprocessing.cpu_count(), 
            num_workers=4,pin_memory=True
        )
        exp_recalls_per_round = {name: [] for name in experiments_names}
        
        for batch in tqdm(val_loader, desc=f"Epoch {epoch} Val Round {dl}"):
            
            target_ids = [path_to_id_map.get(p, -1) for p in batch['target_path']]
            target_ids = torch.tensor(target_ids, dtype=torch.long).unsqueeze(1).to(device)
            text_inputs = tokenizer(text=list(batch['text']), padding="longest", truncation=True, max_length=256,return_tensors="pt").to(device)
            padding_mask = 1 - text_inputs['attention_mask']
            _, text_features = beit3_model(text_description=text_inputs['input_ids'], padding_mask=padding_mask, only_infer=True)
            gen_features_list = [gen_features_cache.get(os.path.basename(p), {}).get('gen_feat', zero_feature) for p in batch['gen_path']]
            gen_features = torch.stack(gen_features_list, dim=0).to(device, non_blocking=True)
            for name, method_type in experiments.items():
                total_scores = _calculate_fused_scores_beit3(method_type, text_features, gen_features, corpus_vectors, dl,
                                                            fusion_strategy=hyper_params['fusion_strategy'],combiner_model=combiner)
                arg_ranks = torch.argsort(total_scores, descending=True, dim=1).long()
                target_recall = _calculate_ranks(arg_ranks, target_ids)
                exp_recalls_per_round[name].append(target_recall)
        
        for name in experiments.keys():
            if exp_recalls_per_round[name]:
                all_rounds_recalls[name].append(torch.cat(exp_recalls_per_round[name]))
            else:
                all_rounds_recalls[name].append(torch.tensor([], dtype=torch.long))

    wandb_val_logs = {} # Create a dictionary to store all metrics

    epoch_results_for_excel = {}
    best_metric_for_checkpoint = 0.0 # Use DAR R@10 (Round 10)
    recall_k_for_excel = 10 # Only R@10 is used here
    
    model_prefix = "EMA" if is_ema else "Reg" # (Regular)


    model_prefix = "EMA" if is_ema else "Reg"

    for name, results_per_round in all_rounds_recalls.items():
        print(f"\n====== Experiment: '{name}' (R@10) ======")
        
        indep_r10_list = []
        
        # a. Independent per-round recall
        print(f"  --- Independent Per Round (R@10) ---")
        for dl, recalls in enumerate(results_per_round): # Iterate over 11 rounds
            total_queries = len(recalls)
            rate = 0.0
            if total_queries > 0:
                num_hits = (recalls < recall_k_for_excel).sum().item()
                rate = (num_hits * 100 / total_queries)
            
            indep_r10_list.append(rate)
            print(f"\tRound {dl}: {rate:.2f}%")
            
            # Fill validation logs with R@10 only
            metric_key = f"z_{model_prefix.lower()}_{name}_Indep_R{dl}_R@10"
            wandb_val_logs[metric_key] = rate
        
        sheet_name_indep = f"{model_prefix}_{name}_Indep"
        epoch_results_for_excel[sheet_name_indep] = indep_r10_list

        # b. Cumulative recall
        print(f"  --- Cumulative (R@10) ---")
        all_recalls_flat = torch.cat([r.cpu() for r in results_per_round if r.numel() > 0])
        
        cumul_r10_list = []
        if all_recalls_flat.numel() > 0:
            cumulative_results = cumulative_hits_per_round(
                all_recalls_flat, num_rounds=num_eval_rounds, hitting_recall=recall_k_for_excel
            ).tolist()
        else:
            cumulative_results = [0.0] * num_eval_rounds
        
        for dl, rate in enumerate(cumulative_results):
            cumul_r10_list.append(rate)
            print(f"\tUp to Round {dl}: {rate:.2f}%")
            
            # Fill validation logs with R@10 only
            metric_key = f"z_{model_prefix.lower()}_{name}_Cumul_R{dl}_R@10"
            wandb_val_logs[metric_key] = rate
            
        # Store cumulative results in the dictionary
        sheet_name_cumul = f"{model_prefix}_{name}_Cumul"
        epoch_results_for_excel[sheet_name_cumul] = cumul_r10_list

    best_metric_for_checkpoint = 0.0 
    fusion_strategy = hyper_params['fusion_strategy']
    fused_metric_name = f"{model_prefix}_BEiT3_Fused_Feature_Indep"
    dar_metric_name = f"{model_prefix}_BEiT3_DAR_Indep"

    if (fusion_strategy == 'combiner' or fusion_strategy == 'add') and fused_metric_name in epoch_results_for_excel:
        indep_r10_list = epoch_results_for_excel[fused_metric_name]
        best_metric_for_checkpoint = indep_r10_list[-1] # R@10 at the final round (Round 10)

    
    elif dar_metric_name in epoch_results_for_excel:
        indep_r10_list = epoch_results_for_excel[dar_metric_name]
        best_metric_for_checkpoint = indep_r10_list[-1]
        print(f"\nUsing DAR R@10 (Round 10) as the best metric: {best_metric_for_checkpoint:.2f}%")
    else:
        print(f"\n[Warning] Could not find '{fused_metric_name}' or '{dar_metric_name}' metric for selecting the best model.")

    return best_metric_for_checkpoint, epoch_results_for_excel, wandb_val_logs

def beit3_collate_fn(batch: list, tokenizer: XLMRobertaTokenizer):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return None, None, None

    ref_pixel_values_1, target_pixel_values, captions = zip(*batch)
    
    ref_pixel_values_1 = torch.stack(ref_pixel_values_1)
    target_pixel_values = torch.stack(target_pixel_values)

    # Text processing with the tokenizer
    text_inputs = tokenizer(
            text=list(captions), 
            padding="longest",  # Pad to the longest sequence in the batch
            truncation=True,       # Enable truncation
            max_length=256,        # Set a reasonable maximum sequence length
            return_tensors="pt"
        )
    
    return ref_pixel_values_1, target_pixel_values, text_inputs

def train_beit3_finetune(
    # experiment: Experiment,
    **hyper_params):

    train_json_path = hyper_params['train_json_path']
    # val_json_path = hyper_params['val_json_path']
    projection_dim = hyper_params['projection_dim']
    hidden_dim = hyper_params['hidden_dim']
    num_epochs = hyper_params['num_epochs']
    combiner_lr = hyper_params['combiner_lr']
    batch_size = hyper_params['batch_size']
    validation_frequency = hyper_params['validation_frequency']
    save_training = hyper_params['save_training']
    # save_best = hyper_params['save_best']
    resume_from = hyper_params.get('resume_from')
    drop_path_rate = hyper_params.get('drop_path', 0.1)

    training_mode = hyper_params['training_mode']
    fusion_strategy = hyper_params['fusion_strategy']
    loss_components = hyper_params['loss_components']
    enable_ref_ref = "ref_ref" in loss_components
    enable_text_text = "text_text" in loss_components

    print(f"--- Current training mode: {training_mode} ---")
    print(f"--- Query fusion strategy: {fusion_strategy} ---")
    if training_mode == 'beit3_only' and fusion_strategy == 'combiner' and not hyper_params.get('combiner_checkpoint_path'):
        raise ValueError("A pretrained Combiner checkpoint must be provided through --combiner-checkpoint-path when using fusion_strategy='combiner' in training_mode='beit3_only'.")
    if training_mode == 'end_to_end' and fusion_strategy == 'add':
        raise ValueError("training_mode='end_to_end' can only be used with fusion_strategy='combiner', not 'add'.")
    if training_mode == 'combiner_only' and fusion_strategy != 'combiner':
        raise ValueError("training_mode='combiner_only' requires fusion_strategy='combiner'.")

    # args = _get_large_config(img_size=384, vocab_size=64010, drop_path_rate=drop_path_rate)
    if training_mode == 'end_to_end':
        model_name_prefix = 'e2e'
    elif training_mode == 'combiner_only':
        model_name_prefix = 'combiner'
    else:  # 'beit3_only' or other default cases
        model_name_prefix = 'beit3'
    args = _get_base_config(
    img_size=hyper_params['input_size'], 
    vocab_size=64010, 
    drop_path_rate=drop_path_rate,
    checkpoint_activations=True  
    )
    beit3_model = modeling_finetune.BEiT3ForRetrieval(args=args)

    beit3_model.to(device)

    training_path = Path(hyper_params['log_base_dir']) / hyper_params['experiment_name']
    training_path.mkdir(exist_ok=True, parents=True)

    hnm_stats_excel_path = training_path / 'hnm_detailed_statistics.xlsx'

    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(hyper_params, file, sort_keys=True, indent=4)
    train_log_path = training_path / 'train_metrics.csv'

    if resume_from and train_log_path.exists():
        print(f"Loading existing training log: {train_log_path}")
        training_log_frame = pd.read_csv(train_log_path)
    else:
        training_log_frame = pd.DataFrame()

    val_excel_path = training_path / 'validation_summary.xlsx'
    validation_dataframes = {}
    if resume_from and val_excel_path.exists():
        validation_dataframes = pd.read_excel(str(val_excel_path), sheet_name=None, index_col=0)

    checkpoint_path = hyper_params['beit3_checkpoint_path']
    utils.load_model_and_may_interpolate(
        ckpt_path=checkpoint_path, 
        model=beit3_model, 
        model_key='model', 
        model_prefix=''
    )  
    # with torch.no_grad():
    #     # log(1/0.07) ≈ 2.659  log(1/0.02) ≈ 3.912
    #     target_scale = np.log(1 / 0.02)
    #     beit3_model.logit_scale.fill_(target_scale)

    beit3_model.to(device).train()

    tokenizer_path = hyper_params['beit3_tokenizer_path']
    tokenizer = XLMRobertaTokenizer(tokenizer_path)

    feature_dim = beit3_model.language_head.out_features

    transform_train = build_beit3_transform(is_train=False, config=hyper_params)
    transform_val = build_beit3_transform(is_train=False, config=hyper_params)
    # 2. --- Data loading and feature preparation ---
    train_dataset = ComposedRetrievalDataset(json_file_path=train_json_path, 
        pil_transform=transform_train,dialogue_format=hyper_params['dialogue_format'], 
        dialogue_round=hyper_params['dialogue_round'],use_random_rounds=hyper_params['use_random_rounds'],
        use_caption_masking=hyper_params['use_caption_masking'],caption_masking_prob=hyper_params['caption_masking_prob'],
        max_samples=hyper_params.get('max_train_samples', 0),
        )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        # num_workers=multiprocessing.cpu_count(),
        num_workers=4,
        pin_memory=True,
        # collate_fn=partial(beit3_collate_fn, tokenizer=tokenizer, preprocessor=preprocessor),
        collate_fn=partial(beit3_collate_fn, tokenizer=tokenizer),
        persistent_workers=True,
        drop_last=True,
        shuffle=True
    )

    combiner = None
    if training_mode == 'end_to_end' or fusion_strategy == 'combiner':
        print("Initializing the Combiner model...")
        feature_dim = beit3_model.language_head.out_features
        combiner = Combiner(feature_dim, hyper_params['projection_dim'], hyper_params['hidden_dim']).to(device)
        
        combiner_ckpt_path = hyper_params.get('combiner_checkpoint_path')
        if combiner_ckpt_path and Path(combiner_ckpt_path).exists():
            print(f"Loading Combiner weights from {combiner_ckpt_path} ...")
            state_dict = torch.load(combiner_ckpt_path, map_location=device)
            # Support two possible checkpoint formats
            if 'combiner_state_dict' in state_dict:
                combiner.load_state_dict(state_dict['combiner_state_dict'])
            elif 'model_state_dict' in state_dict:
                 combiner.load_state_dict(state_dict['model_state_dict'])
            else:
                 combiner.load_state_dict(state_dict)

    if training_mode == 'beit3_only' and combiner:
        print("Mode: beit3_only. Freezing the Combiner.")
        combiner.eval()
        for param in combiner.parameters():
            param.requires_grad = False

    elif training_mode == 'end_to_end' and combiner:
        print("Mode: end_to_end. Setting the Combiner as trainable.")
        combiner.train()
        for param in combiner.parameters():
            param.requires_grad = True

    elif training_mode == 'combiner_only' and combiner:
        print("Mode: combiner_only. Freezing BEiT-3 and training only the Combiner.")

        beit3_model.eval()
        for param in beit3_model.parameters():
            param.requires_grad = False

        combiner.train()
        for param in combiner.parameters():
            param.requires_grad = True
        
    model_ema = None
    if hyper_params.get('model_ema', False) and training_mode != 'combiner_only':
        model_ema = ModelEma(
            beit3_model,
            decay=hyper_params['model_ema_decay'],
            device='cpu' if False else '',
            resume=''
        )
    combiner_ema = None
    # Initialize this only when the combiner exists and EMA is enabled
    if combiner and hyper_params.get('model_ema', False):
        print("Enabling EMA for the Combiner...")
        combiner_ema = ModelEma(
            combiner,
            decay=hyper_params['model_ema_decay'],
            device='cpu' if False else '',
            resume=''
        )

    params_to_optimize = []

    if training_mode in ['beit3_only', 'end_to_end']:
        beit3_lr = hyper_params['beit3_lr']
        layer_decay = hyper_params['layer_decay']
        num_layers = beit3_model.get_num_layers()

        if layer_decay < 1.0:
            lr_scales = list(layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2))
            layer_decay_assigner = LayerDecayValueAssigner(lr_scales)
        else:
            layer_decay_assigner = None

        beit3_param_groups = get_parameter_groups(
            beit3_model,
            hyper_params['weight_decay'],
            beit3_model.no_weight_decay(),
            get_num_layer=layer_decay_assigner.get_layer_id if layer_decay_assigner is not None else None,
            get_layer_scale=layer_decay_assigner.get_scale if layer_decay_assigner is not None else None
        )

        for group in beit3_param_groups:
            group['lr'] = beit3_lr * group.get('lr_scale', 1.0)

        params_to_optimize.extend(beit3_param_groups)

    if training_mode in ['end_to_end', 'combiner_only']:
        combiner_lr = hyper_params['combiner_lr']
        print(f"Adding Combiner parameters to the optimizer with learning rate: {combiner_lr}")

        combiner_param_groups = get_parameter_groups(combiner, hyper_params['weight_decay'])
        for group in combiner_param_groups:
            group['lr'] = combiner_lr

        params_to_optimize.extend(combiner_param_groups)


    optimizer = optim.AdamW(params_to_optimize, eps=1e-8, betas=(0.9, 0.999))
    warmup_epochs = hyper_params.get('warmup_epochs', 2) # Default to 2 epochs
    num_training_steps = (len(train_loader) * num_epochs) // hyper_params['update_freq']
    num_warmup_steps = (len(train_loader) * hyper_params.get('warmup_epochs', 2)) // hyper_params['update_freq']

    scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
    )

    criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler()

    start_epoch = 0
    best_recall_at_10 = 0.0

    if resume_from and Path(resume_from).exists():
        print(f"Resuming training from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location='cpu')

        if 'model' in checkpoint:
            beit3_model.load_state_dict(checkpoint['model'])
        elif 'model_state_dict' in checkpoint:
            beit3_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            if training_mode == 'combiner_only':
                print("Resuming in combiner_only mode: no backbone weights were found in the checkpoint; continue using the frozen BEiT-3 loaded from beit3_checkpoint_path.")
            else:
                raise KeyError("Checkpoint does not contain 'model' or 'model_state_dict'!")

        if combiner and 'combiner_state_dict' in checkpoint:
            combiner.load_state_dict(checkpoint['combiner_state_dict'])

        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if combiner_ema is not None and 'combiner_ema_state_dict' in checkpoint:
            combiner_ema.ema.load_state_dict(checkpoint['combiner_ema_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        best_recall_at_10 = checkpoint.get('best_recall_at_10', 0.0)
        print(f"Resume succeeded. Training will start from Epoch {start_epoch}. Previous best Recall@10 is: {best_recall_at_10:.2f}%")

    # do_sanity_check = True 
    
    # if do_sanity_check:
    #     print("\n--- [Sanity Check] Starting pre-training model validation (Epoch=-1) ---")
    #     # 1. Temporarily switch to evaluation mode
    #     beit3_model.eval()
    #     if combiner: combiner.eval()
        
    #     # 2. Prepare a dictionary for storing results
    #     sanity_metrics_all = {}

    #     # 3. Run validation for the regular model
    #     with torch.no_grad():
    #         _, sanity_reg_metrics = run_eval4_validation(
    #             beit3_model=beit3_model, combiner=combiner, tokenizer=tokenizer,
    #             preprocessor=transform_val, hyper_params=hyper_params, 
    #             experiment=experiment, epoch=-1, device=device, is_ema=False
    #         )
    #         sanity_metrics_all.update(sanity_reg_metrics)

    #         # 4. Optionally run validation for the EMA model
    #         if model_ema is not None:
    #             print("--- [Sanity Check] Validating the EMA model ---")
    #             model_ema.ema.eval()
    #             sanity_combiner_ema = combiner_ema.ema if combiner_ema else None
    #             _, sanity_ema_metrics = run_eval4_validation(
    #                 beit3_model=model_ema.ema, combiner=sanity_combiner_ema, tokenizer=tokenizer,
    #                 preprocessor=transform_val, hyper_params=hyper_params, 
    #                 experiment=experiment, epoch=-1, device=device, is_ema=True
    #             )
    #             sanity_metrics_all.update(sanity_ema_metrics)

    #     # 5. Save to Excel with merge logic
    #     sanity_cols = [f'Round {i}' for i in range(11)]
    #     for s_name, s_data in sanity_metrics_all.items():
    #         s_df = pd.DataFrame([s_data], columns=sanity_cols, index=[-1]) # Set index to -1
    #         s_df.index.name = "Epoch"
            
    #         if s_name in validation_dataframes:
    #             # If the sheet already exists, merge and sort by index (so -1 is placed consistently)
    #             validation_dataframes[s_name] = pd.concat([validation_dataframes[s_name], s_df])
    #             # Deduplicate to avoid index conflicts from repeated runs
    #             validation_dataframes[s_name] = validation_dataframes[s_name][~validation_dataframes[s_name].index.duplicated(keep='last')].sort_index()
    #         else:
    #             validation_dataframes[s_name] = s_df

    #     # 6. Write to file
    #     with pd.ExcelWriter(str(val_excel_path), engine='openpyxl') as writer:
    #         for s_name, df in validation_dataframes.items():
    #             df.to_excel(writer, sheet_name=s_name)
        
    #     print(f"--- [Sanity Check] Done. Results saved to {val_excel_path} (Epoch -1) ---\n")

    print('Training loop started')
    for epoch in range(start_epoch, num_epochs):
        beit3_model.train()
        if training_mode == 'combiner_only':
            beit3_model.eval()
            combiner.train()
        elif training_mode == 'end_to_end':
            beit3_model.train()
            combiner.train()
        else:
            beit3_model.train()
            if combiner:
                combiner.eval()

        epoch_batch_details = []

        train_running_results = {'images_in_epoch': 0, 'accumulated_train_loss': 0.0}
        for loss_name in hyper_params['loss_components']:
            train_running_results[f'accumulated_loss_{loss_name}'] = 0.0
        train_bar = tqdm(train_loader, ncols=150)
        
        for i, (ref_pixel_values, target_pixel_values, text_inputs) in enumerate(train_bar):
            # if ref_pixel_values is None: continue # Skip empty batches

            current_batch_stats = {'batch_idx': i}

            images_in_batch = ref_pixel_values.size(0)
            step = len(train_loader) * epoch + i
            current_step_logs = {
                "train/step": step,
                "train/epoch": epoch + i / len(train_loader),
                "train/learning_rate": optimizer.param_groups[0]['lr'],
                "train/logit_scale": beit3_model.logit_scale.exp().item() # [Added] Log the logit scale
            }
            ref_pixel_values = ref_pixel_values.to(device)
            target_pixel_values = target_pixel_values.to(device)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            
            with torch.cuda.amp.autocast():
                if training_mode == 'combiner_only':
                    with torch.no_grad():
                        reference_features, _ = beit3_model(image=ref_pixel_values, only_infer=True)
                        target_features, _ = beit3_model(image=target_pixel_values, only_infer=True)
                        padding_mask = 1 - text_inputs['attention_mask']
                        _, text_features = beit3_model(
                            text_description=text_inputs['input_ids'],
                            padding_mask=padding_mask,
                            only_infer=True
                        )
                else:
                    reference_features, _ = beit3_model(image=ref_pixel_values, only_infer=True)
                    target_features, _ = beit3_model(image=target_pixel_values, only_infer=True)
                    padding_mask = 1 - text_inputs['attention_mask']
                    _, text_features = beit3_model(
                        text_description=text_inputs['input_ids'],
                        padding_mask=padding_mask,
                        only_infer=True
                    )

                # loss_type = hyper_params['loss_type']
                logit_scale_exp = beit3_model.logit_scale.exp()
                ground_truth = torch.arange(images_in_batch, dtype=torch.long, device=device)
                loss_components_to_use = set(hyper_params['loss_components'])
                loss_weights = dict(zip(hyper_params['loss_components'], hyper_params['loss_weights']))
                all_losses = {}

                reference_features_norm = None
                if 'ref_tgt' in loss_components_to_use or 'ref_text' in loss_components_to_use :
                    reference_features_norm = F.normalize(reference_features, dim=-1, eps=1e-6)

                if 'ref_tgt' in loss_components_to_use or 'text_tgt' in loss_components_to_use or 'fused_tgt' in loss_components_to_use:
                    target_features_norm = F.normalize(target_features, dim=-1, eps=1e-6)
                    
                text_features_norm = None
                if 'text_tgt' in loss_components_to_use or 'ref_text' in loss_components_to_use:
                    text_features_norm = F.normalize(text_features, dim=-1, eps=1e-6)
                
                fused_features_norm = None
                if 'fused_tgt' in loss_components_to_use:
                    # Correctly handle both 'add' and 'combiner' strategies
                    if hyper_params['fusion_strategy'] == 'combiner':
                        predicted_features = combiner.combine_features(reference_features, text_features)
                    else: # 'add'
                        predicted_features = reference_features + text_features
                    fused_features_norm = F.normalize(predicted_features, dim=-1, eps=1e-6)

                use_hnm = hyper_params['use_hnm']
                hnm_weight = hyper_params['hnm_weight']
                hnm_topk = hyper_params['hnm_topk']
                hnm_margin = hyper_params['hnm_margin']
                hnm_temp = hyper_params['hnm_temp']

                pure_losses = {}

                if 'ref_tgt' in loss_components_to_use:
                # 1. First compute pure cosine similarity in [-1, 1]
                    sim_ref_to_tgt = reference_features_norm @ target_features_norm.T
                    sim_tgt_to_ref = sim_ref_to_tgt.T  # Transpose directly; no need to recompute

                    # 2. Apply logit scaling for the InfoNCE loss (used by CrossEntropy)
                    logits_ref_to_tgt = logit_scale_exp * sim_ref_to_tgt
                    logits_tgt_to_ref = logit_scale_exp * sim_tgt_to_ref
                    
                    loss_ref_tgt = criterion(logits_ref_to_tgt, ground_truth)
                    loss_tgt_ref = criterion(logits_tgt_to_ref, ground_truth)
                    loss_ref_tgt_total = (loss_ref_tgt + loss_tgt_ref) / 2

                    pure_losses['ref_tgt'] = loss_ref_tgt_total.detach().item()

                    if use_hnm:
                        # Compute HNM in both directions
                        hnm_ref, pos_ref, neg_ref = compute_hnm_loss(sim_ref_to_tgt, ground_truth, hnm_topk, hnm_margin, hnm_temp)
                        hnm_tgt, pos_tgt, neg_tgt = compute_hnm_loss(sim_tgt_to_ref, ground_truth, hnm_topk, hnm_margin, hnm_temp)
                        hnm_loss_val = (hnm_ref + hnm_tgt) / 2
                        
                        # Log values for display only
                        key_log = 'accumulated_loss_hnm_ref'
                        if key_log not in train_running_results: train_running_results[key_log] = 0.0
                        train_running_results[key_log] += hnm_loss_val.detach().item() * images_in_batch
                        
                        # Add to the total loss with weighting: L = L_nce + lambda * L_hnm
                        loss_ref_tgt_total = loss_ref_tgt_total + hnm_weight * hnm_loss_val

                        avg_pos = (pos_ref + pos_tgt) / 2
                        avg_neg = (neg_ref + neg_tgt) / 2
                        diff_val = avg_pos - avg_neg
                        
                        # Log detailed HNM statistics
                        current_step_logs.update({
                            "hnm_stats/ref_pos": avg_pos,
                            "hnm_stats/ref_neg": avg_neg,
                            "hnm_stats/ref_diff": diff_val,
                            "train/loss_hnm_ref": hnm_loss_val.detach().item()
                        })
                        
                        # Keep current_batch_stats for Excel statistics (local backup)
                        current_batch_stats['ref_pos'] = avg_pos
                        current_batch_stats['ref_neg'] = avg_neg
                        current_batch_stats['ref_diff'] = diff_val

                    all_losses['ref_tgt'] = loss_ref_tgt_total

                # 'text_tgt': (Text_Query) <-> (Target_Image)
                if 'text_tgt' in loss_components_to_use:
                    sim_text_to_tgt = text_features_norm @ target_features_norm.T
                    sim_tgt_to_text = sim_text_to_tgt.T

                        # 2. Logit Scale for InfoNCE
                    logits_text_to_tgt = logit_scale_exp * sim_text_to_tgt
                    logits_tgt_to_text = logit_scale_exp * sim_tgt_to_text

                    loss_text_tgt = criterion(logits_text_to_tgt, ground_truth)
                    loss_tgt_text = criterion(logits_tgt_to_text, ground_truth)
                    loss_text_tgt_total = (loss_text_tgt + loss_tgt_text) / 2

                    pure_losses['text_tgt'] = loss_text_tgt_total.detach().item()

                    if use_hnm:
                            # 3. Key change: pass sim_text_to_tgt
                        hnm_text, pos_text, neg_text = compute_hnm_loss(
                                sim_text_to_tgt, ground_truth, hnm_topk, hnm_margin, hnm_temp
                            )
                        hnm_tgt_text, pos_tgt_text, neg_tgt_text = compute_hnm_loss(
                                sim_tgt_to_text, ground_truth, hnm_topk, hnm_margin, hnm_temp
                            )
                        hnm_loss_val_text = (hnm_text + hnm_tgt_text) / 2
                        
                        key_log = 'accumulated_loss_hnm_text'
                        if key_log not in train_running_results: train_running_results[key_log] = 0.0
                        train_running_results[key_log] += hnm_loss_val_text.detach().item() * images_in_batch

                        loss_text_tgt_total = loss_text_tgt_total + hnm_weight * hnm_loss_val_text


                        avg_pos_text = (pos_text + pos_tgt_text) / 2
                        avg_neg_text = (neg_text + neg_tgt_text) / 2
                        diff_val_text = avg_pos_text - avg_neg_text
                        
                        # Log detailed HNM statistics
                        current_step_logs.update({
                            "hnm_stats/text_pos": avg_pos_text,
                            "hnm_stats/text_neg": avg_neg_text,
                            "hnm_stats/text_diff": diff_val_text,
                            "train/loss_hnm_text": hnm_loss_val_text.detach().item()
                        })
                        
                        # Keep current_batch_stats for Excel statistics (local backup)
                        current_batch_stats['text_pos'] = avg_pos_text
                        current_batch_stats['text_neg'] = avg_neg_text
                        current_batch_stats['text_diff'] = diff_val_text
                        
                    all_losses['text_tgt'] = loss_text_tgt_total

                # 'fused_tgt_sym': (Fused_Query) <-> (Target_Image) [symmetric version]
                if 'fused_tgt' in loss_components_to_use:
                    sim_fused_to_tgt = fused_features_norm @ target_features_norm.T
                    sim_tgt_to_fused = sim_fused_to_tgt.T

                    # 2. Logit Scale for InfoNCE
                    logits_fused_to_tgt = logit_scale_exp * sim_fused_to_tgt
                    logits_tgt_to_fused = logit_scale_exp * sim_tgt_to_fused

                    loss_fused_to_tgt = criterion(logits_fused_to_tgt, ground_truth)
                    loss_tgt_to_fused = criterion(logits_tgt_to_fused, ground_truth)
                    loss_fused_tgt_total = (loss_fused_to_tgt + loss_tgt_to_fused) / 2


                    pure_losses['fused_tgt'] = loss_fused_tgt_total.detach().item()
                    
                    if use_hnm:
                        hnm_fused, pos_fused, neg_fused = compute_hnm_loss(sim_fused_to_tgt, ground_truth, hnm_topk, hnm_margin, hnm_temp)
                        hnm_tgt_fused, pos_tgt_fused, neg_tgt_fused = compute_hnm_loss(sim_tgt_to_fused, ground_truth, hnm_topk, hnm_margin, hnm_temp)
                        hnm_loss_val_fused = (hnm_fused + hnm_tgt_fused) / 2
                        
                        key_log = 'accumulated_loss_hnm_fused'
                        if key_log not in train_running_results: train_running_results[key_log] = 0.0
                        train_running_results[key_log] += hnm_loss_val_fused.detach().item() * images_in_batch

                        loss_fused_tgt_total = loss_fused_tgt_total + hnm_weight * hnm_loss_val_fused

                        avg_pos_fused = (pos_fused + pos_tgt_fused) / 2
                        avg_neg_fused = (neg_fused + neg_tgt_fused) / 2
                        diff_val_fused = avg_pos_fused - avg_neg_fused
                        
                        # Log detailed HNM statistics
                        current_step_logs.update({
                            "hnm_stats/fused_pos": avg_pos_fused,
                            "hnm_stats/fused_neg": avg_neg_fused,
                            "hnm_stats/fused_diff": diff_val_fused,
                            "train/loss_hnm_fused": hnm_loss_val_fused.detach().item()
                        })

                        # Local statistics key:
                        current_batch_stats['fused_pos'] = avg_pos_fused
                        current_batch_stats['fused_neg'] = avg_neg_fused
                        current_batch_stats['fused_diff'] = diff_val_fused

                    all_losses['fused_tgt'] = loss_fused_tgt_total
                    
                # 'ref_text': (Ref_Image) <-> (Text_Query)
                if 'ref_text' in loss_components_to_use:
                    logits_text_to_ref = logit_scale_exp * text_features_norm @ reference_features_norm.T
                    logits_ref_to_text = logit_scale_exp * reference_features_norm @ text_features_norm.T
                    loss_text_ref = criterion(logits_text_to_ref, ground_truth)
                    loss_ref_text = criterion(logits_ref_to_text, ground_truth)
                    all_losses['ref_text'] = (loss_text_ref + loss_ref_text) / 2
                    
                if not all_losses:
                    raise ValueError("Configuration error: 'loss_components' is empty or contains unsupported components.")
                
                total_loss_weighted = 0.0
                total_weight = 0.0
                for name, loss_value in all_losses.items():
                    weight = loss_weights.get(name, 1.0) # Get the weight; default is 1
                    total_loss_weighted += loss_value * weight
                    total_weight += weight
                loss = total_loss_weighted / total_weight
                unscaled_loss = loss # Record the unscaled loss for logging

                current_step_logs["train/loss_total"] = unscaled_loss.detach().item()
                    
                for name, loss_value in all_losses.items():
                    # Build a key such as 'accumulated_loss_ref_tgt'
                    loss_scalar = loss_value.detach().item()
                    current_step_logs[f"train/loss_component_{name}_total"] = loss_scalar

                    # current_step_logs[f"train/loss_component_{name}"] = loss_scalar
                    key = f'accumulated_loss_{name}'
                    if key in train_running_results:
                        train_running_results[key] += loss_scalar * images_in_batch

                for name, val in pure_losses.items():
                    # Log the original unweighted InfoNCE loss
                    current_step_logs[f"train/loss_component_{name}_pure"] = val
                    
                wandb.log(current_step_logs)

            loss = loss / hyper_params['update_freq']

            scaler.scale(loss).backward()
            if (i + 1) % hyper_params['update_freq'] == 0:
                max_norm = hyper_params['clip_grad']

                if max_norm > 0:
                    scaler.unscale_(optimizer)

                    if training_mode in ['beit3_only', 'end_to_end']:
                        torch.nn.utils.clip_grad_norm_(beit3_model.parameters(), max_norm)

                    if training_mode in ['end_to_end', 'combiner_only'] and combiner is not None:
                        torch.nn.utils.clip_grad_norm_(combiner.parameters(), max_norm)

                scaler.step(optimizer)
                scaler.update()

                if model_ema is not None and training_mode != 'combiner_only':
                    model_ema.update(beit3_model)

                if combiner_ema is not None and training_mode in ['end_to_end', 'combiner_only']:
                    combiner_ema.update(combiner)

                optimizer.zero_grad()
                scheduler.step()

            # update_train_running_results(train_running_results, float(unscaled_loss.detach().item()), images_in_batch)
            # Detach from the computation graph and convert to a scalar
            update_train_running_results(train_running_results, unscaled_loss.detach(), images_in_batch)
            set_train_bar_description(train_bar, epoch, num_epochs, train_running_results)

            if use_hnm:
                epoch_batch_details.append(current_batch_stats)
        
        epoch_metrics_data = {'epoch': epoch}
        total_images = train_running_results['images_in_epoch']

        train_epoch_loss = float(train_running_results['accumulated_train_loss'] / total_images)
        epoch_metrics_data['train_epoch_loss'] = train_epoch_loss
        wandb.log({"train/epoch_avg_loss": train_epoch_loss, "epoch": epoch})

        for loss_name in hyper_params['loss_components']:
            key = f'accumulated_loss_{loss_name}'
            if key in train_running_results and total_images > 0:
                epoch_loss_component = train_running_results[key] / total_images
                epoch_metrics_data[f'loss_{loss_name}'] = epoch_loss_component
                
        hnm_suffixes = ['ref', 'text', 'fused'] 
        for suffix in hnm_suffixes:
            key = f'accumulated_loss_hnm_{suffix}' # Corresponds to key_log in the training loop
            if key in train_running_results and total_images > 0:
                epoch_loss_component = train_running_results[key] / total_images
                # Mark HNM explicitly in the CSV column name
                epoch_metrics_data[f'loss_hnm_{suffix}'] = epoch_loss_component

        training_log_frame = pd.concat([
            training_log_frame,
            pd.DataFrame(data=epoch_metrics_data, index=[0])
        ])

        training_log_frame = training_log_frame.drop_duplicates(subset=['epoch'], keep='last')
        training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

        if use_hnm and len(epoch_batch_details) > 0:
            df_epoch_hnm = pd.DataFrame(epoch_batch_details)
            sheet_name = f'Epoch_{epoch}'
            
            mode = 'a' if hnm_stats_excel_path.exists() else 'w'
            # if_sheet_exists='replace' is valid only when mode='a'
            if_sheet_exists = 'replace' if mode == 'a' else None
            
            try:
                with pd.ExcelWriter(hnm_stats_excel_path, engine='openpyxl', mode=mode, if_sheet_exists=if_sheet_exists) as writer:
                    df_epoch_hnm.to_excel(writer, sheet_name=sheet_name, index=False)
            except Exception as e:
                print(f"[Warning] Failed to save detailed HNM statistics to Excel: {e}")

        if epoch % validation_frequency == 0:
            beit3_model.eval()
            all_epoch_metrics = {}
            
            with torch.no_grad():
                current_r10, regular_metrics_log, wandb_reg_logs = run_eval4_validation(
                    beit3_model=beit3_model,
                    combiner=combiner,
                    tokenizer=tokenizer,
                    preprocessor=transform_val, 
                    hyper_params=hyper_params,
                    epoch=epoch,
                    device=device,
                    is_ema=False
                )

            wandb_reg_logs['epoch'] = epoch
            wandb.log(wandb_reg_logs)
        
            all_epoch_metrics.update(regular_metrics_log)
            print(f"Base model R@10 (Round 10, DAR): {current_r10:.2f}%")
            current_best_metric = current_r10

            if model_ema is not None:
                print("\n--- Evaluating the EMA model ---")
                model_ema.ema.eval() # Ensure the EMA model is in evaluation mode
                
                ema_eval_combiner = None
                if combiner_ema is not None:
                    combiner_ema.ema.eval()
                    ema_eval_combiner = combiner_ema.ema

                with torch.no_grad():
                    ema_r10, ema_metrics_log, wandb_ema_logs = run_eval4_validation(
                        beit3_model=model_ema.ema,
                        tokenizer=tokenizer,
                        combiner=ema_eval_combiner,
                        preprocessor=transform_val, 
                        hyper_params=hyper_params,
                        epoch=epoch,
                        device=device,
                        is_ema=True
                    )
                wandb_ema_logs['epoch'] = epoch
                wandb.log(wandb_ema_logs)
            
                all_epoch_metrics.update(ema_metrics_log)
                print(f"EMA model R@10 (Round 10, DAR): {ema_r10:.2f}%")
                current_best_metric = ema_r10 
            col_names = [f'Round {i}' for i in range(11)]
            for sheet_name, new_data_list in all_epoch_metrics.items():
                
                # 3. Convert the new row to a DataFrame indexed by the current epoch
                new_row_df = pd.DataFrame([new_data_list], columns=col_names, index=[epoch])
                new_row_df.index.name = "Epoch"
                
                # 4. Check whether validation_dataframes already has this sheet in memory
                if sheet_name in validation_dataframes:
                    # If it exists, append the new row
                    existing_df = validation_dataframes[sheet_name]
                    # If this epoch already exists, overwrite it
                    if epoch in existing_df.index:
                        existing_df.loc[epoch] = new_data_list
                    else:
                        validation_dataframes[sheet_name] = pd.concat([existing_df, new_row_df])
                else:
                    # If it does not exist, create a new sheet
                    validation_dataframes[sheet_name] = new_row_df
            with pd.ExcelWriter(str(val_excel_path), engine='openpyxl') as writer:
                    for sheet_name, df in validation_dataframes.items():
                        df.to_excel(writer, sheet_name=sheet_name)

            if save_training:
                if current_best_metric > best_recall_at_10:
                    best_recall_at_10 = current_best_metric

                    if training_mode == 'combiner_only':
                        save_combiner_only_checkpoint(
                            f'{model_name_prefix}_best',
                            epoch,
                            combiner,
                            optimizer,
                            scaler,
                            best_recall_at_10,
                            training_path,
                            scheduler,
                            hyper_params,
                            combiner_ema,
                        )
                        if hyper_params.get("export_merged_release_checkpoint", True):
                            export_merged_release_checkpoint(
                                f'{model_name_prefix}_best_{hyper_params.get("release_suffix", "release")}',
                                training_path,
                                beit3_model,
                                combiner,
                                hyper_params,
                            )
                    else:
                        save_checkpoint(
                            f'{model_name_prefix}_best',
                            epoch,
                            beit3_model,
                            optimizer,
                            scaler,
                            best_recall_at_10,
                            training_path,
                            scheduler,
                            combiner,
                            model_ema,
                            combiner_ema,
                        )

                if training_mode == 'combiner_only':
                    save_combiner_only_checkpoint(
                        f'{model_name_prefix}_{epoch}',
                        epoch,
                        combiner,
                        optimizer,
                        scaler,
                        best_recall_at_10,
                        training_path,
                        scheduler,
                        hyper_params,
                        combiner_ema,
                    )
                    if hyper_params.get("export_merged_release_checkpoint", True):
                        export_merged_release_checkpoint(
                            f'{model_name_prefix}_{epoch}_{hyper_params.get("release_suffix", "release")}',
                            training_path,
                            beit3_model,
                            combiner,
                            hyper_params,
                        )
                else:
                    save_checkpoint(
                        f'{model_name_prefix}_{epoch}',
                        epoch,
                        beit3_model,
                        optimizer,
                        scaler,
                        best_recall_at_10,
                        training_path,
                        scheduler,
                        combiner,
                        model_ema,
                        combiner_ema,
                    )

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default=Config.experiment_name, help="WandB experiment name and log folder name")
    parser.add_argument("--log_base_dir", type=str, default=Config.log_base_dir, help="Root directory for all experiment logs")
    
    # --- Data and model path arguments ---
    parser.add_argument("--dialogue_format", type=str, default=Config.dialogue_format, choices=['Summarized', 'VisDial'],
                        help="Dialogue format to use (Summarized or VisDial)")
    parser.add_argument("--dialogue_round", type=int, default=Config.dialogue_round,
                        help="Dialogue round to use (0-10)")
    parser.add_argument('--use_caption_masking', action='store_true', default=Config.use_caption_masking, 
                        help="Enable random caption masking")
    parser.add_argument("--caption_masking_prob", type=float, default=Config.caption_masking_prob, 
                        help="Probability of masking R0")
    parser.add_argument('--use_random_rounds', action='store_true',default=Config.use_random_rounds, help="Enable random dialogue rounds (R3 strategy)")
    parser.add_argument("--train_json_path", type=str, default=Config.train_json_path)
    parser.add_argument(
    "--max_train_samples",type=int,default=Config.max_train_samples,help="Maximum number of training samples; 0 means using all samples")
    parser.add_argument("--val_corpus_json-path", type=str, default=Config.val_corpus_json_path, help="Path to the large-scale validation corpus")
    parser.add_argument("--beit3_checkpoint_path", type=str, default=Config.beit3_checkpoint_path, help="Path to the pretrained BEiT-3 weights (.pth)")
    parser.add_argument("--beit3_tokenizer_path", type=str, default=Config.beit3_tokenizer_path, help="Path to the BEiT-3 tokenizer model (.spm)")
    parser.add_argument("--val_queries_path", type=str, default=Config.val_queries_path)
    parser.add_argument("--val_generated_image_dir", type=str, default=Config.val_generated_image_dir)

    # --- Combiner architecture arguments ---
    parser.add_argument("--projection_dim", default=Config.projection_dim, type=int, help='Combiner projection dimension')
    parser.add_argument("--hidden_dim", default=Config.hidden_dim, type=int, help="Combiner hidden dimension")
    parser.add_argument("--combiner_checkpoint_path", type=str, default=Config.combiner_checkpoint_path)
    
    # --- Training hyperparameters ---
    parser.add_argument("--training_mode", type=str, default=Config.training_mode, choices=['beit3_only', 'end_to_end','combiner_only'], help="Training mode: BEiT-3 only, end-to-end training, or Combiner only")
    parser.add_argument("--num_epochs", default=Config.num_epochs, type=int)
    parser.add_argument("--beit3_lr", default=Config.beit3_lr, type=float)
    parser.add_argument("--combiner_lr", default=Config.combiner_lr, type=float)
    parser.add_argument("--weight_decay", default=Config.weight_decay, type=float, help="Weight decay value (not applied to bias or LayerNorm)")
    parser.add_argument("--warmup_epochs", default=Config.warmup_epochs, type=int, help="Number of warmup epochs")
    parser.add_argument("--layer_decay", default=Config.layer_decay, type=float, help="Layer-wise learning-rate decay coefficient (enabled when < 1.0)")
    parser.add_argument("--drop_path", default=Config.drop_path, type=float, help="BEiT-3 DropPath rate")
    parser.add_argument("--batch_size", default=Config.batch_size, type=int)
    parser.add_argument("--update_freq", default=Config.update_freq, type=int)
    parser.add_argument("--clip_grad", default=Config.clip_grad, type=float)
    parser.add_argument("--model_ema", action='store_true', default=Config.model_ema)
    parser.add_argument("--model_ema-decay", type=float, default=Config.model_ema_decay)
    parser.add_argument("--validation_frequency", default=Config.validation_frequency, type=int)

    parser.add_argument("--resume_from", type=str, default=Config.resume_from, help="Resume training from the specified checkpoint")
    parser.add_argument("--save_training", action='store_true', default=Config.save_training)
    
    # --- Other configuration ---
    parser.add_argument("--input_size", type=int, default=Config.input_size)
    parser.add_argument("--randaug", action='store_true', default=Config.randaug)
    parser.add_argument("--train_interpolation", type=str, default=Config.train_interpolation)
    parser.add_argument("--fusion_strategy", type=str, default=Config.fusion_strategy, choices=['add', 'combiner'])
    # parser.add_argument("--target-ratio", type=float, default=Config.target_ratio)
    # parser.add_argument("--loss-type", type=str, default=Config.loss_type, choices=['crossentropy', 'dual_symmetric_ce','triple_symmetric_ce','quad_symmetric_ce'])
    parser.add_argument("--loss_components", nargs='+', default=Config.loss_components, 
                        help="List of loss components to use (e.g., ref_tgt text_tgt)")
    parser.add_argument("--loss_weights", nargs='+', type=float, default=Config.loss_weights,
                        help="Weights corresponding to loss_components")
    
    parser.add_argument("--use_hnm", action='store_true', default=Config.use_hnm, help="Whether to enable HNM regularization")
    parser.add_argument("--hnm_weight", type=float, default=Config.hnm_weight, help="HNM loss weight")
    parser.add_argument("--hnm_topk", type=int, default=Config.hnm_topk, help="Number of Top-K HNM negatives")
    parser.add_argument("--hnm_margin", type=float, default=Config.hnm_margin, help="HNM Margin")
    parser.add_argument("--hnm_temp", type=float, default=Config.hnm_temp, help="HNM temperature")

    parser.add_argument("--wandb_project", type=str, default=Config.wandb_project, help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default=Config.wandb_entity, help="WandB username/team name")
    parser.add_argument("--wandb_mode", type=str, default=Config.wandb_mode, 
                        choices=['online', 'offline', 'disabled'], 
                        help="WandB mode: online (upload normally), offline (offline mode), disabled (do not log)")
    
    parser.add_argument(
    "--export_merged_release_checkpoint",
    action='store_true',
    default=getattr(Config, "export_merged_release_checkpoint", True)
    )
    parser.add_argument(
        "--release_suffix",
        type=str,
        default=getattr(Config, "release_suffix", "release")
    )

    args = parser.parse_args()

    # Pack all arguments into a dictionary for management and saving
    training_hyper_params = vars(args)
    if len(training_hyper_params['loss_components']) != len(training_hyper_params['loss_weights']):
        raise ValueError("loss_components and loss_weights must have the same length!")

    run_id = args.experiment_name
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.experiment_name,
        id=run_id,         # Force a fixed ID so it remains unchanged after restart
        resume="allow",    # allow: resume if the ID exists; otherwise create a new run
        config=training_hyper_params,
        mode=args.wandb_mode, # Key setting: decide whether to connect online based on arguments
        settings=wandb.Settings(start_method="fork") 
    )
    # Start training
    train_beit3_finetune(**training_hyper_params)