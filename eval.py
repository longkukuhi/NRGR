import torch
from PIL import Image
import os
import json
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import XLMRobertaTokenizer
from typing import Optional
from torch.nn.functional import normalize
import modeling_finetune 
import utils
from timm.models import create_model
from torchvision import transforms
from modeling_utils import _get_large_config, _get_base_config
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
# Run a single dataset and print the results to the console.
CONFIG = {
    # Path configuration
    "generated_image_dir": "./data/generated_images/plugir/sd3/blip3statement", # CB_image_neg_cot  
    # ./data/VD_few_shot_images   HB_few_shot_images  CB_image_neg_cot  ./data/generated_images/VisDial_v1_0_queries_val/sd3/blip3statement
    "queries_path": 'dialogues/plugir.json',    # VD-reformulated.json  plugir.json
    # 'VisDial_v1_0_queries_val.json', 'Human_BLIP2.json', 'ChatGPT_BLIP2.json' FLAN_ALPACA_XXL_BLIP2.json
    "corpus_path": 'ChatIR_Protocol/Search_Space_val_50k.json',  
    "image_size": 224,
    # BEiT-3 model configuration
    "beit3_timm_model_config": "beit3_base_patch16_384_retrieval", # Note: main currently forces the image size to 224.
    "beit3_checkpoint_path": "./experiments/beit3_itc_r10_quad/saved_models/beit3_58.pt",   
    # ./beit3/model/beit3_base_patch16_384_coco_retrieval.pth  ./beit3/model/beit3_base_itc_patch16_224.pth  
    # ./experiments/beit3_itc_r10_quad/saved_models/beit3_58.pt
    "beit3_tokenizer_path": "./beit3/model/beit3.spm",
    
    # Cache paths
    "beit3_cache_path": "temp_beit3/corpus_beit3_itc_base.pth", 
    "beit3_gen_cache_path": "temp_beit3/gen_beit3_itc_base.pth", 

    # Experiment parameters
    "num_eval_rounds": 11,
    "corpus_bs": 256,
    "queries_bs": 128,
    "num_workers": 8,
    "device": 'cuda:0' if torch.cuda.is_available() else 'cpu',
    "sep_token": ', ',
    
    # Experiment branches
    "single_stage_experiments": {
        "Text_Only": {"type": "text"},
        "Image_Only": {"type": "image"},
        "DAR_Fusion": {"type": "dar"},
    },
}

class Corpus(Dataset):
    def __init__(self, corpus_path, preprocessor):
        with open(corpus_path, 'r', encoding='utf-8') as f:
            self.corpus = json.load(f)
        self.preprocessor = preprocessor
        self.path2id = {self.corpus[i]: i for i in range(len(self.corpus))}

    def __len__(self):
        return len(self.corpus)

    def path_to_index(self, path):
        return self.path2id[path]

    def __getitem__(self, i):
        image = Image.open(self.corpus[i]).convert("RGB")
        image_tensor = self.preprocessor(image)
        return {'id': i, 'image': image_tensor}

class Queries(Dataset):
    def __init__(self, cfg, queries_path):
        with open(queries_path, 'r', encoding='utf-8') as f:
            self.queries = json.load(f)
        self.dialog_length = None
        self.cfg = cfg

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, i):
        assert self.dialog_length is not None, "Please set self.dialog_length"
        target_path = self.queries[i]['img']
        text = self.cfg['sep_token'].join(self.queries[i]['dialog'][:self.dialog_length + 1])
        text = self.queries[i]['dialog'][self.dialog_length]  # Use only the current-round text as the query.
        gen_image_filename = f"{i}_{self.dialog_length}.jpg"
        gen_image_path = os.path.join(CONFIG['generated_image_dir'], gen_image_filename)
        return {'text': text, 'target_path': target_path, 'gen_path': gen_image_path}

class GeneratedImagesDataset(Dataset):
    def __init__(self, queries_path, gen_image_dir, num_rounds, preprocessor):
        self.samples = []
        self.preprocessor = preprocessor
        
        # Pre-scan all required files.
        with open(queries_path, 'r', encoding='utf-8') as f:
            queries = json.load(f)
            
        # Flatten the loop to build a list of all (filename, full_path) pairs.
        for query_idx in range(len(queries)):
            for round_idx in range(num_rounds):
                filename = f"{query_idx}_{round_idx}.jpg"
                filepath = os.path.join(gen_image_dir, filename)

                self.samples.append((filename, filepath))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, filepath = self.samples[idx]
        image = Image.open(filepath).convert("RGB")
        if self.preprocessor:
            image = self.preprocessor(image)
        return filename, image

def load_beit3_checkpoint(model, checkpoint_path, target_img_size):
    """
    BEiT-3-specific loading function that automatically interpolates positional embeddings when resolutions mismatch.
    """
    # print(f"Loading BEiT-3 checkpoint from {checkpoint_path} with resize to {target_img_size}...")
    
    # 1. Load the checkpoint.
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 2. Extract the state_dict.
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    # 3. Core step: handle positional embeddings (beit3.encoder.embed_positions.A.weight).
    pos_key = "beit3.encoder.embed_positions.A.weight"
    
    if pos_key in state_dict:
        pos_embed_checkpoint = state_dict[pos_key] # The shape may be [579, 768].
        embedding_size = pos_embed_checkpoint.shape[-1]
        
        # Get the number of patches required by the current target model.
        # Formula: (224 / 16) * (224 / 16) = 14 * 14 = 196.
        num_patches = (target_img_size // 16) ** 2
        
        # Compute the total sequence length of the current target model, usually 196 + 3 = 199.
        # Read the model shape directly to determine the number of extra tokens, making this robust to future changes.
        target_total_len = model.beit3.encoder.embed_positions.A.weight.shape[0]
        num_extra_tokens = target_total_len - num_patches # Expected to be 3.
        
        # Compute the patch grid size in the checkpoint.
        # checkpoint_total_len = 579, num_extra_tokens = 3 => 576 patches
        orig_num_patches = pos_embed_checkpoint.shape[0] - num_extra_tokens
        orig_size = int(orig_num_patches ** 0.5) # sqrt(576) = 24
        new_size = int(num_patches ** 0.5)       # sqrt(196) = 14
        
        # Interpolate if the resolutions do not match.
        if orig_size != new_size:
            print(f"Detected resolution mismatch: Source {orig_size}x{orig_size} -> Target {new_size}x{new_size}")
            print(f"Interpolating position embeddings...")
            
            # A. Separate special tokens (the first 3 tokens).
            extra_tokens = pos_embed_checkpoint[:num_extra_tokens]
            
            # B. Separate image patch tokens (after the first 3 tokens).
            pos_tokens = pos_embed_checkpoint[num_extra_tokens:]
            
            # C. Reshape to (1, Dim, H, W) for F.interpolate.
            # Original shape: [576, 768] -> reshape [24, 24, 768] -> permute [1, 768, 24, 24].
            pos_tokens = pos_tokens.reshape(1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            
            # D. Bicubic interpolation.
            pos_tokens = F.interpolate(
                pos_tokens, 
                size=(new_size, new_size), 
                mode='bicubic', 
                align_corners=False
            )
            
            # E. Restore shape: [1, 768, 14, 14] -> [1, 14, 14, 768] -> flatten [196, 768].
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2).squeeze(0)
            
            # F. Concatenate back into the full positional embedding.
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=0)
            
            # G. Replace the corresponding weight in the state_dict.
            state_dict[pos_key] = new_pos_embed
            print(f"Interpolation done. New shape: {new_pos_embed.shape}")

    # 4. Load the processed weights.
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Weights loaded with msg: {msg}")

class ExperimentEvaluator:
    def __init__(self, cfg, model, preprocessor, tokenizer):
        self.cfg = cfg
        self.model = model
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer

        self.corpus = None
        self.corpus_dataset = Corpus(
            self.cfg['corpus_path'],
            self.preprocessor
        )

        self.generated_features_cache = self._create_or_load_generated_cache()

    def _calculate_fused_scores(self, method, text_features, gen_features, target_features, dialog_length):
        target_features_T = target_features.T

        if method == 'text':
            scores = text_features @ target_features_T
            return scores

        if method == 'image':
            scores = gen_features @ target_features_T
            return scores

        if method == 'dar':
            text_scores = text_features @ target_features_T
            gen_scores = gen_features @ target_features_T

            if dialog_length < 2:
                w_text, w_img = 0.8, 0.2
            else:
                w_text, w_img = 0.5, 0.5
            return w_text * text_scores + w_img * gen_scores
        
        raise ValueError(f"Unknown fusion method: {method}")
    
    def _calculate_ranks(self, ranked_indices, target_ids):
        # ranks = []
        # for i in range(ranked_indices.shape[0]):
        #     rank_tensor = (ranked_indices[i] == target_ids[i]).nonzero(as_tuple=True)[0]
        #     if rank_tensor.numel() > 0:
        #         ranks.append(rank_tensor.squeeze())
        #     else:
        #         ranks.append(torch.tensor(float('inf'), device=self.cfg['device']))
        # return torch.stack(ranks)
        hits = (ranked_indices == target_ids)
        result = torch.full((ranked_indices.size(0),), float('inf'), device=self.cfg['device'])
        batch_indices, rank_indices = hits.nonzero(as_tuple=True)
        result[batch_indices] = rank_indices.float()
        return result

    def _get_recalls_for_all_experiments(self, dataloader, dialog_length):
        dataloader.dataset.dialog_length = dialog_length
        experiments = self.cfg['single_stage_experiments']
        experiment_recalls = {name: [] for name in experiments.keys()}

        for batch in tqdm(dataloader, desc=f"Round {dialog_length} Processing"):
            target_ids = torch.tensor([self.corpus_dataset.path_to_index(p) for p in batch['target_path']]).unsqueeze(1).to(self.cfg['device'])
            dialogue_texts_list = list(batch['text'])
            gen_paths_batch = list(batch['gen_path'])
         
            # Tokenize text.
            text_inputs_generic = self.tokenizer(text=dialogue_texts_list, padding="longest", truncation=False, return_tensors="pt").to(self.cfg['device'])
            
            # BEiT-3 feature dimension.
            feature_dim = self.model.language_head.out_features
            zero_feature = torch.zeros((1, feature_dim), device=self.cfg['device'])

            # BEiT-3 text encoding logic.
            # Note: BEiT-3 expects the inverted attention_mask here, where 1 indicates padding.
            beit3_padding_mask = 1 - text_inputs_generic['attention_mask']
            _, text_features = self.model(
                text_description=text_inputs_generic['input_ids'], 
                padding_mask=beit3_padding_mask,
                only_infer=True
            )
            
            # Get generated image features.
            gen_features_list = [
                self.generated_features_cache.get(os.path.basename(p), {}).get('gen_feat', zero_feature) 
                for p in gen_paths_batch
            ]
            gen_features = torch.cat(gen_features_list, dim=0)

            for name, exp_conf in experiments.items():
                method_type = exp_conf['type']
                total_scores = self._calculate_fused_scores(
                    method=method_type,
                    text_features=text_features,
                    gen_features=gen_features,
                    target_features=self.corpus[1], 
                    dialog_length=dialog_length
                )
                
                arg_ranks = torch.argsort(total_scores, descending=True, dim=1).long()
                target_recall = self._calculate_ranks(arg_ranks, target_ids)
                experiment_recalls[name].append(target_recall)

        for name in experiment_recalls:
            if experiment_recalls[name]:
                experiment_recalls[name] = torch.cat(experiment_recalls[name])
            else:
                experiment_recalls[name] = torch.tensor([], dtype=torch.long)
        return experiment_recalls

    def preprocess_single_image(self, image):
        # BEiT-3 only needs the transform call.
        return self.preprocessor(image)

    def run(self, hits_at=10):
        assert self.corpus, "Prepare corpus first (self.index_corpus())"
        dataset = Queries(self.cfg, self.cfg['queries_path'])
        dataloader = DataLoader(dataset, batch_size=self.cfg['queries_bs'], shuffle=False,
                                num_workers=self.cfg['num_workers'], pin_memory=True, drop_last=False)

        exp_keys = self.cfg['single_stage_experiments'].keys()
        all_rounds_recalls = {name: [] for name in exp_keys}

        for dl in range(self.cfg['num_eval_rounds']):
            experiment_recalls_per_round = self._get_recalls_for_all_experiments(dataloader, dialog_length=dl)
            for name, recalls in experiment_recalls_per_round.items():
                all_rounds_recalls[name].append(recalls.cpu())

        # Keep the result-printing logic unchanged.
        for name, results_per_round in all_rounds_recalls.items():
            print(f"\n====== Results for Experiment: '{name}' (Hits@{hits_at}) ======")
            independent_hit_rates = []
            for recalls in results_per_round:
                if recalls.numel() > 0:
                    num_hits = (recalls < hits_at).sum().item()
                    total_queries = len(recalls)
                    rate = (num_hits * 100 / total_queries) if total_queries > 0 else 0
                else:
                    rate = 0
                independent_hit_rates.append(rate)
            print(f"  --- Independent Per Round ---")
            for dl, rate in enumerate(independent_hit_rates):
                print(f"\tRound {dl}: {rate:.2f}%")
            if results_per_round and results_per_round[0].numel() > 0:
                all_recalls = torch.cat([r.cpu() for r in results_per_round if r.numel() > 0])
                if all_recalls.numel() > 0:
                    cumulative_results = cumulative_hits_per_round(
                        all_recalls, num_rounds=self.cfg['num_eval_rounds'], hitting_recall=hits_at).tolist()
                else:
                    cumulative_results = [0.0] * self.cfg['num_eval_rounds']
            else:
                cumulative_results = [0.0] * self.cfg['num_eval_rounds']
            print(f"  --- Cumulative ---")
            for dl, rate in enumerate(cumulative_results):
                print(f"\tUp to Round {dl}: {rate:.2f}%")

    def index_corpus(self):
        device = self.cfg['device']
        self.cache_path = self.cfg["beit3_cache_path"]

        if os.path.exists(self.cache_path):
            print(f"<<<< Loading cached corpus: {self.cache_path} >>>>")
            self.corpus = torch.load(self.cache_path)
            if isinstance(self.corpus, tuple) and len(self.corpus) == 2:
                self.corpus = (self.corpus[0].to(device), self.corpus[1].to(device))
            return

        dataloader = DataLoader(self.corpus_dataset, batch_size=self.cfg['corpus_bs'], shuffle=False,
                                num_workers=self.cfg['num_workers'], pin_memory=True, drop_last=False)
        print("Preparing corpus (image search space)...")
        corpus_vectors, corpus_ids = [], []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Indexing Corpus"):
                images = batch['image'].to(device)
                # BEiT-3 image inference.
                batch_vectors, _ = self.model(image=images, only_infer=True)

                batch_vectors = F.normalize(batch_vectors, dim=-1)
                corpus_vectors.append(batch_vectors.cpu())
                corpus_ids.append(batch['id'])
        
        corpus_vectors = torch.cat(corpus_vectors)
        corpus_ids = torch.cat(corpus_ids)
        arg_ids = torch.argsort(corpus_ids)
        corpus_vectors = corpus_vectors[arg_ids]
        corpus_ids = corpus_ids[arg_ids]
        self.corpus = corpus_ids.to(device), corpus_vectors.to(device)
        torch.save(self.corpus, self.cache_path)
        print(f"Saved corpus cache to: {self.cache_path}")

    def _create_or_load_generated_cache(self):
        gen_cache_path = self.cfg["beit3_gen_cache_path"]

        if os.path.exists(gen_cache_path):
            print(f"<<<< Loading cached generated features: {gen_cache_path} >>>>")
            # Keep the original loading logic.
            cached_data = torch.load(gen_cache_path, map_location='cpu')
            return {
                k: {fk: fv.to(self.cfg['device']) for fk, fv in v.items()}
                for k, v in cached_data.items()
            }

        print(f"!!!! Cached generated features not found. Creating new cache at: {gen_cache_path} !!!!")
        print("!!!! Using Batch Processing for Speed !!!!")

        # 1. Initialize the dataset and DataLoader.
        gen_dataset = GeneratedImagesDataset(
            queries_path=self.cfg['queries_path'],
            gen_image_dir=self.cfg['generated_image_dir'],
            num_rounds=self.cfg['num_eval_rounds'],
            preprocessor=self.preprocessor  # Make sure the correct preprocessing function is passed here.
        )

        # Use the same batch size as defined in the config.
        gen_loader = DataLoader(
            gen_dataset, 
            batch_size=self.cfg['corpus_bs'], # Use a larger batch size, e.g., 512.
            shuffle=False,
            num_workers=self.cfg['num_workers'], # Enable multi-process loading, e.g., 8 workers.
            pin_memory=True,
            drop_last=False
        )

        cache_data = {}

        # 2. Batched inference.
        with torch.no_grad():
            for filenames, images in tqdm(gen_loader, desc="Caching Generated Image Features (Batched)"):
                images = images.to(self.cfg['device'], non_blocking=True)
                
                # Infer the whole batch at once, e.g., 512 images.
                gen_feats, _ = self.model(image=images, only_infer=True)
                gen_feats = F.normalize(gen_feats, dim=-1)

                # Store the results back into the dictionary.
                # Note: even with batched inference, the storage structure is still keyed by filename.
                for i, filename in enumerate(filenames):
                    cache_data[filename] = {
                        "gen_feat": gen_feats[i].unsqueeze(0).cpu(), # Store on CPU to avoid excessive GPU memory usage.
                    }

        # 3. Save.
        os.makedirs(os.path.dirname(gen_cache_path), exist_ok=True)
        torch.save(cache_data, gen_cache_path)
        print(f"Saved new simplified features cache to: {gen_cache_path}")

        # 4. Move back to GPU on return if GPU memory is sufficient.
        return {
            k: {fk: fv.to(self.cfg['device']) for fk, fv in v.items()}
            for k, v in cache_data.items()
        } 
    
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

def main():
    print(f"Using device: {CONFIG['device']}")
    # Force the script to run the BEiT-3-only logic.
    target_img_size = CONFIG['image_size']
    # 1. Determine the configuration (Base vs. Large).

    args = _get_base_config(img_size=target_img_size, vocab_size=64010)
    
    # 2. Initialize the model.
    model = modeling_finetune.BEiT3ForRetrieval(args=args)
    checkpoint_path = CONFIG['beit3_checkpoint_path']

    # 3. Load weights.
    # print(f"Loading BEiT-3 Checkpoint: {checkpoint_path}")
    # checkpoint = torch.load(checkpoint_path, map_location='cpu')
    # state_dict = None
    # if 'model_state_dict' in checkpoint:
    #     print("Loading weights from 'model_state_dict' (Regular model)...")
    #     state_dict = checkpoint['model_state_dict']
    # elif 'model_ema_state_dict' in checkpoint:
    #     print("Loading weights from 'model_ema_state_dict' (EMA model)...")
    #     state_dict = checkpoint['model_ema_state_dict']
    # else:
    #     print("Could not find 'model_state_dict' or 'model_ema_state_dict', attempting to load root...")
    #     state_dict = checkpoint['model']
    
    # # Remove potential mismatches if needed, or load strictly.
    # model.load_state_dict(state_dict, strict=False) 
    load_beit3_checkpoint(model, checkpoint_path, target_img_size=target_img_size)
    model.to(CONFIG['device']).eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[BEiT-3 Model Stats]")
    print(f"Total Parameters: {total_params / 1e6:.2f} M")

    # 4. Initialize the tokenizer and preprocessor.
    tokenizer_path = CONFIG['beit3_tokenizer_path']
    tokenizer = XLMRobertaTokenizer(tokenizer_path)
    print(f"Official Tokenizer loaded from: {tokenizer_path}")
    
    # Ensure the resize size here is consistent with img_size in args.
    preprocessor = transforms.Compose([
        transforms.Resize((target_img_size, target_img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), 
        # transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    ])

    # 5. Run evaluation.
    with torch.no_grad():
        evaluator = ExperimentEvaluator(
            CONFIG,
            model,
            preprocessor,
            tokenizer
        )
        evaluator.index_corpus()
        evaluator.run(hits_at=10)

if __name__ == '__main__':            

    main()