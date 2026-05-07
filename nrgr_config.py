from pathlib import Path


class Config:
    log_base_dir = "experiments_nrgr"
    experiment_name = "beit3_nrgr"

    dialogue_format = "VisDial"
    dialogue_round = 10
    use_random_rounds = True
    use_caption_masking = False
    caption_masking_prob = 0.2

    train_json_path = "dataset/visdial_1.0_train.json"
    max_train_samples = 0
    val_corpus_json_path = "ChatIR_Protocol/Search_Space_val_50k.json"
    val_queries_path = "dialogues/VisDial_v1_0_queries_val.json"
    val_generated_image_dir = "data/generated_images/VisDial_v1_0_queries_val/sd3/few_shot"

    training_mode = "end_to_end"

    export_merged_release_checkpoint = True
    release_suffix = "release"

    beit3_checkpoint_path = "beit3/model/beit3_base_itc_patch16_224.pth"
    beit3_tokenizer_path = "beit3/model/beit3.spm"
    fusion_strategy = "combiner"
    combiner_checkpoint_path = None


    num_epochs = 50
    batch_size = 256
    update_freq = 8
    val_batch_size = 32
    beit3_lr = 1e-5
    min_lr = 1e-6
    combiner_lr = 5e-5
    modality_weight_lr = 1e-4
    warmup_epochs = 6
    validation_frequency = 1
    weight_decay = 0.05
    layer_decay = 0.75
    drop_path = 0.2
    clip_grad = 3.0
    model_ema = False
    model_ema_decay = 0.9999

    resume_from = None
    save_training = True

    input_size = 224
    train_interpolation = "bicubic"
    randaug = False

    loss_components = ["ref_tgt", "text_tgt", "fused_tgt", "ref_text"]
    loss_weights = [1.0, 1.0, 1.0, 1.0]
    use_learnable_modality_weights = True
    use_hnm = False
    hnm_weight = 1.0
    hnm_topk = 8
    hnm_margin = 0.2
    hnm_temp = 0.1

    wandb_project = "nrgr"
    wandb_entity = None
    wandb_mode = "disabled"
