<h1 align="center">
🧠NRGR: Generation-Noise Robust Framework for Interactive Image Retrieval</h1>

<p align = "center">
<img src="assets/framework.png">
</p>


* **Official PyTorch implementation of NRGR: "Generation isn’t always helpful: Generation-Noise Robust Framework for Interactive Image Retrieval"** <br>

## 📰 Updates
- [2026/05/6] The code of NRGR is released! 🎉

## 🗞️ Table of Contents
- [📰 Updates](#-updates)
- [🗞️ Table of Contents](#️-table-of-contents)
- [🛠️ Setup](#️-setup)
- [⬇️ Download The BEiT-3 Pretrain Weight for Retrieval Task](#️-download-the-beit-3-pretrain-weight-for-retrieval-task)
- [💾 Data Preparation](#-data-preparation)
  - [🌟 Our Constructed Dataset: GA-VisDial Dataset](#-our-constructed-dataset-ga-visdial-dataset)
  - [📚 Source Datasets \& Evaluation Benchmarks](#-source-datasets--evaluation-benchmarks)
- [🗂️ Download our checkpoints](#️-download-our-checkpoints)
- [🚀 Training and Evaluation](#-training-and-evaluation)
  - [Directory Structure](#directory-structure)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [🤝 Acknowledgements](#-acknowledgements)
- [⚖️ License](#️-license)
- [✏️ Citation](#️-citation)

## 🛠️ Setup
First, clone this repository to your local machine, and install the dependencies.
```bash
pip install -r requirements.txt
```
❗ You can modify the PyTorch version to suit your machine.

## ⬇️ Download The BEiT-3 Pretrain Weight for Retrieval Task
This project relies on the official Microsoft BEiT-3 implementation and pretrained weights.
Download the pretrain model weights and the tokenizer model for retrieval task.
   - [`BEiT3-base-itc`](https://github.com/microsoft/unilm/tree/master/beit3#pretrained-models): #layer=12; hidden=768; FFN factor=4x; #head=12; patch=16x16; #parameters: 222M
   - [`beit3.spm`](https://github.com/microsoft/unilm/tree/master/beit3#pretrained-models): the sentencepiece model used for tokenizing texts.

## 💾 Data Preparation
### 🌟 Our Constructed Dataset: GA-VisDial Dataset
   - [`GA-VisDial Dataset`](https://drive.google.com/drive/folders/1JhXEoeiuwKNsVlm6LdJXFcbxYMcaJTw6?usp=sharing): This dataset contains the diffusion-augmented training samples described in our paper.The dataset is currently being prepared for release. We will update this section with the download link upon acceptance.
### 📚 Source Datasets & Evaluation Benchmarks
- [`VisDial v1.0`](https://visualdialog.org/): The source dataset used for constructing GA-VisDial and for in-distribution evaluation.
- [`ChatIR`](https://github.com/levymsn/ChatIR): Out-of-distribution evaluation benchmarks with diverse dialogue styles, including ChatGPT-generated and human-written dialogues.
- [`PlugIR`](https://github.com/Saehyung-Lee/PlugIR): A dataset featuring concise, summary-style queries generated via an interactive pipeline.

## 🗂️ Download our checkpoints
The checkpoints will be made publicly available upon acceptance of the paper.

## 🚀 Training and Evaluation

###  Directory Structure

To ensure the code runs correctly, you can organize your project directory as follows. Alternatively, you can modify the paths in `nrgr_config.py` to match your custom directory structure.

```text
.
├── beit3/                      # Official Microsoft BEiT-3 code
│   ├── modeling_finetune.py    # Core modeling code
│   ├── optim_factory.py        # Optimizer utilities
│   └── ...
├── data/                       # Dataset root directory
│   ├── visdial_1.0_train.json  # VisDial v1.0 training file
│   ├── query_images/           # Training reference images (DA-VisDial)
│   │   ├── train-xxxx_0.jpg
│   │   └── ...
│   ├── ChatIR_Protocol/        # Validation Corpus
│   │   └── Search_Space_val_50k.json
│   ├── dialogues/              # Validation Queries
│   │   └── VisDial_v1_0_queries_val.json
│   └── generated_images/       # Pre-generated images for validation
│       └── VisDial_v1_0_queries_val/
│           └── your_generated_images/
├── model/                      # Pretrained Checkpoints
│   ├── beit3_base_itc_patch16_224.pth
│   └── beit3.spm
├── nrgr_config.py
├── train.py
├── eval_nrgr.py
├──README.md
└── ...
```

### Training
You can adjust the training hyperparameters by passing command-line arguments. Alternatively, you can configure them directly by modifying nrgr_config.py, allowing you to simply run:
```bash
python train.py
```

### Evaluation
To perform a complete evaluation of the experiment, run the following command:
```bash
python eval_nrgr.py
```

## 🤝 Acknowledgements

Our code is built upon the excellent work of [Microsoft BEiT-3](https://github.com/microsoft/unilm/tree/master/beit3). We thank the authors for their open-source contribution.

We also express our gratitude to the following projects for providing datasets and evaluation protocols:
* [VisDial v1.0](https://visualdialog.org/) for the visual dialogue dataset.
* [ChatIR](https://github.com/levymsn/ChatIR) and [PlugIR](https://github.com/Saehyung-Lee/PlugIR) for the interactive text-to-image benchmarks and baselines.

## ⚖️ License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## ✏️ Citation

If you find this code useful for your research, please consider citing our paper:







