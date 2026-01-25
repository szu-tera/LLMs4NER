<div align="center">
  
# Assessment of Generative Named Entity Recognition in the Era of Large Language Models

[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/#TODO)
[![Github](https://img.shields.io/badge/code-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/szu-tera/LLMs4NER)

<div align="center" style="font-family: Arial, sans-serif;">
  <p>
    <a href="#news" style="text-decoration: none; font-weight: bold;">🎉 News</a> •
    <a href="#overview" style="text-decoration: none; font-weight: bold;">📌 Overview</a> •
    <a href="#main-results" style="text-decoration: none; font-weight: bold;">📊 Main Results</a>
  </p>
  <p>
    <a href="#getting-started" style="text-decoration: none; font-weight: bold;">✨ Getting Started</a> •
    <a href="#evaluation" style="text-decoration: none; font-weight: bold;">📃 Evaluation</a>  •
    <a href="#contact" style="text-decoration: none; font-weight: bold;">📨 Contact</a> •
    <a href="#citation" style="text-decoration: none; font-weight: bold;">🎈 Citation</a>
  </p>
</div>

</div>

## 🎉News
- **[2026/1]** We release our paper and code repo.

---

## 📌Overview

We conduct a systematic evaluation of open-source LLMs on both flat and nested NER tasks. We investigate several research questions including the performance gap between generative NER and traditional NER models, the impact of output formats, whether LLMs rely on memorization, and the preservation of general capabilities after fine-tuning. Through experiments across eight LLMs of varying scales and four standard NER datasets, we find that: (1) With parameter-efficient fine-tuning and structured formats like inline bracketed or XML, open-source LLMs achieve performance competitive with traditional encoder-based models and surpass closed-source LLMs like GPT-3; (2) The NER capability of LLMs stems from instruction-following and generative power, not mere memorization of entity-label pairs; and (3) Applying NER instruction tuning has minimal impact on general capabilities of LLMs, even improving performance on datasets like DROP due to enhanced entity understanding. These findings demonstrate that generative NER with LLMs is a promising, user-friendly alternative to traditional methods.
<div align="center">
  <img src="assets/output_formats.png" width="100%" />
</div>

---

## 📊Main Results

<div align="center">
  <img src="assets/main_results.png" width="90%" />
</div>

---

## ✨Getting Started

Clone our repository and install the required environment:

```shell
# Clone the repository
git clone https://github.com/szu-tera/LLMs4NER.git
cd LLMs4NER

# Install the required environment
conda create -n LLMs4NER python=3.10
conda activate LLMs4NER
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[metrics]"
```

Dataset preparation: We employ [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for fine-tuning. You need to replace the `data` folder in the `LLaMA-Factory` repository with our provided version to use datasets. You can set them up using the following commands:

```shell
rm -rf ./LLaMA-Factory/data
cp -r ./data ./LLaMA-Factory/data
```

Training and Inference:

```shell
# Run lora fine-tuning
bash train/run_lora.sh

# Run inference
bash inference/run_inference.sh
```

## 📃Evaluation

After running the inference script, you can evaluate the results using the codes provided in the `evaluation` folder. The script calculates precision, recall, and Micro-F1 scores for NER outputs.

```shell
python evaluation/eval_bracketed_ner.py \
    --file /path/generated_predictions.jsonl \
    --preset conll2003 # dataset
```

*   `--file` (required): Path to the `generated_predictions.jsonl` file produced by the inference script.
*   `--preset` (required): The dataset preset to use for per-label reporting.
*   `--labels` (optional): An explicit list of labels to report on, separated by spaces. If not provided, the script will use the labels defined by the `--preset`.
*   `--output_dir` (optional): The directory where the evaluation results (`results.json`) will be saved. Defaults to the same directory as the input `--file`.

Evaluate the generation ability of the fine-tuned model:

```shell
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
pip install "lm_eval[hf]"

lm-eval run --config evaluation/eval_config.yaml # evaluate
```

---

## 📨Contact

- Qi Zhan: qzhan65@gmail.com

---

## 🎈Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@article{#TODO,
  title={Assessment of Generative Named Entity Recognition in the Era of Large Language Models},
  author={Zhan, Qi and Wang, Yile and Huang, Hui},
  journal={#TODO},
  year={2026}
}
```
