# Dual-Space Smoothness for Robust and Balanced LLM Unlearning

🛡️ We propose **PRISM**, a unified framework that enforces dual-space smoothness in representation and parameter spaces to improve robustness and balance unlearning metrics. PRISM consists of two smoothness optimization stages: (i) a representation space stage that employs a robustly trained probe to defend against jailbreak attacks, and (ii) a parameter-space stage that decouples retain-forget gradient conflicts, reduces imbalance, and smooths the parameter space to mitigate relearning attacks.

📄 **Check out our paper:** [Dual-Space Smoothness for Robust and Balanced LLM Unlearning](https://https://arxiv.org/abs/2509.23362).

---

🙏 **Citation**: If you find this work useful, please cite our paper

```bibtex
@article{yan2025dual,
  title={Dual-Space Smoothness for Robust and Balanced LLM Unlearning},
  author={Yan, Han and Liu, Zheyuan and Jiang, Meng},
  journal={arXiv preprint arXiv:2509.23362},
  year={2025}
}
```

## Installation

To create a conda environment for Python 3.10, run:
```bash
conda env create -f environment.yml
conda activate muse
```

## Get the data & origin models

- Two corpora `News` and `Books` and the corresponding target models are available as follows from the MUSE benchmark:
    | Domain | <div style="text-align: center">Target Model for Unlearning</div> | Dataset |
    |----------|:------------------------------:|----------| 
    | News | [Target model](https://huggingface.co/muse-bench/MUSE-News_target) | [Dataset](https://huggingface.co/datasets/muse-bench/MUSE-News) |
    | Books | [Target model](https://huggingface.co/muse-bench/MUSE-Books_target) | [Dataset](https://huggingface.co/datasets/muse-bench/MUSE-Books) | 

- Before proceeding, please load all the data from HuggingFace to the root of this repostiory through:
    ```
    python load_data.py
    ```

> **Note:**  
> The WMDP data may not be available for now due to copyright restrictions and access limitations, as obtaining it requires access approval from CAIS. （[WMDP Benchmark](https://github.com/centerforaisafety/wmdp))

## Get the unlearned model
- Run `run_muse_unlearn.sh` in the `MUSE/baselines/`.
    - `algo`: Unlearning algorithm to run (`npo_gdr` or `sam_npo_gdr`, which is refer to PRISM in our work).
    - `model_dir`: Directory of the target model.
    - `tokenizer_dir`: Directory of the tokenizer.
    - `data_file`: Forget set.
    - `retain_data_file`: Retain set for regularizations.
    - `out_dir`: Directory to save the unlearned model (default: `ckpt`).
    - `max_len`: Maximum input length (default: 2048).
    - `per_device_batch_size`, `epochs`, `lr`: Hyperparameters.
    - `per_device_batch_size`, `epochs`, `lr`, `pretrained_probe_path`, `adv_gamma`, `select_layer`: Hyperparameters for controlling the training process and model behavior, especially for the probe's layer selections.


## Get the relearned model
- Run `run_muse_relearn.sh` in the `MUSE/baselines/`.


## Evaluate the unlearned model

- To evaluate the resulting model(s) in MUSE, we provide an example script in `MUSE/eval.sh` with the following command-line arguments:

    - `--model_dirs`: A list of directories containing the unlearned models. 
    - `--names`: A unique name assigned to each unlearned model in `--model_dirs`. The length of `--names` should match the length of `--model_dirs`.
    - `--corpus`: The corpus to use for evaluation. Options are `news` or `books`.
    - `--out_file`: The name of the output file. The file will be in CSV format, with each row corresponding to an unlearning method from `--model_dirs`, and columns representing the metrics specified by `--metrics`.
    - `--tokenizer_dir` (Optional): The directory of the tokenizer. Defaults to `meta-llama/Llama-2-7b-hf`, which is the default tokenizer for LLaMA.
    - `--metrics` (Optional): The metrics to evaluate. Options are `verbmem_f` (VerbMem Forget), `privleak` (PrivLeak), `knowmem_f` (KnowMem Forget), and `knowmem_r` (Knowmem Retain, i.e., Utility). Defaults to evaluating all these metrics.
    - `--temp_dir` (Optional): The directory for saving intermediate computations. Defaults to `temp`.

## Jailbreak

- To evaluate the resulting model(s) in jailbreaking performances, we provide an example script in `Jailbreak/generate_response.sh` with the following command-line arguments:

    - `--mode`: Specifies the evaluation mode used by `gen_response.py`. Different modes correspond to different attack settings and data formats.
        - `general`: Used for general single-turn evaluation. With `xstest_prompts.csv`, it is mainly for **overrefusal** evaluation, testing whether the model refuses harmless prompts too often.
        - `prefill`: Used for **prefill attacks**. With `llama_harmful-prefix.jsonl`, it evaluates whether an adversarial prefix can steer the model into unsafe behavior. This setting can usually support different **prefill lengths**.
        - `harmbench`: Used for harmful prompt attack evaluation. With `mistral_autodan.json`, it mainly tests **AutoDAN** style jailbreak attacks.
        - `multi_turn`: Used for **multi-turn attacks**. With `llama_multi.json`, it evaluates whether the model can be gradually jailbroken through multiple rounds of conversation.

    - `--model_path`: The path to the model being evaluated. This can be a local model directory or a model identifier.

    - `--eval_path`: The path to the evaluation dataset. The file should match the format required by the selected `--mode`.

- Data files

    - `xstest_prompts.csv`: Used for **overrefusal** evaluation. It contains harmless prompts that the model should normally answer, so it measures whether the model is overly conservative.

    - `llama_harmful-prefix.jsonl`: Used for **prefill attacks**. It adds a harmful or adversarial prefix before the actual prompt to test whether the model is more vulnerable under prefilling. Different **prefill lengths** can be tested.

    - `mistral_autodan.json`: Used for **AutoDAN** evaluation. It contains automatically optimized jailbreak prompts designed to induce harmful responses.

    - `llama_multi.json`: Used for **multi-turn attack** evaluation. It simulates gradual jailbreak attempts across several dialogue turns, which is closer to realistic attack scenarios.
