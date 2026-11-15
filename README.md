# Quality-Aware Self-Training for Semi-Supervised Hate Speech Detection

Code for the paper *A Quality-Aware Self-Training Framework for Semi-Supervised Hate Speech Detection*.

## Notebooks

| Notebook | Contents |
|---|---|
| `quality_aware_hatexplain.ipynb` | Data preparation, supervised BERT–MLP, fixed-encoder self-training, proposed framework and retrained BERT head on HateXplain |
| `quality_aware_dynahate_davidson.ipynb` | The same four systems on DynaHate and binary Davidson |
| `qwen_zero_shot_hatexplain.ipynb` | Zero-shot Qwen baseline on the HateXplain test partition |
| `qwen_zero_shot_davidson_dynahate.ipynb` | Zero-shot Qwen baseline on the Davidson and DynaHate test partitions |

The two training notebooks download the datasets and the pretrained model, and need a GPU. The reported runs used one NVIDIA Tesla T4. The Qwen notebooks need a Groq API key and no GPU.

## Reported results

Test macro-F1. The first four columns are means over three repetitions (seeds 42, 43, 44); Qwen is one evaluation per dataset.

| Dataset | Supervised BERT–MLP | Fixed-encoder self-training | Proposed ensemble | Retrained BERT head | Zero-shot Qwen |
|---|---|---|---|---|---|
| HateXplain | 0.7539 | 0.7512 | 0.7619 | 0.7632 | 0.7149 |
| DynaHate | 0.7347 | 0.7326 | 0.7541 | 0.7529 | 0.8377 |
| Davidson | 0.9360 | 0.9355 | 0.9425 | 0.9423 | 0.8856 |

## Data

The datasets are available from their original publications: HateXplain (Mathew et al., 2021), DynaHate (Vidgen et al., 2021) and Davidson et al. (2017). The notebooks contain the code that produces the binary label mappings, the class balancing and the partitions used in the paper.

## Notes

- The training notebooks are published without stored outputs. Running them reproduces the tables and charts.
- The Qwen notebooks keep the outputs of their final evaluation session. Each restores the predictions of earlier sessions from an embedded compressed checkpoint.
- The final self-training round is fixed in advance. No round or model is selected using test scores.
