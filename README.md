# Search to Pass Messages for Temporal Knowledge Graph Completion
<p align="center">
<a href="https://preview.aclanthology.org/emnlp-22-ingestion/2022.findings-emnlp.458/"><img src="https://img.shields.io/badge/EMNLP%202022-Findings-brightgreen.svg" alt="emnlp paper"></a>
<a href="https://arxiv.org/abs/2210.16740"><img src="http://img.shields.io/badge/arxiv-abs-green.svg" alt="arxiv"></a>
</p>

---

## Overview
This repository contains the code for [Search to Pass Messages for Temporal Knowledge Graph Completion](https://preview.aclanthology.org/emnlp-22-ingestion/2022.findings-emnlp.458/) in Findings of EMNLP 2022.

## Requirements
```text
python=3.8
torch==1.9.0+cu111
dgl+cu111==0.6.1
```

## Instructions to run the experiment

### Search process
```shell
# Random baseline
python main.py --train_mode search --search_mode random --encoder SPATune --max_epoch 200

# SPA supernet training
python main.py --train_mode search --search_mode spos --encoder SPASPOSSearch --search_max_epoch 800

# SPA architecture search
python main.py --train_mode search --search_mode spos_search --encoder SPASPOSSearch --arch_sample_num 1000 --weight_path <xx.pt>
```
### Fine-tuning process
```shell
python main.py --train_mode tune --encoder SPATune --search_res_file <xxx.json>
```
## Citation

Readers are welcomed to fork this repository to reproduce the experiments and follow our work. Please kindly cite our paper

```bibtex
@inproceedings{wang2022search,
    title={Search to Pass Messages for Temporal Knowledge Graph Completion},
    author={Wang, Zhen and Du, Haotong and Yao, Quanming and Li, Xuelong},
    booktitle={Findings of the Association for Computational Linguistics: EMNLP 2022},
    pages={6160--6172},
    year={2022}
}
```

## Contact
If you have any questions, feel free to contact me at [duhaotong@mail.nwpu.edu.cn](mailto:duhaotong@mail.nwpu.edu.cn).

## Acknowledgement

The codes of this paper are partially based on the codes of [SANE](https://github.com/AutoML-Research/SANE) and [TeMP](https://github.com/JiapengWu/TeMP). We thank the authors of above work.