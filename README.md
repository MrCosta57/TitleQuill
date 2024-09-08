<div align="center">

# TitleQuill: Unified Framework for Titles and Keywords Generation using Pre-Trained Model

<a href="https://huggingface.co/"><img alt="HuggingFace" src="https://img.shields.io/badge/-HuggingFace-FDEE21?style=for-the-badge&logo=HuggingFace&logoColor=black"></a>
<a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://wandb.ai/"><img alt="WeightsAndBiases" src="https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white"></a>

</div>

## Description
This repository contains the implementation of TitleQuill, a novel approach for keyword extraction and title generation, reinterpreted as two forms of summarization. The project leverages the Flan-T5 model, fine-tuned using two distinct strategies: simultaneous training on both tasks and divided task training with combined losses. The approach is built on the T5 idea, framing both tasks as text-to-text transformations, enabling the use of a single model for both. The repository includes scripts for model training, data preparation, and evaluation, along with pre-trained model checkpoints and instructions for reproducing the experiments.

## Installation
```bash
# [OPTIONAL] Create conda environment
conda create -n myenv python=3.11
conda activate myenv

# Install pytorch according to instructions
# https://pytorch.org/get-started/

# Install requirements
pip install -r requirements.txt
```

## Download data

Using the following command is possible to download the dataset used in the project. The script apply also a post-processing to the files, changing the extension from `.txt` to `.jsonl` adjust their names properly.

```bash
python src/datamodule/download.py
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for download.py</span></summary>

  #### --data_dir
  Directory to save the dataset
  #### --url
  URL to download the dataset
  #### --old_ext_postproc
  Old extension of the files to postprocess
  #### --new_ext_postproc
  New extension of the files to postprocess
</details>


## Data statistics

The statistics about the dataset can be obtained using the following command:
```bash
python src/datamodule/stats.py
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for stats.py</span></summary>

  #### --data_dir
    Directory to save the dataset
  #### --out_dir
    Directory to save the plots
</details>



## Model weights

The models weights can be obtained from this [link](https://drive.google.com/drive/folders/1yVKkDVj1UrwRl_EAPzG9yUr4zOoRRdjO?usp=drive_link). Please place the weights in the `output/` directory.


## How to run

All the scripts can be configured by modifying the configuration files in the [configs/](configs/) directory. The configuration files are written in YAML format. The scripts parameters can be overridden from the command line.

The configuration files are validated using Hydra, a powerful configuration management tool. For more information about Hydra, please refer to the official [documentation](https://hydra.cc/docs/intro/)

### Training TitleQuill
```bash
python src/run_titlequill.py
```

### Other scripts

- Baseline
```bash
python src/run_baseline.py
```
- Qwen2
```bash
python src/run_qwen2.py
```
- TextRank
```bash
python src/run_textrank.py
```


## Demo

The project includes a Streamlit GUI for the TitleQuill model. To run the GUI, execute the following command:
```bash
# Activate your environment
streamlit run src/app.py
```

## References

```
@misc{flan_t5,
      title={Scaling Instruction-Finetuned Language Models}, 
      author={Hyung Won Chung and Le Hou and Shayne Longpre and Barret Zoph and Yi Tay and William Fedus and Yunxuan Li and Xuezhi Wang and Mostafa Dehghani and Siddhartha Brahma and Albert Webson and Shixiang Shane Gu and Zhuyun Dai and Mirac Suzgun and Xinyun Chen and Aakanksha Chowdhery and Alex Castro-Ros and Marie Pellat and Kevin Robinson and Dasha Valter and Sharan Narang and Gaurav Mishra and Adams Yu and Vincent Zhao and Yanping Huang and Andrew Dai and Hongkun Yu and Slav Petrov and Ed H. Chi and Jeff Dean and Jacob Devlin and Adam Roberts and Denny Zhou and Quoc V. Le and Jason Wei},
      year={2022},
      eprint={2210.11416},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2210.11416}, 
}

@inproceedings{text_rank,
    title = "{T}ext{R}ank: Bringing Order into Text",
    author = "Mihalcea, Rada  and
      Tarau, Paul",
    editor = "Lin, Dekang  and
      Wu, Dekai",
    booktitle = "Proceedings of the 2004 Conference on Empirical Methods in Natural Language Processing",
    month = jul,
    year = "2004",
    address = "Barcelona, Spain",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W04-3252",
    pages = "404--411",
}


@misc{qwen2,
      title={Qwen Technical Report}, 
      author={Jinze Bai and Shuai Bai and Yunfei Chu and Zeyu Cui and Kai Dang and Xiaodong Deng and Yang Fan and Wenbin Ge and Yu Han and Fei Huang and Binyuan Hui and Luo Ji and Mei Li and Junyang Lin and Runji Lin and Dayiheng Liu and Gao Liu and Chengqiang Lu and Keming Lu and Jianxin Ma and Rui Men and Xingzhang Ren and Xuancheng Ren and Chuanqi Tan and Sinan Tan and Jianhong Tu and Peng Wang and Shijie Wang and Wei Wang and Shengguang Wu and Benfeng Xu and Jin Xu and An Yang and Hao Yang and Jian Yang and Shusheng Yang and Yang Yao and Bowen Yu and Hongyi Yuan and Zheng Yuan and Jianwei Zhang and Xingxuan Zhang and Yichang Zhang and Zhenru Zhang and Chang Zhou and Jingren Zhou and Xiaohuan Zhou and Tianhang Zhu},
      year={2023},
      eprint={2309.16609},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2309.16609}, 
}