# ERGO: Event Relational Graph Transformer for Document-level Event Causality Identification
This is the implementation of the paper _ERGO: Event Relational Graph Transformer for Document-level Event Causality Identification_. Accepted by COLING'2022.

## Environment
python == 3.6.13\
pytorch == 1.8.1\
transformers == 4.11.3

Then, create a ./ckpts directory and download the Longformer/BERT model from [HuggingFace](https://huggingface.co/allenai/longformer-base-4096) into the directory.

Or use the following command:
```bash
mkdir ckpts
cd ./ckpts
git lfs install
git clone https://huggingface.co/allenai/longformer-base-4096
```

## Run the Experiments

Please make sure your folder structure is as below.
```bash
CHEER
  ├── ckpts
  │   ├──longformer-base-4096
  ├── data
  │   └── cached_eci
  │       ├── EventStoryLine_longformer_intra_and_inter_dev_events.pkl
  │       ├── EventStoryLine_longformer_intra_and_inter_dev_features.pkl
  │       ├── EventStoryLine_longformer_intra_and_inter_train_events.pkl
  │       └── EventStoryLine_longformer_intra_and_inter_train_features.pkl
  └── model.py
  └── processor.py
  └── README.md
  └── train.py
  └── utils.py
  
```

Then just run:

`python train.py --num_heads 4 --dropout_att 0.1 --dropout_emb 0.3 
`

The hyper-parameters are selected by a grid search using the validation data.

You could check the outputs in `./outputs_res/log.txt`.

## Citation
Please cite our paper if this repository inspires your work.
```bibtex
@inproceedings{chen-etal-2022-ergo,
    title = "{ERGO}: Event Relational Graph Transformer for Document-level Event Causality Identification",
    author = "Chen, Meiqi  and
      Cao, Yixin  and
      Deng, Kunquan  and
      Li, Mukai  and
      Wang, Kun  and
      Shao, Jing  and
      Zhang, Yan",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.185",
    pages = "2118--2128",}
```
## Contact
- meiqichen@stu.pku.edu.cn