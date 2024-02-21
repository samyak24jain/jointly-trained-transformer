# Joint Multi-Task Training with Transformers

This repository provides the code, data and scripts for jointly training a [vanilla transformer](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) model from scratch in PyTorch. We train the model to learn two tasks simultaneously.

## Tasks:
- BIO Slot Tagging (multi-class token classification)
- Core Relation extraction (multi-label sequence classification)

## Dataset:
The dataset is generated based on film schema of Freebase knowledge graph. There are two files [data/hw1_train.csv](https://github.com/samyak24jain/jointly-trained-transformer/blob/main/data/hw1_train.csv) and [data/hw1_test.csv](https://github.com/samyak24jain/jointly-trained-transformer/blob/main/data/hw1_test.csv). The train csv file has three columns: utterances, IOB Slot tags and	Core Relations. The test csv file has only the utterances. The dataset looks like this:
![Dataset example](https://github.com/samyak24jain/jointly-trained-transformer/assets/10193535/af4af764-c2c5-4673-bec1-80530a547d14)

## Model Architecture:
![Transformer architecture for multi-task joint training](https://github.com/samyak24jain/jointly-trained-transformer/assets/10193535/5cdc49d5-0bed-44d4-833b-02036f6889dc)

## How to run:

### Requirements
Install the required libraries using the following command:

```
pip install -r requirements.txt
```

### Train:
Run the train script using the following command:

```
./scripts/train.sh
```

### Train:
Run the test script using the following command:

```
./scripts/test.sh
```



