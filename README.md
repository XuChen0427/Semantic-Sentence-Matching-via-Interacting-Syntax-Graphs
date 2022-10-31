# Syntactic-Informed-Graph-Networks-for-Sentence-Matching
## Xu Chen, joint Ph.D. student of Renming University of China, GSAI and University of Montreal, RALI
Any question, please mail to xc_chen@ruc.edu.cn or chen.xu@umontreal.ca\
Implementation of Semantic Sentence Matching via Interacting Syntax Graphs in COLING 2022

# Note for the additional package:
We only provide the small dataset: Scitail,
For QQP and SNLI dataset, please download from https://www.kaggle.com/c/quora-question-pairs and https://nlp.stanford.edu/projects/snli

For the semantic parser, please donwload the semantic parser model to ~/sugar_model\
url: https://github.com/yzhangcs/parser

please download the BERT/RoBERTa model to ~/modeling_bert/\
url: https://github.com/huggingface/transformers

please download the stanford CoreNLP parser to any dir\
url: https://stanfordnlp.github.io/CoreNLP/

The graph matching module is provided by ThinkLab at Shanghai Jiao Tong University
Any wrong for the module, please ref the following link
url: https://github.com/Thinklab-SJTU/ThinkMatch

## 1. enter the stanford CoreNLP parser dir:
 ```bash
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port 9000 -port 9000 -timeout 1500000
 ```

## 2. change the data dir to your own at configs/ and set your parameters

## 3. Preprepare the syntactic structures:
```bash
python build_graph.py configs/scitail.json5
```
note that we prepare the parsed scitail data to graphs/, if you need extra dataset, please run the parser

## 4. train the model:
```bash
python train.py configs/scitail.json5
```

##For citation, please cite the following bib
```
@inproceedings{xu-etal-2022-semantic,
title = "Semantic Sentence Matching via Interacting Syntax Graphs",
author = "Xu, Chen  and
Xu, Jun  and
Dong, Zhenhua  and
Wen, Ji-Rong",
booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
month = oct,
year = "2022",
address = "Gyeongju, Republic of Korea",
publisher = "International Committee on Computational Linguistics",
url = "https://aclanthology.org/2022.coling-1.78",
pages = "938--949",
}
```

