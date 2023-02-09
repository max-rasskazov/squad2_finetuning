# solving SQuAD v2.0 task with pretrained language models

## SQuAD v2.0 task
Stanford Question Answering Dataset is a dataset for an extractive question-answering task, consisting of questions posed by crowdworkers on a set of Wikipedia articles where the answer to every question is a segment of text, or span, from the corresponding reading passage. In version 2.0, some questions might be unanswerable, unlike in the previous version 1.1.  
https://rajpurkar.github.io/SQuAD-explorer/

## About this solution
Here is a solution built on fine-tuning a pretrained neural language model for the task of question answering.

### Base Model
The backbone of this model is a BERT-like transformer model DeBERTa V3 in its smallest size with ~21M parameters in a transformer part and ~49M parameters in embeddings. 
https://arxiv.org/abs/2111.09543

### Stability
Some effort was put to counteract a known issue of instability while fine-tuning transformer architectures. This problem manifests in a big variation of model performance for similar training hyperparameter sets and even for the same set with different random seeds.
https://arxiv.org/pdf/2006.04884.pdf

#### Layer-wise learning rate decay
For the first 3/4 of the training, the top layers had a higher learning rate than the lower ones. The learning rate was exponentially decreased from the top layers to the bottom ones.   

#### Top layers reinitialization
Three top layers were reinitialized using the original initialization.

#### Stochastic Weight Averaging
During the last 1/4 of the training frequent snapshots of the weights were taken. The resulting model is the product of averaging those snapshots.
https://arxiv.org/pdf/1803.05407.pdf

### Comparison to existing fine-tuned models
There is a rich variety of fine-tuned transformer models solving the  SQuAD v2.0 task.
This model was compared to three other fine-tuned models for SQuAD2:
* [deepset/tinyroberta-squad2](https://huggingface.co/deepset/tinyroberta-squad2) - another small fine-tuned model
* [deepset/deberta-v3-large-squad2](https://huggingface.co/deepset/deberta-v3-large-squad2) - a bigger version of the same backbone model holding the best result on [paperswithcode leaderboard](https://paperswithcode.com/sota/question-answering-on-squad-v2)
* [nlpconnect/deberta-v3-xsmall-squad2](https://huggingface.co/nlpconnect/deberta-v3-xsmall-squad2) - the same pretrained model fine-tuned by someone else

The evaluation was conducted on the dev set using the official [SQuAD evaluation script](https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/)

|    | params.experiment_id                           |   metrics.n_params |   metrics.n_params_wo_embeddings |   metrics.best_f1 |   metrics.best_exact |
|---:|:-----------------------------------------------|-------------------:|---------------------------------:|------------------:|---------------------:|
|  3 | pretrained/deepset/tinyroberta-squad2          |           81529346 |                         42528770 |           81.4648 |              78.8006 |
|  2 | pretrained/deepset/deberta-v3-large-squad2     |          434014210 |                        302837762 |           88.7495 |              85.5723 |
|  1 | pretrained/nlpconnect/deberta-v3-xsmall-squad2 |           70682882 |                         21491714 |           81.6735 |              78.9017 |
|  0 | finetuned                                      |           70682882 |                         21491714 |           77.9783 |              75.6085 |


### Deployment
This solution's model was deployed using huggingface spaces and can be played with:
https://huggingface.co/spaces/fungi/QA-deberta-v3-xsmall-squad2

The model is available on the huggingface model hub:
https://huggingface.co/fungi/deberta-v3-xsmall-squad2
