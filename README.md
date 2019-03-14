# EmoContext

A Shared Task on Contextual Emotion Detection in Text.

## Usage

There are two different notebooks. Each of them requires a different configuration to run.   

For `notebooks/EmoContext_DeepMojiModels.ipynb` use the following the method(Just a heads up: installing DeepMoji is not a trivial task).
```
docker build -t emo .
nvidia-docker run -it -v "$PWD":/app -p 8888:8888 emo
```

For `notebooks/EmoContext_Elmo.ipynb`, use [this notebook on colab](https://colab.research.google.com/drive/1NDn4j9wvSewxriGpHWO_NdQYgxzb1Gcd#scrollTo=w6u-c9hA8PVd)


## Files Desprictiom

- `notebooks/EmoContext_DeepMojiModels.ipynb`: Contains experiments using DeepMoji 
- `notebooks/EmoContext_Elmo.ipynb`: Contains experiments using Elmo (Tested on colab only)
- `utills.py`: contains a few important utills
- `data/train.txt`: Our training dataset
- `data/devwithoutlabels.txt`: Our testing dataset
- `models/*py`: contains some old file that are required anymore

## Model Desciption

![Model Photo](https://raw.githubusercontent.com/sedflix/EmoContext/master/results/photo_2019-03-14_11-15-14.jpg)

- Input: DeepMoji embedding of the give sentence.
- Augmentation techniques used: random word switching and removal
- DeepMoji with an increased vocabulary size of 3000 along with an augmented Training Dataset


## Experiment Details:

Details all of ours experiments can be found in the following two documents.
- [Slides(google slides)](https://docs.google.com/presentation/d/1W4-2ZhBncycvIDqbRNiCX5K8UIdH6J9rSvHTv8mhl5Y/edit?usp=sharing) 
- [Report(pdf)](https://github.com/sedflix/EmoContext/raw/master/report.pdf)

## Results

Our best model had a f1 score of ~0.68 on the `devwithoutlabels.txt`.  More details can see [on codalab competitions result](https://competitions.codalab.org/competitions/19790#results). Our team name is `chaicoffee`.  
