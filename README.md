# EmoContext


## Usage
```
docker build -t emo .
nvidia-docker run -it -v "$PWD":/app -p 8888:8888 emo
```

## Files

- `EmoContext_DeepMojiModels.ipynb`: Contains experiments using DeepMoji (This should be run within the docker or you should have DeepMoji installed)
- `EmoContext_Elmo.ipynb`: Contains experiments using Elmo (Test on CoLab)
- `train.txt`: Our training dataset
- `devwithoutlabels.txt`: Our testing dataset
- `utills.py`: contains a few utilss