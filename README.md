You may want to remove/change wandb settings You will need to create a wandb account and project at https://app.wandb.ai/ in order to use wandb

# Installation 

## Packages
```
pip install -r requirements.txt
```

## Get the training data

Go to https://www.dropbox.com/s/opuvvdv3uligypx/fer2013.tar and click Download.

Move the file into the root directory of this project and then run

```
tar xf fer2013.tar
```

# Training

```
wandb run python train_emotion_classifier.py
```

# Testing
```
python run_classifier.py test.jpg
```

# Things to try:

## Look at the data!
  - How is it distributed?  How well would a random model do?
  - Can the model learn on a tiny subset of the data?
## Better models
  - Better architecture
## Data cleanup
  - Normalize
## Data generation
  - https://keras.io/preprocessing/image/
## Reduce learning rate on plateau
  - https://keras.io/callbacks/#reducelronplateau
## Find more training data online?
## Anything else?
