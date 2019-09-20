# Keras Accent Trainer [![CometML](https://img.shields.io/badge/comet.ml-track-brightgreen.svg)](https://www.comet.ml)

### About
Every individual has their own dialects or mannerisms in which they speak. This project revolves around the detection of backgrounds of every individual using their speeches. The goal in this project is to classify various types of accents, specifically foreign accents, by the native language of the speaker. This project allows to detect the demographic and linguistic backgrounds of the speakers by comparing different speech outputs with the speech accent archive dataset in order to determine which variables are key predictors of each accent. The speech accent archive demonstrates that accents are systematic rather than merely mistaken speech. Given a recording of a speaker speaking a known script of English words, this project predicts the speaker’s native language.

### Dataset
All of the speech files used for this project come from the Speech Accent Archive, a repository of spoken English hosted by George Mason University. Over 2000 speakers representing over 100 native languages read a common elicitation paragraph in English:

```
'Please call Stella. Ask her to bring these things with her from the store: Six spoons of fresh
snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob. We also need 
a small plastic snake and a big toy frog for the kids. She can scoop these things into three red 
bags, and we will go meet her Wednesday at the train station.'
```

The dataset contained **.mp3** audio files which were converted to **.wav** audio files which allowed easy extraction of the **MFCC (Mel Frequency Cepstral Coefficients)** features to build a 2-D convolution neural network.

The MFCC was fed into a 2-Dimensional Convolutional Neural Network (CNN) to predict the native language class.

# Running The Demo Project

# Comet.ml Integration
We added integration to Comet.ml which allows you to see all your hyper-params, metrics, graphs, dependencies and more including real-time metric.

Add your API key [in the configuration file](configs/all_english_speakers.json#L15):

For example:  `"{"api": {"comet": {"api_key": "your key here"}}}`

# Project Architecture

<div align="center">

<img align="center" width="600" src="https://github.com/Ahmkel/Keras-Project-Template/blob/master/figures/ProjectArchitecture.jpg?raw=true">

</div>

# Training configurations
- **all_english_speakers.json**
  - The training set contains all speaksers, where USA natives are in one class and all others are matched to the other class as foreign speakers
- **usa_english_speakers.json** 
  - The training set contains only USA natives as one class, without any other english speakers.

# Dockerized environment

Use [keras_accent_deployment](https://github.com/guyeshet/accent_training_deployment) for the fully dockerized environment

# Local Installation

1. First we need to download locally the accent audio files
   ```
   pip install -r requirements.txt
   ```

# Project Execution

1. First we need to download locally the accent audio files. Its a long operation that downloads over 1000 audio files
   ```
   python accent_dataset/create.py
   ```
2. Train the model by the requested configuration. At first it's a long process, as we need to convert the wav
   files into MFCC. The MFCCs are cached so future trainings are faster: 
   ```
   python train_from_config.py -c configs/usa_english_speakers.json
   ```

# Acknowledgements
This project template is based on [Ahmkel](https://github.com/Ahmkel)'s [Keras Project Template](https://github.com/Ahmkel/Keras-Project-Template).

The model and implementation is inspired by [yatharthgarg](https://github.com/yatharthgarg)'s [Speech-Accent-Recognition](https://github.com/yatharthgarg/Speech-Accent-Recognition).
