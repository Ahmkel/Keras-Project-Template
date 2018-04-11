# Keras Project Template
A project template to simplify building and training deep learning models using Keras.

# Table of contents

- [Getting Started](#getting-started)
- [Running The Demo Project](#running-the-demo-project)
- [Template Details](#template-details)
    - [Project Architecture](#project-architecture)
    - [Folder Structure](#folder-structure)
    - [Main Components](#main-components)
- [Future Work](#future-work)
- [Other Examples](#more-advanced-example)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

# Getting Started
This template allows you to simply build and train deep learning models with checkpoints and tensorboard visualization.

In order to use the template you have to:
1. Define a data loader class.
2. Define a model class that inherits from BaseModel.
3. Define a trainer class that inherits.
4. Define a configuration file with the parameters needed in an experiment.
5. Run the model using:
```shell
python main.py -c [path to configuration file]
```

# Running The Demo Project
A simple model for the mnist dataset is available to test the template.
To run the demo project:
1. Start the training using:
```shell
python main.py -c configs/simple_mnist_config.json
```
2. Start Tensorboard visualization using:
```shell
tensorboard --logdir=experiments/simple_mnist/logs
```

<div align="center">

<img align="center" width="600" src="https://github.com/Ahmkel/Keras-Project-Template/blob/master/figures/Tensorboard_demo.PNG?raw=true">

</div>


# Template Details

## Project Architecture

<div align="center">

<img align="center" width="600" src="https://github.com/Ahmkel/Keras-Project-Template/blob/master/figures/ProjectArchitecture.jpg?raw=true">

</div>


## Folder Structure

```
├── main.py             - here's an example of main that is responsible for the whole pipeline.
│
│
├── base                - this folder contains the abstract classes of the project components
│   ├── base_data_loader.py   - this file contains the abstract class of the data loader.
│   ├── base_model.py   - this file contains the abstract class of the model.
│   └── base_train.py   - this file contains the abstract class of the trainer.
│
│
├── model               - this folder contains the models of your project.
│   └── simple_mnist_model.py
│
│
├── trainer             - this folder contains the trainers of your project.
│   └── simple_mnist_trainer.py
│
|
├── data_loader         - this folder contains the data loaders of your project.
│   └── simple_mnist_data_loader.py
│
│
├── configs             - this folder contains the experiment and model configs of your project.
│   └── simple_mnist_config.json
│
│
├── datasets            - this folder might contain the datasets of your project.
│
│
└── utils               - this folder contains any utils you need.
     ├── config.py      - util functions for parsing the config files.
     ├── dirs.py        - util functions for creating directories.
     └── utils.py       - util functions for parsing arguments.
```


## Main Components

### Models
You need to:
1. Create a model class that inherits from **BaseModel**.
2. Override the ***build_model*** function which defines your model.
3. Call ***build_model*** function from the constructor.


### Trainers
You need to:
1. Create a trainer class that inherits from **BaseTrainer**.
2. Override the ***train*** function which defines the training logic.

**Note:** To add functionalities after each training epoch such as saving checkpoints or logs for tensorboard using Keras callbacks:
1. Declare a callbacks array in your constructor.
2. Define an ***init_callbacks*** function to populate your callbacks array and call it in your constructor.
3. Pass the callbacks array to the ***fit*** function on the model object.

**Note:** You can use ***fit_generator*** instead of ***fit*** to support generating new batches of data instead of loading the whole dataset at one time.

### Data Loaders
You need to:
1. Create a data loader class that inherits from **BaseDataLoader**.
2. Override the ***get_train_data()*** and the ***get_test_data()*** functions to return your train and test dataset splits.

**Note:** You can also define a different logic where the data loader class has a function ***get_next_batch*** if you want the data reader to read batches from your dataset each time.

### Configs
You need to define a .json file that contains your experiment and model configurations such as the experiment name, the batch size, and the number of epochs.


### Main
Responsible for building the pipeline.
1. Parse the config file
2. Create an instance of your data loader class.
3. Create an instance of your model class.
4. Create an instance of your trainer class.
5. Train your model using ".Train()" function on the trainer object.

# Example Projects
* [Toxic comments classification using Convolutional Neural Networks and Word Embedding](https://github.com/Ahmkel/Toxic-Comments-Competition-Kaggle)


# Future Work
Create a command line tool for Keras project scaffolding where the user defines a data loader, a model, a trainer and runs the tool to generate the whole project.


# Contributing
Any contributions are welcome including improving the template and example projects.

# Acknowledgements
This project template is based on [MrGemy95](https://github.com/MrGemy95)'s [Tensorflow Project Template](https://github.com/MrGemy95/Tensorflow-Project-Template).


Thanks for my colleagues [Mahmoud Khaled](https://github.com/MahmoudKhaledAli), [Ahmed Waleed](https://github.com/Rombux) and [Ahmed El-Gammal](https://github.com/AGammal) who worked on the initial project that spawned this template.
