import json
import pathlib
import shutil

from keras_preprocessing.image import ImageDataGenerator

from base.base_trainer import BaseTrain
import os
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard


class AccentTrainer(BaseTrain):

    def __init__(self, model,
                 training_data,
                 validation_data,
                 config):
        super(AccentTrainer, self).__init__(model,
                                            training_data,
                                            validation_data,
                                            config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.experiment_id = ""
        self.init_callbacks()

    def init_callbacks(self):
        # Stops training if accuracy does not change at least 0.005 over 10 epochs
        # self.callbacks.append(
        #     EarlyStopping(monitor='acc', min_delta=.005, patience=10, verbose=1, mode='auto')
        # )

        self.callbacks.append(
                TensorBoard(
                    log_dir=self.config.callbacks.tensorboard_log_dir,
                    write_graph=self.config.callbacks.tensorboard_write_graph,
                )
        )

        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.callbacks.checkpoint_dir, '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp.name),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                verbose=self.config.callbacks.checkpoint_verbose,
            )
        )

        # log experiments to comet.ml
        if hasattr(self.config.api, "comet"):
            from comet_ml import Experiment
            experiment = Experiment(api_key=self.config.api.comet.api_key,
                                    project_name=self.config.api.comet.exp_name)
            experiment.disable_mp()
            experiment.log_parameters(self.config)
            self.experiment_id = experiment.id
            self.callbacks.append(experiment.get_callback('keras'))

    def train(self):

        # Image shifting
        # used to augement in the input data
        datagen = ImageDataGenerator(width_shift_range=0.05)

        # steps per epoch is the number of rounds the generator goes within one epoch
        steps_per_epoch = len(self.training_data[0]) / self.config.trainer.batch_size

        # using a generator to load the data
        history = self.model.fit_generator(
            datagen.flow(self.training_data[0], self.training_data[1],
                         batch_size=self.config.trainer.batch_size),
            # self.training_data,
            # batch_size=self.config.trainer.batch_size,
            epochs=self.config.trainer.num_epochs,
            steps_per_epoch=steps_per_epoch,
            verbose=self.config.trainer.verbose_training,
            validation_data=(self.validation_data[0], self.validation_data[1]),
            callbacks=self.callbacks,
        )

        self.loss.extend(history.history['loss'])
        self.acc.extend(history.history['acc'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_acc.extend(history.history['val_acc'])

    def save_model(self):

        if not self.experiment_id:
            import datetime
            self.experiment_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        model_type_path = os.path.join("saved_models", self.config.exp.name, self.experiment_id)

        pathlib.Path(model_type_path).mkdir(parents=True, exist_ok=True)

        name = "model.h5"
        model_path = os.path.join(model_type_path, name)
        self.model.save(model_path)

        self.copy_context()

    def copy_context(self):
        model_type_path = os.path.join("saved_models", self.config.exp.name, self.experiment_id)

        pathlib.Path(model_type_path).mkdir(parents=True, exist_ok=True)

        # copying CSV file
        csv_name = self.config.data_loader.data_file
        path_from = os.path.join("datasets/training_files", csv_name)
        path_to = os.path.join(model_type_path, csv_name)
        shutil.copy(path_from, path_to)

        # dumping the config
        json_config = json.dumps(self.config.toDict(), indent=4)
        path_to = os.path.join(model_type_path, "config.json")
        with open(path_to, "w") as f:
            f.write(json_config)
