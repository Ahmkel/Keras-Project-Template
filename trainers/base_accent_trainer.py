from keras_preprocessing.image import ImageDataGenerator

from base.base_trainer import BaseTrain
import os
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping


class BaseAccentTrainer(BaseTrain):
    def __init__(self, model, data, config):
        super(BaseAccentTrainer, self).__init__(model, data, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_callbacks()

    def init_callbacks(self):
        # Stops training if accuracy does not change at least 0.005 over 10 epochs
        self.callbacks.append(
            EarlyStopping(monitor='acc', min_delta=.005, patience=10, verbose=1, mode='auto')
        )

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

        if hasattr(self.config.api, "telegram"):
            from bot.dl_bot import DLBot
            from bot.telegram_bot_callback import TelegramBotCallback

            # Create a DLBot instance
            user_id = None if not self.config.api.telegram.user_id else self.config.api.telegram.user_id
            bot = DLBot(token=self.config.api.telegram.token,
                        user_id=user_id)
            # Create a TelegramBotCallback instance
            self.callbacks.append(TelegramBotCallback(bot))
        #
        # # # Creates log file for graphical interpretation using TensorBoard
        # tb = TensorBoard(log_dir='../logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=True,
        #                  write_images=True, embeddings_freq=0, embeddings_layer_names=None,
        #                  embeddings_metadata=None)

        # log experiments to comet.ml
        if hasattr(self.config.api, "comet"):
            from comet_ml import Experiment
            experiment = Experiment(api_key=self.config.api.comet.api_key,
                                    project_name=self.config.api.comet.exp_name)
            experiment.disable_mp()
            experiment.log_parameters(self.config)
            self.callbacks.append(experiment.get_callback('keras'))

    def train(self):

        # Image shifting
        # used to augement in the input data
        datagen = ImageDataGenerator(width_shift_range=0.05)

        # steps per epoch is the number of rounds the generator goes within one epoch
        steps_per_epoch = len(self.data[0]) / self.config.trainer.batch_size

        # using a generator to load the data
        history = self.model.fit_generator(
            datagen.flow(self.data[0], self.data[1],
                         batch_size=self.config.trainer.batch_size),
            self.data[0], self.data[1],
            epochs=self.config.trainer.num_epochs,
            steps_per_epoch=steps_per_epoch,
            verbose=self.config.trainer.verbose_training,
            validation_split=self.config.trainer.validation_split,
            callbacks=self.callbacks,
        )
        #
        # self.loss.extend(history.history['loss'])
        # self.acc.extend(history.history['acc'])
        # self.val_loss.extend(history.history['val_loss'])
        # self.val_acc.extend(history.history['val_acc'])

        # # Fit model using ImageDataGenerator
        # self.model.fit_generator(datagen.flow(self.data[0], y_train, batch_size=batch_size),
        #                          steps_per_epoch=len(X_train) / 32
        #                          , epochs=EPOCHS,
        #                          callbacks=[es, tb, bot], validation_data=(X_validation, y_validation))


