from base.base_trainer import BaseTrain
import os
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard


class SimpleMnistModelTrainer(BaseTrain):
    def __init__(self, model, data, config):
        super(SimpleMnistModelTrainer, self).__init__(model, data, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_callbacks()

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.checkpoint_dir, '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp_name),
                monitor=self.config.checkpoint_monitor,
                mode=self.config.checkpoint_mode,
                save_best_only=self.config.checkpoint_save_best_only,
                save_weights_only=self.config.checkpoint_save_weights_only,
                verbose=self.config.checkpoint_verbose,
            )
        )

        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.tensorboard_log_dir,
                write_graph=self.config.tensorboard_write_graph,
            )
        )
    
    def train(self):
        history = self.model.fit(
            self.data[0], self.data[1],
            epochs=self.config.num_epochs,
            verbose=self.config.verbose_training,
            batch_size=self.config.batch_size,
            validation_split=self.config.validation_split,
            callbacks=self.callbacks,
        )
        self.loss.extend(history.history['loss'])
        self.acc.extend(history.history['acc'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_acc.extend(history.history['val_acc'])
