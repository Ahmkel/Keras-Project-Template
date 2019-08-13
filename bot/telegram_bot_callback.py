""" Deep Learning Telegram bot
DLBot and TelegramBotCallback classes for the monitoring and control
of a Keras \ Tensorflow training process using a Telegram bot
By: Eyal Zakkay, 2019
https://eyalzk.github.io/
"""

from keras.callbacks import Callback
import keras.backend as K

from accent_detector.dl_bot import DLBot


class TelegramBotCallback(Callback):
    """Callback that sends metrics and responds to Telegram Bot.
    Supports the following commands:
     /start: activate automatic updates every epoch
     /help: get a reply with all command options
     /status: get a reply with the latest epoch's results
     /getlr: get a reply with the current learning rate
     /setlr: change the learning rate (multiply by a factor of 0.5,0.1,2 or 10)
     /plot: get a reply with the loss convergence plot image
     /quiet: stop getting automatic updates each epoch
     /stoptraining: kill Keras training process

    # Arguments
        kbot: Instance of the DLBot class, holding the appropriate bot token

    # Raises
        TypeError: In case kbot is not a DLBot instance.
    """

    def __init__(self, kbot):
        assert isinstance(kbot, DLBot), 'Bot must be an instance of the DLBot class'
        super(TelegramBotCallback, self).__init__()
        self.kbot = kbot


    def on_train_begin(self, logs=None):
        logs['lr'] = K.get_value(self.model.optimizer.lr)  # Add learning rate to logs dictionary
        self.kbot.lr = logs['lr']  # Update bot's value of current LR
        self.kbot.activate_bot()  # Activate the telegram bot
        self.epochs = self.params['epochs']  # number of epochs
        # loss history tracking
        self.loss_hist = []
        self.val_loss_hist = []

    def on_train_end(self, logs=None):
        self.kbot.send_message('Train Completed!')
        self.kbot.stop_bot()

    def on_epoch_begin(self, epoch, logs=None):
        # Check if learning rate should be changed
        if self.kbot.modify_lr != 1:
            if not hasattr(self.model.optimizer, 'lr'):
                raise ValueError('Optimizer must have a "lr" attribute.')
            lr = float(K.get_value(self.model.optimizer.lr))  # get current lr
            # new LR
            lr = lr*self.kbot.modify_lr
            K.set_value(self.model.optimizer.lr, lr)
            self.kbot.modify_lr = 1  # Set multiplier back to 1

            message = '\nEpoch %05d: setting learning rate to %s.' % (epoch + 1, lr)
            print(message)
            self.kbot.send_message(message)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # Did user invoke STOP command
        if self.kbot.stop_train_flag:
            self.model.stop_training = True
            self.kbot.send_message('Training Stopped!')
            print('Training Stopped! Stop command sent via Telegram bot.')

        # LR handling
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        self.kbot.lr = logs['lr']  # Update bot's value of current LR

        # Epoch message handling
        tlogs = ', '.join([k+': '+'{:.4f}'.format(v) for k, v in zip(logs.keys(), logs.values())])  # Clean logs string
        message = 'Epoch %d/%d \n' % (epoch + 1, self.epochs) + tlogs
        # Send epoch end logs
        if self.kbot.verbose:
            self.kbot.send_message(message)
        # Update status message
        self.kbot.set_status(message)

        # Loss tracking
        # Track loss to export as an image
        self.loss_hist.append(logs['loss'])
        if 'val_loss' in logs:
            self.val_loss_hist.append(logs['val_loss'])
        self.kbot.loss_hist = self.loss_hist
        self.kbot.val_loss_hist = self.val_loss_hist
