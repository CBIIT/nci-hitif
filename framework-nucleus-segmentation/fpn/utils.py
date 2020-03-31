
from keras.callbacks import Callback
import numpy as np

class Checkpoints(Callback):
    def __init__(self, filepath, monitor='val_loss', verbose=0, mode='auto'):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.filepath = filepath
        self.verbose = verbose
        
        if mode not in ['auto', 'min', 'max']:
          logging.warning('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.', mode)
          mode = 'auto'
          
          
        if mode == 'min':
          self.monitor_op = np.less
          self.best = np.Inf
        elif mode == 'max':
          self.monitor_op = np.greater
          self.best = -np.Inf
        else:
          if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
            self.monitor_op = np.greater
            self.best = -np.Inf
          else:
            self.monitor_op = np.less
            self.best = np.Inf
            
    def set_model(self, model):
      self.model = model
      
    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
          logging.warning('Can save best model only with %s available, '
                          'skipping.', self.monitor)
        else:
          if self.monitor_op(current, self.best):
             filepath = self.filepath.format(epoch=epoch + 1, **logs)
             if self.verbose > 0:
                print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                      ' saving model to %s' % (epoch + 1, self.monitor, 
                                               self.best,
                                               current, filepath))
             self.best = current
             np.save(filepath, self.model.get_weights()) 
          else:
            if self.verbose > 0:
              print('\nEpoch %05d: %s did not improve from %0.5f' %
                    (epoch + 1, self.monitor, self.best))            
