"""
Callback inspired in https://github.com/mlflow/mlflow/blob/master/examples/hyperparam/train.py
"""
import math
import keras
import mlflow
import mlflow.keras


# train_losses = ['regression_loss', 'classification_loss', 'loss']
# val_metric = 'mAP'
# minimize_val_metric = False
# other_metrics = 'lr'


class MLflowCheckpoint(keras.callbacks.Callback):
    """
        Example of Keras MLflow logger.
        Logs training metrics and final model with MLflow.
        We log metrics provided by Keras during training and keep track of the best model (best loss
        on validation dataset). Every improvement of the best model is also evaluated on the test set.
        At the end of the training, log the best model with MLflow.
        """
    def __init__(self, train_losses, val_metric, minimize_val_metric, other_metrics=None):
        self.train_losses = train_losses
        self.val_metric = val_metric
        self.minimize_val_metric = minimize_val_metric
        self.other_metrics = other_metrics
        if self.minimize_val_metric:
            self._best_val_metric = math.inf
        else:  # maximize val metric
            self._best_val_metric = -math.inf
        self._best_model = None
        super(MLflowCheckpoint, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        """
        Log Keras metrics with MLflow. If model improved on the validation data, evaluate it on
        a test set and store it as the best model.
        """
        if not logs:
            return

        for train_loss in self.train_losses:
            mlflow.log_metric('train_{}'.format(train_loss), logs[train_loss])

        val_metric_value = logs[self.val_metric]
        mlflow.log_metric('validation_{}'.format(self.val_metric), val_metric_value)

        if self.other_metrics:
            for metric in self.other_metrics:
                mlflow.log_metric(metric, logs[metric])

        if (val_metric_value < self._best_val_metric and self.minimize_val_metric) or \
           (val_metric_value > self._best_val_metric and not self.minimize_val_metric):
            # The result improved in the validation set.
            # Log the model with mlflow and also evaluate and log on test set.
            self._best_val_metric = val_metric_value
            mlflow.keras.log_model(self.model, "best_model")
