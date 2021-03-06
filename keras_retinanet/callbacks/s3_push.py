"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import datetime
import os

import keras

from ..utils import s3_tools as s3t


class S3StoreResults(keras.callbacks.Callback):
    def __init__(
        self,
        s3_bucket,
        snapshots_dir=None,
        tensorboard_dir=None,
        verbose=1
    ):
        """ Callback for storing tensorboard logs and model snapsots to S3.

        # Arguments
            verbose          : Set the verbosity level, by default this is set to 1.
        """
        self.s3_bucket = s3_bucket
        self.snapshots_dir = snapshots_dir
        self.tensorboard_dir = tensorboard_dir
        self.verbose = verbose

        training_job_name = os.environ.get('TRAINING_JOB_NAME', datetime.datetime.utcnow().strftime("%Y%m%d_%H%M"))
        self.s3_root_key = 'machine_learning/models/retinanet/{}'.format(training_job_name)
        self.s3_snapshots_key = os.path.join(self.s3_root_key, 'snapshots')
        self.s3_tensorboard_key = os.path.join(self.s3_root_key, 'tensorboard')
        self.uploaded_weights = []

        super(S3StoreResults, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        if self.snapshots_dir is not None:
            for snps_file in os.listdir(self.snapshots_dir):
                src_path = os.path.join(self.snapshots_dir, snps_file)
                if snps_file in self.uploaded_weights:
                    os.remove(src_path)
                    print('Weights {} removed'.format(snps_file))
                else:
                    self.uploaded_weights.append(snps_file)
                    s3t.upload_file_to_bucket(src_path, self.s3_snapshots_key, self.s3_bucket, verbose=self.verbose)

        if self.tensorboard_dir is not None:
            for tnsb_file in os.listdir(self.tensorboard_dir):
                src_path = os.path.join(self.tensorboard_dir, tnsb_file)
                s3t.upload_file_to_bucket(src_path, self.s3_tensorboard_key, self.s3_bucket, verbose=self.verbose)

    def on_train_end(self, logs=None):
        # Just in case last weights are stored locally after this callback is called
        self.on_epoch_end(-1, logs=logs)
