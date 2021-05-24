from azureml.core import Model, Workspace, Experiment, ScriptRunConfig
from azureml.core.run import Run
from azureml.core import Dataset as AzureDataset
from azureml.train.dnn import TensorFlow
from azureml.core.compute import ComputeTarget
from azureml.core.compute_target import ComputeTargetException
from azureml.widgets import RunDetails
from azure.storage.blob import BlobServiceClient


import tensorflow as tf
import os


class AzureLoggingCallback(tf.keras.callbacks.Callback):

    """ Keras callback for azure logging """

    def __init__(self, azure_run):
        self.azure_run = azure_run

    def on_epoch_end(self, epoch, logs=None):
        train_loss = logs['loss']
        valid_loss = logs['val_loss']

        # Common metrics
        self.azure_run.log_table("Validation loss", {'epoch': [epoch], 'loss': [valid_loss]})
        self.azure_run.log_table("Train loss", {'epoch': [epoch], 'loss': [train_loss]})
        self.azure_run.log_table("Learning Rate", {'epoch': [epoch], 'learning rate': [logs['lr']]})


""" SETTING UP GPU NUMBER FOR TENSORFLOW """

try:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(physical_devices[1], 'GPU')
    logical_devices = tf.config.list_logical_devices('GPU')
    # Logical device was not created for first GPU
    assert len(logical_devices) == len(physical_devices) - 1
except Exception as err:
    print('Bullshit happened when tried to set up GPU number!')
    print(err)

""" CREATE AN AzureML EXPERIMENT """

workspace = Workspace.from_config() # local

experiment = Experiment(workspace = workspace, name = 'experiment_name')

run = experiment.start_logging(outputs = None, snapshot_directory = '.')

""" RUN TAGS """
run.set_tags({'type': 'self-supervised', 'input_size': '224x224x3',
              'loss': 'cross-entropy', 'network': 'resnet50',
              'batch_size': '4', 'learning_rate': '0.0001'})


""" RUN EXPERIMENT AS A SCRIPT ON SOME COMPUTE INSTANCE """

compute_target = ComputeTarget(workspace = workspace, name = 'compute_instance_name')

estimator = TensorFlow(source_directory = '.',
                       script_params = None,
                       entry_script = './scripts/train_network.py',
                       pip_packages = ['numpy','opencv-python==4.2.0.34', 'pandas',
                                       'scikit-image', 'addict',
                                       'git+https://github.com/tensorflow/examples.git', 'segmentation-models',
                                       'albumentations'],
                       compute_target = compute_target,
                       use_gpu = True,
                       framework_version = '2.1')

experiment = Experiment(workspace = workspace, name = 'experiment_name')

run = experiment.submit(estimator)

# get details about experiment run
RunDetails(run).show()


""" DATA MOUNTING """

root = os.getcwd()
workspace = Workspace.from_config()

try:
    dataset = AzureDataset.get_by_name(workspace, name = 'my_super_dataset')
    mount_context = dataset.mount(os.path.join(root, 'data'))
    mount_context.start()
except Exception as err:
    print(err)
    print("Can't mount the dataset")

mount_context.stop()


""" MANUALLY STOP RUNNING RUNS INSIDE AN EXPERIMENT """

workspace = Workspace.from_config()
experiment = Experiment(workspace = workspace, name = 'experiment_name')

for run in experiment.get_runs():
    if(run.status == 'Running'):
        #print(run.id)
        run.complete()


""" UPLOAD LOCAL FILE TO AN AZURE BLOB STORAGE """

bs = BlobServiceClient(
    account_url="https://storagename.blob.core.windows.net",
    credential="sdfsdfsdf/sdfsdf/23423asdf2+asdfasdf23==",
)

container_client = bs.get_container_client("data")

file_path = './data/images/image.jpeg'
blob_name = 'image/image_blob.jpeg'

with open(file_path, "rb") as data:
        blob_client = container_client.upload_blob(name=blob_name, data=data)


""" DOWNLOAD BLOB FROM AN AZURE BLOB STORAGE IN BYTES FORM """

bs = BlobServiceClient(
    account_url="https://storagename.blob.core.windows.net",
    credential="sdfsdfsdf/sdfsdf/23423asdf2+asdfasdf23==",
)

container_client = bs.get_container_client("data")

file_path = './data/images/image.jpeg'
blob_name = 'image/image_blob.jpeg'

blob = container_client.download_blob(blob_name)
file_bytes = blob.readall()

