import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import device_lib

virtual_device_gpu_options = config_pb2.GPUOptions(
  visible_device_list='0',
  experimental=config_pb2.GPUOptions.Experimental(
    virtual_devices=[config_pb2.GPUOptions.Experimental.VirtualDevices(memory_limit_mb=[200, 300])])
    
config = config_pb2.ConfigProto(gpu_options=virtual_device_gpu_options)

device_lib.list_local_devices(session_config=config)

with tf.Session(config=config) as sess:
  with tf.device('/gpu:1'):
    result = sess.run(tf.constand(42))
