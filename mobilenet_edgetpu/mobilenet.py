import tensorflow as tf
import coremltools as ct

from coremltools.proto.FeatureTypes_pb2 import ArrayFeatureType
import convert_pb_to_saved_model as pb_to_sm

input_height = 224
input_width = 224
input_node = "images"
output_node = "Softmax"
num_classes = 1001

pb_to_sm.convert_pb_to_server_model('frozen_pb/edge_frozen_graph.pb', 'saved_model', input_node+":0", output_node+":0")

m = tf.saved_model.load("saved_model")
p = m.prune(input_node+":0", [output_node+":0"])

inputs=[ct.ImageType(name=input_node, shape=(1, input_height, input_width, 3), bias=[-1, -1, -1], scale=1/127.5)]
mobilenet_edgetpu_original = ct.convert(
    [p],
    "tensorflow",
    inputs=inputs,
    classifier_config=ct.ClassifierConfig("frozen_pb/labels.txt")
)

mobilenet_edgetpu_original.save("MobilenetEdgeTPU.mlmodel")
model = ct.models.MLModel("MobilenetEdgeTPU.mlmodel")

model.author = u"Converted by Koan-Sin Tan. Original Authers: Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam"
model.short_description = 'MobileNet EdgeTPUconverted from https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz. Detects the dominant objects present in an image from a set of 1000 categories such as trees, animals, food, vehicles, person etc.'
model.license = 'Apache License. More information available at https://github.com/tensorflow/models/blob/master/LICENSE '
model.input_description[input_node] = 'Input image to be classified'
model.output_description['classLabel'] = 'Most likely image category'
model.output_description[output_node] = 'Probability'
model.save("MobilenetEdgeTPU.mlmodel")

inputs_multiarray = [ct.TensorType(name=input_node, shape=(1, input_height, input_width, 3))]
mobilenet_edgetpu_original = ct.convert(
    [p],
    "tensorflow",
    inputs=inputs_multiarray,
)

mobilenet_edgetpu_original.save("MobilenetEdgeTPU_multi_arrays.mlmodel")
model = ct.models.MLModel("MobilenetEdgeTPU_multi_arrays.mlmodel")

model.author = u"Converted by Koan-Sin Tan. Original Authers: Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam"
model.short_description = 'MobileNet EdgeTPUconverted from https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz. Detects the dominant objects present in an image from a set of 1000 categories such as trees, animals, food, vehicles, person etc.'
model.license = 'Apache License. More information available at https://github.com/tensorflow/models/blob/master/LICENSE '
model.input_description[input_node] = 'Input image to be classified'
model.output_description[output_node] = 'Probability'

model.save("MobilenetEdgeTPU_multi_arrays.mlmodel")
spec = model.get_spec()

spec.description.output[0].type.multiArrayType.dataType = ArrayFeatureType.FLOAT32
spec.description.output[0].type.multiArrayType.shape.append(1)
spec.description.output[0].type.multiArrayType.shape.append(num_classes)

model = ct.models.MLModel(spec)
model.save("MobilenetEdgeTPU_multi_arrays.mlmodel")
