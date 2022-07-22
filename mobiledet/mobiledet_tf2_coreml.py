#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import coremltools as ct
import numpy as np

input_height = 320
input_width = 320

input_node = 'Preprocessor/map/TensorArrayStack/TensorArrayGatherV3'
bbox_output_node = 'Squeeze'
class_output_node = 'Postprocessor/convert_scores'

original_model = tf.saved_model.load('mobiledet')
pruned_model = original_model.prune(input_node+':0', [bbox_output_node+':0', class_output_node+':0'])

inputs=[ct.ImageType(name=input_node, shape=(1, input_height, input_width, 3), bias=[-1, -1, -1], scale=1/127.5)]
ssd_model = ct.convert([pruned_model], source='tensorflow', inputs=inputs)

spec = ssd_model.get_spec()

spec.description.input[0].name = "image"
spec.description.output[0].name = "scores"
spec.description.output[0].shortDescription = "Predicted class scores for each bounding box"
spec.description.output[1].name = "boxes"
spec.description.output[1].shortDescription = "Predicted coordinates for each bounding box"

spec.neuralNetwork.preprocessing[0].featureName = "image"

for i in range(len(spec.neuralNetwork.layers)):
    # print(spec.neuralNetwork.layers[i].input)
    if len(spec.neuralNetwork.layers[i].input) > 0:
        if spec.neuralNetwork.layers[i].input[0] == input_node.replace('/','_'):
            spec.neuralNetwork.layers[i].input[0] = "image"

    # print(spec.neuralNetwork.layers[i].output)
    # if len(spec.neuralNetwork.layers[i].output) > 0:
    if spec.neuralNetwork.layers[i].output[0] == class_output_node.replace('/','_'):
        spec.neuralNetwork.layers[i].output[0] = "scores"
    if spec.neuralNetwork.layers[i].output[0] == bbox_output_node.replace('/','_'):
        spec.neuralNetwork.layers[i].output[0] = "boxes"

spec.neuralNetwork.preprocessing[0].featureName = "image"

num_classes = 90
num_anchors = 2034

spec.description.output[0].type.multiArrayType.shape.append(num_anchors)
spec.description.output[0].type.multiArrayType.shape.append(num_classes + 1)

spec.description.output[1].type.multiArrayType.shape.append(num_anchors)
spec.description.output[1].type.multiArrayType.shape.append(4)

ssd_model = ct.models.MLModel(spec)
ssd_model.save('/tmp/mobiledet.mlmodel')


def get_anchors(start_tensor, end_tensor):
    """
    Computes the list of anchor boxes by sending a fake image through the graph.
    Outputs an array of size (4, num_anchors) where each element is an anchor box
    given as [ycenter, xcenter, height, width] in normalized coordinates.
    """
    anchors_model = original_model.prune(start_tensor, [end_tensor])
    box_corners = tf.squeeze(anchors_model(tf.zeros((1, input_height, input_width, 3), tf.uint8)))

    # The TensorFlow graph gives each anchor box as [ymin, xmin, ymax, xmax]. 
    # Convert these min/max values to a center coordinate, width and height.
    ymin, xmin, ymax, xmax = np.transpose(box_corners)
    width = xmax - xmin
    height = ymax - ymin
    ycenter = ymin + height / 2.
    xcenter = xmin + width / 2.
    return np.stack([ycenter, xcenter, height, width])

# Read the anchors into a (4, 2034) tensor.
start_tensor = "image_tensor:0"
anchors_tensor = "Concatenate/concat:0"
anchors = get_anchors(start_tensor, anchors_tensor)
assert(anchors.shape[1] == num_anchors)

from coremltools.models import datatypes
from coremltools.models import neural_network

# MLMultiArray inputs of neural networks must have 1 or 3 dimensions. 
# We only have 2, so add an unused dimension of size one at the back.
input_features = [ ("scores", datatypes.Array(num_anchors, num_classes + 1, 1)),
                   ("boxes", datatypes.Array(num_anchors, 4, 1)) ]

# The outputs of the decoder model should match the inputs of the next
# model in the pipeline, NonMaximumSuppression. This expects the number
# of bounding boxes in the first dimension.
output_features = [ ("raw_confidence", datatypes.Array(num_anchors, num_classes)),
                    ("raw_coordinates", datatypes.Array(num_anchors, 4)) ]

builder = neural_network.NeuralNetworkBuilder(input_features, output_features, use_float_arraytype=True)

# (num_anchors, num_classes+1, 1) --> (1, num_anchors, num_classes+1)
builder.add_permute(name="permute_scores",
                    dim=(0, 3, 1, 2),
                    input_name="scores",
                    output_name="permute_scores_output")

# Strip off the "unknown" class (at index 0).
builder.add_slice(name="slice_scores",
                  input_name="permute_scores_output",
                  output_name="raw_confidence",
                  axis="width",
                  start_index=1,
                  end_index=num_classes + 1)

# (num_anchors, 4, 1) --> (4, num_anchors, 1)

builder.add_permute(name="permute_boxed",
                    dim=(0, 2, 1, 3),
                    input_name="boxes",
                    output_name="permute_boxes_output")

# Grab the y, x coordinates (channels 0-1).
builder.add_slice(name="slice_yx",
                  input_name="permute_boxes_output",
                  output_name="slice_yx_output",
                  axis="channel",
                  start_index=0,
                  end_index=2)

# boxes_yx / 10
builder.add_elementwise(name="scale_yx",
                        input_names="slice_yx_output",
                        output_name="scale_yx_output",
                        mode="MULTIPLY",
                        alpha=0.1)

# Split the anchors into two (2, 2034, 1) arrays.
anchors_yx = np.expand_dims(anchors[:2, :], axis=-1)
anchors_hw = np.expand_dims(anchors[2:, :], axis=-1)

builder.add_load_constant(name="anchors_yx",
                          output_name="anchors_yx",
                          constant_value=anchors_yx,
                          shape=[2, num_anchors, 1])

builder.add_load_constant(name="anchors_hw",
                          output_name="anchors_hw",
                          constant_value=anchors_hw,
                          shape=[2, num_anchors, 1])

# (boxes_yx / 10) * anchors_hw
builder.add_elementwise(name="yw_times_hw",
                        input_names=["scale_yx_output", "anchors_hw"],
                        output_name="yw_times_hw_output",
                        mode="MULTIPLY")

# (boxes_yx / 10) * anchors_hw + anchors_yx
builder.add_elementwise(name="decoded_yx",
                        input_names=["yw_times_hw_output", "anchors_yx"],
                        output_name="decoded_yx_output",
                        mode="ADD")

# Grab the height and width (channels 2-3).
builder.add_slice(name="slice_hw",
                  input_name="permute_boxes_output",
                  output_name="slice_hw_output",
                  axis="channel",
                  start_index=2,
                  end_index=4)

# (boxes_hw / 5)
builder.add_elementwise(name="scale_hw",
                        input_names="slice_hw_output",
                        output_name="scale_hw_output",
                        mode="MULTIPLY",
                        alpha=0.2)

# exp(boxes_hw / 5)
builder.add_unary(name="exp_hw",
                  input_name="scale_hw_output",
                  output_name="exp_hw_output",
                  mode="exp")

# exp(boxes_hw / 5) * anchors_hw
builder.add_elementwise(name="decoded_hw",
                        input_names=["exp_hw_output", "anchors_hw"],
                        output_name="decoded_hw_output",
                        mode="MULTIPLY")

# The coordinates are now (y, x) and (height, width) but NonMaximumSuppression
# wants them as (x, y, width, height). So create four slices and then concat
# them into the right order.
builder.add_slice(name="slice_y",
                  input_name="decoded_yx_output",
                  output_name="slice_y_output",
                  axis="channel",
                  start_index=0,
                  end_index=1)

builder.add_slice(name="slice_x",
                  input_name="decoded_yx_output",
                  output_name="slice_x_output",
                  axis="channel",
                  start_index=1,
                  end_index=2)

builder.add_slice(name="slice_h",
                  input_name="decoded_hw_output",
                  output_name="slice_h_output",
                  axis="channel",
                  start_index=0,
                  end_index=1)

builder.add_slice(name="slice_w",
                  input_name="decoded_hw_output",
                  output_name="slice_w_output",
                  axis="channel",
                  start_index=1,
                  end_index=2)

builder.add_elementwise(name="concat",
                        input_names=["slice_x_output", "slice_y_output", 
                                     "slice_w_output", "slice_h_output"],
                        output_name="concat_output",
                        mode="CONCAT")

# (4, num_anchors, 1) --> (1, num_anchors, 4)
builder.add_permute(name="permute_output",
                    dim=(0, 3, 2, 1),
                    input_name="concat_output",
                    output_name="raw_coordinates")

decoder_model = ct.models.MLModel(builder.spec)
decoder_model.save("/tmp/Decoder.mlmodel")

nms_spec = ct.proto.Model_pb2.Model()
nms_spec.specificationVersion = 5

for i in range(2):
    decoder_output = decoder_model._spec.description.output[i].SerializeToString()

    nms_spec.description.input.add()
    nms_spec.description.input[i].ParseFromString(decoder_output)

    nms_spec.description.output.add()
    nms_spec.description.output[i].ParseFromString(decoder_output)
    
nms_spec.description.output[0].name = "confidence"
nms_spec.description.output[1].name = "coordinates"

output_sizes = [num_classes, 4]
for i in range(2):
    ma_type = nms_spec.description.output[i].type.multiArrayType
    ma_type.shapeRange.sizeRanges.add()
    ma_type.shapeRange.sizeRanges[0].lowerBound = 0
    ma_type.shapeRange.sizeRanges[0].upperBound = -1
    ma_type.shapeRange.sizeRanges.add()
    ma_type.shapeRange.sizeRanges[1].lowerBound = output_sizes[i]
    ma_type.shapeRange.sizeRanges[1].upperBound = output_sizes[i]
    del ma_type.shape[:]

nms = nms_spec.nonMaximumSuppression
nms.confidenceInputFeatureName = "raw_confidence"
nms.coordinatesInputFeatureName = "raw_coordinates"
nms.confidenceOutputFeatureName = "confidence"
nms.coordinatesOutputFeatureName = "coordinates"
nms.iouThresholdInputFeatureName = "iouThreshold"
nms.confidenceThresholdInputFeatureName = "confidenceThreshold"

default_iou_threshold = 0.3
default_confidence_threshold = 0.3
nms.iouThreshold = default_iou_threshold
nms.confidenceThreshold = default_confidence_threshold

nms.pickTop.perClass = True

labels = np.loadtxt("mobiledet/coco_labels.txt", dtype=str, usecols=False)
nms.stringClassLabels.vector.extend(labels)

nms_model = ct.models.MLModel(nms_spec)
nms_model.save("/tmp/NMS.mlmodel")


from coremltools.models.pipeline import *

coreml_model_path = "MobileDet.mlmodel"

input_features = [ ("image", datatypes.Array(3, input_height, input_height)),
                   ("iouThreshold", datatypes.Double()),
                   ("confidenceThreshold", datatypes.Double()) ]

output_features = [ "confidence", "coordinates" ]

pipeline = Pipeline(input_features, output_features)

# We added a dimension of size 1 to the back of the inputs of the decoder 
# model, so we should also add this to the output of the SSD model or else 
# the inputs and outputs do not match and the pipeline is not valid.
ssd_output = ssd_model._spec.description.output
ssd_output[0].type.multiArrayType.shape[:] = [num_anchors, num_classes + 1, 1]
ssd_output[1].type.multiArrayType.shape[:] = [num_anchors, 4, 1]

pipeline.add_model(ssd_model)
pipeline.add_model(decoder_model)
pipeline.add_model(nms_model)

# The "image" input should really be an image, not a multi-array.
pipeline.spec.description.input[0].ParseFromString(ssd_model._spec.description.input[0].SerializeToString())

# Copy the declarations of the "confidence" and "coordinates" outputs.
# The Pipeline makes these strings by default.
pipeline.spec.description.output[0].ParseFromString(nms_model._spec.description.output[0].SerializeToString())
pipeline.spec.description.output[1].ParseFromString(nms_model._spec.description.output[1].SerializeToString())

# Add descriptions to the inputs and outputs.
pipeline.spec.description.input[1].shortDescription = "(optional) IOU Threshold override"
pipeline.spec.description.input[2].shortDescription = "(optional) Confidence Threshold override"
pipeline.spec.description.output[0].shortDescription = u"Boxes \xd7 Class confidence"
pipeline.spec.description.output[1].shortDescription = u"Boxes \xd7 [x, y, width, height] (relative to image size)"

# Add metadata to the model.
pipeline.spec.description.metadata.versionString = "ssdlite_mobiledet_edgetpu_320x320_coco_2020_05_19"
pipeline.spec.description.metadata.shortDescription = "MobileDet, trained on COCO"
pipeline.spec.description.metadata.author = "Converted to Core ML by Koan-Sin Tan. Original Authors: Yunyang Xiong, Hanxiao Liu, Suyog Gupta, Berkin Akin, Gabriel Bender, Yongzhe Wang, Pieter-Jan Kindermans, Mingxing Tan, Vikas Singh, Bo Chen"
pipeline.spec.description.metadata.license = "https://github.com/tensorflow/models/blob/master/research/object_detection"
  
# Add the list of class labels and the default threshold values too.
user_defined_metadata = {
    "iou_threshold": str(default_iou_threshold),
    "confidence_threshold": str(default_confidence_threshold),
    "classes": ",".join(labels)
}
pipeline.spec.description.metadata.userDefined.update(user_defined_metadata)
pipeline.spec.specificationVersion = 5

final_model = ct.models.MLModel(pipeline.spec)
final_model.save(coreml_model_path)

print(final_model)
print("Done!")

