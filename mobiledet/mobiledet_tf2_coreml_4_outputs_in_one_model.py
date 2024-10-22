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

inputs=[ct.TensorType(name=input_node, shape=(1, input_height, input_width, 3))]
ssd_model = ct.convert([pruned_model], source='tensorflow', inputs=inputs)

spec = ssd_model.get_spec()

ct.utils.rename_feature(spec, input_node.replace('/', '_'), 'image')
ct.utils.rename_feature(spec, class_output_node.replace('/', '_'), 'scores')
ct.utils.rename_feature(spec, bbox_output_node.replace('/', '_'), 'boxes')

spec.description.output[0].shortDescription = "Predicted class scores for each bounding box"
spec.description.output[1].shortDescription = "Predicted coordinates for each bounding box"

num_classes = 90
num_anchors = 2034

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
from coremltools.proto.FeatureTypes_pb2 import ArrayFeatureType

spec = ssd_model.get_spec()
builder = neural_network.NeuralNetworkBuilder(spec=spec, use_float_arraytype=True)

builder.add_permute(name="permute_boxed",
                    dim=(0, 3, 2, 1),
                    # input_name="expanded_boxes",
                    input_name='boxes',
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

builder.add_elementwise(name="concat2",
                        input_names=["slice_x_output", "slice_y_output", 
                                     "slice_w_output", "slice_h_output"],
                        output_name="concat_output",
                        mode="CONCAT")

builder.add_permute(name="permute_output",
                    dim=(0, 1, 2, 3),
                    input_name="concat_output",
                    output_name="raw_coordinates")

input_names = ['raw_coordinates', 'scores']
output_names = ['coordinates', 'confidence','box_index', 'number_of_boxes']
builder.add_nms('nms', input_names, output_names, 0.3, 0.3, 10, True)

output_features = [
    ('coordinates', datatypes.Array(1, 10, 4)),
    ('confidence', datatypes.Array(1, 10, num_classes + 1)),    
    ('box_index', datatypes.Array(1, 10)),
    ('number_of_boxes', datatypes.Array(1))
]

spec.description.output.add()
spec.description.output.add()

output_names = ['coordinates', 'confidence', 'box_index', 'number_of_boxes']
output_dims = [(1, 10, 4), (1, 10, 91), (1, 10), (1,)]
builder.set_output(output_names, output_dims)
for i in range(4):
    spec.description.output[i].type.multiArrayType.dataType = ArrayFeatureType.FLOAT32

spec.description.metadata.versionString = "ssdlite_mobiledet_edgetpu_320x320_coco_2020_05_19"
spec.description.metadata.shortDescription = "MobileDet, trained on COCO"
spec.description.metadata.author = "Converted to Core ML by Koan-Sin Tan. Original Authors: Yunyang Xiong, Hanxiao Liu, Suyog Gupta, Berkin Akin, Gabriel Bender, Yongzhe Wang, Pieter-Jan Kindermans, Mingxing Tan, Vikas Singh, Bo Chen"
spec.description.metadata.license = "https://github.com/tensorflow/models/blob/master/research/object_detection"


final_model = ct.models.MLModel(builder.spec)
final_model.save('MobileDet_4_outputs_in_one_model.mlmodel')

print(final_model)
print("Done!")

