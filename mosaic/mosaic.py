#!/usr/bin/env python
# coding: utf-8

import json

import tensorflow as tf
import coremltools as ct
import coremltools.proto.FeatureTypes_pb2 as ft
from coremltools.proto.FeatureTypes_pb2 import ArrayFeatureType

pruned = tf.saved_model.load("saved_model").prune("sub_2:0", "ArgMax:0")
inputs = [
    ct.ImageType(name="sub_2",
                 shape=(1, 512, 512, 3),
                 bias=[-1, -1, -1],
                 scale=1 / 127.5)
]

mosaic_original = ct.convert([pruned], "tensorflow", inputs)

spec = mosaic_original.get_spec()

spec.description.input[0].name = "image"
spec.description.output[0].name = "ArgMax"
spec.description.output[0].shortDescription = "ArgMax"
# an issue in coremltools casts int32 to fp16
# set it back.
spec.description.output[0].type.multiArrayType.dataType = ArrayFeatureType.INT32

spec.neuralNetwork.preprocessing[0].featureName = "image"

for i in range(len(spec.neuralNetwork.layers)):
  # print(spec.neuralNetwork.layers[i].input)
  if len(spec.neuralNetwork.layers[i].input) > 0:
    if spec.neuralNetwork.layers[i].input[0] == "sub_2":
      spec.neuralNetwork.layers[i].input[0] = "image"

  # print(spec.neuralNetwork.layers[i].output)
  # if len(spec.neuralNetwork.layers[i].output) > 0:
  if spec.neuralNetwork.layers[i].output[0] == "ArgMax":
    spec.neuralNetwork.layers[i].output[0] = "ArgMax"

spec.neuralNetwork.preprocessing[0].featureName = "image"

model = ct.models.MLModel(spec)

labels_json = {
    "labels": [
        "background",
        "wall",
        "building, edifice",
        "sky",
        "floor, flooring",
        "tree",
        "ceiling",
        "road, route",
        "bed ",
        "windowpane, window ",
        "grass",
        "cabinet",
        "sidewalk, pavement",
        "person, individual, someone, somebody, mortal, soul",
        "earth, ground",
        "door, double door",
        "table",
        "mountain, mount",
        "plant, flora, plant life",
        "curtain, drape, drapery, mantle, pall",
        "chair",
        "car, auto, automobile, machine, motorcar",
        "water",
        "painting, picture",
        "sofa, couch, lounge",
        "shelf",
        "house",
        "sea",
        "mirror",
        "rug, carpet, carpeting",
        "field",
        "armchair",
    ]
}

model.user_defined_metadata[
    "com.apple.coreml.model.preview.type"] = "imageSegmenter"
model.user_defined_metadata[
    "com.apple.coreml.model.preview.params"] = json.dumps(labels_json)
model.save("mosaic.mlmodel")
