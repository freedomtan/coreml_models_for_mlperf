import tensorflow as tf
import numpy as np
import coremltools as ct

from coremltools.proto.FeatureTypes_pb2 import ArrayFeatureType

mobilebert_model = ct.convert('mobilebert_squad_savedmodels/float/', source='tensorflow', tags={'serve'})

spec = mobilebert_model.get_spec()

spec.description.input[0].type.multiArrayType.dataType = ArrayFeatureType.INT32
spec.description.input[1].type.multiArrayType.dataType = ArrayFeatureType.INT32
spec.description.input[2].type.multiArrayType.dataType = ArrayFeatureType.INT32

spec.description.output[0].type.multiArrayType.shape.append(1)
spec.description.output[0].type.multiArrayType.shape.append(384)
spec.description.output[1].type.multiArrayType.shape.append(1)
spec.description.output[1].type.multiArrayType.shape.append(384)

spec.description.metadata.versionString = "MobileBERT fp32"
spec.description.metadata.shortDescription = "MobileBERT"
spec.description.metadata.author = "Converted to Core ML by Koan-Sin Tan. Original Authors: Zhiqing Sun, Hongkun Yu, Xiaodan Song, Renjie Liu, Yiming Yang, Denny Zhou"
spec.description.metadata.license = "Apache, https://github.com/mlcommons/mobile_open/blob/main/vision/mosaic/LICENCE.md"

model = ct.models.MLModel(spec)
model.save('MobileBERT.mlmodel')
print(model)
