# MobileDet

It works in Xcode Core ML model preview, ![](mobiledet_in_xcode_model_previvew.png)

Tested with TensorFlow 2.9.1 + coremltools 6.0b1
The saved_model from what [Google contributed to MLPerf Mobile]( https://github.com/mlcommons/mobile/tree/master/vision/mobiledet/fp32/export_inference_graph/saved_model)

Most of the code is derived from @hollance's "[MobileNetV2 + SSDLite with Core ML](https://machinethink.net/blog/mobilenet-ssdlite-coreml/)". Most significant changes are:
* use TF 2.x instead of 1.x (this simplified the feature extraction part and some of the decoding part)
* parameters, e.g., (1, 300, 300, 3) -> (1, 320, 320, 3)


test_images/: images from TensorFlow Objection Dection API, https://github.com/tensorflow/models/tree/master/research/object_detection/test_images
