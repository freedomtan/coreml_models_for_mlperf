# MOSAIC Image Segmentation

![](mosaic/mosaic_ade20k_in_xcode_model_preview.png)

* [frozen_pb](mosaic/frozen_pb): frozen_pb from [mlperf mobile wg](https://github.com/mlcommons/mobile_open/tree/main/vision/mosaic/models_and_checkpoints/R4)
* [saved_model](mosaic/saved_model): converted from frozen pb with [convert_pb_to_saved_model.py](convert_pb_to_saved_model.py)

* [mosaic.mlmodel](mosaic/mosaic.mlmodel): input: 512x512x3 RGB, output: 513x513 INT32, works with Xcode model preview
* [mosaic_multi_arrays.mlmodel](mosaic/mosaic.mlmodel): input: 512x512x3 float32, output: 513x513x1 INT32
