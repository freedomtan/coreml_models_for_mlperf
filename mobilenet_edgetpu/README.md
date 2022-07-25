# MobileNet EdgeTPU

* [frozen_pb](frozen_pb): frozen pb from https://github.com/mlcommons/mobile_open/blob/main/vision/mobilenet/models_and_code/checkpoints/float/edge_frozen_graph.pb
* [saved_model](saved_model): converted from frozen pb
* [mobilenet.py](mobilenet.py): converting frozen pb to saved_model, then to [MoiblenetEdgeTPU.model](MoiblenetEdgeTPU.mlmodel) and [MoiblenetEdgeTPU_multi_arrays.mlmodel](MoiblenetEdgeTPU_multi_arrays.mlmodel)
