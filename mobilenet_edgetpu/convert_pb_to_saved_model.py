import os
import shutil

import tensorflow.compat.v1 as tf1
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

def convert_pb_to_server_model(pb_model_path, export_dir, input_name='input:0', output_name='output:0'):
    graph_def = read_pb_model(pb_model_path)
    convert_pb_saved_model(graph_def, export_dir, input_name, output_name)

def read_pb_model(pb_model_path):
    with tf1.gfile.GFile(pb_model_path, "rb") as f:
        graph_def = tf1.GraphDef()
        graph_def.ParseFromString(f.read())
        return graph_def

def convert_pb_saved_model(graph_def, export_dir, input_name='input:0', output_name='output:0'):
    if os.path.isdir(export_dir):
        shutil.rmtree(export_dir)
    builder = tf1.saved_model.builder.SavedModelBuilder(export_dir)

    sigs = {}
    with tf1.Session(graph=tf1.Graph()) as sess:
        tf1.import_graph_def(graph_def, name="")
        g = tf1.get_default_graph()
        inp = g.get_tensor_by_name(input_name)
        out = g.get_tensor_by_name(output_name)

        sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
            tf1.saved_model.signature_def_utils.predict_signature_def(
                {"input": inp}, {"output": out})

        builder.add_meta_graph_and_variables(sess,
                                             [tag_constants.SERVING],
                                             signature_def_map=sigs)
        builder.save()
