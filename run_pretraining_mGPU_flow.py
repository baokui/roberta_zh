# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import modeling
import optimization
import tensorflow as tf
import time
import json
from flow.glow_1x1 import AttrDict, Glow
from optimization_bert_flow import AdamWeightDecayOptimizer
import tokenization
flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer(
    "max_predictions_per_seq", 20,
    "Maximum number of masked LM predictions per sequence. "
    "Must match data generation.")

flags.DEFINE_integer(
    "n_gpus", 4,
    "number of gpus")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

# sentence embedding related parameters
flags.DEFINE_string("sentence_embedding_type", "avg-last-2", "avg, cls, ...")

# flow parameters
flags.DEFINE_integer("flow", 1, "use flow or not")
flags.DEFINE_integer("flow_loss", 1, "use flow loss or not")
flags.DEFINE_float("flow_learning_rate", 1e-3, "The initial learning rate for Adam.")
flags.DEFINE_string("flow_model_config", "config_l3_d3_w32", None)

def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    masked_lm_positions = features["masked_lm_positions"]
    masked_lm_ids = features["masked_lm_ids"]
    masked_lm_weights = features["masked_lm_weights"]
    next_sentence_labels = features["next_sentence_labels"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    (masked_lm_loss,
     masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
         bert_config, model.get_sequence_output(), model.get_embedding_table(),
         masked_lm_positions, masked_lm_ids, masked_lm_weights)

    (next_sentence_loss, next_sentence_example_loss, # TODO TODO TODO 可以计算单不算成绩
     next_sentence_log_probs) = get_next_sentence_output(
         bert_config, model.get_pooled_output(), next_sentence_labels)
    # batch_size=masked_lm_log_probs.shape[0]
    # next_sentence_example_loss=tf.zeros((batch_size)) #tf.constant(0.0,dtype=tf.float32)
    # next_sentence_log_probs=tf.zeros((batch_size,2))
    total_loss = masked_lm_loss # TODO remove next sentence loss 2019-08-08, + next_sentence_loss

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    print("init_checkpoint:",init_checkpoint)
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
      logging_hook = tf.train.LoggingTensorHook({"loss": total_loss}, every_n_iter=10)
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          training_hooks=[logging_hook],
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                    masked_lm_weights, next_sentence_example_loss,
                    next_sentence_log_probs, next_sentence_labels):
        """Computes the loss and accuracy of the model."""
        masked_lm_log_probs = tf.reshape(masked_lm_log_probs,[-1, masked_lm_log_probs.shape[-1]])
        masked_lm_predictions = tf.argmax(masked_lm_log_probs, axis=-1, output_type=tf.int32)
        masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
        masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
        masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
        masked_lm_accuracy = tf.metrics.accuracy(
            labels=masked_lm_ids,
            predictions=masked_lm_predictions,
            weights=masked_lm_weights)
        masked_lm_mean_loss = tf.metrics.mean(
            values=masked_lm_example_loss, weights=masked_lm_weights)

        next_sentence_log_probs = tf.reshape(
            next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])
        next_sentence_predictions = tf.argmax(
            next_sentence_log_probs, axis=-1, output_type=tf.int32)
        next_sentence_labels = tf.reshape(next_sentence_labels, [-1])
        next_sentence_accuracy = tf.metrics.accuracy(
            labels=next_sentence_labels, predictions=next_sentence_predictions)
        next_sentence_mean_loss = tf.metrics.mean(
            values=next_sentence_example_loss)

        return {
            "masked_lm_accuracy": masked_lm_accuracy,
            "masked_lm_loss": masked_lm_mean_loss,
            "next_sentence_accuracy": next_sentence_accuracy,
            "next_sentence_loss": next_sentence_mean_loss,
        }

      # next_sentence_example_loss=0.0 TODO
      # next_sentence_log_probs=0.0 # TODO
      eval_metrics = (metric_fn, [
          masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
          masked_lm_weights, next_sentence_example_loss,
          next_sentence_log_probs, next_sentence_labels
      ])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

    return output_spec

  return model_fn


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
  """Get loss and log probs for the masked LM."""
  input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        "output_bias",
        shape=[bert_config.vocab_size],
        initializer=tf.zeros_initializer())
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator

  return (loss, per_example_loss, log_probs)


def get_next_sentence_output(bert_config, input_tensor, labels):
  """Get loss and log probs for the next sentence prediction."""

  # Simple binary classification. Note that 0 is "next sentence" and 1 is
  # "random sentence". This weight matrix is not used after pre-training.
  with tf.variable_scope("cls/seq_relationship"):
    output_weights = tf.get_variable(
        "output_weights",
        shape=[2, bert_config.hidden_size],
        initializer=modeling.create_initializer(bert_config.initializer_range))
    output_bias = tf.get_variable(
        "output_bias", shape=[2], initializer=tf.zeros_initializer())

    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    labels = tf.reshape(labels, [-1])
    one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor


def input_fn_builder(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     num_cpu_threads=4):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    name_to_features = {
        "input_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "masked_lm_positions":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_ids":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_weights":
            tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
        "next_sentence_labels":
            tf.FixedLenFeature([1], tf.int64),
    }

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
      d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
      d = d.repeat()
      d = d.shuffle(buffer_size=len(input_files))

      # `cycle_length` is the number of parallel files that get read.
      cycle_length = min(num_cpu_threads, len(input_files))

      # `sloppy` mode means that the interleaving is not exact. This adds
      # even more randomness to the training pipeline.
      d = d.apply(
          tf.contrib.data.parallel_interleave(
              tf.data.TFRecordDataset,
              sloppy=is_training,
              cycle_length=cycle_length))
      d = d.shuffle(buffer_size=100)
    else:
      d = tf.data.TFRecordDataset(input_files)
      # Since we evaluate for a fixed number of steps we don't want to encounter
      # out-of-range exceptions.
      d = d.repeat()

    # We must `drop_remainder` on training because the TPU requires fixed
    # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
    # and we *don't* want to drop the remainder, otherwise we wont cover
    # every sample.
    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=True))
    return d

  return input_fn


def _decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.to_int32(t)
    example[name] = t

  return example

def input_fn(input_files,
             batch_size,
             max_seq_length,
             max_predictions_per_seq,
             is_training,
             num_cpu_threads=4):
    """The actual input function."""
    # batch_size = params["batch_size"]

    name_to_features = {
        "input_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "masked_lm_positions":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_ids":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_weights":
            tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
        "next_sentence_labels":
            tf.FixedLenFeature([1], tf.int64),
    }

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
        d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
        d = d.repeat()
        d = d.shuffle(buffer_size=len(input_files))

        # `cycle_length` is the number of parallel files that get read.
        cycle_length = min(num_cpu_threads, len(input_files))

        # `sloppy` mode means that the interleaving is not exact. This adds
        # even more randomness to the training pipeline.
        d = d.apply(
            tf.contrib.data.parallel_interleave(
                tf.data.TFRecordDataset,
                sloppy=is_training,
                cycle_length=cycle_length))
        d = d.shuffle(buffer_size=100)
    else:
        d = tf.data.TFRecordDataset(input_files)
        # Since we evaluate for a fixed number of steps we don't want to encounter
        # out-of-range exceptions.
        d = d.repeat()

    # We must `drop_remainder` on training because the TPU requires fixed
    # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
    # and we *don't* want to drop the remainder, otherwise we wont cover
    # every sample.
    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=True))
    return d
def parse_input_fn_result(result):
    """Gets features, labels, and hooks from the result of an Estimator input_fn.
    Args:
      result: output of an input_fn to an estimator, which should be one of:
        * A 'tf.data.Dataset' object: Outputs of `Dataset` object must be a
            tuple (features, labels) with same constraints as below.
        * A tuple (features, labels): Where `features` is a `Tensor` or a
          dictionary of string feature name to `Tensor` and `labels` is a
          `Tensor` or a dictionary of string label name to `Tensor`. Both
          `features` and `labels` are consumed by `model_fn`. They should
          satisfy the expectation of `model_fn` from inputs.
    Returns:
      Tuple of features, labels, and input_hooks, where features are as described
      above, labels are as described above or None, and input_hooks are a list
      of SessionRunHooks to be included when running.
    Raises:
      ValueError: if the result is a list or tuple of length != 2.
    """
    try:
        # We can't just check whether this is a tf.data.Dataset instance here,
        # as this is plausibly a PerDeviceDataset. Try treating as a dataset first.
        iterator = result.make_initializable_iterator()
    except AttributeError:
        # Not a dataset or dataset-like-object. Move along.
        pass
    else:
        result = iterator.get_next()
    return result,iterator


def average_gradients(tower_grads, batch_size, options):
    # calculate average gradient for each shared variable across all GPUs
    average_grads = []
    count = 0
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        # We need to average the gradients across each GPU.
        count += 1
        g0, v0 = grad_and_vars[0]

        if g0 is None:
            # no gradient for this variable, skip it
            average_grads.append((g0, v0))
            continue

        if isinstance(g0, tf.IndexedSlices):
            # If the gradient is type IndexedSlices then this is a sparse
            #   gradient with attributes indices and values.
            # To average, need to concat them individually then create
            #   a new IndexedSlices object.
            indices = []
            values = []
            for g, v in grad_and_vars:
                indices.append(g.indices)
                values.append(g.values)
            all_indices = tf.concat(indices, 0)
            avg_values = tf.concat(values, 0) / len(grad_and_vars)
            # deduplicate across indices
            av, ai = _deduplicate_indexed_slices(avg_values, all_indices)
            grad = tf.IndexedSlices(av, ai, dense_shape=g0.dense_shape)

        else:
            # a normal tensor can just do a simple average
            grads = []
            for g, v in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)
                # Append on a 'tower' dimension which we will average over
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

        # the Variables are redundant because they are shared
        # across towers. So.. just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)

        average_grads.append(grad_and_var)

    assert len(average_grads) == len(list(zip(*tower_grads)))

    return average_grads


def clip_by_global_norm_summary(t_list, clip_norm, norm_name, variables):
    # wrapper around tf.clip_by_global_norm that also does summary ops of norms

    # compute norms
    # use global_norm with one element to handle IndexedSlices vs dense
    norms = [tf.global_norm([t]) for t in t_list]

    # summary ops before clipping
    summary_ops = []
    for ns, v in zip(norms, variables):
        name = 'norm_pre_clip/' + v.name.replace(":", "_")
        summary_ops.append(tf.summary.scalar(name, ns))

    # clip
    clipped_t_list, tf_norm = tf.clip_by_global_norm(t_list, clip_norm)

    # summary ops after clipping
    norms_post = [tf.global_norm([t]) for t in clipped_t_list]
    for ns, v in zip(norms_post, variables):
        name = 'norm_post_clip/' + v.name.replace(":", "_")
        summary_ops.append(tf.summary.scalar(name, ns))

    summary_ops.append(tf.summary.scalar(norm_name, tf_norm))

    return clipped_t_list, tf_norm, summary_ops


def clip_grads(grads, all_clip_norm_val, do_summaries):
    # grads = [(grad1, var1), (grad2, var2), ...]
    def _clip_norms(grad_and_vars, val, name):
        # grad_and_vars is a list of (g, v) pairs
        grad_tensors = [g for g, v in grad_and_vars]
        vv = [v for g, v in grad_and_vars]
        scaled_val = val
        if do_summaries:
            clipped_tensors, g_norm, so = clip_by_global_norm_summary(
                grad_tensors, scaled_val, name, vv)
        else:
            so = []
            clipped_tensors, g_norm = tf.clip_by_global_norm(
                grad_tensors, scaled_val)

        ret = []
        for t, (g, v) in zip(clipped_tensors, grad_and_vars):
            ret.append((t, v))

        return ret, so

    ret, summary_ops = _clip_norms(grads, all_clip_norm_val, 'norm_grad')

    assert len(ret) == len(grads)

    return ret, summary_ops
def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  if not FLAGS.do_train and not FLAGS.do_eval: # 必须是训练或验证的类型
    raise ValueError("At least one of `do_train` or `do_eval` must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file) # 从json文件中获得配置信息

  tf.gfile.MakeDirs(FLAGS.output_dir)

  input_files = [] # 输入可以是多个文件，以“逗号隔开”；可以是一个匹配形式的，如“input_x*”
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Input Files ***")
  for input_file in input_files:
    tf.logging.info("  %s" % input_file)

  # tpu_cluster_resolver = None
  # if FLAGS.use_tpu and FLAGS.tpu_name:
  #   tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver( # TODO
  #       tpu=FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
  #
  # print("###tpu_cluster_resolver:",tpu_cluster_resolver,";FLAGS.use_tpu:",FLAGS.use_tpu,";FLAGS.tpu_name:",FLAGS.tpu_name,";FLAGS.tpu_zone:",FLAGS.tpu_zone)
  # # ###tpu_cluster_resolver: <tensorflow.python.distribute.cluster_resolver.tpu_cluster_resolver.TPUClusterResolver object at 0x7f4b387b06a0> ;FLAGS.use_tpu: True ;FLAGS.tpu_name: grpc://10.240.1.83:8470

  # is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  # run_config = tf.contrib.tpu.RunConfig(
  #     keep_checkpoint_max=None, # 10
  #     cluster=tpu_cluster_resolver,
  #     master=FLAGS.master,
  #     model_dir=FLAGS.output_dir,
  #     save_checkpoints_steps=FLAGS.save_checkpoints_steps,
  #     tpu_config=tf.contrib.tpu.TPUConfig(
  #         iterations_per_loop=FLAGS.iterations_per_loop,
  #         num_shards=FLAGS.num_tpu_cores,
  #         per_host_input_for_training=is_per_host))

  # model_fn = model_fn_builder(
  #     bert_config=bert_config,
  #     init_checkpoint=FLAGS.init_checkpoint,
  #     learning_rate=FLAGS.learning_rate,
  #     num_train_steps=FLAGS.num_train_steps,
  #     num_warmup_steps=FLAGS.num_warmup_steps,
  #     use_tpu=FLAGS.use_tpu,
  #     use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  # estimator = tf.contrib.tpu.TPUEstimator(
  #     use_tpu=FLAGS.use_tpu,
  #     model_fn=model_fn,
  #     config=run_config,
  #     train_batch_size=FLAGS.train_batch_size,
  #     eval_batch_size=FLAGS.eval_batch_size)

  if FLAGS.do_train:
      mode = tf.estimator.ModeKeys.TRAIN
      use_one_hot_embeddings = FLAGS.use_tpu
      tf.logging.info("***** Running training *****")
      tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
      n_gpus = FLAGS.n_gpus
      batch_size = FLAGS.train_batch_size
      d = input_fn(input_files, batch_size * n_gpus, FLAGS.max_seq_length,
                   FLAGS.max_predictions_per_seq, True)
      features, iterator = parse_input_fn_result(d)
      # train_input_fn = input_fn_builder(
      #     input_files=input_files,
      #     max_seq_length=FLAGS.max_seq_length,
      #     max_predictions_per_seq=FLAGS.max_predictions_per_seq,
      #     is_training=True)
      # estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)

      input_ids_list = tf.split(features["input_ids"], n_gpus, axis=0)
      input_mask_list = tf.split(features["input_mask"], n_gpus, axis=0)
      segment_ids_list = tf.split(features["segment_ids"], n_gpus, axis=0)
      masked_lm_positions_list = tf.split(features["masked_lm_positions"], n_gpus, axis=0)
      masked_lm_ids_list = tf.split(features["masked_lm_ids"], n_gpus, axis=0)
      masked_lm_weights_list = tf.split(features["masked_lm_weights"], n_gpus, axis=0)
      next_sentence_labels_list = tf.split(features["next_sentence_labels"], n_gpus, axis=0)

      # multi-gpu train

      # optimizer = optimization_gpu.create_optimizer(
      #     None, FLAGS.learning_rate, FLAGS.num_train_steps, FLAGS.num_warmup_steps, False)
      # optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
      optimizer = AdamWeightDecayOptimizer(
          learning_rate=FLAGS.learning_rate,
          weight_decay_rate=0.01,
          beta_1=0.9,
          beta_2=0.999,
          epsilon=1e-6,
          exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
      flow_optimizer = AdamWeightDecayOptimizer(
          learning_rate=FLAGS.flow_learning_rate,  # learning_rate / init_lr *
          weight_decay_rate=0.01,
          beta_1=0.9,
          beta_2=0.999,
          epsilon=1e-6,
          exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
      # calculate the gradients on each GPU
      tower_grads_lm = []
      tower_grads_flow = []
      loss_print = tf.get_variable(
          'train_perplexity', [],
          initializer=tf.constant_initializer(0.0), trainable=False)
      loss_print_lm = tf.get_variable(
          'train_perplexity_lm', [],
          initializer=tf.constant_initializer(0.0), trainable=False)
      loss_print_flow = tf.get_variable(
          'train_perplexity_flow', [],
          initializer=tf.constant_initializer(0.0), trainable=False)
      k = 0
      global_step = tf.train.get_or_create_global_step()
      with tf.device('/gpu:%d' % k):
          with tf.variable_scope('lm', reuse=k > 0):
              # calculate the loss for one model replica and get
              #   lstm states
              input_ids = input_ids_list[k]
              input_mask = input_mask_list[k]
              segment_ids = segment_ids_list[k]
              masked_lm_positions = masked_lm_positions_list[k]
              masked_lm_ids = masked_lm_ids_list[k]
              masked_lm_weights = masked_lm_weights_list[k]
              #next_sentence_labels = next_sentence_labels_list[k]
              is_training = (mode == tf.estimator.ModeKeys.TRAIN)
              model = modeling.BertModel(
                  config=bert_config,
                  is_training=is_training,
                  input_ids=input_ids,
                  input_mask=input_mask,
                  token_type_ids=segment_ids,
                  use_one_hot_embeddings=use_one_hot_embeddings)
              (masked_lm_loss,
               masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
                  bert_config, model.get_sequence_output(), model.get_embedding_table(),
                  masked_lm_positions, masked_lm_ids, masked_lm_weights)
              # flow loss
              flow_loss_batch = 0
              if FLAGS.flow:
                  pooled = 0
                  n_last = int(FLAGS.sentence_embedding_type[-1])
                  input_mask_ = tf.cast(tf.expand_dims(input_mask, axis=-1), dtype=tf.float32)
                  for i in range(n_last):
                      sequence = model.all_encoder_layers[-i]  # [batch_size, seq_length, hidden_size]
                      pooled += tf.reduce_sum(sequence * input_mask_, axis=1) / tf.reduce_sum(input_mask_, axis=1)
                  pooled /= float(n_last)
                  # load model and train config
                  with open(os.path.join("./flow/config", FLAGS.flow_model_config + ".json"), 'r') as jp:
                      flow_model_config = AttrDict(json.load(jp))
                  flow_model_config.is_training = is_training
                  flow_model = Glow(flow_model_config)
                  flow_loss_example = flow_model.body(pooled, is_training)  # no l2 normalization here any more
                  flow_loss_batch = tf.math.reduce_mean(flow_loss_example)
                  embedding = tf.identity(tf.squeeze(flow_model.z, [1, 2]))
              ##################
              total_loss = masked_lm_loss+flow_loss_batch
      tvars = [v for v in tf.trainable_variables() if not v.name.startswith("lm/flow")]
      grads = tf.gradients(masked_lm_loss, tvars)
      (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
      train_op = optimizer.apply_gradients(
          zip(grads, tvars), global_step=global_step)

      flow_tvars = [v for v in tf.trainable_variables() if v.name.startswith("lm/flow")]
      flow_grads = tf.gradients(flow_loss_batch, flow_tvars)
      (flow_grads, _) = tf.clip_by_global_norm(flow_grads, clip_norm=1.0)
      flow_train_op = flow_optimizer.apply_gradients(
          zip(flow_grads, flow_tvars), global_step=global_step)

      new_global_step = global_step + 1
      train_op = tf.group(train_op, flow_train_op, [global_step.assign(new_global_step)])

      init = tf.global_variables_initializer()
      saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
      with tf.Session(config=tf.ConfigProto(
              allow_soft_placement=True)) as sess:
          sess.run(init)
          sess.run(iterator.initializer)
          #saver.restore(sess, init_checkpoint)
          #checkpoint_path = os.path.join(FLAGS.output_dir, 'model.ckpt')
          checkpoint_path = FLAGS.init_checkpoint
          if checkpoint_path:
              tvars = tf.trainable_variables()
              initialized_variable_names = {}
              print("init_checkpoint:", checkpoint_path)
              print("trainable vars:",tvars)
              if checkpoint_path:
                  (assignment_map, initialized_variable_names
                   ) = modeling.get_assignment_map_from_checkpoint(tvars, checkpoint_path)
                  print("assignment_map",assignment_map)
                  tf.train.init_from_checkpoint(checkpoint_path, assignment_map)
              tf.logging.info("**** Trainable Variables ****")
              for var in tvars:
                  init_string = ""
                  if var.name in initialized_variable_names:
                      init_string = ", *INIT_FROM_CKPT*"
                  tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                                  init_string)
              #saver.restore(sess, checkpoint_path)

          #if os.path.exists(FLAGS.output_dir):
              #saver.restore(sess, checkpoint_path)

          count = 0
          t0 = time.time()
          sum = 0
          sum_lm = 0
          sum_flow = 0
          while True:
              try:
                _, loss_print_, loss_print_lm_,loss_print_flow_ = sess.run([train_op, total_loss,masked_lm_loss,flow_loss_batch])
              except:
                sess.run(iterator.initializer)
                print('Iterator initialized')
              # optimistic_restore(sess, checkpoint_path + "-0")
              # loss_print_2 = sess.run([loss_print])
              sum += loss_print_
              sum_lm += loss_print_lm_
              sum_flow += loss_print_flow_
              count += 1
              if count % 300 == 0:
                  print("------------")
                  print(time.time() - t0, " s")
                  t0 = time.time()
                  print("loss,loss_lm,loss_flow is %0.4f,%0.4f,%0.4f"%(sum / count,sum_lm/count,sum_flow/count))
                  sum = 0
                  sum_lm = 0
                  sum_flow = 0
                  count = 0
                  checkpoint_path = os.path.join(FLAGS.output_dir, 'model.ckpt')
                  saver.save(sess, checkpoint_path,global_step=global_step)

def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  output_layer = model.get_pooled_output()

  sequence_output = model.get_sequence_output()

  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1) # todo 08-29 try temp-loss
    ###############bi_tempered_logistic_loss############################################################################
    # print("##cross entropy loss is used...."); tf.logging.info("##cross entropy loss is used....")
    # t1=0.9 #t1=0.90
    # t2=1.05 #t2=1.05
    # per_example_loss=bi_tempered_logistic_loss(log_probs,one_hot_labels,t1,t2,label_smoothing=0.1,num_iters=5) # TODO label_smoothing=0.0
    #tf.logging.info("per_example_loss:"+str(per_example_loss.shape))
    ##############bi_tempered_logistic_loss#############################################################################

    loss = tf.reduce_mean(per_example_loss)

    return (sequence_output,output_layer,loss, per_example_loss, logits, probabilities,model.all_encoder_layers)
def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  import re,collections
  """Compute the union of the current variables and checkpoint variables."""
  initialized_variable_names = {}
  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var
  init_vars = tf.train.list_variables(init_checkpoint)
  assignment_map = collections.OrderedDict()
  vars_others = []
  for x in init_vars:
    (name, var) = (x[0], x[1])
    if name[3:] in name_to_variable:
      assignment_map[name] = name[3:]
    elif name in name_to_variable:
      assignment_map[name] = name
    else:
      vars_others.append(name)
      continue
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1
  return (assignment_map, initialized_variable_names,vars_others)
class InputExample(object):
  """A single training/test example for simple sequence classification."""
  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label
def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()
class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example
class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """
def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""
  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_id=0,
        is_real_example=False)

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = label_map[example.label]
  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      is_real_example=True)
  return feature
def sentEmb(S,bert_config_file,vocab_file,path_checkpoint,max_seq_length = FLAGS.max_seq_length):
    label_list = ['0','1']
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=True)
    tf.reset_default_graph()
    input_ids = tf.placeholder(tf.int32, shape=[None, max_seq_length], name='input_ids')
    input_mask = tf.placeholder(tf.int32, shape=[None, max_seq_length], name='input_mask')
    segment_ids = tf.placeholder(tf.int32, shape=[None, max_seq_length], name='segment_ids')
    labels = tf.placeholder(tf.int32, shape=[None, ], name='labels')
    sequence_output, output_layer0, loss, per_example_loss, logits, probabilities, all_encoder_layers= create_model(bert_config, False,
                                                                                                 input_ids, input_mask,
                                                                                                 segment_ids, labels, 2,
                                                                                                 False)
    pooled = 0
    n_last = int(FLAGS.sentence_embedding_type[-1])
    input_mask_ = tf.cast(tf.expand_dims(input_mask, axis=-1), dtype=tf.float32)
    for i in range(n_last):
        sequence = all_encoder_layers[-i]  # [batch_size, seq_length, hidden_size]
        pooled += tf.reduce_sum(sequence * input_mask_, axis=1) / tf.reduce_sum(input_mask_, axis=1)
    pooled /= float(n_last)
    # load model and train config
    with open(os.path.join("./flow/config", FLAGS.flow_model_config + ".json"), 'r') as jp:
        flow_model_config = AttrDict(json.load(jp))
    flow_model_config.is_training = False
    flow_model = Glow(flow_model_config)
    flow_loss_example = flow_model.body(pooled, False)  # no l2 normalization here any more
    embedding = tf.identity(tf.squeeze(flow_model.z, [1, 2]))
    checkpoint_path = tf.train.latest_checkpoint(path_checkpoint)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    if checkpoint_path:
        tvars = tf.trainable_variables()
        print("init_checkpoint:", checkpoint_path)
        print("trainable vars:", tvars)
        if checkpoint_path:
            (assignment_map, initialized_variable_names,vars_others) = get_assignment_map_from_checkpoint(tvars, checkpoint_path)
            print("assignment_map", assignment_map)
            tf.train.init_from_checkpoint(checkpoint_path, assignment_map)
    output = {'bertflow': embedding}
    T = []
    for i in range(len(S)):
        if i % 100 == 0:
            print(i, len(S))
        text_a = S[i]
        example = InputExample(guid='guid', text_a=text_a, label='0')
        feature = convert_single_example(10, example, label_list, max_seq_length, tokenizer)
        feed_dict = {input_ids: [feature.input_ids], segment_ids: [feature.segment_ids],
                     input_mask: [feature.input_mask]}
        y = {key: sess.run(output[key], feed_dict=feed_dict)[0] for key in output}
        T.append([i, S[i], y])
    return T
if __name__ == "__main__":
  # flags.mark_flag_as_required("input_file")
  # flags.mark_flag_as_required("bert_config_file")
  # flags.mark_flag_as_required("output_dir")
  #tf.app.run()
  main(0)