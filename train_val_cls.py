#!/usr/bin/python3
"""Training and Validation On Classification Task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import random
import shutil
import argparse
import importlib
import numpy as np
import pointfly as pf
import tensorflow as tf
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-t', help='Path to data', required=True) # train_file.txt
    parser.add_argument('--path_val', '-v', help='Path to validation data') # test_files.txt
    parser.add_argument('--load_ckpt', '-l', help='Path to a check point file for load')
    parser.add_argument('--save_folder', '-s', help='Path to folder for saving check points and summary', required=True) # ../../models/cls or seg
    parser.add_argument('--model', '-m', help='Model to use', required=True) # pointcnn_cls or pointcnn_seg
    parser.add_argument('--setting', '-x', help='Setting to use', required=True) # modelnet_x3_l4
    args = parser.parse_args()

    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # e.g. "pointcnn_cls_modelnet_x3_l4_6150_2018-04-25-11-43-14"
    root_folder = os.path.join(args.save_folder, '%s_%s_%d_%s' % (args.model, args.setting, os.getpid(), time_string))
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    # write the stdout to 'log.txt'
    sys.stdout = open(os.path.join(root_folder, 'log.txt'), 'w')
    # write down PID
    print('PID:', os.getpid())

    print(args)
    # use "import_module" to import from -m "pointcnn_cls", module in "pointcnn_cls.py" is imported
    model = importlib.import_module(args.model)
    # "/data/code4/mil_pointcnn/pointcnn_cls" -> os.path.dirname(__file__), args.model="pointcnn_cls"
    # setting_path = "/data/code4/mil_pointcnn/pointcnn_cls"
    setting_path = os.path.join(os.path.dirname(__file__), args.model)
    sys.path.append(setting_path) # "/data/code4/mil_pointcnn/pointcnn_cls" path is added
    setting = importlib.import_module(args.setting) # module in "modelnet_x3_l4.py" is imported

    num_epochs = setting.num_epochs # (modelnet_x3_l4): 2048
    batch_size = setting.batch_size # (modelnet_x3_l4): 200
    sample_num = setting.sample_num # (modelnet_x3_l4): 1024
    step_val = setting.step_val # (modelnet_x3_l4): 500
    num_class = setting.num_class # (modelnet_x3_l4): 40
    rotation_range = setting.rotation_range # (modelnet_x3_l4): [0, math.pi, 0, 'u']
    rotation_range_val = setting.rotation_range_val # (modelnet_x3_l4): rotation_range_val = [0, 0, 0, 'u']
    jitter = setting.jitter # (modelnet_x3_l4): 0.0
    jitter_val = setting.jitter_val # (modelnet_x3_l4): 0.0

    # Prepare inputs
    print('{}-Preparing datasets...'.format(datetime.now()))
    # call "data_utils.load_cls_train_val()"
    import data_utils # import from "data_utils.py"
    data_train, label_train, data_val, label_val = data_utils.load_cls_train_val("../../data/modelnet/train_files.txt", "../../data/modelnet/test_files.txt")
    data_train, label_train, data_val, label_val = setting.load_fn(args.path, args.path_val)
    # data_train.shape = (9840, 2048, 6), label_train.shape = (9840,)
    # data_val.shape = (2468, 2048, 6), label_val.shape = (2468,)

    # when save_ply_fn is enabled
    if setting.save_ply_fn is not None:
        folder = os.path.join(root_folder, 'pts') # create "pts" directory
        print('{}-Saving samples as .ply files to {}...'.format(datetime.now(), folder))
        sample_num_for_ply = min(512, data_train.shape[0])  # save 512 samples
        if setting.map_fn is None:
            data_sample = data_train[:sample_num_for_ply]  # sample the first 512 examples
        else:  # when map_fn is enabled
            data_sample_list = []
            for idx in range(sample_num_for_ply):
                data_sample_list.append(setting.map_fn(data_train[idx], 0)[0])
            data_sample = np.stack(data_sample_list)
        setting.save_ply_fn(data_sample, folder)

    num_train = data_train.shape[0] # 9840
    point_num = data_train.shape[1] # 2048 points
    num_val = data_val.shape[0] # 2468
    print('{}-{:d}/{:d} training/validation samples.'.format(datetime.now(), num_train, num_val))

    ######################################################################
    # Placeholders
    indices = tf.placeholder(tf.int32, shape=(None, None, 2), name="indices")
    xforms = tf.placeholder(tf.float32, shape=(None, 3, 3), name="xforms")
    rotations = tf.placeholder(tf.float32, shape=(None, 3, 3), name="rotations")
    jitter_range = tf.placeholder(tf.float32, shape=(1), name="jitter_range")
    global_step = tf.Variable(0, trainable=False, name='global_step')
    is_training = tf.placeholder(tf.bool, name='is_training')

    data_train_placeholder = tf.placeholder(data_train.dtype, data_train.shape)
    label_train_placeholder = tf.placeholder(label_train.dtype, label_train.shape)
    data_val_placeholder = tf.placeholder(data_val.dtype, data_val.shape)
    label_val_placeholder = tf.placeholder(label_val.dtype, label_val.shape)
    handle = tf.placeholder(tf.string, shape=[])

    ######################################################################
    # prepare for training dataset "dataset_train"
    # Creates a `Dataset` dataset_train whose elements are slices of the given tensors.
    dataset_train = tf.data.Dataset.from_tensor_slices((data_train_placeholder, label_train_placeholder))
    if setting.map_fn is not None: # when map_fn is enabled
        dataset_train = dataset_train.map(lambda data, label: tuple(tf.py_func(
            setting.map_fn, [data, label], [tf.float32, label.dtype])), num_parallel_calls=setting.num_parallel_calls)
    dataset_train = dataset_train.shuffle(buffer_size=batch_size * 4)

    if setting.keep_remainder: # enabled by default
        dataset_train = dataset_train.batch(batch_size) # batch_size = 200
        batch_num_per_epoch = math.ceil(num_train / batch_size)  # math.ceil(9840 / 200) = 50, run 50 batches to cover through training set
    else:
        dataset_train = dataset_train.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
        batch_num_per_epoch = math.floor(num_train / batch_size)
    batch_num = batch_num_per_epoch * num_epochs # total number of batch needed, 50*2048 = 102,400
    print('{}-{:d} training batches.'.format(datetime.now(), batch_num))

    dataset_train = dataset_train.repeat() # Repeats this dataset, for the elements to be repeated indefinitely.
    iterator_train = dataset_train.make_initializable_iterator() # Creates an `Iterator` for enumerating the elements of this dataset.

    # prepare for validation dataset "dataset_val"
    dataset_val = tf.data.Dataset.from_tensor_slices((data_val_placeholder, label_val_placeholder))
    if setting.map_fn is not None:
        dataset_val = dataset_val.map(lambda data, label: tuple(tf.py_func(
            setting.map_fn, [data, label], [tf.float32, label.dtype])), num_parallel_calls=setting.num_parallel_calls)
    if setting.keep_remainder:
        dataset_val = dataset_val.batch(batch_size) # batch_size = 200
        batch_num_val = math.ceil(num_val / batch_size) # math.ceil(2468 / 200) = 13, run 13 batches to cover through val set
    else:
        dataset_val = dataset_val.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
        batch_num_val = math.floor(num_val / batch_size)
    iterator_val = dataset_val.make_initializable_iterator() # Creates an `Iterator`

    # Creates a new, uninitialized `Iterator` based on the given handle
    iterator = tf.data.Iterator.from_string_handle(string_handle=handle, output_types=dataset_train.output_types,
                                                   output_shapes=dataset_train.output_shapes)
    (pts_fts, labels) = iterator.get_next() # get a batch of points_features and labels
    # pts_fts (?, 2048, 6)
    # labels (?,)

    features_augmented = None
    if setting.data_dim > 3: # data_dim = 6
        # split into points and features
        points, features = tf.split(pts_fts, [3, setting.data_dim - 3], axis=-1, name='split_points_features')
        # points (?, 2048, 3), features (?, 2048, 3)
        if setting.use_extra_features: # By default, use_extra_features = False
            features_sampled = tf.gather_nd(features, indices=indices, name='features_sampled')
            if setting.with_normal_feature:
                features_augmented = pf.augment(features_sampled, rotations)
            else:
                features_augmented = features_sampled
    else:
        points = pts_fts # if data_dim is no greater than 3 dim, the features is only the position
    # get sampled points
    # Gather slices from "points" into a Tensor with shape specified by indices (None, None, 2)
    points_sampled = tf.gather_nd(points, indices=indices, name='points_sampled') # indices: (None, None, 2)
    # 把 points_sampled 乘上 xforms變換後，並添加jitter
    points_augmented = pf.augment(points_sampled, xforms, jitter_range)


    # Construct the net structure
    # def __init__(self, points, features, num_class, is_training, setting, task="classification")
    net = model.Net(points=points_augmented, features=features_augmented, num_class=num_class,
                    is_training=is_training, setting=setting)
    logits, probs = net.logits, net.probs
    labels_2d = tf.expand_dims(labels, axis=-1, name='labels_2d')
    labels_tile = tf.tile(labels_2d, (1, tf.shape(probs)[1]), name='labels_tile')
    loss_op = tf.losses.sparse_softmax_cross_entropy(labels=labels_tile, logits=logits)
    t_1_acc_op = pf.top_1_accuracy(probs, labels_tile)

    _ = tf.summary.scalar('loss/train', tensor=loss_op, collections=['train'])
    _ = tf.summary.scalar('t_1_acc/train', tensor=t_1_acc_op, collections=['train'])

    loss_val_avg = tf.placeholder(tf.float32)
    t_1_acc_val_avg = tf.placeholder(tf.float32)
    _ = tf.summary.scalar('loss/val', tensor=loss_val_avg, collections=['val'])
    _ = tf.summary.scalar('t_1_acc/val', tensor=t_1_acc_val_avg, collections=['val'])

    lr_exp_op = tf.train.exponential_decay(setting.learning_rate_base, global_step, setting.decay_steps,
                                           setting.decay_rate, staircase=True)
    lr_clip_op = tf.maximum(lr_exp_op, setting.learning_rate_min)
    _ = tf.summary.scalar('learning_rate', tensor=lr_clip_op, collections=['train'])
    reg_loss = setting.weight_decay * tf.losses.get_regularization_loss()
    if setting.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=lr_clip_op, epsilon=setting.epsilon)
    elif setting.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=lr_clip_op, momentum=setting.momentum, use_nesterov=True)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss_op + reg_loss, global_step=global_step)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    saver = tf.train.Saver(max_to_keep=None)

    # backup all code
    code_folder = os.path.abspath(os.path.dirname(__file__))
    shutil.copytree(code_folder, os.path.join(root_folder, os.path.basename(code_folder)))

    folder_ckpt = os.path.join(root_folder, 'ckpts')
    if not os.path.exists(folder_ckpt):
        os.makedirs(folder_ckpt)

    folder_summary = os.path.join(root_folder, 'summary')
    if not os.path.exists(folder_summary):
        os.makedirs(folder_summary)

    parameter_num = np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
    print('{}-Parameter number: {:d}.'.format(datetime.now(), parameter_num))

    with tf.Session() as sess:
        summaries_op = tf.summary.merge_all('train')
        summaries_val_op = tf.summary.merge_all('val')
        summary_writer = tf.summary.FileWriter(folder_summary, sess.graph)

        sess.run(init_op)

        # Load the model
        if args.load_ckpt is not None:
            saver.restore(sess, args.load_ckpt)
            print('{}-Checkpoint loaded from {}!'.format(datetime.now(), args.load_ckpt))

        handle_train = sess.run(iterator_train.string_handle())
        handle_val = sess.run(iterator_val.string_handle())

        sess.run(iterator_train.initializer, feed_dict={
            data_train_placeholder: data_train,
            label_train_placeholder: label_train,
        })

        for batch_idx_train in range(batch_num):
            ######################################################################
            # Validation
            if (batch_idx_train != 0 and batch_idx_train % step_val == 0) or batch_idx_train == batch_num - 1:
                sess.run(iterator_val.initializer, feed_dict={
                    data_val_placeholder: data_val,
                    label_val_placeholder: label_val,
                })
                filename_ckpt = os.path.join(folder_ckpt, 'iter')
                saver.save(sess, filename_ckpt, global_step=global_step)
                print('{}-Checkpoint saved to {}!'.format(datetime.now(), filename_ckpt))

                losses = []
                t_1_accs = []
                for batch_idx_val in range(batch_num_val):
                    if not setting.keep_remainder or num_val % batch_size == 0 or batch_idx_val != batch_num_val - 1:
                        batch_size_val = batch_size
                    else:
                        batch_size_val = num_val % batch_size
                    xforms_np, rotations_np = pf.get_xforms(batch_size_val, rotation_range=rotation_range_val,
                                                                order=setting.order)
                    _, loss_val, t_1_acc_val = \
                        sess.run([update_ops, loss_op, t_1_acc_op],
                                 feed_dict={
                                     handle: handle_val,
                                     indices: pf.get_indices(batch_size_val, sample_num, point_num),
                                     xforms: xforms_np,
                                     rotations: rotations_np,
                                     jitter_range: np.array([jitter_val]),
                                     is_training: False,
                                 })
                    losses.append(loss_val * batch_size_val)
                    t_1_accs.append(t_1_acc_val * batch_size_val)
                    print('{}-[Val  ]-Iter: {:06d}  Loss: {:.4f}  T-1 Acc: {:.4f}'.format
                          (datetime.now(), batch_idx_val, loss_val, t_1_acc_val))
                    sys.stdout.flush()

                loss_avg = sum(losses) / num_val
                t_1_acc_avg = sum(t_1_accs) / num_val
                summaries_val = sess.run(summaries_val_op,
                                         feed_dict={
                                             loss_val_avg: loss_avg,
                                             t_1_acc_val_avg: t_1_acc_avg,
                                         })
                summary_writer.add_summary(summaries_val, batch_idx_train)
                print('{}-[Val  ]-Average:      Loss: {:.4f}  T-1 Acc: {:.4f}'
                      .format(datetime.now(), loss_avg, t_1_acc_avg))
                sys.stdout.flush()
            ######################################################################

            ######################################################################
            # Training
            if not setting.keep_remainder or num_train % batch_size == 0 or (batch_idx_train % batch_num_per_epoch) != (batch_num_per_epoch - 1):
                batch_size_train = batch_size
            else:
                batch_size_train = num_train % batch_size
            offset = int(random.gauss(0, sample_num // 8))
            offset = max(offset, -sample_num // 4)
            offset = min(offset, sample_num // 4)
            sample_num_train = sample_num + offset
            xforms_np, rotations_np = pf.get_xforms(batch_size_train, rotation_range=rotation_range,
                                                        order=setting.order)
            _, loss, t_1_acc, summaries = \
                sess.run([train_op, loss_op, t_1_acc_op, summaries_op],
                         feed_dict={
                             handle: handle_train,
                             indices: pf.get_indices(batch_size_train, sample_num_train, point_num),
                             xforms: xforms_np,
                             rotations: rotations_np,
                             jitter_range: np.array([jitter]),
                             is_training: True,
                         })
            summary_writer.add_summary(summaries, batch_idx_train)
            print('{}-[Train]-Iter: {:06d}  Loss: {:.4f}  T-1 Acc: {:.4f}'
                  .format(datetime.now(), batch_idx_train, loss, t_1_acc))
            sys.stdout.flush()
            ######################################################################
        print('{}-Done!'.format(datetime.now()))


if __name__ == '__main__':
    main()
