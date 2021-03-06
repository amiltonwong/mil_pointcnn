from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import pointfly as pf
import tensorflow as tf

"""
1. Move P to local coordinate of system of p.
2. Indiviually lift each point into C_delta dim. space.
3. Concatenate F_delta and F into F_*, which is a K X (C_delta + C1) matrix
4. Learn the K x K X-transformation matrix
5. Weight and permute F_* with the learnt X-transformation into F_X (X-transformed feature matrix of input neighborhood points)
6. Finally, typical convolution between K and F_X
"""


""" 
core layer: X-conv (X_-transformation + Conv layer)
input: 
    pts: input points, (neighborhood) point matrix
    fts: input point features, (neighborhood point) feature matrix
    qrs: representative points / ? receptive field of representative points
    K: nunmber of neighboring points around representative point by K-Nearest Neighbors search
     `size of trainable tensor (kernel weights)
    N: batch size
    P: number of output (representative) points
    D: dilation rate
    C: number of channels of output feature, equals to C_2 in paper
    C_pts_fts: number of channels of point's feature, C_delta in paper
    is_training:
    with_X_transformation:
    depth_multiplier: used in separable_convolution
    sorting_method: not yet implemented 
    with_global: true/false

output:
    fts_conv_3d feature map

"""
def xconv(pts, fts, qrs, tag, N, K, D, P, C, C_pts_fts, is_training, with_X_transformation, depth_multiplier,
          sorting_method=None, with_global=False):
    if D == 1: # no dilation
        # use K-Nearest Neighbors search to find K neighboring points
        # qrs : output representative points, pts : input points, K neighboring points
        _, indices = pf.knn_indices_general(queries=qrs, points=pts, k=K, sort=True)  # return shape is (N, P, K, 2)
    else: # with dilation
        _, indices_dilated = pf.knn_indices_general(queries=qrs, points=pts, k=K * D, sort=True)
        indices = indices_dilated[:, :, ::D, :]

    # indices: sampled K neigboring points, shape is (N, P, K, 2). If dilation enabled, increase multiple of D

    if sorting_method is not None: # by default, no sorting method
        indices = pf.sort_points(pts, indices, sorting_method)

    # "Gather slices from `params` into a Tensor with shape specified by `indices`
    # nn_pts: P個 representative points, K個 neigboring points
    nn_pts = tf.gather_nd(params=pts, indices=indices, name=tag + 'nn_pts')  # (N, P, K, 3)
    # nn_pts_center: representatives ponts, used as to convert to local coordinates
    nn_pts_center = tf.expand_dims(qrs, axis=2, name=tag + 'nn_pts_center')  # (N, P, 1, 3)
    # Move (neighborhood) point matrix to local coordinate system of representative point
    # nn_pts_local : local coordinate  w.r.t representative points : PointMatrixlocal <- PointMatrix - RepresentativePoint
    # P' <= P - p, in paper
    nn_pts_local = tf.subtract(nn_pts, nn_pts_center, name=tag + 'nn_pts_local')  # (N, P, K, 3)
    # end of "1. Move P to local coordinate of system of p."


    # Prepare features to be transformed.
    # "2. Indiviually lift each point into C_delta dim. space." by using MLP_delta()
    # MLP_delta(P − p)
    # C_pts_fts: C_delta in paper
    nn_fts_from_pts_0 = pf.dense(nn_pts_local, C_pts_fts, tag + 'nn_fts_from_pts_0', is_training)
    # 2nd lifting
    nn_fts_from_pts = pf.dense(nn_fts_from_pts_0, C_pts_fts, tag + 'nn_fts_from_pts', is_training)
    # end of obtaining F_delta

    if fts is None: # if features matrix "F" is none, pass F_delta to fts
        nn_fts_input = nn_fts_from_pts
    else:
        # nn_fts_from_prev: F
        nn_fts_from_prev = tf.gather_nd(fts, indices, name=tag + 'nn_fts_from_prev')
        # 3. Concatenate F_delta and F into F_*, which is a K X (C_delta + C1) matrix
        # nn_fts_from_pts: F_delta,   nn_fts_from_prev: F,  nn_fts_input: F_*
        nn_fts_input = tf.concat([nn_fts_from_pts, nn_fts_from_prev], axis=-1, name=tag + 'nn_fts_input')

    # 4. Learn the K x K X-transformation matrix
    # MLP(P − p)
    if with_X_transformation:  # by default, = True
        ######################## X-transformation #########################
        # nn_pts_local.shape = (N, P, K, 3), perform general convolution
        X_0 = pf.conv2d(input=nn_pts_local, output=K * K, name=tag + 'X_0', is_training=is_training, kernel_size=(1, K))
        # reshape to (N, P, K, K)
        X_0_KK = tf.reshape(X_0, (N, P, K, K), name=tag + 'X_0_KK')
        # perform separable convolution
        X_1 = pf.depthwise_conv2d(input=X_0_KK, depth_multiplier=K, name=tag + 'X_1', is_training=is_training, kernel_size=(1, K))
        # reshape to (N, P, K, K)
        X_1_KK = tf.reshape(X_1, (N, P, K, K), name=tag + 'X_1_KK')
        # perform separable convolution, no activation function applied
        X_2 = pf.depthwise_conv2d(input=X_1_KK, depth_multiplier=K, name=tag + 'X_2', is_training=is_training, kernel_size=(1, K), activation=None)
        # reshape to (N, P, K, K), X_2_KK : X
        X_2_KK = tf.reshape(X_2, (N, P, K, K), name=tag + 'X_2_KK')
        # 5. Weight and permute F_* with the learnt X-transformation (X) into F_X,  fts_X: F_X
        # F_X <= X x F_*
        # MLP(P − p) x [MLP_delta(P − p), F]
        fts_X = tf.matmul(X_2_KK, nn_fts_input, name=tag + 'fts_X')
        ###################################################################
    else:
        fts_X = nn_fts_input  # use F_*
    # 6. Finally, typical convolution between Kernel and transformed feature matrix: F_X
    # fts_conv: features after X-conv layer
    fts_conv = pf.separable_conv2d(input=fts_X, output=C, name=tag + 'fts_conv', is_training=is_training,
                                   kernel_size=(1, K), depth_multiplier=depth_multiplier)
    # Removes dimensions of size 1
    fts_conv_3d = tf.squeeze(fts_conv, axis=2, name=tag + 'fts_conv_3d') # turn it as compact tensor (dim-1 removed)

    if with_global: # when layer_idx == len(xconv_params) - 1 hence 4-1=3. At last layer, then apply "fts_global" on qrs
        # densely-connected layer.
        # number of channels of output layer is  C//4 ?
        fts_global_0 = pf.dense(input=qrs, output=C // 4, name=tag + 'fts_global_0', is_training=is_training)
        # densely-connected layer. (2nd)
        fts_global = pf.dense(input=fts_global_0, output=C // 4, name=tag + 'fts_global_', is_training=is_training)
        # concat along the last dimension
        return tf.concat([fts_global, fts_conv_3d], axis=-1, name=tag + 'fts_conv_3d_with_global')
    else:
        return fts_conv_3d



"""
modelnet40
x = 3 
xconv_params = [dict(zip(xconv_param_name, xconv_param)) for xconv_param in
                [(8, 1, -1, 16 * x, []),
                 (12, 2, 384, 32 * x, []),
                 (16, 2, 128, 64 * x, []),
                 (16, 3, 128, 128 * x, [])]]
                 
xconv_param            = [(8, 1, -1, 32), (12, 2, 384, 64), (16, 2, 128, 128), (16, 3, 128, 256)]
                 
fc_params = [dict(zip(fc_param_name, fc_param)) for fc_param in
             [(128 * x, 0.0),
              (64 * x, 0.5)]]    

members:
    layer_pts
    layer_fts
    
"""

class PointCNN:
    def __init__(self, points, features, num_class, is_training, setting, task):
        xconv_params = setting.xconv_params
        fc_params = setting.fc_params
        with_X_transformation = setting.with_X_transformation # =True
        sorting_method = setting.sorting_method # = None
        N = tf.shape(points)[0] # number of points
        # ModelNet40, setting.sampling = "random"
        if setting.sampling == 'fps': #Furthest point sampling
            from sampling import tf_sampling

        self.layer_pts = [points] # assign input points as list, assigned to self.layer_pts
        if features is None: # by default
            self.layer_fts = [features]
        else:
            C_fts = xconv_params[0]['C'] // 2
            features_hd = pf.dense(features, C_fts, 'features_hd', is_training)
            self.layer_fts = [features_hd]

        for layer_idx, layer_param in enumerate(xconv_params):
            tag = 'xconv_' + str(layer_idx + 1) + '_'   # xconv_1 , xconv_2, xconv_3
            # Read xconv_params
            K = layer_param['K'] # (8, 12, 16, 16)
            D = layer_param['D'] # (1, 2, 2, 3)
            P = layer_param['P'] # (-1, 384, 128, 128)
            C = layer_param['C'] # (16*3, 32*3, 64*3, 128*3)
            links = layer_param['links'] #([], [], [], [])
            if setting.sampling != 'random' and links:
                print('Error: flexible links are supported only when random sampling is used!')
                exit()

            # get k-nearest points
            pts = self.layer_pts[-1]
            fts = self.layer_fts[-1]
            # P: number of output (representative) points
            #                                current P == prev_layer P
            if P == -1 or (layer_idx > 0 and P == xconv_params[layer_idx - 1]['P']):
                qrs = self.layer_pts[-1] # qrs: representative points
            else:
                # By default,  ModelNet40, setting.sampling = "random"
                if setting.sampling == 'fps':
                    # use farthest_point_sampling to sample P個 representative points
                    indices = tf_sampling.farthest_point_sample(P, pts)
                    # gather sampling points into representative points in position coordinate
                    qrs = tf_sampling.gather_point(pts, indices)  # (N,P,3)
                elif setting.sampling == 'ids':
                    indices = pf.inverse_density_sampling(pts, K, P)
                    qrs = tf.gather_nd(pts, indices)
                elif setting.sampling == 'random': # byu default, use "random"
                    # slice the P個 representative points
                    qrs = tf.slice(pts, (0, 0, 0), (-1, P, -1), name=tag + 'qrs')  # (N, P, 3)
                else:
                    print('Unknown sampling method!')
                    exit()
            self.layer_pts.append(qrs) # add qrs "representative points" into layer_pts

            if layer_idx == 0: # For the first layer
                # C_pts_fts, number of channels of point's feature, C_delta in paper
                # 取 C 的1/2, 或1/4
                C_pts_fts = C // 2 if fts is None else C // 4 # by default, C_pts_fts = C // 2 = 48 // 2 = 24
                depth_multiplier = 4  # What purpose?
            else:
                # 取前一層的channel數
                C_prev = xconv_params[layer_idx - 1]['C'] # read C_prev param
                # 取前一層的channel數的1/4 作為C_delta
                C_pts_fts = C_prev // 4
                depth_multiplier = math.ceil(C / C_prev)
            #### Apply X_transform on fts
            fts_xconv = xconv(pts, fts, qrs, tag, N, K, D, P, C, C_pts_fts, is_training, with_X_transformation,
                              depth_multiplier, sorting_method, layer_idx == len(xconv_params) - 1) # "layer_idx == len(xconv_params) - 1" 作為with_global的依據
            fts_list = []
            for link in links:  # link = edge?
                fts_from_link = self.layer_fts[link]
                if fts_from_link is not None:
                    fts_slice = tf.slice(fts_from_link, (0, 0, 0), (-1, P, -1),
                                         name=tag + 'fts_slice_' + str(-link))
                    C_forward = math.ceil(fts_slice.get_shape().as_list()[-1] / (-link))
                    fts_forward = pf.dense(fts_slice, C_forward, tag + 'fts_forward_' + str(-link), is_training)
                    fts_list.append(fts_forward)

            if fts_list:
                fts_list.append(fts_xconv)
                self.layer_fts.append(tf.concat(fts_list, axis=-1, name=tag + 'fts_list_concat'))
            else:
                self.layer_fts.append(fts_xconv)

        if task == 'segmentation':
            for layer_idx, layer_param in enumerate(setting.xdconv_params):
                tag = 'xdconv_' + str(layer_idx + 1) + '_'
                K = layer_param['K']
                D = layer_param['D']
                pts_layer_idx = layer_param['pts_layer_idx']
                qrs_layer_idx = layer_param['qrs_layer_idx']

                pts = self.layer_pts[pts_layer_idx + 1]
                fts = self.layer_fts[pts_layer_idx + 1] if layer_idx == 0 else self.layer_fts[-1]
                qrs = self.layer_pts[qrs_layer_idx + 1]
                fts_qrs = self.layer_fts[qrs_layer_idx + 1]
                P = xconv_params[qrs_layer_idx]['P']
                C = xconv_params[qrs_layer_idx]['C']
                C_prev = xconv_params[pts_layer_idx]['C']
                C_pts_fts = C_prev // 4
                depth_multiplier = 1
                fts_xdconv = xconv(pts, fts, qrs, tag, N, K, D, P, C, C_pts_fts, is_training, with_X_transformation,
                                   depth_multiplier, sorting_method)
                fts_concat = tf.concat([fts_xdconv, fts_qrs], axis=-1, name=tag + 'fts_concat')
                fts_fuse = pf.dense(fts_concat, C, tag + 'fts_fuse', is_training)
                self.layer_pts.append(qrs)
                self.layer_fts.append(fts_fuse)

        self.fc_layers = [self.layer_fts[-1]]
        # fc_params:
        #     [(128 * x, 0.0),
        #      (64 * x, 0.5)]], use dropout
        for layer_idx, layer_param in enumerate(fc_params):
            C = layer_param['C']
            dropout_rate = layer_param['dropout_rate']
            fc = pf.dense(input=self.fc_layers[-1], output=C, name='fc{:d}'.format(layer_idx), is_training=is_training)
            # use dropout
            fc_drop = tf.layers.dropout(fc, dropout_rate, training=is_training, name='fc{:d}_drop'.format(layer_idx))
            self.fc_layers.append(fc_drop)

        if task == 'classification':
            fc_mean = tf.reduce_mean(self.fc_layers[-1], axis=1, keep_dims=True, name='fc_mean')
            # Return `true_fn()` if the predicate `pred` is true else `false_fn()`
            self.fc_layers[-1] = tf.cond(is_training, lambda: self.fc_layers[-1], lambda: fc_mean)
        # compute logits
        self.logits = pf.dense(self.fc_layers[-1], num_class, 'logits', is_training, with_bn=False, activation=None)
        # compute probs.
        self.probs = tf.nn.softmax(self.logits, name='probs')
