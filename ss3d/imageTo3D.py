from __future__ import absolute_import, division, print_function

import tensorflow as tf

tf.enable_eager_execution()
arg_scope = tf.contrib.framework.arg_scope
import math as m
import os
import sys

from binvox_rw import *
import cv2
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
import model_iterator
import numpy as np
from PIL import Image
import rospy
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from sensor_msgs.msg import Image
import tensorflow.contrib.layers as layers
import utils

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

flags = tf.app.flags
flags.DEFINE_string("dataset", "chair", "Dataset name that is to be used for training and evaluation.")
flags.DEFINE_string(
    "log_dir",
    "/tmp/net_test/",
    "Directory path for saving trained models and other data.",
)
flags.DEFINE_bool("volume64", False, "Use/Predict (64,64,64) voxels")
flags.DEFINE_bool("use_mean_shape", True, "Use mean shape as prior and compute residuals")
FLAGS = flags.FLAGS


class eval_ss3d:
    def __init__(self, sess):
        self.sess = sess
        self.is_training = False
        self.input_shape = [64, 64, 3]
        self.volume64 = FLAGS.volume64
        if self.volume64:
            self.vox_shape = [64, 64, 64, 1]
        else:
            self.vox_shape = [32, 32, 32, 1]
        self.reg_loss = tf.zeros(dtype=tf.float32, shape=[])
        self.max_iter = 500000
        self.focal_length = 0.866
        self.focal_range = 1.732
        self.dataset = FLAGS.dataset
        self.class_dict = {
            "plane": "02691156",
            "car": "02958343",
            "chair": "03001627",
            "table": "04379243",
            "bottle": "02876657",
            "mug": "03797390",
            "bowl": "02880940",
            "can": "02946921",
        }
        self.use_mean_shape = FLAGS.use_mean_shape
        self.log_dir = FLAGS.log_dir
        self.eval_dir = self.log_dir + "/eval/"
        if not os.path.exists(self.eval_dir):
            os.makedirs(self.eval_dir)
        if self.use_mean_shape:
            self.mean_shape = np.load("mean_shape/mug.npy").astype(np.float32)
            self.mean_shape = tf.expand_dims(self.mean_shape, 0)
            self.mean_shape = tf.tile(self.mean_shape, [1, 1, 1, 1])
            self.mean_shape = tf.expand_dims(self.mean_shape, -1)

    def forward(self, img):
        outputs_ = self.sess.run(
            [self.voxels_1, self.azimuth_1, self.elevation_1, self.quat_pose],
            feed_dict={self.images: img},
        )
        return outputs_

    def buildModel(self):
        self.images = tf.placeholder(tf.float32, [None] + self.input_shape, name="input_images")
        self.bn = False
        f_dim = 64
        fc_dim = int(1024)
        z_dim = int(512)
        image_size = self.images.get_shape().as_list()[1]
        # convert to [-1,1] range
        self.images_normalized = self.images / 255.0
        with tf.variable_scope("encoder"):
            init_w = (
                lambda x: tf.truncated_normal_initializer(stddev=0.02, seed=1)
                if x
                else tf.contrib.layers.xavier_initializer(uniform=False, seed=1)
            )
            # convolutional encoder
            with arg_scope(
                [layers.conv2d, layers.fully_connected],
                weights_initializer=init_w(self.bn),
            ):
                h0 = layers.conv2d(self.images_normalized, f_dim, [5, 5], stride=2)
                h0 = tf.nn.leaky_relu(h0)
                h1 = layers.conv2d(h0, f_dim * 2, [5, 5], stride=2)
                h1 = tf.nn.leaky_relu(h1)
                h2 = layers.conv2d(h1, f_dim * 4, [5, 5], stride=2)
                h2 = tf.nn.leaky_relu(h2)
                # Reshape layer
                s8 = image_size // 8
                h2 = tf.reshape(h2, [-1, s8 * s8 * f_dim * 4])
                h3a = layers.fully_connected(h2, fc_dim)
                h3a = tf.nn.leaky_relu(h3a)
                h4a = layers.fully_connected(h3a, fc_dim)
                h4a = tf.nn.leaky_relu(h4a)
                bottleneck = layers.fully_connected(h4a, z_dim)
                bottleneck = tf.nn.leaky_relu(bottleneck)

        with tf.variable_scope("pose_regression"):
            with arg_scope(
                [layers.fully_connected],
                weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=1),
            ):
                self.azimuth_1 = layers.fully_connected(bottleneck, 1, activation_fn=tf.nn.sigmoid) * 2 * m.pi
                # 3*m.pi/9 #[0,60]#2*m.pi/9 #elev range [0,40] degrees
                self.elevation_1 = layers.fully_connected(bottleneck, 1, activation_fn=tf.nn.sigmoid) * 2 * m.pi / 9
                self.quat_pose = utils.eulerToQuat(self.azimuth_1, self.elevation_1)

        with tf.variable_scope("decoder"):
            with arg_scope(
                [layers.conv3d_transpose, layers.fully_connected],
                weights_initializer=init_w(self.bn),
            ):
                h0_dec = layers.fully_connected(bottleneck, 4 * 4 * 4 * f_dim * 8)
                h0_dec = tf.nn.relu(h0_dec)
                h1_dec = tf.reshape(h0_dec, [-1, 4, 4, 4, f_dim * 8])
                h1_dec = layers.conv3d_transpose(h1_dec, f_dim * 4, [4, 4, 4], stride=2)
                h1_dec = tf.nn.relu(h1_dec)
                h2_dec = layers.conv3d_transpose(h1_dec, 128, [5, 5, 5], stride=2)
                h2_dec = tf.nn.relu(h2_dec)
                if self.volume64:
                    h4_dec = layers.conv3d_transpose(h2_dec, 8, [5, 5, 5], stride=2)
                    h4_dec = tf.nn.relu(h4_dec)
                    self.voxels_1 = layers.conv3d_transpose(
                        h4_dec,
                        1,
                        [6, 6, 6],
                        stride=2,
                        activation_fn=tf.nn.sigmoid,
                        biases_initializer=tf.constant_initializer(0),
                    )
                else:
                    self.voxels_1 = layers.conv3d_transpose(
                        h2_dec,
                        1,
                        [6, 6, 6],
                        stride=2,
                        activation_fn=tf.nn.sigmoid,
                        biases_initializer=tf.constant_initializer(0),
                    )
                if self.use_mean_shape:
                    self.voxels_1 = self.voxels_1 + self.mean_shape


def rad2deg(angle_rad):
    return angle_rad * 180 / m.pi


class PredNode:
    def __init__(self, sess):
        self.bridge = CvBridge()
        rospy.Subscriber("/segmented_mug_rgb", Image, self.callback)
        self.pub_vox = rospy.Publisher("predicted_voxel_np", numpy_msg(Floats), queue_size=1)
        self.pub_pose = rospy.Publisher("predicted_pose_np", numpy_msg(Floats), queue_size=1)
        self.sess = sess
        self.net = eval_ss3d(self.sess)
        self.net.buildModel()
        tf.set_random_seed(0)
        tf.logging.set_verbosity(tf.logging.INFO)
        self.saver = tf.train.Saver()
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)
        log_dir = "trained_models"
        for checkpoint_path in model_iterator.stepfiles_iterator(log_dir, wait_minutes=5, min_steps=5000):
            print(checkpoint_path)
            self.saver.restore(self.sess, checkpoint_path.filename)
        print("finished initializing")

    def callback(self, data):
        im = self.bridge.imgmsg_to_cv2(data, "bgr8")
        assert im.shape == (64, 64, 3), "wrong shape:" + str(im.shape)
        img = np.expand_dims(im, axis=0)
        out = self.net.forward(img)
        voxels = out[0][0, :, :, :, 0]
        print("azi: ", rad2deg(out[1]))
        print("elev: ", rad2deg(out[2]))
        pose = np.array([out[1], out[2]], dtype=np.float32)
        # np.save("/home/meeso/mugs_pr2/pred.npy", voxels)
        vox = np.rint(voxels)
        voxels_flat = np.transpose(vox, (0, 2, 1)).flatten()
        self.pub_vox.publish(voxels_flat)
        self.pub_pose.publish(pose)


def main(args):
    rospy.init_node("tensorflow", anonymous=True)
    with tf.Session() as sess:
        odn = PredNode(sess)
        rospy.loginfo("Pred  node started")
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down")


if __name__ == "__main__":
    main(sys.argv)
