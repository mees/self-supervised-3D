import matplotlib
import numpy as np
from PIL import Image
import StringIO
import tensorflow as tf

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# expects both inputs in radians
def euler_2_quat(azi, elev):
    c1 = tf.cos(azi / 2)
    s1 = tf.sin(azi / 2)
    c2 = tf.cos(elev / 2)
    s2 = tf.sin(elev / 2)
    c1c2 = c1 * c2
    s1s2 = s1 * s2
    w = c1c2
    x = s1s2
    y = s1 * c2
    z = c1 * s2
    result = tf.concat([w, x, y, z], axis=1)
    return result


# expects both inputs in radians
def euler_2_quat_np(azi, elev):
    c1 = np.cos(azi / 2)
    s1 = np.sin(azi / 2)
    c2 = np.cos(elev / 2)
    s2 = np.sin(elev / 2)
    c1c2 = c1 * c2
    s1s2 = s1 * s2
    w = c1c2
    x = s1s2
    y = s1 * c2
    z = c1 * s2
    result = np.concatenate([w, x, y, z], axis=1)
    return result


def save_image(image, image_file):
    """Function that dumps the image to disk."""
    buf = StringIO.StringIO()
    image.save(buf, format="JPEG")
    with open(image_file, "w") as f:
        f.write(buf.getvalue())
    return None


def save_np_image(inp_array, image_file):
    """Function that dumps the image to disk."""
    inp_array = np.clip(inp_array, 0, 255).astype(np.uint8)
    image = Image.fromarray(inp_array)
    buf = StringIO.StringIO()
    image.save(buf, format="JPEG")
    with open(image_file, "w") as f:
        f.write(buf.getvalue())
    return None


def vis_pose_pred_dist(pred_azimuths, step, elev, log_dir, suffix):
    angles_deg = np.concatenate(pred_azimuths, axis=0)
    fig = plt.figure(1)
    if elev:
        x_tck = [5, 10, 15, 20, 25, 30, 35, 40]
        count, bins, ignored = plt.hist(angles_deg, 8)
    else:
        x_tck = [
            15,
            30,
            45,
            60,
            75,
            90,
            105,
            120,
            135,
            150,
            165,
            180,
            195,
            210,
            225,
            240,
            255,
            270,
            285,
            300,
            315,
            330,
            345,
            360,
        ]
        count, bins, ignored = plt.hist(angles_deg, 24)
    count2 = count / np.max(count)
    plt.bar(x_tck, count2, width=14, align="edge")
    fig.canvas.draw()
    fig.savefig(log_dir + "/" + str(step) + "_" + str(suffix) + ".png")
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    image = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    image = np.expand_dims(image, 0)
    plt.close(fig)
    fig.clf()
    return image
