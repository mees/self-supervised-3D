# Self-supervised 3D Shape and Viewpoint Estimation from Single Images for Robotics
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Code for deploying the self-supervised single-image 3D shape model from the paper "Self-supervised 3D Shape and Viewpoint Estimation from Single Images for Robotics" (IROS 2019).
Concretely, we showcase how using the hallucinated 3D object shapes improve the performance on the task of grasping real-world objects with a PR2 robot.

## Reference
If you find the code helpful please consider citing our work
```
@INPROCEEDINGS{mees19iros,
  author = {Oier Mees and Maxim Tatarchenko and Thomas Brox and Wolfram Burgard},
  title = {Self-supervised 3D Shape and Viewpoint Estimation from Single Images for Robotics},
  booktitle = {Proceedings of the International Conference on Intelligent Robots and Systems (IROS)},
  year = 2019,
  address = {Macao, China},
}
```

# Installation
  - Follow the instructions to install [mask-rcnn](seg_every_thing) in its own conda environment with python2.
  - Install Tensorflow with  ```conda create -n tf-gpu tensorflow-gpu==1.13.1```.
  - Install Robot Operating System ([ROS](https://www.ros.org/))
  - Download and extract pretrained model ```sh ss3d/trained_models/download_model.sh```

# Deployment
  - First we need to detect and segment the mugs in the scene. With mask-rcnn we segment the mugs and after padding them we publish the image via [ROS](https://www.ros.org/).
    You can then visualize segmentation results in Rviz under the topic ```/segmented_mug_rgb```.
   <pre>
    conda activate caffe2_python2
    python tools/infer_simple.py --cfg configs/bbox2mask_vg/eval_sw_R101/runtest_clsbox_2_layer_mlp_nograd_R101.yaml     --output-dir /tmp/detectron-visualizations-vg3k-R101     --image-ext jpg     --thresh 0.1 --use-vg3k     --wts /home/meeso/seg_every_thing/lib/datasets/data/trained_models/33219850_model_final_coco2vg3k_seg.pkl     demo_vg3k
   </pre>

- Predict the 3D shape and pose given the input images with:
  <pre>
  conda activate tf-gpu
  python ss3d/imageTo3D.py
  </pre>
- Convert the numpy voxel grid to a point cloud and transform it to the robot frame
  ```rosrun mesh2cloud mesh2cloud_node ```
