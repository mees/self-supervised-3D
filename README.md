# Self-supervised 3D Shape and Viewpoint Estimation from Single Images for Robotics
Code for deploying the self-supervised single-image 3D shape model from the paper "Self-supervised 3D Shape and Viewpoint Estimation from Single Images for Robotics" (IROS 2019).
Concretely, we showcase how to using the hallucinated 3D object shapes improve the performance on the task of grasping real-world objects with a PR2 robot.

#Installation


#Instructions
start mask-rcnn with caffe2

conda activate caffe2_python2

CUDA_VISIBLE_DEVICES=1 python tools/infer_simple.py --cfg configs/bbox2mask_vg/eval_sw_R101/runtest_clsbox_2_layer_mlp_nograd_R101.yaml     --output-dir /tmp/detectron-visualizations-vg3k-R101     --image-ext jpg     --thresh 0.1 --use-vg3k     --wts /home/meeso/seg_every_thing/lib/datasets/data/trained_models/33219850_model_final_coco2vg3k_seg.pkl     demo_vg3k

check segmentation in rviz topic /segmented_mug_rgb
