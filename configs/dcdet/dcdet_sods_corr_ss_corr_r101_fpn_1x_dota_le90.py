_base_ = './oriented_rcnn_sdc_ss_corr_r50_fpn_1x_dota_le90.py'

#model
model = dict(pretrained='torchvision://resnet101',backbone=dict(depth=101))