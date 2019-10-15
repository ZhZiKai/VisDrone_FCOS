python setup.py build develop

python tools/test_net.py \
    --config-file configs/10012_1100_cocopretrained_2x_P2.yaml \
    MODEL.WEIGHT training_dir/10012_1100_cocopretrained_2x_P2/model_0080000.pth \
    TEST.IMS_PER_BATCH 2
    
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.380
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.611
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.394
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.294
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.491
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.599
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.138
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.422
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.600
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.521
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.713
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.793
# Maximum f-measures for classes:
# [-0.0, 0.7156237298797176, 0.6451260509885888, 0.4759035095358356, 0.8595600533704361, 0.6093999240410178, 0.5601194921583271, 0.5431049287308948, 0.3668335856527309, 0.732407468934813, 0.702405968293441, -0.0]
# Score thresholds for classes (used in demos for visualization purposes):
# [-1.0, 0.28051263093948364, 0.2513558566570282, 0.26865154504776, 0.2974579930305481, 0.3063768446445465, 0.2857804000377655, 0.26119425892829895, 0.24696026742458344, 0.33234840631484985, 0.27221328020095825, -1.0]
# 2019-10-10 10:31:09,611 maskrcnn_benchmark.inference INFO: OrderedDict([('bbox', OrderedDict([('AP', 0.37972144595440815), ('AP50', 0.6107789530121627), ('AP75', 0.3938617744057577), ('APs', 0.29397640606335607), ('APm', 0.4908982622565456), ('APl', 0.5987360971577397)]))])
