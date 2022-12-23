#! bash

# MPNet
python test_MPNet.py --logdir /root/autodl-tmp/MPNet/logs/Rain1400/MPNet_second --save_path results/Rain1400/MPNet_second --data_path datasets/test/Rain1400/rainy_image

# MPNet_r
python test_MPNet_r.py --logdir logs/Rain1400/MPNet_r --save_path results/Rain1400/MPNet_r --data_path datasets/test/Rain1400/rainy_image

