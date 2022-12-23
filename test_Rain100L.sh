#! bash 

# MPNet
python test_MPNet.py --logdir /root/autodl-tmp/MPNet/logs/Rain100L/MPNet_second --save_path results/Rain100L/MPNet_second --data_path datasets/test/Rain100L/rainy

# MPNet_r
python test_MPNet_r.py --logdir logs/Rain100L/MPNet_r --save_path results/Rain100L/MPNet_r --data_path datasets/test/Rain100L/rainy


