#! bash

# Rain100H
python train_MPNet_r.py --save_path logs/Rain100H/MPNet_r --data_path datasets/train/RainTrainH

# Rain100L
python train_MPNet_r.py --save_path logs/Rain100L/MPNet_r --data_path datasets/train/RainTrainL

# Rain12600
python train_MPNet_r.py --save_path logs/Rain1400/MPNet_r --data_path datasets/train/Rain12600
