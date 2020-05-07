# Setup
<!-- git clone https://github.com/openai/train-procgen.git -->
conda env update --name train-procgen --file train-procgen/environment.yml
conda activate train-procgen
pip install https://github.com/openai/baselines/archive/9ee399f5b20cd70ac0a871927a6cf043b478193f.zip
pip install -e train-procgen

# Train2.py
## Command Line Arguments
* --env_name : Defaults to fruitbot (for us, leave blank)
* --distribution_mode : Defaults to easy (for us, leave blank)
* --num_levels : Defaults to 50. Num levels to train on
* --start_level : Defaults to 0. Start Level is what it sounds like. We want to use this to make sure that we test on different levels.
* --test_worker_interval : Defaults to 0. This is like every nth worker will be for testing. Generally use 1
* --timesteps_total : defaults to 50M. Total number of timesteps.
* --save_interval : defaults to 0. Set to 1 to save periodically. I have not figured out what defines this period (with just one hundred timesteps for example, it does not save, the program terminates. I arbitrarily have been using 20,000 timesteps). We might want to change default to 1, to always save. Note that if run_dir is not changed, or left blank, this WILL overwrite the previous model.
* --load_path : set this to load an existing model. Put full directory
* --run_dir : this is the 'name' of the run. Ie "PPO_with_Dropout_1". It defaults to './train_procgen/models/default', so put the FULL (or relative) file path you desire here. Made it like this to work better on colab. Make sure it's a legal directory. Checkpoints will be saved as run_dir/checkpoints/00001

## Sample Commands (to be run from train-procgen directory)
python -m train_procgen.train2 --timesteps_total 20000 --save_interval 1
python -m train_procgen.train2 --timesteps_total 20000 --save_interval 1 --load_path ./train_procgen/run/checkpoints/00001
python -m train_procgen.train2 --timesteps_total 20000 --save_interval 1 --load_path ./train_procgen/run/checkpoints/00002 --run_dir test2
python -m train_procgen.train2 --timesteps_total 20000 --save_interval 1 --run_dir test2 --test_worker_interval 1


## Example workflow. (repeat per model tested).
* Setup the model
* Train the model for 50M timesteps (default is 50M, but I put it as parameter in case want to change for testing)
  - Trains on --num_levels 50 as default
  - Save every 1 interval
  - Logs at the same interval
python -m train_procgen.train2 --timesteps_total 50000000 --save_interval 1 --run_dir "train_procgen/models/This_Model_1"
* If training pauses, if necessary, resume training:
  - Changed name to not override previous checkpoints
  - Might want to decrease timesteps_total based on how much progress was made previously
  - Note the number of the last checkpoint saved by previous run
python -m train_procgen.train2 --timesteps_total 50000000 --save_interval 1 --run_dir "train_procgen/models/This_Model_1_1" --load_path /train_procgen/This_Model_1_1/checkpoints/00009
* Test on new, unseen levels
  - Not sure what timesteps_total needs to be. Random testing suggests, something between 100 and 20000...
  - Note checkpoint.
  - Might need to do this in a loop, multiple times for each model checkpoint to get change over time. Will be kind of annoying to post process unfortunately. Possibly there is a way to do it in a loop of a train2.py type of file (ie test.py) that sets some of these default values, and then can use the same logger to iterate through each checkpoint and compile a single csv. Worth investigating.
  - -- start_level 50 so that there is no overlap of levels. Need to change this if trained OG model on more levels.
python -m train_procgen.train2 --timesteps_total 50000000 --save_interval 1 --run_dir "train_procgen/models/This_Model_1_Test" --load_path /train_procgen/This_Model_1_1/checkpoints/00100 --start_level 50
