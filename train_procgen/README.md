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
* --start_level : Defaults to 0. Start Level
* --test_worker_interval : Defaults to 0. ROLE TBD. Probably some sort of testing. Setting to 1 sets is_test_worker to true. Have not tried other values yet
* --timesteps_total : defaults to 50M. Total number of timesteps.
* --save_interval : defaults to 0. Set to 1 to save periodically. I have not figured out what defines this period (with just one hundred timesteps for example, it does not save, the program terminates. I arbitrarily have been using 20,000 timesteps). We might want to change default to 1, to always save. Note that if run_dir is not changed, or left blank, this WILL overwrite the previous model.
* --load_path : set this to load an existing model. Put full directory
* --run_dir : this is the 'name' of the run. Ie "PPO_with_Dropout_1". It defaults to './train_procgen/models/default', so put the FULL (or relative) file path you desire here. Made it like this to work better on colab. Make sure it's a legal directory. Checkpoints will be saved as run_dir/checkpoints/00001

## Sample Commands (to be run from train-procgen directory)
python -m train_procgen.train2 --timesteps_total 20000 --save_interval 1
python -m train_procgen.train2 --timesteps_total 20000 --save_interval 1 --load_path ./train_procgen/run/checkpoints/00001
python -m train_procgen.train2 --timesteps_total 20000 --save_interval 1 --load_path ./train_procgen/run/checkpoints/00002 --run_dir test2
python -m train_procgen.train2 --timesteps_total 20000 --save_interval 1 --run_dir test2 --test_worker_interval 1
