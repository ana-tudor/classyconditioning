Running the Procgen Fruitbot Environment and Training Our models
Additional Tools used for visualization, processing, and development are described below.

Please use requirements.txt for a full list of requirements which can be quickly accessed via pip install -r requirements.txt
Contained within the file are additional requirements which cannot be installed via the previous command:
  pip install https://github.com/openai/baselines/archive/9ee399f5b20cd70ac0a871927a6cf043b478193f.zip
  git clone https://github.com/openai/train-procgen.git
  pip install -e train-procgen
Additionally, a .yml file is included for a pre-built environment.

The code for our model is stored within the fruitbot_ppo folder

Quick run with fruitbot environment and defaults:

Train run with logged results:
python -m run_agent --num_levels 50 --timesteps_total 1000000 --run_dir "YOUR_RUN_DIRECTORY HERE"

Test run on unseen levels with best benchmark model and logged results:
python -m run_agent --test_mode True --num_levels 500 --start_level 50 --timesteps_total 20000 --load_path models\ppo2_train_run\checkpoints\00610 --run_dir "YOUR_RUN_DIRECTORY HERE"

*Important to note: the test-time convnet sizes must match the training convnet sizes*

Test visualization with best benchmark model:
*coming soon!*

run_agent.py specifications:

usage: run_agent.py [-h] [--env_name ENV_NAME]
                    [--distribution_mode {easy,hard,exploration,memory,extreme}]
                    [--num_levels NUM_LEVELS] [--start_level START_LEVEL]
                    [--timesteps_total TIMESTEPS_TOTAL]
                    [--save_interval SAVE_INTERVAL] [--load_path LOAD_PATH]
                    [--run_dir RUN_DIR] [--test_mode TEST_MODE]
                    [--variable_oi VARIABLE_OI]
                    [--values_oi VALUES_OI [VALUES_OI ...]]
                    [--num_envs NUM_ENVS] 
                    [--epopt_timestep] [--paths]


Process fruitbot_ppo agent training arguments.

optional arguments:
  -h, --help            show this help message and exit
  --env_name ENV_NAME   Provide an environment name available in procgen
  --distribution_mode {easy,hard,exploration,memory,extreme}
  --num_levels NUM_LEVELS
                        Number of levels to run in the environment
  --start_level START_LEVEL
                        The point in the list of levels available to the
                        environment at which to index into, eg. --num_levels
                        50 --start_level 50 makes levels 50-99 available to
                        this environment
  --timesteps_total TIMESTEPS_TOTAL
                        The desired number of total timesteps spent training
                        or testing
  --save_interval SAVE_INTERVAL
                        The interval spent in between checkpoints saved, 0
                        will save none, and 1 will save checkpoints after
                        every model update.
  --load_path LOAD_PATH
                        The relative or absolute path to a model checkpoint if
                        an initial load from this checkpoint is desired
  --run_dir RUN_DIR     The relative or absolute path to the directory where
                        results should be logged
  --test_mode TEST_MODE
                        True if the model should run as a testing agent, and
                        should not be updated
  --variable_oi VARIABLE_OI
                        A global variable name of interest for hyperparameter
                        searching
  --values_oi VALUES_OI [VALUES_OI ...]
                        Values of interest for hyperparameter searching
  --num_envs NUM_ENVS   The number of environments across which the agent
                        should be run in parallel
  --epopt_timestep EPOPT_TIMESTEP
                        The number of burn-in timesteps before EPOpt sampling 
                        begins
  --paths PATHS         The number of trajectories sampled with EPOpt of which to 
                        take the minimum 

Tools - Please look inside respective files for detailed documentation. 

  Visualize Observations: Model observations are different than rendered images. This script shows them side-by-side for comparison.
  python visualize_observations.py

  Plotting: This function neatly plots training and testing performance from csvs. Useful for comparing different model performance.
  python plot.py --diff_models True --model_names acer_explore1 a2c_test PPO2_initial_test --columns 14 1 1 --time_columns -1 -2 -1 --single_plot True

  Run Averaging: We ran multiple runs with different seeds. To get a better idea of average performance we use this script to average the log files.
  python average_logs.py --model_names 50_epopt_ckpt05 50_epopt_ckpt10 --save_name 50_epopt_avg_ckpts --across_time True

  Run Baseline: This file runs to baseline PPO2 that is implemented with Procgen. It includes the original hyperparameters as well.
  python run_baseline.py --timesteps_total 10000000 --save_interval 610 --run_dir "YOUR_RUN_DIRECTORY"

  Game Visualization: This file allows for the loading of a model checkpoint to visualize how the agent plays in the fruitbot game environment.
  Can also save gifs of games.
  python visualize.py --timesteps 1000 --load_path "PATH TO MODEL CHECKPOINT" --save_path "PATH TO SAVE GIF" --num_envs 4
