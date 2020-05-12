Describes your project, and how to run the project if there are any requirements.

Quick run with fruitbot environment and defaults:

Train run with logged results:
python -m run_agent --num_levels 50 --timesteps_total 1000000 --run_dir "YOUR_RUN_DIRECTORY HERE"

Test run on unseen levels with best benchmark model and logged results:
python -m run_agent --test_mode True --num_levels 500 --start_level 50 --timesteps_total 20000 --load_path models\ppo2_train_run\checkpoints\00610 --run_dir "YOUR_RUN_DIRECTORY HERE"

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
