# Installation And Setup
1. Download CoppeliaSim_Edu_V4_1_0_Ubuntu18_04/ at https://www.coppeliarobotics.com/downloads# in the current directory
2. `git pull --recursive https://github.com/xli4217/lof_robot_learning.git`
3. conda env create -f env.yml
4. source setup_path.sh
5. Follow https://github.com/stepjam/PyRep to install PyRep (installed in libs/PyRep)
6. sudo apt install xvfb (for training headless)

# Train And Run
1. `conda activate lof`
2. To start training run `python src/train.py` (by default training is done headless, if visualization of the training is desired, in src/train.py modify `HEADLESS=False` )
3. To test trained policy run `python src/rollout.py`
4. To plot learning curve, in package root directory run `python -m spinup.run plot experiments/<experiment name>/
`
