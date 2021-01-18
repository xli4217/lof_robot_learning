# Installation And Setup
1. sudo apt install xvfb (for training headless)
2. `git clone --recurse-submodules https://github.com/xli4217/lof_robot_learning.git`
3. Download CoppeliaSim_Edu_V4_1_0_Ubuntu18_04/ at https://www.coppeliarobotics.com/downloads# in the lof_robot_learning directory
4. Follow https://github.com/stepjam/PyRep to install PyRep (installed in libs/PyRep)
5. conda env create -f env.yml
6. source setup_path.sh

# Train And Run
1. `conda activate lof`
2. To start training run `python src/train.py` (by default training is done headless, if visualization of the training is desired, in src/train.py modify `HEADLESS=False` )
3. To test trained policy run `python src/rollout.py`
4. To plot learning curve, in package root directory run `python -m spinup.run plot experiments/<experiment name>/
`
