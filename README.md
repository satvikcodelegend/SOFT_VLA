# SOFT_VLA
MuJoCo-based soft continuum robot control project with trajectory tracking, inverse pressure modeling, and VLA-based pressure residual control.

Project Structure

SOFT_VLA/
│
├── config/                 # Project configuration and parameters
├── control/                # Robot control, kinematics, pressure model, inverse MLP
├── data/                   # Logging utilities; generated experiment data is ignored
├── dataset/                # Dataset generators and dataset metadata/manifests
├── learning/               # Inverse-model dataset generation and training
├── model/                  # MuJoCo scene, robot meshes, model metrics
├── models/                 # Training/evaluation metrics and model-related files
├── soft_robot_vla/         # VLA model, dataset, training and prediction code
├── tools/                  # Mesh and dataset utility scripts
│
├── main_figure8.py         # Figure-8 experiment
├── main_heart.py           # Heart trajectory experiment
├── main_random.py          # Random/smooth trajectory experiment
│
├── plot_figure8.py         # Figure-8 result plotting
├── plot_heart.py           # Heart result plotting
├── plot_random.py          # Random result plotting
│
├── experiment_logger.py    # Experiment logging
└── README.md

What Each Main Folder Is For

Folder

Purpose

config/

Configuration files and robot/controller parameters

control/

Main control components used by the robot

dataset/

Dataset generation scripts, manifests and metadata

learning/

Training pipeline for the inverse pressure model

soft_robot_vla/

VLA dataset/model/training/prediction code

model/

MuJoCo XML scene and robot meshes

models/

Saved metrics/evaluation information

data/

Logging and experiment-data utilities

tools/

Supporting mesh/dataset scripts

The main_*.py files are the easiest place to start when running trajectory experiments.

Using the Project on Another Laptop

1. Clone the repository

git clone https://github.com/satvikcodelegend/SOFT_VLA.git
cd SOFT_VLA

2. Create/activate the Python environment

The project was developed using the openvla-oft Conda environment.

conda create -n openvla-oft python=3.x
conda activate openvla-oft

Install the Python packages required by the project in this environment.

The repository currently does not contain a requirements.txt, so the exact package versions are not fixed in GitHub.

3. Check the MuJoCo model

The MuJoCo scene and robot meshes are already inside:

model/
model/meshes/

Run Python scripts from the root of the repository:

cd SOFT_VLA

Datasets and Large Model Files

Large generated files are intentionally not stored in GitHub.

The repository ignores:

*.npz
*.npy
*.pt
*.pth
*.ckpt
dataset/data/
data/experiments/
results/

This keeps the GitHub repository manageable.

If the datasets are needed

They must either be:

obtained separately from the project owner, or

regenerated using the dataset-generation scripts.

Important dataset-generation/training scripts include:

python dataset/generate_inverse_dataset.py
python learning/train_inverse_mlp.py
python soft_robot_vla/generate_curved_dataset.py
python soft_robot_vla/train.py

The exact order depends on which trained model/checkpoint is being used. In particular, VLA data generation can depend on the trained inverse model.

Trained checkpoints

Trained model checkpoints are also excluded from GitHub. A new user therefore needs the required checkpoint files separately, or must retrain the models.

Running the Experiments

After the environment, MuJoCo setup, required datasets and checkpoints are available:

Figure-8

python main_figure8.py

Heart trajectory

python main_heart.py

Random trajectory

python main_random.py

Results can then be visualized using:

python plot_figure8.py
python plot_heart.py
python plot_random.py

Run these commands from the project root.

Dataset / Model Workflow

For a new setup, the general workflow is:

Clone repository
      ↓
Set up Python/Conda environment
      ↓
Make required datasets available
      ↓
Make trained checkpoints available
      ↓
Run trajectory experiment
      ↓
Inspect / plot results

What Is Stored in GitHub

Stored

Python source code

Controller code

Dataset-generation code

Training code

VLA code

Configuration files

MuJoCo XML/model files

Robot meshes

Dataset manifests and metadata

Evaluation/training metrics

Plotting scripts

README and project utilities

Not stored

Large .npz / .npy datasets

Model checkpoint files such as .pt, .pth, .ckpt

Generated experiment data

Generated results

Python cache files

Important

Before running experiments on another laptop, make sure the required:

Python packages

MuJoCo environment

datasets

trained checkpoints

are available.

The GitHub repository contains the code and project structure, while large generated data and trained checkpoint files are kept outside Git.
