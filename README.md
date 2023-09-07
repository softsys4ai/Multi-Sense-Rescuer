# Multi-Sense-Rescuer

[![website](https://img.shields.io/badge/Project-Website-blue)](https://kksinghal.github.io/multi-sense-rescuer/)

Extending beyond the task of single sound source seeking, this work tackles the task of navigating to multiple-sound emitting destinations, akin to search and rescue scenarios.
This repository contains the modified version of SoundSpaces, Habitat-Sim and Habitat-Lab for supporting multi-destination navigation.

## Motivation
Navigation to a specific sound source has been extensively studied, to the best of our knowledge, the multi-targeted
counterpart of this problem has not received any attention in the existing literature. Unlike simpler single-source navigation,
tackling multiple sound sources becomes difficult due to the inherent challenge of planning the optimal
next target while simultaneously receiving audio signals from multiple emitting sources.
We build on the work of Chen et al., which primarily dealt with navigation to a single sound source, and extend it
to navigation to multiple audio goals—the scenario more akin to real-world search and rescue (SAR) applications.

Motivated by this application, we perceive the problem as a
scenario in which multiple victims, located at random places
in an unknown environment, seek help by making noise. The
agent’s goal is to visit these victims, reflecting the urgency
of delivering crucial supplies and lifesaving information to
them in the real world. The agent complements vision with
aural sensory to infer the location of the sound sources and
the geometry of the surrounding environment, enhancing its
search capabilities.

## Installation 
These are combined installation instructions from [sound-spaces installation instructions](/sound-spaces/INSTALLATION.md) followed by [additional downloads](/sound-spaces/soundspaces/README.md).

```
git clone https://github.com/softsys4ai/Multi-Sense-Rescuer.git

# conda env setup
conda create -n ss python=3.9 cmake=3.14.0 -y
conda activate ss

# Habitat-Sim installation
cd habitat-sim
python setup.py install --headless --audio --with-cuda
cd ..

# Habitat-Lab installation
cd habitat-lab
pip install -e .
cd ..

# SoundSpaces installation
cd sound-spaces
pip install -e .
mkdir data && cd data
mkdir scene_datasets && cd scene_datasets
```

### Dataset
Follow [instructions](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md) to download scene datasets in the folder, e.g., Replica, Matteport3D, Gibson, HM3D. Make sure to download the SceneDatasetConfig file for each dataset.
### Additional download
```
cd ../data
wget https://raw.githubusercontent.com/facebookresearch/rlr-audio-propagation/main/RLRAudioPropagationPkg/data/mp3d_material_config.json
wget http://dl.fbaipublicfiles.com/SoundSpaces/metadata.tar.xz && tar xvf metadata.tar.xz
wget http://dl.fbaipublicfiles.com/SoundSpaces/sounds.tar.xz && tar xvf sounds.tar.xz
wget http://dl.fbaipublicfiles.com/SoundSpaces/datasets.tar.xz && tar xvf datasets.tar.xz
wget http://dl.fbaipublicfiles.com/SoundSpaces/pretrained_weights.tar.xz && tar xvf pretrained_weights.tar.xz
cd metadata

# Replace the <dataset-folder> dataset folder name
ln -s <dataset-folder> default
```

## Usage

Let the maximum number of sound sources in any episode by `N`
To set the number of destinations, go the task config [file](https://github.com/softsys4ai/Multi-Sense-Rescuer/tree/main/sound-spaces/configs/audionav/av_nav) and set the DATASET.NUM_GOALS and SIMULATOR.NUM_GOALS to `N`.
Similarly, set the [num_goals](https://github.com/softsys4ai/Multi-Sense-Rescuer/blob/2a7822664128980a28f84735f60fe2ad6ebacce4/sound-spaces/soundspaces/datasets/audionav_dataset.py#L106) property to `N`.

By default, the number of sources in each episode are sampled using `Uniform(1,N)` fistribution. To fix the number of sources in each episode, comment this [line](https://github.com/softsys4ai/Multi-Sense-Rescuer/blob/2a7822664128980a28f84735f60fe2ad6ebacce4/sound-spaces/soundspaces/datasets/audionav_dataset.py#L239C2-L239C2) and uncomment [next line](https://github.com/softsys4ai/Multi-Sense-Rescuer/blob/2a7822664128980a28f84735f60fe2ad6ebacce4/sound-spaces/soundspaces/datasets/audionav_dataset.py#L240).

Below are some example commands for training and evaluating AudioGoal with depth sensor on Replica. (From sound-spaces README)
1. Training
```
python ss_baselines/av_nav/run.py --exp-config ss_baselines/av_nav/config/audionav/replica/train_telephone/audiogoal_depth.yaml --model-dir data/models/replica/audiogoal_depth CONTINUOUS True
```
2. Validation (evaluate each checkpoint and generate a validation curve)
```
python ss_baselines/av_nav/run.py --run-type eval --exp-config ss_baselines/av_nav/config/audionav/replica/val_telephone/audiogoal_depth.yaml --model-dir data/models/replica/audiogoal_depth CONTINUOUS True
```
3. Test the best validation checkpoint based on validation curve
```
python ss_baselines/av_nav/run.py --run-type eval --exp-config ss_baselines/av_nav/config/audionav/replica/test_telephone/audiogoal_depth.yaml --model-dir data/models/replica/audiogoal_depth EVAL_CKPT_PATH_DIR data/models/replica/audiogoal_depth/data/ckpt.XXX.pth CONTINUOUS True
```
4. Generate demo video with audio
```
python ss_baselines/av_nav/run.py --run-type eval --exp-config ss_baselines/av_nav/config/audionav/replica/test_telephone/audiogoal_depth.yaml --model-dir data/models/replica/audiogoal_depth EVAL_CKPT_PATH_DIR data/models/replica/audiogoal_depth/data/ckpt.220.pth VIDEO_OPTION [\"disk\"] TASK_CONFIG.SIMULATOR.USE_RENDERED_OBSERVATIONS False TASK_CONFIG.TASK.SENSORS [\"POINTGOAL_WITH_GPS_COMPASS_SENSOR\",\"SPECTROGRAM_SENSOR\",\"AUDIOGOAL_SENSOR\"] SENSORS [\"RGB_SENSOR\",\"DEPTH_SENSOR\"] EXTRA_RGB True TASK_CONFIG.SIMULATOR.CONTINUOUS_VIEW_CHANGE True DISPLAY_RESOLUTION 512 TEST_EPISODE_COUNT 1 CONTINUOUS True
```
5. Interactive demo
```
python scripts/interactive_demo.py CONTINUOUS True
```
5. ***[New]*** Training continuous navigation agent 
```
python ss_baselines/av_nav/run.py --exp-config ss_baselines/av_nav/config/audionav/mp3d/train_telephone/audiogoal_depth_ddppo.yaml --model-dir data/models/ss2/mp3d/dav_nav CONTINUOUS True
```

## Questions?
Feel free to ask questions by creating an issue or emailing the author.

## License
Multi-Sense-Rescuer is MIT-licensed, as found in the [LICENSE](LICENSE) file.
