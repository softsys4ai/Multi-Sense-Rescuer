# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Any, List, Optional
from abc import ABC
from collections import defaultdict, namedtuple
import logging
import time
import pickle
import os
import json

import librosa
import psutil
import scipy
from scipy.io import wavfile
from scipy.signal import fftconvolve
import numpy as np
import networkx as nx
from gym import spaces

from itertools import permutations

from habitat.core.registry import registry
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis, quat_from_coeffs, quat_to_angle_axis
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSimSensor, overwrite_config
from habitat.core.simulator import (
    AgentState,
    Config,
    Observations,
    SensorSuite,
    ShortestPathPoint,
    Simulator,
)
from habitat.tasks.nav.nav import (
    NavigationGoal
)

import csv
def writer(data):
    f = open("./logs.txt", "a")
    f.write(data)
    f.close()
    
from soundspaces.utils import load_metadata
from soundspaces.mp3d_utils import HouseReader


def calculate_mem_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024.0 / 1024 / 1024  # in GB


def crossfade(x1, x2, sr):
    crossfade_samples = int(0.05 * sr)  # 30 ms
    x2_weight = np.arange(crossfade_samples + 1) / crossfade_samples
    x1_weight = np.flip(x2_weight)
    x3 = [x1[:, :crossfade_samples+1] * x1_weight + x2[:, :crossfade_samples+1] * x2_weight, x2[:, crossfade_samples+1:]]

    return np.concatenate(x3, axis=1)


@registry.register_simulator()
class ContinuousSoundSpacesSim(Simulator, ABC):
    r"""Changes made to simulator wrapper over habitat-sim

    This simulator first loads the graph of current environment and moves the agent among nodes.
    Any sounds can be specified in the episode and loaded in this simulator.
    Args:
        config: configuration for initializing the simulator.
    """

    def action_space_shortest_path(self, source: AgentState, targets: List[AgentState], agent_id: int = 0) -> List[
            ShortestPathPoint]:
        pass

    def __init__(self, config: Config) -> None:
        self.config = self.habitat_config = config
        agent_config = self._get_agent_config()
        sim_sensors = []

        """print(agent_config)
                HEIGHT: 1.5
                IS_SET_START_STATE: False
                RADIUS: 0.1
                SENSORS: ['RGB_SENSOR']
                START_POSITION: [0, 0, 0]
                START_ROTATION: [0, 0, 0, 1]
        """
        
        ## RGB sensor, when using interactive_demo
        for sensor_name in agent_config.SENSORS:
            sensor_cfg = getattr(self.config, sensor_name)
            sensor_type = registry.get_sensor(sensor_cfg.TYPE)

            assert sensor_type is not None, "invalid sensor type {}".format(
                sensor_cfg.TYPE
            )
            sim_sensors.append(sensor_type(sensor_cfg))

        self._sensor_suite = SensorSuite(sim_sensors)
        self.sim_config = self.create_sim_config(self._sensor_suite)
        self._current_scene = self.sim_config.sim_cfg.scene_id
        self._action_space = spaces.Discrete(
            len(self.sim_config.agents[0].action_space)
        )
        self._prev_sim_obs = None

        self.goals = None ## KS; set episode goal positions in reset() in environements.py 

        self._source_position_index = None
        self._receiver_position_index = None
        self._rotation_angle = None
        self._current_sound = None
        self._offset = None
        self._duration = None
        self._audio_index = None
        self._audio_length = None
        self._source_sound_dict = dict()
        self._sampling_rate = None
        self._node2index = None
        self._scene_observations = None
        self._episode_step_count = None
        self._is_episode_active = None
        self._position_to_index_mapping = dict()
        self._previous_step_collided = False
        self._house_readers = dict()

        self.points, self.graph = load_metadata(self.metadata_dir)
        self._sim = habitat_sim.Simulator(config=self.sim_config)
        
        self.add_acoustic_config()
        self._last_rir = None
        self._current_sample_index = 0

    def add_acoustic_config(self):
        ## Add NUM_GOALS sensors. In reconfigure(), map first n sensors to n goals for each episode
        ## audio_sensor1, audio_sensor2, audio_sensor3, ...
  
        for i in range(self.config.NUM_GOALS):         
            audio_sensor_spec = habitat_sim.AudioSensorSpec()
            audio_sensor_spec.uuid = "audio_sensor"+str(i+1)
            audio_sensor_spec.enableMaterials = False
            audio_sensor_spec.channelLayout.type = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Binaural
            audio_sensor_spec.channelLayout.channelCount = 2
            audio_sensor_spec.acousticsConfig.sampleRate = self.config.AUDIO.RIR_SAMPLING_RATE
            audio_sensor_spec.acousticsConfig.threadCount = 10
            audio_sensor_spec.acousticsConfig.indirectRayCount = 500
            audio_sensor_spec.acousticsConfig.temporalCoherence = True
            audio_sensor_spec.acousticsConfig.transmission = True
            self._sim.add_sensor(audio_sensor_spec)


    def create_sim_config(
        self, _sensor_suite: SensorSuite
    ) -> habitat_sim.Configuration:
        sim_config = habitat_sim.SimulatorConfiguration()
        # Check if Habitat-Sim is post Scene Config Update
        if not hasattr(sim_config, "scene_id"):
            raise RuntimeError(
                "Incompatible version of Habitat-Sim detected, please upgrade habitat_sim"
            )
        overwrite_config(
            config_from=self.config.HABITAT_SIM_V0,
            config_to=sim_config,
            # Ignore key as it gets propogated to sensor below
            ignore_keys={"gpu_gpu"},
        )
        sim_config.scene_id = self.config.SCENE
        sim_config.enable_physics = False
        # sim_config.scene_dataset_config_file = 'data/scene_datasets/mp3d/mp3d.scene_dataset_config.json'
        # sim_config.scene_dataset_config_file = 'data/scene_datasets/replica/replica.scene_dataset_config.json'
        agent_config = habitat_sim.AgentConfiguration()
        overwrite_config(
            config_from=self._get_agent_config(),
            config_to=agent_config,
            # These keys are only used by Hab-Lab
            ignore_keys={
                "is_set_start_state",
                # This is the Sensor Config. Unpacked below
                "sensors",
                "start_position",
                "start_rotation",
                "goal_position",
                "offset",
                "duration",
                "sound_id",
                "mass",
                "linear_acceleration",
                "angular_acceleration",
                "linear_friction",
                "angular_friction",
                "coefficient_of_restitution",
                "distractor_sound_id",
                "distractor_position_index"
            },
        )

        sensor_specifications = []
        for sensor in _sensor_suite.sensors.values():
            assert isinstance(sensor, HabitatSimSensor)
            sim_sensor_cfg = sensor._get_default_spec()  # type: ignore[misc]
            overwrite_config(
                config_from=sensor.config,
                config_to=sim_sensor_cfg,
                # These keys are only used by Hab-Lab
                # or translated into the sensor config manually
                ignore_keys=sensor._config_ignore_keys,
                # TODO consider making trans_dict a sensor class var too.
                trans_dict={
                    "sensor_model_type": lambda v: getattr(
                        habitat_sim.FisheyeSensorModelType, v
                    ),
                    "sensor_subtype": lambda v: getattr(
                        habitat_sim.SensorSubType, v
                    ),
                },
            )
            sim_sensor_cfg.uuid = sensor.uuid
            sim_sensor_cfg.resolution = list(
                sensor.observation_space.shape[:2]
            )

            # TODO(maksymets): Add configure method to Sensor API to avoid
            # accessing child attributes through parent interface
            # We know that the Sensor has to be one of these Sensors
            sim_sensor_cfg.sensor_type = sensor.sim_sensor_type
            sim_sensor_cfg.gpu2gpu_transfer = (
                self.config.HABITAT_SIM_V0.GPU_GPU
            )
            sensor_specifications.append(sim_sensor_cfg)

        agent_config.sensor_specifications = sensor_specifications
        agent_config.action_space = registry.get_action_space_configuration(
            self.config.ACTION_SPACE_CONFIG
        )(self.config).get()

        return habitat_sim.Configuration(sim_config, [agent_config])

    @property
    def sensor_suite(self) -> SensorSuite:
        return self._sensor_suite

    def _update_agents_state(self) -> bool:
        is_updated = False
        for agent_id, _ in enumerate(self.config.AGENTS):
            agent_cfg = self._get_agent_config(agent_id)
            if agent_cfg.IS_SET_START_STATE:
                self.set_agent_state(
                    agent_cfg.START_POSITION,
                    agent_cfg.START_ROTATION,
                    agent_id,
                )
                is_updated = True

        return is_updated

    def _get_agent_config(self, agent_id: Optional[int] = None) -> Any:
        if agent_id is None:
            agent_id = self.config.DEFAULT_AGENT_ID
        agent_name = self.config.AGENTS[agent_id]
        agent_config = getattr(self.config, agent_name)
        return agent_config

    def get_agent_state(self, agent_id: int = 0) -> habitat_sim.AgentState:
        return self._sim.get_agent(agent_id).get_state()

    def set_agent_state(
        self,
        position: List[float],
        rotation: List[float],
        agent_id: int = 0,
        reset_sensors: bool = True,
    ) -> bool:
        agent = self._sim.get_agent(agent_id)
        new_state = self.get_agent_state(agent_id)
        new_state.position = position
        new_state.rotation = rotation

        # NB: The agent state also contains the sensor states in _absolute_
        # coordinates. In order to set the agent's body to a specific
        # location and have the sensors follow, we must not provide any
        # state for the sensors. This will cause them to follow the agent's
        # body
        new_state.sensor_states = {}
        agent.set_state(new_state, reset_sensors)
        return True

    @property
    def binaural_rir_dir(self):
        return os.path.join(self.config.AUDIO.BINAURAL_RIR_DIR, self.config.SCENE_DATASET, self.current_scene_name)

    @property
    def source_sound_dir(self):
        return self.config.AUDIO.SOURCE_SOUND_DIR

    @property
    def distractor_sound_dir(self):
        return self.config.AUDIO.DISTRACTOR_SOUND_DIR

    @property
    def metadata_dir(self):
        return os.path.join(self.config.AUDIO.METADATA_DIR, self.config.SCENE_DATASET, self.current_scene_name)

    @property
    def current_scene_name(self):
        # config.SCENE (_current_scene) looks like 'data/scene_datasets/replica/office_1/habitat/mesh_semantic.ply'
        return self._current_scene.split('/')[3]

    @property
    def current_scene_observation_file(self):
        return os.path.join(self.config.SCENE_OBSERVATION_DIR, self.config.SCENE_DATASET,
                            self.current_scene_name + '.pkl')

    @property
    def current_source_sound(self):
        return self._source_sound_dict[self._current_sound]

    @property
    def is_silent(self):
        return self._episode_step_count > self._duration

    @property
    def pathfinder(self):
        return self._sim.pathfinder

    def get_agent(self, agent_id):
        return self._sim.get_agent(agent_id)

    def reconfigure(self, config: Config, goals: List[NavigationGoal]) -> None:
        self.config = config
        self.num_goals = len(goals)
        self.goals = goals

        if hasattr(self.config.AGENT_0, 'OFFSET'):
            self._offset = int(self.config.AGENT_0.OFFSET)
        else:
            self._offset = 0
        if self.config.AUDIO.EVERLASTING:
            self._duration = 500
        else:
            assert hasattr(self.config.AGENT_0, 'DURATION')
            self._duration = int(self.config.AGENT_0.DURATION)
        self._audio_index = 0
        is_same_sound = config.AGENT_0.SOUND_ID == self._current_sound
        if not is_same_sound:
            self._current_sound = self.config.AGENT_0.SOUND_ID
            self._load_single_source_sound()
            logging.debug("Switch to sound {} with duration {} seconds".format(self._current_sound, self._duration))

        is_same_scene = config.SCENE == self._current_scene
        if not is_same_scene:
            self._current_scene = config.SCENE
            logging.debug('Current scene: {} and sound: {}'.format(self.current_scene_name, self._current_sound))

            self._sim.close()
            del self._sim
            self.sim_config = self.create_sim_config(self._sensor_suite)
            self._sim = habitat_sim.Simulator(self.sim_config)
            self.add_acoustic_config()
            for i in range(self.num_goals):
                audio_sensor = self._sim.get_agent(0)._sensors["audio_sensor"+str(i+1)]
                audio_sensor.setAudioMaterialsJSON("data/mp3d_material_config.json")
            
            logging.debug('Loaded scene {}'.format(self.current_scene_name))

            self.points, self.graph = load_metadata(self.metadata_dir)

        self._update_agents_state()

        self.goals = self.set_goals(goals)
        self.goals_complete = [False] * len(self.goals)

        for i in range(len(self.goals)):
            audio_sensor = self._sim.get_agent(0)._sensors["audio_sensor"+str(i+1)]
            # writer("\ngoal:"+str(self.goals[i].position)+"\n")
            audio_sensor.setAudioSourceTransform(self.goals[i].position + np.array([0, 1.5, 0])) #1.5 is the offset for the height
        
        self._episode_step_count = 0
        self._last_rir = None
        self._current_sample_index = np.random.randint(0, self.config.AUDIO.RIR_SAMPLING_RATE * self.config.STEP_TIME, len(self.goals))

    def set_goals(self, goals):
        agent_position = self.get_agent_state()
        perm = permutations(goals, len(goals))
        min_dist = np.inf
        best_seq = None
        
        for seq in list(perm):
            dist = self._trajectory_length(agent_position, seq)
            if dist == np.inf: print(dist)
            if dist < min_dist:
                min_dist = dist
                best_seq = seq

        return best_seq

    @staticmethod
    def position_encoding(position):
        return '{:.2f}_{:.2f}_{:.2f}'.format(*position)

    def _position_to_index(self, position):
        if self.position_encoding(position) in self._position_to_index_mapping:
            return self._position_to_index_mapping[self.position_encoding(position)]
        else:
            raise ValueError("Position misalignment.")

    def reset(self):
        logging.debug('Reset simulation')
        sim_obs = self._sim.reset()
        if self._update_agents_state():
            sim_obs = self._sim.get_sensor_observations()

        self._is_episode_active = True
        self._prev_sim_obs = sim_obs
        self._previous_step_collided = False
        # Encapsule data under Observations class
        observations = self._sensor_suite.get_observations(sim_obs)

        return observations

    def step(self, action):
        """
        All angle calculations in this function is w.r.t habitat coordinate frame, on X-Z plane
        where +Y is upward, -Z is forward and +X is rightward.
        Angle 0 corresponds to +X, angle 90 corresponds to +y and 290 corresponds to 270.

        :param action: action to be taken
        :return:
        Dict of observations
        """
        assert self._is_episode_active, (
            "episode is not active, environment not RESET or "
            "STOP action called previously"
        )

        self._last_rir = []
        for i in range(len(self.goals)):
                binaural_rir = np.transpose(np.array(self._prev_sim_obs["audio_sensor"+str(i+1)]))
                self._last_rir.append(binaural_rir)

        sim_obs = self._sim.step(action)
        self._prev_sim_obs = sim_obs
        observations = self._sensor_suite.get_observations(sim_obs)
        self._episode_step_count += 1
        self._current_sample_index = ((self._current_sample_index + self.config.AUDIO.RIR_SAMPLING_RATE *
                                         self.config.STEP_TIME).astype(np.int64) % self._sound_source_length[:len(self.goals)])

        return observations

    def _load_source_sounds(self):
        # load all mono files at once
        sound_files = os.listdir(self.source_sound_dir)
        for sound_file in sound_files:
            sound = sound_file.split('.')[0]
            audio_data, sr = librosa.load(os.path.join(self.source_sound_dir, sound),
                                          sr=self.config.AUDIO.RIR_SAMPLING_RATE)
            self._source_sound_dict[sound] = audio_data
            self._audio_length = audio_data.shape[0] // self.config.AUDIO.RIR_SAMPLING_RATE

    def _load_single_source_sound(self):
        # if self._current_sound not in self._source_sound_dict:
        #     audio_data, sr = librosa.load(os.path.join(self.source_sound_dir, self._current_sound),
        #                                   sr=self.config.AUDIO.RIR_SAMPLING_RATE)
        #     if audio_data.shape[0]//self.config.AUDIO.RIR_SAMPLING_RATE == 1:
        #         audio_data = np.concatenate([audio_data] * 3, axis=0)  # duplicate to be longer than longest RIR
        #     self._source_sound_dict[self._current_sound] = audio_data
        # self._audio_length = self._source_sound_dict[self._current_sound].shape[0]//self.config.AUDIO.RIR_SAMPLING_RATE
        self.current_sounds = ["telephone.wav", "telephone.wav", "telephone.wav", "telephone.wav", "baby_cry_loop.wav", "baby_crying.mp3", "help1.wav", "help2.wav"]
        self._audio_length = []
        self._sound_source_length = []
        for sound in self.current_sounds:
            audio_data, sr = librosa.load(os.path.join(self.source_sound_dir, sound),
                                          sr=self.config.AUDIO.RIR_SAMPLING_RATE)
            if audio_data.shape[0]//self.config.AUDIO.RIR_SAMPLING_RATE == 1:
                audio_data = np.concatenate([audio_data] * 3, axis=0)  # duplicate to be longer than longest RIR
            self._source_sound_dict[sound] = audio_data
            self._audio_length.append(self._source_sound_dict[sound].shape[0]//self.config.AUDIO.RIR_SAMPLING_RATE)
            self._sound_source_length.append(self._source_sound_dict[sound].shape[0])

    def _compute_audiogoal(self):
        sampling_rate = self.config.AUDIO.RIR_SAMPLING_RATE
        if self._episode_step_count > self._duration:
            logging.debug('Step count is greater than duration. Empty spectrogram.')
            audiogoal = np.zeros((2, sampling_rate))
        else:
            audiogoal = np.zeros((2, sampling_rate))
            for i in range(len(self.goals)):
                if self.goals_complete[i]: continue
                binaural_rir = np.transpose(np.array(self._prev_sim_obs["audio_sensor"+str(i+1)]))
                sound_source = self._source_sound_dict[self.current_sounds[i]]
                audiogoal += self._convolve_with_rir(binaural_rir, sound_source, self._current_sample_index[i])

            if self.config.AUDIO.CROSSFADE and self._last_rir is not None:
                audiogoal_from_last_rir = np.zeros((2, sampling_rate))
                for i in range(len(self.goals)):
                    if self.goals_complete[i]: continue
                    sound_source = self._source_sound_dict[self.current_sounds[i]]
                    audiogoal_from_last_rir += self._convolve_with_rir(self._last_rir[i], sound_source, self._current_sample_index[i])

                audiogoal = crossfade(audiogoal_from_last_rir, audiogoal, sampling_rate)

        return audiogoal

    def _convolve_with_rir(self, rir, source_sound, sound_sample_index):
        sampling_rate = self.config.AUDIO.RIR_SAMPLING_RATE
        num_sample = int(sampling_rate * self.config.STEP_TIME)

        index = sound_sample_index
        if index - rir.shape[0] < 0:
            sound_segment = source_sound[: index + num_sample]
            binaural_convolved = np.array([fftconvolve(sound_segment, rir[:, channel]
                                                       ) for channel in range(rir.shape[-1])])
            audiogoal = binaural_convolved[:, index: index + num_sample]
        else:
            # include reverb from previous time step
            if index + num_sample < source_sound.shape[0]:
                sound_segment = source_sound[index - rir.shape[0] + 1: index + num_sample]
            else:
                wraparound_sample = index + num_sample - source_sound.shape[0]
                sound_segment = np.concatenate([source_sound[index - rir.shape[0] + 1:],
                                                source_sound[: wraparound_sample]])
            # sound_segment = source_sound[index - rir.shape[0] + 1: index + num_sample]
            binaural_convolved = np.array([fftconvolve(sound_segment, rir[:, channel], mode='valid',
                                                       ) for channel in range(rir.shape[-1])])
            audiogoal = binaural_convolved

        # audiogoal = np.array([fftconvolve(source_sound, rir[:, channel], mode='full',
        #                                   ) for channel in range(rir.shape[-1])])
        # audiogoal = audiogoal[:, self._episode_step_count * num_sample: (self._episode_step_count + 1) * num_sample]
        audiogoal = np.pad(audiogoal, [(0, 0), (0, sampling_rate - audiogoal.shape[1])])

        return audiogoal

    def get_current_audiogoal_observation(self):
        return self._compute_audiogoal()

    def get_current_spectrogram_observation(self, audiogoal2spectrogram):
        return audiogoal2spectrogram(self.get_current_audiogoal_observation())

    def geodesic_distance(self, position_a, position_bs, episode=None):
        ## if episode is None or episode._shortest_path_cache is None:
        ##     path = habitat_sim.MultiGoalShortestPath()
        ##     path.requested_ends = np.array(
        ##         [np.array(position_bs[0], dtype=np.float32)]
        ##     )
        ## else:
        ##     path = episode._shortest_path_cache
        
        path = habitat_sim.MultiGoalShortestPath()

        if len(np.array(position_bs).shape) == 1: ## for calls from habitat sim
            path.requested_ends = [np.array(position_bs, dtype=np.float32)]
        else:
            path.requested_ends = [np.array(position_b, dtype=np.float32) for position_b in position_bs]

        path.requested_start = np.array(position_a, dtype=np.float32)

        self.pathfinder.find_path(path)

        # if episode is not None:
        #     episode._shortest_path_cache = path

        return path.geodesic_distance

    @property
    def previous_step_collided(self):
        return self._previous_step_collided

    def seed(self, seed):
        self._sim.seed(seed)

    def get_observations_at(
            self,
            position: Optional[List[float]] = None,
            rotation: Optional[List[float]] = None,
            keep_agent_at_new_pose: bool = False,
    ) -> Optional[Observations]:
        current_state = self.get_agent_state()
        if position is None or rotation is None:
            success = True
        else:
            success = self.set_agent_state(
                position, rotation, reset_sensors=False
            )

        if success:
            sim_obs = self._sim.get_sensor_observations()

            self._prev_sim_obs = sim_obs

            observations = self._sensor_suite.get_observations(sim_obs)
            if not keep_agent_at_new_pose:
                self.set_agent_state(
                    current_state.position,
                    current_state.rotation,
                    reset_sensors=False,
                )
            return observations
        else:
            return None

    def _trajectory_length(self, start, trajectory):
        dist = 0
        for next_target in trajectory:
            dist += self.geodesic_distance(start.position, next_target.position)
            start = next_target
        return dist
        
    def get_straight_shortest_path_points(self, position_a, position_b):
        points = []
        all_goals = position_b

        agent_position = NavigationGoal(**{"position": position_a.tolist(), "radius": 1e-5})

        perm = permutations(all_goals, len(all_goals))
        min_dist = np.inf
        best_seq = None
        for seq in list(perm):
            dist = self._trajectory_length(agent_position, seq)
            if dist < min_dist:
                min_dist = dist
                best_seq = seq

        start = position_a
        for i in range(len(best_seq)):
            path = habitat_sim.ShortestPath()
            path.requested_start = start
            path.requested_end = best_seq[i].position
            self.pathfinder.find_path(path)
            points += path.points
            start = best_seq[i].position

        return points

    def make_greedy_follower(self, *args, **kwargs):
        return self._sim.make_greedy_follower(*args, **kwargs)
