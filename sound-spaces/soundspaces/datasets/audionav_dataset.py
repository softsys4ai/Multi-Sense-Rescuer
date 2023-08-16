# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import os
import logging
from typing import List, Optional
import time
import numpy as np

import quaternion
import habitat_sim.sim
from soundspaces.continuous_simulator import ContinuousSoundSpacesSim

from habitat.config import Config
from habitat.core.dataset import Dataset
from habitat.core.registry import registry
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    ShortestPathPoint,
)
import random

import csv
def writer(data):
    f = open("./logs.txt", "a")
    f.write(data)
    f.close()

ALL_SCENES_MASK = "*"
CONTENT_SCENES_PATH_FIELD = "content_scenes_path"
DEFAULT_SCENE_PATH_PREFIX = "data/scene_dataset/"

import yaml
import munch
MY_CONFIG = None
with open("dataset_config.yaml", "r") as stream:
    try:
        MY_CONFIG = munch.munchify(yaml.safe_load(stream))
    except yaml.YAMLError as exc:
        print(exc)

"""#
Reads json files. Each episode parameters (start loc, goal loc, ..) are fixed in these json files. It just reads them and creates the dataset.
For a mp3d scene that I am using in interactive demo, there are unique 38 locations in that scene from which it create 1030 different 
combinations of start and goal location.
"""
@registry.register_dataset(name="AudioNav")
class AudioNavDataset(Dataset):
    r"""Class inherited from Dataset that loads Audio Navigation dataset.
    """

    episodes: List[NavigationEpisode]
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"

    @staticmethod
    def check_config_paths_exist(config: Config) -> bool:
        return os.path.exists(
            config.DATA_PATH.format(version=config.VERSION, split=config.SPLIT)
        ) and os.path.exists(config.SCENES_DIR)

    @staticmethod
    def get_scenes_to_load(config: Config) -> List[str]:
        r"""Return list of scene ids for which dataset has separate files with
        episodes.
        """
        assert AudioNavDataset.check_config_paths_exist(config), \
            (config.DATA_PATH.format(version=config.VERSION, split=config.SPLIT), config.SCENES_DIR)
        dataset_dir = os.path.dirname(
            config.DATA_PATH.format(version=config.VERSION, split=config.SPLIT)
        )

        cfg = config.clone()
        cfg.defrost()
        cfg.CONTENT_SCENES = []
        dataset = AudioNavDataset(cfg)
        return AudioNavDataset._get_scenes_from_folder(
            content_scenes_path=dataset.content_scenes_path,
            dataset_dir=dataset_dir,
        )

    @staticmethod
    def _get_scenes_from_folder(content_scenes_path, dataset_dir):
        scenes = []
        content_dir = content_scenes_path.split("{scene}")[0]
        scene_dataset_ext = content_scenes_path.split("{scene}")[1]
        content_dir = content_dir.format(data_path=dataset_dir)
        if not os.path.exists(content_dir):
            return scenes

        for filename in os.listdir(content_dir):
            if filename.endswith(scene_dataset_ext):
                scene = filename[: -len(scene_dataset_ext)]
                scenes.append(scene)
        scenes.sort()
        return scenes

    def __init__(self, config: Optional[Config] = None) -> None:
        self.episodes = []
        self._config = config
        self.num_goals = 3

        if config is None:
            return

        datasetfile_path = config.DATA_PATH.format(version=config.VERSION, split=config.SPLIT)

        with gzip.open(datasetfile_path, "rt") as f:
            self.from_json(f.read(), scenes_dir=config.SCENES_DIR, scene_filename=datasetfile_path)

        # Read separate file for each scene
        dataset_dir = os.path.dirname(datasetfile_path)
        scenes = config.CONTENT_SCENES
        if ALL_SCENES_MASK in scenes:
            scenes = AudioNavDataset._get_scenes_from_folder(
                content_scenes_path=self.content_scenes_path,
                dataset_dir=dataset_dir,
            )

        last_episode_cnt = 0
        for scene in scenes:
            scene_filename = self.content_scenes_path.format(
                data_path=dataset_dir, scene=scene
            )
        
            with gzip.open(scene_filename, "rt") as f:
                self.from_json(f.read(), scenes_dir=config.SCENES_DIR, scene_filename=scene_filename)

            num_episode = len(self.episodes) - last_episode_cnt
            last_episode_cnt = len(self.episodes)
            logging.info('Sampled {} from {}'.format(num_episode, scene))
        
        random.shuffle(self.episodes)

    def filter_by_ids(self, scene_ids):
        episodes_to_keep = list()

        for episode in self.episodes:
            for scene_id in scene_ids:
                scene, ep_id = scene_id.split(',')
                if scene in episode.scene_id and ep_id == episode.episode_id:
                    episodes_to_keep.append(episode)

        self.episodes = episodes_to_keep

    # filter by scenes for data collection
    def filter_by_scenes(self, scene):
        episodes_to_keep = list()

        for episode in self.episodes:
            episode_scene = episode.scene_id.split("/")[3]
            if scene == episode_scene:
                episodes_to_keep.append(episode)

        self.episodes = episodes_to_keep

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None, scene_filename: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        goal_radius = 1e-5
        goal_poses = set()
        for episode in deserialized["episodes"]:
            goal_poses.add(tuple(episode["goals"][0]["position"]))

        self.sim = None

        episode_cnt = 0
        # episode_scenes_added = {}
        for episode in deserialized["episodes"]:
            # # a temporal workaround to set scene_dataset_config attribute
            # episode.scene_dataset_config = self._config.SCENES_DIR.split('/')[-1]

            episode = NavigationEpisode(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX):
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            # if episode.scene_id in episode_scenes_added.keys() and episode_scenes_added[episode.scene_id] > 1:
            #     continue

            if self.sim == None or episode.scene_id != MY_CONFIG.SCENE:
                MY_CONFIG.SCENE = episode.scene_id
                if self.sim:
                    self.sim._sim.close()
                    del self.sim._sim
                    del self.sim
                self.sim = ContinuousSoundSpacesSim(MY_CONFIG)
            
            first_goal = np.array(episode.goals[0]["position"])
            if self.sim.geodesic_distance(episode.start_position, [first_goal]) == np.inf:
                continue

            goals = []
          
            for g_index, goal in enumerate(episode.goals):
                # episode.goals[g_index] = NavigationGoal(**goal)
                goals.append(tuple(goal["position"]))
            
            ## Add randomly chosen goals
            couldn_t_find_goals = False
            for _ in range(self.num_goals - len(goals)):
                next_goal = random.sample(goal_poses, 1)[0]
                i = 1
                # print((next_goal in goals), ((np.sqrt(np.square(np.array(next_goal) - np.array(goals)).sum(1)) > 4).sum() < len(goals)), (np.array(next_goal)[1] != np.array(episode.start_position)[1]), not sim.pathfinder.find_path(path))
                # time.sleep(0.5)

                while (next_goal in goals) or \
                        ((np.sqrt(np.square(np.array(next_goal) - np.array(goals)).sum(1)) > 4).sum() < len(goals)) or \
                        (np.sqrt(np.square(np.array(next_goal) - np.array(episode.start_position)).sum()) < 4) or \
                        (next_goal[1] != episode.start_position[1]) or \
                        (self.sim.geodesic_distance(episode.start_position, [np.array(list(next_goal))])) == np.inf:# or \
                    # (not self.min_dist_to_obstacles_compatible(shortest_path_points)): 
                    next_goal = random.sample(goal_poses, 1)[0]
                    i += 1
                    if i > 100:
                        couldn_t_find_goals = True
                        break
                if couldn_t_find_goals:
                    break
                goals.append(next_goal)

            if couldn_t_find_goals:
                continue

            N = random.randint(1, self.num_goals) # Random number of sound sources
            # N = self.num_goals

            episode.goals = [NavigationGoal(**{"position": list(goal), "radius": goal_radius}) for goal in goals][:N]

            if episode.shortest_paths is not None:
                for path in episode.shortest_paths:
                    for p_index, point in enumerate(path):
                        path[p_index] = ShortestPathPoint(**point)

            if hasattr(self._config, 'CONTINUOUS') and self._config.CONTINUOUS:
                # TODO: fix
                for i in range(len(episode.goals)):
                    episode.goals[i].position[1] += 0.1

            self.episodes.append(episode)
            episode_cnt += 1
            # if episode.scene_id not in episode_scenes_added.keys():
            #     episode_scenes_added[episode.scene_id] = 0
            # episode_scenes_added[episode.scene_id] += 1
            
        print("TEST scenss" , episode_cnt)
        if self.sim:
            self.sim._sim.close()
            del self.sim._sim
            del self.sim
        # exit()
        