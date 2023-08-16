#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

r"""
This file hosts task-specific or trainer-specific environments for trainers.
All environments here should be a (direct or indirect ) subclass of Env class
in habitat. Customized environments should be registered using
``@baseline_registry.register_env(name="myEnv")` for reusability
"""

from typing import Optional, Type
import logging
import math
import numpy as np
from itertools import permutations

import habitat
from habitat import Config, Dataset
from ss_baselines.common.baseline_registry import baseline_registry

import csv
def writer(data):
    f = open("./logs.txt", "a")
    f.write(data)
    f.close()


def get_env_class(env_name: str) -> Type[habitat.RLEnv]:
    r"""Return environment class based on name.

    Args:
        env_name: name of the environment.

    Returns:
        Type[habitat.RLEnv]: env class.
    """
    return baseline_registry.get_env(env_name)


@baseline_registry.register_env(name="AudioNavRLEnv")
class AudioNavRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG

        self._continuous = config.CONTINUOUS
        self._previous_target_distance = None
        self._previous_action = None
        self._episode_distance_covered = None
        self._success_distance = self._core_env_config.TASK.SUCCESS.SUCCESS_DISTANCE
        super().__init__(self._core_env_config, dataset)
        self._env.sim._distance_target = self._distance_target
        self._env.sim._episode_success = self._episode_success

    def reset(self):
        self.new_goals_reached = 0
        self._previous_action = None
        self._env.sim.goals_complete = [False] * 3 # TODO:
        observations = super().reset()
        logging.debug(super().current_episode)

        self._env.sim.goals_complete = [False] * len(self._env.sim.goals)

        if self._continuous:
            self._previous_target_distance = self._distance_target()

        return observations

    def step(self, *args, **kwargs):
        self.new_goals_reached = 0 # at this time step
        self._previous_action = kwargs["action"]
        x = super().step(*args, **kwargs)
        # writer(str(x))
        return x

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        reward = 0

        if self._rl_config.WITH_TIME_PENALTY:
            reward += self._rl_config.SLACK_REWARD

        if self._rl_config.WITH_DISTANCE_REWARD:
            current_target_distance = self._distance_target()
            reward += (self._previous_target_distance - current_target_distance) * self._rl_config.DISTANCE_REWARD_SCALE
            self._previous_target_distance = current_target_distance

        self.update_completed_goals()
        if self.new_goals_reached > 0:
            reward += self.new_goals_reached * self._rl_config.SUCCESS_REWARD
            logging.debug('Reaching goal!')

        if math.isnan(reward):
            writer("shit, big trouble: reward nan\n")

        # assert not math.isnan(reward)
        return reward
    
    def _trajectory_length(self, start, trajectory):
        dist = 0
        for next_target in trajectory:
            dist += self._env.sim.geodesic_distance(start.position, next_target.position)
            start = next_target
        return dist
    
    def _distance_target(self):
        all_goals = np.array(self._env.sim.goals)
        goals_complete = np.array(self._env.sim.goals_complete)
        remaining_goals = all_goals[goals_complete == False]

        agent_position = self._env.sim.get_agent_state()

        perm = permutations(remaining_goals, len(remaining_goals))
        min_dist = np.inf
        for seq in list(perm):
            dist = self._trajectory_length(agent_position, seq)
            min_dist = min(min_dist, dist)

        return min_dist

    def update_completed_goals(self): ## previous name _distance_target
        agent_position = self._env.sim.get_agent_state().position

        prev_completed_goals = self._env.sim.goals_complete
        one_sound_source_left = (len(prev_completed_goals) - sum(prev_completed_goals)) == 1 
        for i in range(len(self._env.sim.goals)):
            if prev_completed_goals[i] == True: continue
            goal = self._env.sim.goals[i]
            d = self._env.sim.geodesic_distance(agent_position, goal.position)
            if ((one_sound_source_left and self._env.task.is_stop_called) or \
			        (not one_sound_source_left)) and d < self._success_distance: # We know it can be reduced further
                self._env.sim.goals_complete[i] = True
                self.new_goals_reached += 1


    def _episode_success(self):
        self.update_completed_goals()
        episode_successful = True
        for completed in self._env.sim.goals_complete:
            if completed == False:
                episode_successful = False
                break

        return episode_successful

    def get_done(self, observations):
        self.update_completed_goals()
        done = False

        if self._env.episode_over or self._episode_success():
            done = True

        return done

    def get_info(self, observations):
        # writer(str(self.habitat_env.get_metrics()))
        return self.habitat_env.get_metrics()

    # for data collection
    def get_current_episode_id(self):
        return self.habitat_env.current_episode.episode_id
