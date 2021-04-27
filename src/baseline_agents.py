# -*- coding: utf-8 -*-
import logging
import os
import random
import math

from environment import WebBotEnv, Utils


# Hill climbing algorithm
class HillClimbing:
    def __init__(self, n_episodes=100, data_dir=None, log_dir=None, model_dir=None, **kwargs):
        self.logger = logging.getLogger("HillClimbing")
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        self.model_dir = model_dir if model_dir else self.data_dir
        self.log_dir = log_dir if log_dir else self.model_dir
        self.n_episodes = n_episodes

    def save_model(self):
        pass

    def load_model(self):
        pass

    def _get_action_score(self, task, action):
        if not action.element:
            return 0
        if not hasattr(self, "_action_score_cache"):
            self._action_score_cache = {}
        task_action_key = "%s/%s|%s" % (task.state.state_str, action.element.own_text, action.value_text)
        if task_action_key not in self._action_score_cache:
            score = task.get_action_usefulness(action)
            self._action_score_cache[task_action_key] = score
        return self._action_score_cache[task_action_key]

    def _set_action_score(self, task, action, score):
        if not action.element:
            return
        if not hasattr(self, "_action_score_cache"):
            self._action_score_cache = {}
        task_action_key = "%s/%s|%s" % (task.state.state_str, action.element.own_text, action.value_text)
        self._action_score_cache[task_action_key] = score
        return self._action_score_cache[task_action_key]

    def train(self, tasks, browser):
        env = WebBotEnv(tasks=tasks, browser=browser)
        stats = []

        found_tasklets = {}
        try:
            for episode in range(1, self.n_episodes + 1):
                env.reset()
                task = env.current_task.snapshot()
                self.logger.info("Episode %d/%d, task: %s" % (episode, self.n_episodes, task.task_str))
                max_reward = 0
                while True:
                    if task.done or task.reward < -10:
                        break
                    env.render()

                    actions = task.get_preferred_actions()
                    if len(actions) == 0:
                        actions = task.state.possible_actions
                    candidate_actions = []
                    action_scores = {}
                    for action in actions:
                        action_score = self._get_action_score(task, action)
                        action_scores[action] = action_score
                        if action_score > 0.1:
                            candidate_actions.append(action)
                    if len(candidate_actions) == 0:
                        candidate_actions = Utils.top_n(action_scores, 5, reverse=True)
                    action = random.choice(candidate_actions)

                    env.step(action)
                    self._set_action_score(task, action, score=task.reward)

                    task_ = env.current_task.snapshot()
                    task = task_
                    self.logger.info("\taction:%s, %s" % (action, task.get_reward_str()))

                    tasklet = task.get_tasklet()
                    if tasklet not in found_tasklets:
                        found_tasklets[tasklet] = (task.total_reward, episode, task.state.screenshot)
                    if task.total_reward > max_reward:
                        max_reward = task.total_reward

                stats.append([episode, max_reward])
                self.logger.info("Episode %d/%d, max_reward %.2f" % (episode, self.n_episodes, max_reward))
            env.destroy()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.logger.info("failed with error: %s" % e)
        return found_tasklets


# Monte Carlo Tree Search (MCTS) algorithm
class MonteCarlo:
    def __init__(self, n_episodes=100, data_dir=None, log_dir=None, model_dir=None, **kwargs):
        self.logger = logging.getLogger("MonteCarlo")
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        self.model_dir = model_dir if model_dir else self.data_dir
        self.log_dir = log_dir if log_dir else self.model_dir
        self.n_episodes = n_episodes
        self._visit_count = {}
        self._total_visit_count = 0
        self._average_score = {}

    def save_model(self):
        pass

    def load_model(self):
        pass

    def _get_action_score(self, task, action):
        if not action.element:
            return 0
        if not hasattr(self, "_action_score_cache"):
            self._action_score_cache = {}
        task_action_key = "%s/%s|%s" % (task.state.state_str, action.element.own_text, action.value_text)
        if task_action_key not in self._action_score_cache:
            score = task.get_action_usefulness(action)
            self._action_score_cache[task_action_key] = score
        return self._action_score_cache[task_action_key]

    def _set_action_score(self, task, action, score):
        if not action.element:
            return
        if not hasattr(self, "_action_score_cache"):
            self._action_score_cache = {}
        task_action_key = "%s/%s|%s" % (task.state.state_str, action.element.own_text, action.value_text)
        self._action_score_cache[task_action_key] = score
        return self._action_score_cache[task_action_key]

    def _get_action_ucb(self, task, action):
        if not action.element:
            return 0
        action_key = "%s/%s" % (task.state.state_str, action.unique_id)
        action_visit_count = self._visit_count[action_key] if action_key in self._visit_count else 0
        if action_visit_count == 0:
            return 10000
        action_average_score = self._average_score[action_key] if action_key in self._average_score else 0
        ucb = action_average_score + 2 * math.sqrt(math.log(self._total_visit_count) / action_visit_count)
        return ucb

    def _backpropagation(self, action_keys, total_rewards):
        for i, action_key in enumerate(action_keys):
            max_reward_after_action = max(total_rewards[i:])
            action_visit_count = self._visit_count[action_key] if action_key in self._visit_count else 0
            new_visit_count = action_visit_count + 1
            action_average_score = self._average_score[action_key] if action_key in self._average_score else 0
            new_average_score = (action_average_score * action_visit_count + max_reward_after_action) / new_visit_count
            self._visit_count[action_key] = new_visit_count
            self._average_score[action_key] = new_average_score

    def train(self, tasks, browser):
        env = WebBotEnv(tasks=tasks, browser=browser)
        stats = []

        found_tasklets = {}
        try:
            for episode in range(1, self.n_episodes + 1):
                self._total_visit_count = episode
                env.reset()
                task = env.current_task.snapshot()
                self.logger.info("Episode %d/%d, task: %s" % (episode, self.n_episodes, task.task_str))
                max_reward = 0
                action_keys = []
                total_rewards = []
                while True:
                    if task.done or task.reward < -10:
                        break
                    env.render()

                    actions = task.get_preferred_actions()
                    if len(actions) == 0:
                        actions = task.state.possible_actions
                    candidate_actions = []
                    action_scores = {}
                    for action in actions:
                        action_score = self._get_action_score(task, action)
                        action_scores[action] = action_score
                        if action_score > 0.1:
                            candidate_actions.append(action)
                    if len(candidate_actions) == 0:
                        candidate_actions = Utils.top_n(action_scores, 5, reverse=True)

                    action_ucbs = {}
                    for action in candidate_actions:
                        action_ucbs[action] = self._get_action_ucb(task, action)
                    max_ucb = max(action_ucbs.values())
                    max_ucb_actions = []
                    for action in action_ucbs:
                        if action_ucbs[action] == max_ucb:
                            max_ucb_actions.append(action)
                    action = random.choice(max_ucb_actions)
                    action_key = "%s/%s" % (task.state.state_str, action.unique_id)

                    env.step(action)

                    task_ = env.current_task.snapshot()
                    task = task_
                    self.logger.info("\taction:%s, %s" % (action, task.get_reward_str()))

                    tasklet = task.get_tasklet()
                    if tasklet not in found_tasklets:
                        found_tasklets[tasklet] = (task.total_reward, episode, task.state.screenshot)
                    if task.total_reward > max_reward:
                        max_reward = task.total_reward
                    action_keys.append(action_key)
                    total_rewards.append(task.total_reward)
                self._backpropagation(action_keys=action_keys, total_rewards=total_rewards)

                stats.append([episode, max_reward])
                self.logger.info("Episode %d/%d, max_reward %.2f" % (episode, self.n_episodes, max_reward))
            env.destroy()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.logger.info("failed with error: %s" % e)
        return found_tasklets

