# -*- coding: utf-8 -*-
import json
import logging
import os
import random

import keras
import numpy as np
from keras import backend as K
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D, Input, Lambda
from keras.models import Sequential, Model

from environment import WebBotEnv, Task, State, Utils, Action
from feature import FeatureExtractor
from form_agent import FormAction, FormManager


class Transition:
    def __init__(self, task, action, task_, error=None):
        self.task = task
        self.task_ = task_
        self.action = action
        self.error = error

    def get_error(self):
        return self.error if self.error else np.inf

    def set_error(self, error):
        self.error = float(error)


class ReplayMemory:
    """
    Simplified version of prioritized replay
    """

    def __init__(self, memory_size=1000, prioritized_replay=True):
        self.transitions = []
        self.memory_size = memory_size
        self.prioritized_replay = prioritized_replay
        self.counter = 0

    def sample(self, n_samples):
        """
        return indices and samples
        :param n_samples: number of samples
        :return:
        """
        if len(self.transitions) < n_samples:
            n_samples = len(self.transitions)
        if self.prioritized_replay:
            return sorted(self.transitions, key=lambda x: x.get_error(), reverse=True)[:n_samples]
        else:
            return random.sample(self.transitions, n_samples)

    def store_transition(self, transition):
        # Replace the old memory with new memory
        if self.counter > self.memory_size:
            index = self.counter % self.memory_size
            self.transitions[index] = transition
        else:
            self.transitions.append(transition)
        self.counter += 1

    def save(self, memory_dir):
        if memory_dir is None:
            return
        if not os.path.exists(memory_dir):
            os.makedirs(memory_dir)
        states_dir = os.path.join(memory_dir, "states")
        memory_file = open(os.path.join(memory_dir, "memory.json"), "w")
        transition_dicts = []
        memory_dict = {
            "memory_size": self.memory_size,
            "prioritized_replay": self.prioritized_replay,
            "counter": self.counter,
            "transitions": transition_dicts
        }
        for transition in self.transitions:
            task, action, task_, error = transition.task, transition.action, transition.task_, transition.error
            transition_dicts.append([task.to_dict(), action.action_str, task_.to_dict(), error])
            for state in task.state_history + task_.state_history + [task.state, task_.state]:
                state.save(state_dir=states_dir)
        json.dump(memory_dict, memory_file, indent=2)
        memory_file.close()

    def load(self, memory_dir):
        states_dir = os.path.join(memory_dir, "states")
        memory_file = os.path.join(memory_dir, "memory.json")
        if memory_dir is None or not os.path.exists(memory_file):
            return
        memory_dict = json.load(open(memory_file))
        for task_dict, action_str, task_dict_, error in memory_dict["transitions"]:
            task = self._load_task(task_dict, states_dir)
            task_ = self._load_task(task_dict_, states_dir)
            action = self._load_action(task.state, action_str)
            transition = Transition(task=task, action=action, task_=task_, error=error)
            self.store_transition(transition)

    def _load_task(self, task_dict, states_dir):
        task = Task(resume_utg=False, **task_dict)
        for i in range(len(task_dict["state_history"])):
            state_str = task_dict["state_history"][i]
            action_str = task_dict["action_history"][i]
            state = State.load(state_dir=states_dir, state_str=state_str)
            state.setup(task)
            action = self._load_action(state, action_str)
            task.state_history.append(state)
            task.action_history.append(action)
        task.state = State.load(state_dir=states_dir, state_str=task_dict["state"])
        task.state.setup(task)
        task.reward = task_dict["reward"]
        task.total_reward = task_dict["total_reward"]
        task.done = task_dict["done"]
        return task

    def _load_action(self, state, action_str):
        for action in state.possible_actions:
            if action.action_str == action_str:
                return action
        return state.finish_action

    def update_rewards(self, new_task):
        for transition in self.transitions:
            transition.task.update_reward(new_task)
            transition.task_.update_reward(new_task)


class ExploreStrategy:
    def __init__(self, n_episodes=100, explore_rate=0.1, exploit_rate=0.0, feature_extractor=None,
                 epsilon_decay_policy="linear", epsilon_step_decay=False,
                 explore_policy="supervised", supervised_model=None,
                 **kwargs):
        self.logger = logging.getLogger("ExploreStrategy")
        self.n_episodes = n_episodes
        self.n_explore_episodes = int(explore_rate * n_episodes)
        self.n_exploit_episodes = int(exploit_rate * n_episodes)

        self.fe = feature_extractor
        self.epsilon_decay_policy = epsilon_decay_policy
        self.epsilon_step_decay = epsilon_step_decay

        self.explore_policy = explore_policy
        self.supervised_model = supervised_model
        if supervised_model is None and explore_policy == "supervised":
            self.logger.warning("supervised_model is None, using similarity-based strategy instead")
            self.explore_policy = "similarity"

    def get_epsilon(self, episode, task=None):
        if episode <= self.n_explore_episodes:
            epsilon = 1.0
        elif episode > self.n_episodes - self.n_exploit_episodes:
            epsilon = 0.0
        else:
            progress = float(episode - self.n_explore_episodes) / \
                       (self.n_episodes - self.n_explore_episodes - self.n_exploit_episodes)
            if self.epsilon_decay_policy == "linear":
                epsilon = 1.0 - progress
            elif self.epsilon_decay_policy == "exp":
                epsilon = np.exp(-4 * progress)
            else:
                raise RuntimeError("Unknown epsilon_decay_policy: " + self.epsilon_decay_policy)
        if self.epsilon_step_decay and task:
            n_remain_steps = task.step_limit - len(task.action_history)
            if n_remain_steps < 1:
                n_remain_steps = 1
            epsilon = epsilon ** n_remain_steps
        return epsilon

    def _get_action_probability(self, task, action):
        if not hasattr(self, "_action_probability_cache"):
            self._action_probability_cache = {}
        task_action_key = "%s/%s/%s" % (task.task_str, task.state.state_str, action.action_str)
        if task_action_key not in self._action_probability_cache:
            action_probability = self.supervised_model.predict(task, [action])
            self._action_probability_cache[task_action_key] = action_probability[action]
        return self._action_probability_cache[task_action_key]

    def _get_action_usefulness(self, task, action):
        if not action.element:
            return 0
        if not hasattr(self, "_action_score_cache"):
            self._action_score_cache = {}
        task_action_key = "%s/%s/%s|%s" % (task.task_str, task.state.state_str, action.element.own_text, action.value_text)
        if task_action_key not in self._action_score_cache:
            action_score = task.get_action_usefulness(action)
            self._action_score_cache[task_action_key] = action_score
        return self._action_score_cache[task_action_key]

    def choose_action_to_explore(self, task):
        preferred_actions = task.get_preferred_actions()
        if len(preferred_actions) == 0:
            return random.choice(task.state.possible_actions)

        if "supervised" in self.explore_policy:
            if np.random.uniform() < 0.5:
                action_probabilities = self.supervised_model.predict(task, preferred_actions)
                return Utils.weighted_choice(action_probabilities)
            else:
                return np.random.choice(preferred_actions)
        elif "similarity" in self.explore_policy:
            if np.random.uniform() < 0.5:
                action_scores = {}
                for action in preferred_actions:
                    action_scores[action] = self._get_action_usefulness(task, action)
                return Utils.weighted_choice(action_scores)
            else:
                return np.random.choice(preferred_actions)
        elif "full_sim" in self.explore_policy:
            action_scores = {}
            for action in task.state.possible_actions:
                action_scores[action] = self._get_action_usefulness(task, action)
            return Utils.weighted_choice(action_scores)
        elif "full_rand" in self.explore_policy:
            return np.random.choice(task.state.possible_actions)
        elif "half_sim_half_rand" in self.explore_policy:
            if np.random.uniform() < 0.5:
                action_scores = {}
                for action in task.state.possible_actions:
                    action_scores[action] = self._get_action_usefulness(task, action)
                return Utils.weighted_choice(action_scores)
            else:
                return np.random.choice(task.state.possible_actions)
        return np.random.choice(task.state.possible_actions)


# Q Learning
class TestQTable:
    def __init__(self, gamma=0.99, lr=0.1, no_sub_policy=False,
                 n_episodes=100, n_backup_episodes=10, resume=False, supervised_model=None,
                 data_dir=None, log_dir=None, model_dir=None, **kwargs):
        self.logger = logging.getLogger("TestQTable")
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        self.model_dir = model_dir if model_dir else self.data_dir
        self.log_dir = log_dir if log_dir else self.model_dir

        self.fe = FeatureExtractor(**kwargs)
        self.et = ExploreStrategy(n_episodes=n_episodes, feature_extractor=self.fe,
                                  supervised_model=supervised_model, **kwargs)
        self.gamma = gamma
        self.lr = lr
        self.no_sub_policy = no_sub_policy
        self.n_episodes = n_episodes
        self.n_backup_episodes = n_backup_episodes
        self.resume = resume
        self.q_table = {}
        self.form_manager = FormManager()

    def save_model(self):
        if not self.model_dir:
            return
        model_path = os.path.join(self.model_dir, 'q_table.json')
        json.dump(self.q_table, open(model_path, "w"), indent=2)

    def load_model(self):
        model_path = os.path.join(self.model_dir, 'q_table.json')
        if not self.model_dir or not os.path.exists(model_path):
            self.logger.warning("Failed to load model.")
            return
        self.q_table = json.load(open(model_path))

    def _task_id(self, task):
        # task_id = "->".join(action.action_str for action in task.action_history)
        return "%s // %s" % (task.task_str, task.state.state_str)

    def _action_id(self, task, action):
        return action.action_str

    def _get_q(self, task, action):
        task_id = self._task_id(task)
        if task_id not in self.q_table:
            self.q_table[task_id] = {}
        action_id = self._action_id(task, action)
        if action_id not in self.q_table[task_id]:
            self.q_table[task_id][action_id] = 0
        return self.q_table[task_id][action_id]

    def _set_q(self, task, action, q):
        task_id = self._task_id(task)
        if task_id not in self.q_table:
            self.q_table[task_id] = {}
        action_id = self._action_id(task, action)
        self.q_table[task_id][action_id] = q

    def _get_max_q(self, task):
        task_id = self._task_id(task)
        if task_id not in self.q_table:
            return 0
        else:
            return max(self.q_table[task_id].values())

    def get_candidate_actions(self, task):
        # actions = task.get_preferred_actions()
        all_actions = task.get_preferred_actions()
        if len(all_actions) == 0:
            all_actions = task.state.possible_actions
        if self.no_sub_policy:
            return all_actions
        # filter out input actions and replace with form actions
        forms = self.form_manager.get_forms(task)
        form_actions = [FormAction(form) for form in forms]
        other_actions = []
        for action in all_actions:
            if action.action_type in [Action.INPUT_TEXT, Action.SELECT]:
                continue
            other_actions.append(action)
        return form_actions + other_actions

    def choose_action_to_explore(self, task, candidate_actions):
        if np.random.uniform() < 0.5:
            action_scores = {}
            for action in candidate_actions:
                action_scores[action] = len(action.form.input_candidates) \
                    if isinstance(action, FormAction) else task.get_action_usefulness(action)
            return Utils.weighted_choice(action_scores)
        else:
            return random.choice(candidate_actions)

    def choose_action_with_model(self, task, candidate_actions):
        qs = [self._get_q(task, action) for action in candidate_actions]
        i = int(np.argmax(qs))
        return candidate_actions[i], qs[i]

    def _learn(self, task, action, task_):
        q_predict = self._get_q(task, action)
        if task_.done:
            q_target = task_.reward
        else:
            q_ = self._get_max_q(task_)
            q_target = task_.reward + self.gamma * q_
        self._set_q(task, action, q_predict + self.lr * (q_target - q_predict))

    def train(self, tasks, browser):
        env = WebBotEnv(tasks=tasks, browser=browser)
        stats = []

        def save_progress(save_stats=True, save_fig=True, save_model=True):
            try:
                if save_stats:
                    stats_path = os.path.join(self.model_dir, "training_stats.json")
                    json.dump(stats, open(stats_path, "w"), indent=2)
                if save_fig:
                    stats_png_path = os.path.join(self.log_dir, "training_stats.png")
                    self._plot_training_stats(stats, stats_png_path)
                if save_model:
                    self.save_model()
            except Exception as e:
                self.logger.warning(e)

        def resume_progress():
            # resume model
            self.load_model()
            stats_path = os.path.join(self.model_dir, "training_stats.json")
            if os.path.exists(stats_path):
                stats.append(json.load(open(stats_path)))

        if self.resume:
            resume_progress()

        found_tasklets = {}
        try:
            for episode in range(1, self.n_episodes + 1):
                env.reset()
                task = env.current_task.snapshot()
                self.logger.info("Episode %d/%d, task: %s" % (episode, self.n_episodes, task.task_str))

                max_reward = 0
                max_reward_task_snapshot = None
                tried_form_actions = []
                while True:
                    if task.done or task.reward < -10:
                        break
                    env.render()
                    epsilon = self.et.get_epsilon(episode, task)
                    # if episode == 1 and self.et.supervised_model and self.et.explore_policy == "supervised":
                    #     action_type = "Guided"
                    #     action = self.choose_action_with_supervised_model(task)
                    # el
                    candidate_actions = []
                    interacted_form_ids = [form.unique_id for form, _ in tried_form_actions]
                    for candidate_action in self.get_candidate_actions(task):
                        if isinstance(candidate_action, FormAction) and \
                                candidate_action.form.unique_id in interacted_form_ids:
                            continue
                        candidate_actions.append(candidate_action)

                    if len(candidate_actions) == 0:
                        break
                    rand = np.random.uniform()
                    action_category, action, q = "Unknown", None, 0
                    if rand > epsilon:
                        action_category = "Exploit"
                        action, q = self.choose_action_with_model(task, candidate_actions)
                    if rand <= epsilon or q == 0:
                        action_category = "Explore"
                        action = self.choose_action_to_explore(task, candidate_actions)
                    # self.fe.plot_feature(task, action)
                    if action is None:
                        break

                    init_task = task.snapshot()
                    if isinstance(action, FormAction):
                        form = action.form
                        form_actions, action_categories = form.try_solve(epsilon)
                        init_reward = task.total_reward
                        for i, form_action in enumerate(form_actions):
                            form_action_element = task.state.get_element_by_locator(form_action.element.locator)
                            if form_action_element is None:
                                form_action.value = None
                            if form_action.value is None:
                                continue
                            env.step(form_action)
                            task = env.current_task.snapshot()
                            self.logger.info("\t%s, epsilon:%.3f, action:%s, %s" %
                                             (action_categories[i], epsilon, form_action, task.get_reward_str()))
                        tried_form_actions.append((form, form_actions))
                        self.logger.info(f" {action} achieved {task.total_reward - init_reward:.2f}")
                    else:
                        env.step(action)
                        task = env.current_task.snapshot()
                        self.logger.info("\t%s, epsilon:%.3f, action:%s, %s" %
                                         (action_category, epsilon, action, task.get_reward_str()))
                    self._learn(init_task, action, task)

                    if task.total_reward > max_reward:
                        max_reward = task.total_reward
                        max_reward_task_snapshot = task.snapshot()

                if max_reward_task_snapshot is not None:
                    max_reward_tasklet = max_reward_task_snapshot.get_tasklet()
                    if max_reward_tasklet not in found_tasklets:
                        found_tasklets[max_reward_tasklet] = \
                            (max_reward_task_snapshot.total_reward, episode, max_reward_task_snapshot.state.screenshot)

                # learn form
                for (form, form_actions) in tried_form_actions:
                    form.store_actions_actual_reward(form_actions, max_reward)
                self.form_manager.learn()

                epsilon = self.et.get_epsilon(episode=episode)
                stats.append([episode, epsilon, max_reward])
                if episode % self.n_backup_episodes == 0:
                    save_progress(save_fig=True, save_model=False)
                self.logger.info("Episode %d/%d, epsilon %.3f, max_reward %.2f" %
                                 (episode, self.n_episodes, epsilon, max_reward))
            save_progress(save_fig=True, save_model=True)
            env.destroy()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.logger.info("failed with error: %s" % e)
        return found_tasklets

    def _plot_training_stats(self, stats, stats_file_path=None):
        from mpl_toolkits.axes_grid1 import host_subplot
        from mpl_toolkits import axisartist
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        episodes, epsilons, rewards = zip(*stats)
        y_range = int(np.min(rewards)) - 1, int(np.max(rewards)) + 1

        # plt.rcParams["figure.figsize"] = (10, 8)
        par0 = host_subplot(111, axes_class=axisartist.Axes)
        par1 = par0.twinx()
        plt.subplots_adjust(right=0.85)

        new_fixed_axis = par1.get_grid_helper().new_fixed_axis
        par1.axis["right"] = new_fixed_axis(loc="right", axes=par1, offset=(0, 0))
        par1.axis["right"].toggle(all=True)

        par0.set_xlabel("Episode")
        par0.set_ylabel("Epsilon")
        par1.set_ylabel("Reward")

        par0.plot(episodes, epsilons, label="Epsilon")
        par1.plot(episodes, rewards, label="Reward", marker='o', linewidth=0, markersize=2)

        par0.set_ylim([0, 1])
        par0.xaxis.set_major_locator(MaxNLocator(integer=True))
        par1.set_ylim(y_range)
        par0.legend(loc="upper left")

        plt.draw()
        if stats_file_path:
            plt.savefig(stats_file_path)
        else:
            plt.show()
        plt.gcf().clear()

    def execute(self, tasks, browser, visualize=False):
        env = WebBotEnv(tasks=tasks, browser=browser, visualize=visualize)
        for task in tasks:
            # initial observation
            env.reset(new_task=task)
            task = env.current_task.snapshot()
            self.logger.info("Executing task: %s" % task.task_str)
            while True:
                if task.done:
                    break
                env.render()
                action, q = self.choose_action_with_model(task)
                env.step(action)
                # self.fe.plot_feature(task, action)
                task_ = env.current_task.snapshot()
                task = task_
                self.logger.info("\tExploit, action:%s, reward:%.2f, done:%s" % (action, task.reward, task.done))
            self.logger.info("Got total_reward %.2f in task: %s" % (task.total_reward, task.task_str))
        self.logger.info("Done executing tasks.")


# Deep Q Network
class TestDQN:
    def __init__(
            self,
            double_dqn=True,
            dueling_dqn=False,

            gamma=0.9,
            replace_target_iter=10,
            batch_size=32,
            memory_size=1000,
            prioritized_replay=True,

            n_episodes=100,
            n_backup_episodes=10,
            resume=False,
            demo_dir=None,
            demo_pretrain_steps=50,

            data_dir=None,
            log_dir=None,
            model_dir=None,

            supervised_model=None,
            **kwargs
    ):
        self.logger = logging.getLogger("TestDQN")
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        self.model_dir = model_dir if model_dir else self.data_dir
        self.log_dir = log_dir if log_dir else self.model_dir

        self.replay_memory = ReplayMemory(memory_size=memory_size, prioritized_replay=prioritized_replay)
        self.demo_memory = ReplayMemory(memory_size=memory_size, prioritized_replay=prioritized_replay)

        self.fe = FeatureExtractor(**kwargs)
        self.et = ExploreStrategy(n_episodes=n_episodes, feature_extractor=self.fe,
                                  supervised_model=supervised_model, **kwargs)

        self.double_dqn = double_dqn
        self.dueling_dqn = dueling_dqn
        self.q_eval, self.q_next = self._build_net()

        self.gamma = gamma
        self.replace_target_iter = replace_target_iter
        self.batch_size = batch_size
        self.n_episodes = n_episodes
        self.n_backup_episodes = n_backup_episodes
        self.resume = resume
        self.demo_dir = demo_dir
        self.demo_pretrain_steps = demo_pretrain_steps
        self._n_learn_steps = 0

    def _build_net(self):
        def build_q_func_old():
            s, a = [Input(shape=shape) for shape in self.fe.get_feature_shape_old()]
            # inputs = Input(shape=self.fe.get_feature_shape())
            if self.dueling_dqn:
                sa = keras.layers.concatenate([s, a], -1)
                v = build_cnn(self.fe.task_dim)(s)
                a = build_cnn(self.fe.action_dim + self.fe.task_dim)(sa)
                q = Lambda(lambda va: K.expand_dims(va[0] + (va[1] - K.mean(va[1])), -1), output_shape=(1,))([v, a])
            else:
                sa = keras.layers.concatenate([s, a], -1)
                q = build_cnn(self.fe.task_dim + self.fe.action_dim)(sa)
            return Model(inputs=[s, a], outputs=q)

        def build_cnn_old(n_dims):
            model = Sequential()
            model.add(Conv2D(32, (3, 3), input_shape=(self.fe.height, self.fe.width, n_dims)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Conv2D(32, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Conv2D(64, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
            model.add(Dense(64))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            model.add(Dense(1))
            return model

        def build_q_func():
            dom_img, action_img, query_vec, action_vec = [Input(shape=shape) for shape in self.fe.get_feature_shape()]
            # inputs = Input(shape=self.fe.get_feature_shape())
            dom_action_img = keras.layers.concatenate([dom_img, action_img], -1)
            dom_action_img_dims = self.fe.dom_feature_image_n_channel + self.fe.action_feature_image_n_channel
            dom_action_vec = build_cnn(dom_action_img_dims)(dom_action_img)
            query_action_vec = keras.layers.concatenate([query_vec, action_vec], -1)
            query_action_vec = Dense(32)(query_action_vec)
            feature_vec = keras.layers.concatenate([dom_action_vec, query_action_vec], -1)
            # feature_vec = Dense(32)(feature_vec)
            # feature_vec = Dense(32, activation='relu')(feature_vec)
            p = Dense(1)(feature_vec)
            model = Model(inputs=[dom_img, action_img, query_vec, action_vec], outputs=p)
            return model

        def build_cnn(n_dims):
            model = Sequential()

            model.add(Conv2D(32, (3, 3), input_shape=(self.fe.height, self.fe.width, n_dims)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Conv2D(32, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Conv2D(32, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Conv2D(16, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            # model.add(Dropout(0.5))
            model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
            model.add(Dense(32))
            return model

        q_eval = build_q_func()
        q_next = build_q_func()
        q_eval.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        print(q_eval.summary())
        return q_eval, q_next

    def save_model(self):
        if not self.model_dir:
            return
        model_path = os.path.join(self.model_dir, 'q.h5')
        if os.path.exists(model_path):
            os.remove(model_path)
        self.q_eval.save(model_path, overwrite=True)

    def load_model(self):
        model_path = os.path.join(self.model_dir, 'q.h5')
        if not self.model_dir or not os.path.exists(model_path):
            self.logger.warning("Failed to load model.")
            return
        self.q_eval = keras.models.load_model(model_path)
        self.q_next.set_weights(self.q_eval.get_weights())

    def choose_action_with_model(self, task, q_func=None):
        actions = task.get_preferred_actions()
        if not actions:
            actions = task.state.possible_actions
        tasks = [task] + [task] * len(actions)
        actions = [task.state.finish_action] + actions
        qs = self.predict(tasks, actions, q_func)
        i = int(np.argmax(qs))
        return actions[i], qs[i]

    def predict(self, tasks, actions, q_func):
        if q_func is None:
            q_func = self.q_eval
        features = self.fe.get_feature(tasks, actions)
        return q_func.predict(x=features).squeeze(-1)

    def _learn(self, memory_source=None):
        """
        Fit Q function
        :param memory_source the memory buffer to learn from. Could be `replay`, `demo`, or `hybrid`
        :return max_q and q_error
        """
        # sample batch memory from all memory
        if memory_source == "replay":
            batch_memory = self.replay_memory.sample(self.batch_size)
        elif memory_source == "demo":
            batch_memory = self.demo_memory.sample(self.batch_size)
        else:
            demo_samples = self.demo_memory.sample(self.batch_size / 3)
            replay_samples = self.replay_memory.sample(self.batch_size - len(demo_samples))
            batch_memory = demo_samples + replay_samples

        # check to replace target parameters
        self._n_learn_steps += 1
        if self._n_learn_steps % self.replace_target_iter == 0:
            self.q_next.set_weights(self.q_eval.get_weights())

        tasks = []
        actions = []
        q_targets = []
        for transition in batch_memory:
            task, action, task_ = transition.task, transition.action, transition.task_
            if task_.done:
                q_target = task_.reward
            else:
                if self.double_dqn:
                    action_, q_ = self.choose_action_with_model(task_, q_func=self.q_eval)
                    q_ = self.predict([task_], [action_], self.q_next)[0]
                else:
                    action_, q_ = self.choose_action_with_model(task_, q_func=self.q_next)
                q_target = task_.reward + self.gamma * q_
            tasks.append(task)
            actions.append(action)
            q_targets.append(q_target)

        self.q_eval.fit(
            x=self.fe.get_feature(tasks, actions),
            y=np.array(q_targets),
            epochs=1,
            verbose=0)

        q_predicts = self.predict(tasks, actions, q_func=self.q_eval)
        errors = np.abs(q_predicts - q_targets)

        if self.replay_memory.prioritized_replay:
            for i in range(len(batch_memory)):
                batch_memory[i].error = errors[i]
        for t, a, q_t, q_p in zip(tasks, actions, list(q_targets), list(q_predicts)):
            self.logger.debug(u"Q_predict=%.3f, Q_target=%.3f, State:%s, Action:%s" % (q_p, q_t, t.state.state_str, a))
        return float(np.max(q_predicts)), float(np.mean(errors))

    def train(self, tasks, browser):
        env = WebBotEnv(tasks=tasks, browser=browser)
        stats = []

        def save_progress(save_stats=True, save_fig=True, save_model=False, save_memory=False):
            try:
                if save_stats:
                    stats_path = os.path.join(self.model_dir, "training_stats.json")
                    json.dump(stats, open(stats_path, "w"), indent=2)
                if save_fig:
                    stats_png_path = os.path.join(self.log_dir, "training_stats.png")
                    self._plot_training_stats(stats, self.et.n_explore_episodes, stats_png_path)
                if save_model:
                    self.save_model()
                if save_memory:
                    self.replay_memory.save(self.model_dir)
            except Exception as e:
                self.logger.warning(e)

        def resume_progress():
            # resume model
            self.load_model()
            # resume memory
            self.replay_memory.load(self.model_dir)
            # resume stats
            stats_path = os.path.join(self.model_dir, "training_stats.json")
            if os.path.exists(stats_path):
                stats.append(json.load(open(stats_path)))

        if self.resume:
            resume_progress()

        if self.demo_dir:
            self.demo_memory.load(self.demo_dir)
            for task in tasks:
                self.demo_memory.update_rewards(task)
            for i in range(self.demo_pretrain_steps):
                self._learn(memory_source="demo")
            self.logger.info("Done pre-training on demos.")

        found_tasklets = {}
        for episode in range(1, self.n_episodes + 1):
            # initial observation
            env.reset()
            task = env.current_task.snapshot()
            self.logger.info("Episode %d/%d, task: %s" % (episode, self.n_episodes, task.task_str))

            max_reward = 0
            while True:
                # break while loop when end of this episode
                if task.done or task.reward < -10:
                    break
                env.render()
                epsilon = self.et.get_epsilon(episode, task)

                # RL choose action based on current task snapshot
                if np.random.uniform() < epsilon:
                    action_type = "Explore"
                    action = self.et.choose_action_to_explore(task)
                else:
                    action_type = "Exploit"
                    action, q = self.choose_action_with_model(task, q_func=self.q_eval)
                env.step(action)

                # self.fe.plot_feature(task, action)
                task_ = env.current_task.snapshot()
                self.replay_memory.store_transition(Transition(task=task, action=action, task_=task_))
                # swap observation
                task = task_
                self.logger.info("\t%s, epsilon:%.3f, action:%s, %s" %
                                 (action_type, epsilon, action, task.get_reward_str()))

                tasklet = task.get_tasklet()
                if tasklet not in found_tasklets:
                    found_tasklets[tasklet] = (task.total_reward, episode, task.state.screenshot)
                if task.total_reward > max_reward:
                    max_reward = task.total_reward

            if episode > self.et.n_explore_episodes:
                max_q, q_error = self._learn()
            else:
                max_q, q_error = None, None
            epsilon = self.et.get_epsilon(episode=episode)
            stats.append([episode, epsilon, max_reward, max_q, q_error])
            self.logger.info("Episode %d/%d, epsilon %.3f, max_reward %.2f, max_q %.3f, q_error %.3f" %
                             (episode, self.n_episodes, epsilon, max_reward, max_q or np.nan, q_error or np.nan))
            if episode % self.n_backup_episodes == 0:
                save_progress(save_fig=True, save_model=False, save_memory=False)
        save_progress(save_fig=True, save_model=True, save_memory=False)
        env.destroy()
        return found_tasklets

    def _plot_training_stats(self, stats, n_explore_episodes, stats_file_path):
        from mpl_toolkits.axes_grid1 import host_subplot
        import mpl_toolkits.axisartist as AA
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        episodes, epsilons, rewards, max_qs, errors = zip(*stats)
        if len(stats) <= n_explore_episodes + 1:
            y_values = np.array(rewards)
        else:
            y_values = np.concatenate([rewards, max_qs[n_explore_episodes + 1:], errors[n_explore_episodes + 1:]])
        y_range = int(np.min(y_values)) - 1, int(np.max(y_values)) + 1

        # plt.rcParams["figure.figsize"] = (10, 8)
        par0 = host_subplot(111, axes_class=AA.Axes)
        par1 = par0.twinx()
        par2 = par0.twinx()
        par3 = par0.twinx()
        plt.subplots_adjust(right=0.85)

        new_fixed_axis = par1.get_grid_helper().new_fixed_axis
        par1.axis["right"] = new_fixed_axis(loc="right", axes=par1, offset=(0, 0))
        par1.axis["right"].toggle(all=True)

        par0.set_xlabel("Episode")
        par0.set_ylabel("Epsilon")
        par1.set_ylabel("Reward, Q max, Q error")

        par0.plot(episodes, epsilons, label="Epsilon")
        par1.plot(episodes, rewards, label="Reward", marker='o', linewidth=0, markersize=2)
        par2.plot(episodes, max_qs, label="Q max", marker='.', linewidth=1, markersize=1)
        par3.plot(episodes, errors, label="Q error", marker='.', linewidth=1, markersize=1)

        par0.set_ylim([0, 1])
        par0.xaxis.set_major_locator(MaxNLocator(integer=True))
        par1.set_ylim(y_range)
        par2.set_ylim(y_range)
        par3.set_ylim(y_range)
        par0.legend(loc="upper left")

        plt.draw()
        # plt.show(block=False)
        if stats_file_path:
            plt.savefig(stats_file_path)
        else:
            plt.show()
        plt.gcf().clear()

    def execute(self, tasks, browser):
        env = WebBotEnv(tasks=tasks, browser=browser)
        for task in tasks:
            # initial observation
            env.reset(new_task=task)
            task = env.current_task.snapshot()
            self.logger.info("Executing task: %s" % task.task_str)
            while True:
                if task.done:
                    break
                env.render()
                action, q = self.choose_action_with_model(task, q_func=self.q_eval)
                env.step(action)
                # self.fe.plot_feature(task, action)
                task_ = env.current_task.snapshot()
                task = task_
                self.logger.info("\tExploit, action:%s, reward:%.2f, done:%s" % (action, task.reward, task.done))
            self.logger.info("Got total_reward %.2f in task: %s" % (task.total_reward, task.task_str))
        self.logger.info("Done executing tasks.")


class SupervisedAgent:
    def __init__(self, **kwargs):
        self.logger = logging.getLogger("SupervisedAgent")
        from model import Qda2pModel
        self.model = Qda2pModel(**kwargs)

    def train(self, tasks, browser, **kwargs):
        self.model.train(tasks, browser)

    def execute(self, tasks, browser):
        env = WebBotEnv(tasks=tasks, browser=browser)
        for task in tasks:
            env.reset(new_task=task)
            task = env.current_task.snapshot()

            while True:
                if task.done:
                    break
                env.render()

                actions = task.get_preferred_actions()
                action2p = self.model.predict(task, actions)
                action = Utils.weighted_choice(action2p)
                env.step(action)
                task_ = env.current_task.snapshot()
                task = task_
                self.logger.info("\tExploit, action:%s, reward:%.2f, done:%s" % (action, task.reward, task.done))
            self.logger.info("Got total_reward %.2f in task." % task.total_reward)
        env.destroy()
        self.logger.info("Done testing tasks.")
