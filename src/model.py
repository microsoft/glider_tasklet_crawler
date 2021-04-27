# coding=utf-8
import logging
import keras
import os
import random
import numpy as np
from keras import backend as K
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D, Input, Lambda
from keras.models import Sequential, Model
from feature import FeatureExtractor


class ExplorationModel:
    def __init__(self,
                 data_dir, log_dir=None, model_dir=None,
                 batch_size=64, n_episodes=200, n_backup_episodes=10, resume=False,
                 **kwargs):
        self.logger = logging.getLogger("ExplorationModel")
        self.fe = FeatureExtractor(**kwargs)
        self.model = None
        self.batch_size = batch_size
        self.n_episodes = n_episodes
        self.n_backup_episodes = n_backup_episodes
        self.resume = resume

        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        self.model_dir = model_dir if model_dir else self.data_dir
        self.log_dir = log_dir if log_dir else self.model_dir

        self._build_net()

    def _build_net(self):
        pass

    def _do_train(self, positive_samples, negative_samples):
        pass

    def _do_predict(self, task, actions):
        pass

    def save_model(self):
        if not self.model_dir:
            return
        model_path = os.path.join(self.model_dir, 'supervised.h5')
        if os.path.exists(model_path):
            os.remove(model_path)
        self.model.save(model_path, overwrite=True)

    def load_model(self):
        model_path = os.path.join(self.model_dir, 'supervised.h5')
        if not self.model_dir or not os.path.exists(model_path):
            self.logger.warning("Model file supervised.h5 does not exist.")
            return
        self.model = keras.models.load_model(model_path)

    def train(self, tasks, browser):
        positive_samples = []
        negative_samples = []
        samples = list(self._load_samples(tasks, browser))
        positive_labels = set()
        for task, action, is_positive in samples:
            if is_positive:
                label = "%s|%s" % (task.task_str, action.action_str)
                positive_labels.add(label)
        for task, action, is_positive in samples:
            if is_positive:
                positive_samples.append((task, action))
            else:
                # label = "%s|%s" % (task.task_str, actions[0].norm_action_str)
                # if label in positive_labels:
                #     continue
                negative_samples.append((task, action))
        self.logger.info("# Positive samples: %d" % len(positive_samples))
        self.logger.info("# Negative samples: %d" % len(negative_samples))
        self._do_train(positive_samples, negative_samples)
        self.save_model()

    def predict(self, task, actions):
        return self._do_predict(task, actions)

    def _load_samples(self, tasks, browser):
        from environment import WebBotEnv
        for task in tasks:
            env = WebBotEnv(tasks=[task], browser=browser)
            env.replay()
            for i in range(len(task.state_history)):
                task_i = task.snapshot(step=i)
                action_i = task.action_history[i]
                if not task_i.state or not action_i or action_i.element is None:
                    continue
                # self.fe.plot_feature(task_i, action_i)
                # feature = self.fe.get_new_feature([task_i], [action_i])
                # feature_shape = self.fe.get_new_feature_shape()
                yield task_i, action_i, 1
                for action in task_i.state.possible_actions:
                    if action.action_str != action_i.action_str:
                        yield task_i, action, 0


class Qda2pModel(ExplorationModel):
    """
    Input: Task state and action
    Output: Probability
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = logging.getLogger("SA2PModel")

    def _build_net(self):
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
        p = Dense(1, activation='sigmoid')(feature_vec)
        model = Model(inputs=[dom_img, action_img, query_vec, action_vec], outputs=p)

        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        self.model = model

    def _do_train(self, positive_samples, negative_samples):
        half_batch_size = min(int(self.batch_size / 2), len(positive_samples), len(negative_samples))
        for i in range(1, self.n_episodes + 1):
            positive_batch = random.sample(positive_samples, half_batch_size)
            negative_batch = random.sample(negative_samples, half_batch_size)
            x_samples = positive_batch + negative_batch
            x = self.fe.get_feature(*zip(*x_samples))
            # x = self.fe.get_cluster_feature(*zip(*x_samples))
            y = np.array([1.0] * half_batch_size + [0.0] * half_batch_size)
            history = self.model.fit(x=x, y=y, shuffle=True, epochs=2, verbose=0)
            self.logger.info(
                "Episode %d/%d, loss %.4f, acc %.4f" %
                (i, self.n_episodes, history.history['loss'][-1], history.history['acc'][-1])
            )

    def _do_predict(self, task, actions):
        tasks = [task] * len(actions)
        p = self.model.predict(x=self.fe.get_feature(tasks, actions)).squeeze(-1)
        action2p = {}
        # self.logger.info("Testing task: \t%s" % task.task_str)
        for i, action in enumerate(actions):
            # self.logger.info("% 8d (%.4f): \t%s" % (i, p[i], actions[i]))
            action2p[action] = p[i]
        return action2p


class S2AModel(ExplorationModel):
    """
    Input: task state
    Output: action
    """
    pass


class NaiveModel(ExplorationModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = logging.getLogger("NaiveModel")

    def _build_net(self):
        def build_cnn(n_dims):
            model = Sequential()
            model.add(Conv2D(32, (1, 1), input_shape=(self.fe.height, self.fe.width, n_dims)))

            model.add(Conv2D(32, (3, 3)))
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
            model.add(Dense(1, activation='sigmoid'))
            return model

        s, a = [Input(shape=shape) for shape in self.fe.get_feature_shape_old()]
        # inputs = Input(shape=self.fe.get_feature_shape())
        sa = keras.layers.concatenate([s, a], -1)
        p = build_cnn(self.fe.task_dim + self.fe.action_dim)(sa)
        model = Model(inputs=[s, a], outputs=p)

        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        return model

    def do_train(self, positive_samples, negative_samples, **kwargs):
        half_batch_size = min(int(self.batch_size / 2), len(positive_samples), len(negative_samples))
        for i in range(1, self.n_episodes + 1):
            positive_batch = random.sample(positive_samples, half_batch_size)
            negative_batch = random.sample(negative_samples, half_batch_size)
            x_samples = positive_batch + negative_batch
            x = self.fe.get_feature_old(*zip(*x_samples))
            # x = self.fe.get_cluster_feature(*zip(*x_samples))
            y = np.array([1.0] * half_batch_size + [0.0] * half_batch_size)
            history = self.model.fit(x=x, y=y, shuffle=True, epochs=2, verbose=0)
            self.logger.info("Episode %d/%d, loss %.4f, acc %.4f"
                             % (i, self.n_episodes, history.history['loss'][-1], history.history['acc'][-1]))
        self.save_model()

    def _do_predict(self, task, actions):
        tasks = [task] * len(actions)
        p = self.model.predict(x=self.fe.get_feature_old(tasks, actions)).squeeze(-1)
        action2p = {}
        for i, action in enumerate(actions):
            action2p[action] = p[i]
        return action2p

