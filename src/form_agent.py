# -*- coding: utf-8 -*-
import random
import numpy as np
import functools
import itertools
from collections import OrderedDict

import keras
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Input, Lambda, Dense, Flatten
from keras.models import Sequential, Model

from environment import Utils, Action, Task
from feature import FeatureExtractor


def lazy_property(func):
    attribute = '_lazy_' + func.__name__

    @property
    @functools.wraps(func)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, func(self))
        return getattr(self, attribute)

    return wrapper


class FormAction:
    def __init__(self, form):
        self.form = form

    @lazy_property
    def action_str(self):
        return "fill_form @ " + self.form.unique_id

    def __str__(self):
        return self.action_str


class FormManager:
    def __init__(self):
        self.forms = {}

    def get_form(self, task):
        """
        Deprecated, use get_forms instead
        :param task: the task to extract form from
        :return:
        """
        input_candidates = FormManager.extract_input_candidates(task)
        form = Form(task, input_candidates)
        if form.unique_id not in self.forms:
            self.forms[form.unique_id] = form
        return form

    @staticmethod
    def extract_input_candidates(task):
        input_candidates = {}
        selectable_values = set()
        for action in task.state.possible_actions:
            action_type, action_ele, action_value = action.action_type, action.element, action.value
            if action_type not in [Action.INPUT_TEXT, Action.SELECT]:
                continue
            if (action_type, action_ele) not in input_candidates:
                input_candidates[(action_type, action_ele)] = [None]
            if action_value in input_candidates[(action_type, action_ele)]:
                continue
            if action_type == Action.SELECT:
                action_value_useful = False
                for word in task.all_words_parsed:
                    word_sim = Utils.text_similarity(word, action.value_text_parsed)
                    if word_sim > 0.5:
                        selectable_values.add(word)
                        action_value_useful = True
                if not action_value_useful:
                    continue
            input_candidates[(action_type, action_ele)].append(action_value)
        input_candidates = OrderedDict(sorted(input_candidates.items(), key=lambda x: x[0][1].id))

        for (action_type, action_ele) in input_candidates:
            values = input_candidates[(action_type, action_ele)]
            values_parsed = [
                None if value is None else Action(action_ele, action_type, value).value_text_parsed
                for value in values
            ]

            # keep the max-similarity value for each parameter
            filtered_values = [None]
            for word in task.all_words_parsed:
                if action_type == Action.INPUT_TEXT and word in selectable_values:
                    continue
                max_score = 0
                max_score_value = None
                for i, value_parsed in enumerate(values_parsed):
                    if value_parsed is None:
                        continue
                    value_score = Utils.text_similarity(word, value_parsed)
                    if value_score > max_score:
                        max_score = value_score
                        max_score_value = values[i]
                if max_score_value is not None and max_score_value not in filtered_values:
                    filtered_values.append(max_score_value)
            values = filtered_values

            values = sorted(values, key=lambda x: str(x))
            input_candidates[(action_type, action_ele)] = values
        return input_candidates

    def get_forms(self, task):
        all_input_candidates = FormManager.extract_input_candidates(task)
        all_input_set = set(Form.get_input_candidates_strs(all_input_candidates))

        found_forms = {} # key is the form, value is the input set

        # deal with existing forms
        for form_id in self.forms:
            form = self.forms[form_id]
            form_input_set = set(form.input_candidates_strs)
            # if len(set(form.input_candidates_strs).intersection(all_input_set)) > 0:
            if form_input_set.issubset(all_input_set):
                is_subset_of_previous_form = False
                found_forms_to_remove = []
                for found_form in found_forms:
                    found_form_input_set = found_forms[found_form]
                    if form_input_set.issubset(found_form_input_set):
                        is_subset_of_previous_form = True
                    if form_input_set.issuperset(found_form_input_set):
                        found_forms_to_remove.append(found_form)
                for _ in found_forms_to_remove:
                    del found_forms[_]
                if not is_subset_of_previous_form:
                    found_forms[form] = form_input_set

        found_forms_input_set = set()
        for found_form in found_forms:
            found_forms_input_set = found_forms_input_set.union(found_forms[found_form])

        # generate new forms
        remain_forms = {}
        for (action_type, action_ele) in all_input_candidates:
            values = all_input_candidates[(action_type, action_ele)]
            id_str = f"{action_type}-{action_ele.xpath}-{values}"
            if id_str not in found_forms_input_set:
                parent_form = action_ele.parent_form
                parent_form_id = "root" if parent_form is None else parent_form.xpath
                if parent_form_id not in remain_forms:
                    remain_forms[parent_form_id] = OrderedDict()
                remain_forms[parent_form_id][(action_type, action_ele)] = values
        found_forms = list(found_forms.keys())
        for parent_form_id in remain_forms:
            new_form = Form(task, remain_forms[parent_form_id])
            self.forms[new_form.unique_id] = new_form
            found_forms.append(new_form)
            print(f"new form discovered: {new_form.unique_id}\n" + "\n".join(new_form.input_candidates_strs))

        # filter out huge forms
        filered_forms = [form for form in found_forms if len(form.input_candidates) <= 10]
        return filered_forms

    def learn(self):
        for form_id in self.forms:
            form = self.forms[form_id]
            r = form.learn()
            print(f"form:{form.unique_id} learn_result: {r}")

    @staticmethod
    def test_learn(form):
        for episode in range(1, 100):
            epsilon = 1 - episode / 100
            actions, action_categories = form.try_solve(epsilon=epsilon)
            actual_reward = form.simulate_actions(actions) + 10
            form.store_actions_actual_reward(actions, actual_reward)
            r = form.learn()
            print(f"form:{form.unique_id} {r}")


class Form:
    def __init__(self, task, input_candidates):
        self.task = task.snapshot()
        self.input_candidates = input_candidates
        self.input_candidates_strs = Form.get_input_candidates_strs(input_candidates)
        self.unique_id = Utils.md5("\n".join(self.input_candidates_strs))

        self.form_actor = FormActor(self)
        self.form_critic = FormCritic(self)

        self.tried_input_combs = {}
        self.best_input_comb, _ = self.generate_input_comb()
        self.best_reward = -np.inf

    @staticmethod
    def get_input_candidates_strs(input_candidates):
        id_strs = []
        for (action_type, action_ele) in input_candidates:
            values = input_candidates[(action_type, action_ele)]
            id_strs.append(f"{action_type}-{action_ele.xpath}-{values}")
        return id_strs

    @lazy_property
    def greedy_input_comb_rewards(self):
        greedy_input_comb_rewards = OrderedDict()
        for (action_type, action_ele) in self.input_candidates:
            values = self.input_candidates[(action_type, action_ele)]
            greedy_input_comb_rewards[(action_type, action_ele)] = {}
            for value in values:
                action = Action(action_ele, action_type, value)
                action_reward = self.simulate_actions([action])
                greedy_input_comb_rewards[(action_type, action_ele)][value] = action_reward
        return greedy_input_comb_rewards

    @lazy_property
    def all_input_combs(self):
        all_values_products = itertools.product(*self.input_candidates.values())
        input_combs = []
        for input_values in all_values_products:
            input_comb = OrderedDict()
            for i, _ in enumerate(self.input_candidates):
                input_comb[_] = input_values[i]
            input_combs.append(input_comb)
        return input_combs

    def store_actions_actual_reward(self, actions, reward):
        input_comb = Form.convert_actions_to_input_comb(actions)
        input_comb_id = self.get_input_comb_str(input_comb)
        # if input_comb_id in self.tried_input_combs and reward <= self.tried_input_combs[input_comb_id][-1]:
        #     return
        simulation_reward = self.simulate_actions(actions)
        self.tried_input_combs[input_comb_id] = (input_comb, simulation_reward, reward)
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_input_comb = input_comb

    @staticmethod
    def convert_input_comb_to_actions(input_comb):
        actions = []
        for action_type, action_ele in input_comb:
            action_value = input_comb[(action_type, action_ele)]
            action = Action(action_type=action_type, element=action_ele, value=action_value)
            actions.append(action)
        return actions

    def get_input_comb_str(self, input_comb=None):
        value_strs = []
        for _ in self.input_candidates:
            value_str = "N/A"
            if input_comb is not None and _ in input_comb:
                value_str = str(input_comb[_])
            value_strs.append(value_str)
        return ",".join(value_strs)

    def get_input_comb_vec(self, input_comb=None):
        action_vecs = []
        for _ in self.input_candidates:
            value_candidates = self.input_candidates[_]
            action_vec = np.zeros(len(value_candidates))
            if input_comb is not None and _ in input_comb:
                action_value = input_comb[_]
                action_value_idx = value_candidates.index(action_value)
                action_vec[action_value_idx] = 1.0
            action_vecs.append(action_vec)
        return np.concatenate(action_vecs)

    @staticmethod
    def convert_actions_to_input_comb(actions):
        input_comb = OrderedDict()
        for action in actions:
            input_comb[(action.action_type, action.element)] = action.value
        return input_comb

    def simulate_actions(self, actions):
        task = self.task.snapshot()
        init_reward = task.total_reward
        for action in actions:
            if action.value is None:
                continue
            fake_state = Utils.create_fake_state(current_state=task.state, action=action)
            task.state_history.append(task.state)
            task.action_history.append(action)
            task.state = fake_state
        task._evaluate()
        final_reward = task.total_reward
        return final_reward - init_reward

    def try_solve(self, epsilon, n_simulation_steps=30):
        input_combs_and_action_categories = []
        if n_simulation_steps >= len(self.all_input_combs):
            action_categories = ["Traverse"] * len(self.input_candidates)
            for input_comb in self.all_input_combs:
                input_combs_and_action_categories.append((input_comb, action_categories))
        else:
            for i in range(n_simulation_steps):
                input_comb, action_categories = self.generate_input_comb(
                    epsilon=epsilon,
                    eval_func=self.form_actor.evaluate
                )
                input_combs_and_action_categories.append((input_comb, action_categories))

        # best_input_comb = self.best_input_comb
        # best_reward = self.form_critic.evaluate_input_comb(self.best_input_comb)
        best_input_comb = None
        best_action_categories = None
        best_reward = -np.inf
        for input_comb, action_categories in input_combs_and_action_categories:
            reward = self.form_critic.evaluate_input_comb(input_comb)
            if reward > best_reward:
                best_reward = reward
                best_input_comb = input_comb
                best_action_categories = action_categories
        return Form.convert_input_comb_to_actions(best_input_comb), best_action_categories

    def generate_input_comb(self, epsilon=1.0, eval_func=None):
        input_comb = OrderedDict()
        action_categories = []
        previous_actions = []
        for action_type, action_ele in self.input_candidates:
            previous_values = [
                None if previous_action.value is None else previous_action.value_text_parsed
                for previous_action in previous_actions
            ]
            value_candidates = []
            for value in self.input_candidates[(action_type, action_ele)]:
                if value is not None:
                    value_text = Action(action_ele, action_type, value).value_text_parsed
                    if value_text in previous_values:
                        continue
                value_candidates.append(value)
            if np.random.uniform() <= epsilon:
                action_category = "Explore"
                if np.random.uniform() <= 0.5:
                    action_value = random.choice(value_candidates)
                else:
                    greedy_input_comb_rewards = self.greedy_input_comb_rewards[(action_type, action_ele)]
                    value_candidate_rewards = {
                        value_candidate: greedy_input_comb_rewards[value_candidate]
                        for value_candidate in value_candidates
                    }
                    action_value = Utils.weighted_choice(value_candidate_rewards)
            else:
                action_category = "Exploit"
                max_q_score = -1
                best_value = random.choice(value_candidates)
                for value_candidate in value_candidates:
                    action_candidate = Action(element=action_ele, action_type=action_type, value=value_candidate)
                    q_score = eval_func(self, previous_actions, action_candidate)
                    if q_score > max_q_score:
                        max_q_score = q_score
                        best_value = value_candidate
                action_value = best_value
            previous_actions.append(Action(element=action_ele, action_type=action_type, value=action_value))
            input_comb[(action_type, action_ele)] = action_value
            action_categories.append(action_category)
        return input_comb, action_categories

    @staticmethod
    def eval_action_greedy(form, previous_actions, action):
        if action.value is None:
            return 0
        else:
            actions = previous_actions + [action]
            reward = form.simulate_actions(actions=actions)
        return reward

    def learn(self):
        if len(self.tried_input_combs) == 0:
            return None

        critic_loss = self.form_critic.learn()
        actor_loss = self.form_actor.learn()

        actor_input_comb, _ = self.generate_input_comb(epsilon=0, eval_func=self.form_actor.evaluate)
        critic_reward = self.form_critic.evaluate_input_comb(actor_input_comb)
        return {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "actor_input": self.get_input_comb_str(actor_input_comb),
            "critic_reward": critic_reward
        }


class FormActor:
    def __init__(self, form):
        self.form = form
        self.fe = FeatureExtractor(
            # state representation
            use_screenshot=False, use_dom_style=False, use_dom_type=False, use_dom_embed=True, use_dom_interacted=False,
            # query representation
            use_query_embed=True, use_query_score=True,
            # action representation
            use_action_type=False, use_action_query_sim=True,
            # query intersection representation
            use_sim_non_para=True, use_sim_para_val=False, use_sim_para_anno=True, merge_para_sim=False,
            # feature dimensions
            feature_width=100, feature_height=100, text_embed_dim=10, n_para_dim=10,
            # misc
            disable_text_embed=False, feature_cache_size=5000, work_dir=None
        )
        self.net = self._build_net()

    def _build_net(self):
        # input: form, actions
        action_loc = Input(shape=(self.fe.height, self.fe.width, 1))
        dom_sim = Input(shape=(self.fe.height, self.fe.width, 1))
        input_comb_vec = Input(shape=(len(self.form.get_input_comb_vec()),))
        dom_embed = Input(shape=(self.fe.height, self.fe.width, self.fe.text_embed_dim))
        para_value_embed = Input(shape=(self.fe.text_embed_dim,))
        para_anno_embed = Input(shape=(self.fe.text_embed_dim,))

        dom_conv = Sequential(layers=[
            Conv2D(1, (7, 7), padding="same", activation="relu"),
            Conv2D(1, (7, 7), padding="same", activation="relu"),
            Conv2D(1, (7, 7), padding="same", activation="sigmoid")
        ])

        action_conv = Sequential(layers=[
            Conv2D(1, (3, 3), padding="valid", activation="relu"),
            MaxPooling2D(),
            Conv2D(1, (3, 3), padding="valid", activation="relu"),
            MaxPooling2D(),
            Conv2D(1, (3, 3), padding="valid", activation="relu"),
            MaxPooling2D(),
            Conv2D(1, (3, 3), padding="valid", activation="relu"),
            MaxPooling2D(),
            Flatten()
        ])

        def compute_sim(x):
            channel_product = K.prod(K.concatenate(x, axis=-1), axis=-1)
            return K.mean(channel_product, axis=[1,2])

        def get_action_vec(x):
            return K.concatenate(x, axis=-1)

        dom_sim_loc = dom_conv(dom_sim)
        sim_q = Lambda(compute_sim)([action_loc, dom_sim_loc])

        action_loc_vec = action_conv(action_loc)
        action_vec = Lambda(get_action_vec)([action_loc_vec, para_anno_embed, para_value_embed])
        action_q = Dense(1, activation="sigmoid")(action_vec)

        input_comb_q = Dense(1, activation="sigmoid")(input_comb_vec)
        phi = 0.1

        # q = Lambda(lambda x: K.sum([K.expand_dims(x[0], 1), phi * x[1]], axis=0))([sim_q, action_q])
        # net = Model(inputs=[action_loc, dom_sim, para_value_embed, para_anno_embed], outputs=q)
        q = Lambda(lambda x: K.sum([K.expand_dims(x[0], 1), phi * x[1]], axis=0))([sim_q, input_comb_q])
        net = Model(inputs=[action_loc, dom_sim, input_comb_vec], outputs=q)

        optimizer = keras.optimizers.Adam(lr=0.01)
        net.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        # print(net.summary())
        return net

    def learn(self):
        best_input_comb = self.form.best_input_comb
        positive_samples = []
        negative_samples = []
        previous_actions = []
        for (action_type, action_ele) in best_input_comb:
            value_candidates = self.form.input_candidates[(action_type, action_ele)]
            best_value = best_input_comb[(action_type, action_ele)]
            best_action = None
            for value in value_candidates:
                action = Action(action_ele, action_type, value)
                encoding = self.encode(self.form, previous_actions, action)
                if value == best_value:
                    positive_samples.append(encoding)
                else:
                    negative_samples.append(encoding)
            if best_action:
                previous_actions.append(best_action)
        # negative_samples = random.sample(negative_samples, len(positive_samples))
        samples = positive_samples + negative_samples
        if len(samples) <= 0:
            return None
        samples = self.zip_inputs(samples)
        labels = [1.0] * len(positive_samples) + [0.0] * len(negative_samples)
        history = self.net.fit(x=samples, y=np.array(labels), epochs=5, verbose=0)
        # i = random.randint(0, len(positive_samples) - 1)
        # self.fe.show_image("output/positive_action_loc.png", positive_samples[i][0])
        # self.fe.show_image("output/positive_dom_sim.png", positive_samples[i][1])
        return history.history["loss"][-1]

    def zip_inputs(self, inputs):
        return list([np.array(x) for x in zip(*inputs)])

    def encode(self, form, previous_actions, action):
        task = form.task
        assert isinstance(task, Task)
        action_loc = self.fe.get_action_feature_image(action)
        dom_sim = np.zeros([self.fe.height, self.fe.width, 1])
        value_text, value_annotation = "", ""
        if action.value is not None:
            value_text = action.value_text_parsed
            value_annotation = task.get_parameter_surrounding_word_parsed(value_text)
            dom_sim = self.get_dom_similarity_feature(task.state, value_annotation)
        # para_value_vec = Utils.vec(value_text)[:self.fe.text_embed_dim]
        # para_anno_vec = Utils.vec(value_annotation)[:self.fe.text_embed_dim]
        # return action_loc, dom_sim, para_value_vec, para_anno_vec
        input_comb_vec = self.form.get_input_comb_vec(Form.convert_actions_to_input_comb(previous_actions + [action]))
        return action_loc, dom_sim, input_comb_vec

    def evaluate(self, form, previous_actions, action):
        input = self.encode(form, previous_actions, action)
        q = self.net.predict(x=self.zip_inputs([input]))
        return float(q)

    def get_dom_similarity_feature(self, state, text_parsed):
        def compute_feature(element, text_parsed):
            if not element.own_text_parsed or not text_parsed:
                return None
            return Utils.text_similarity(text_parsed, element.own_text_parsed)

        def merge_features(old_ele_feature, new_ele_feature):
            return np.max([old_ele_feature, new_ele_feature], axis=0)

        feature = np.zeros([self.fe.height, self.fe.width, 1])
        self.fe._render_feature_recursively(feature, merge_features, compute_feature, state.root_element, text_parsed)
        return feature


class FormCritic:
    def __init__(self, form):
        self.form = form
        self.net = self._build_net()

    def _build_net(self):
        model = Sequential()
        model.add(Dense(1, input_shape=(len(self.form.get_input_comb_vec()),)))
        optimizer = keras.optimizers.Adam(lr=0.01)
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def evaluate_input_comb(self, input_comb):
        if input_comb is None:
            return 0
        actions = Form.convert_input_comb_to_actions(input_comb)
        simulation_reward = self.form.simulate_actions(actions)
        input_comb_vec = self.form.get_input_comb_vec(input_comb)
        predict_reward = self.net.predict(x=np.array([input_comb_vec]))
        reward = simulation_reward + float(predict_reward)
        return reward

    def learn(self):
        X = []
        Y = []
        for input_comb, simulation_reward, actual_reward in self.form.tried_input_combs.values():
            X.append(self.form.get_input_comb_vec(input_comb))
            Y.append(actual_reward - simulation_reward)
        history = self.net.fit(np.array(X), np.array(Y), epochs=5, verbose=0)
        return history.history["loss"][-1]


if __name__ == "__main__":
    form_net = FormActor(form=None)

