# coding=utf-8
import time
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from environment import Action, Utils


class FeatureExtractor:
    ELE_TYPE_INPUT = "input"
    ELE_TYPE_NAVIGATE = "navigate"
    ACTION_TYPE_MAPPING = {
        Action.CLICK: ELE_TYPE_NAVIGATE,
        Action.PRESS_ENTER: ELE_TYPE_NAVIGATE,
        Action.CHECK: ELE_TYPE_NAVIGATE,
        Action.INPUT_TEXT: ELE_TYPE_INPUT,
        Action.SELECT: ELE_TYPE_INPUT
    }
    ACTION_TYPES = [ELE_TYPE_INPUT, ELE_TYPE_NAVIGATE]

    def __init__(self,
                 # state representation
                 use_screenshot=True, use_dom_style=True, use_dom_type=True, use_dom_embed=True, use_dom_interacted=True,
                 # query representation
                 use_query_embed=True, use_query_score=True,
                 # action representation
                 use_action_type=True, use_action_query_sim=True,
                 # query intersection representation
                 use_sim_non_para=True, use_sim_para_val=True, use_sim_para_anno=True, merge_para_sim=True,
                 # feature dimensions
                 feature_width=100, feature_height=100, text_embed_dim=10, n_para_dim=10, screenshot_dim=1,
                 # misc
                 disable_text_embed=False, feature_cache_size=5000, work_dir=None, **kwargs):
        # State
        self.use_screenshot = use_screenshot
        self.use_dom_style = use_dom_style
        self.use_dom_type = use_dom_type
        self.use_dom_embed = use_dom_embed
        self.use_dom_interacted = use_dom_interacted

        # Query
        self.use_query_embed = use_query_embed
        self.use_query_score = use_query_score

        # Action
        self.use_action_type = use_action_type
        self.use_action_query_sim = use_action_query_sim

        # Query intersection
        self.use_sim_non_para = use_sim_non_para
        self.use_sim_para_val = use_sim_para_val
        self.use_sim_para_anno = use_sim_para_anno
        self.merge_para_sim = merge_para_sim

        if disable_text_embed:
            self.use_dom_embed = False
            self.use_query_embed = False

        # Feature dimensions
        self.width = feature_width
        self.height = feature_height
        self.text_embed_dim = text_embed_dim
        self.n_para_dim = n_para_dim
        self.screenshot_dim = screenshot_dim
        self.n_dim_para_sim = 1 if self.merge_para_sim else self.n_para_dim

        self.action_dim = \
            (len(self.ACTION_TYPES) if self.use_action_type else 0) \
            + (1 if self.use_sim_non_para else 0) \
            + (self.n_para_dim if self.use_sim_para_val else 0)
        self.task_dim = \
            (self.screenshot_dim if self.use_screenshot else 0) \
            + (1 if self.use_dom_style else 0) \
            + (len(self.ACTION_TYPES) if self.use_dom_type else 0) \
            + (self.text_embed_dim if self.use_dom_embed else 0) \
            + (1 if self.use_dom_interacted else 0) \
            + (1 if self.use_sim_non_para else 0) \
            + (self.n_dim_para_sim if self.use_sim_para_anno else 0) \
            + (self.n_dim_para_sim if self.use_sim_para_val else 0)

        self.dom_feature_image_n_channel = self.task_dim
        self.action_feature_image_n_channel = 1
        self.query_vector_length = \
            ((self.text_embed_dim * (1 + self.n_para_dim)) if self.use_query_embed else 0) \
            + ((1 + self.n_para_dim) if self.use_query_score else 0)
        self.action_vector_length = \
            (len(self.ACTION_TYPES) if self.use_action_type else 0) \
            + ((1 + self.n_para_dim) if self.use_action_query_sim else 0)

        self.work_dir = work_dir
        self._feature_cache_size = feature_cache_size
        self._feature_cache = {}

    def get_feature_shape_old(self):
        return [(self.height, self.width, self.task_dim), (self.height, self.width, self.action_dim)]

    def get_feature_shape(self):
        return [
            (self.height, self.width, self.dom_feature_image_n_channel), # dom feature image
            (self.height, self.width, self.action_feature_image_n_channel), # action feature image
            (self.query_vector_length, ), # query vector
            (self.action_vector_length, ) # action vector
        ]

    def get_feature(self, tasks, actions):
        dom_feature_images = []
        action_feature_images = []
        query_vectors = []
        action_vectors = []
        for task, action in zip(tasks, actions):
            dom_feature_images.append(self.get_dom_feature_image(task))
            action_feature_images.append(self.get_action_feature_image(action))
            query_vectors.append(self.get_query_vector(task))
            action_vectors.append(self.get_action_vector(action, task))
        return [
            np.array(dom_feature_images),
            np.array(action_feature_images),
            np.array(query_vectors),
            np.array(action_vectors)
        ]

    def get_feature_old(self, tasks, actions):
        task_features = []
        action_features = []
        for task, action in zip(tasks, actions):
            task_features.append(self.get_task_feature(task))
            action_features.append(self.get_action_feature(task, action))
        return [np.array(task_features), np.array(action_features)]

    def _save_feature_to_cache(self, feature_id, feature):
        if self._feature_cache_size <= 0:
            return
        timestamp = time.time()
        self._feature_cache[feature_id] = (feature, timestamp)
        if len(self._feature_cache) > self._feature_cache_size:
            timestamps = [v[1] for v in self._feature_cache.values()]
            median_timestamp = np.median(timestamps)
            ids_to_delete = []
            for i in self._feature_cache:
                if self._feature_cache[i][1] < median_timestamp:
                    ids_to_delete.append(i)
            for i in ids_to_delete:
                self._feature_cache.pop(i, None)

    def _get_feature_from_cache(self, feature_id):
        return self._feature_cache[feature_id][0] if feature_id in self._feature_cache else None

    def get_query_vector(self, task):
        query_vecs = []
        if self.use_query_embed:
            non_para_embeds = []
            for non_para in task.non_parameters_parsed:
                non_para_embed = Utils.vec(non_para)[:self.text_embed_dim]
                non_para_embeds.append(non_para_embed)
            query_vecs.append(np.mean(non_para_embeds, axis=0))
            for i in range(self.n_para_dim):
                para_embed = np.zeros(self.text_embed_dim)
                if i < len(task.parameter_annotations_parsed):
                    para_annos = task.parameter_annotations_parsed[i]
                    para_embed = np.mean([Utils.vec(anno)[:self.text_embed_dim] for anno in para_annos], axis=0)
                query_vecs.append(para_embed)
        if self.use_query_score:
            sim_non_para, sim_paras = task.query_achieved_scores()
            query_score = np.zeros(1 + self.n_para_dim)
            query_score[0] = sim_non_para
            for i, sim_para in enumerate(sim_paras):
                if i + 1 < len(query_score):
                    query_score[i+1] = sim_para
            query_vecs.append(query_score)
        return np.concatenate(query_vecs)

    def get_action_vector(self, action, task):
        action_vecs = []
        if self.use_action_type:
            action_type_vec = np.zeros(len(self.ACTION_TYPES))
            if action.action_type in self.ACTION_TYPE_MAPPING:
                action_type = self.ACTION_TYPE_MAPPING[action.action_type]
                action_type_index = self.ACTION_TYPES.index(action_type)
                action_type_vec[action_type_index] = 1.
            action_vecs.append(action_type_vec)
        if self.use_action_query_sim:
            action_sim_non_para = 0
            if action.is_input:
                action_sim_non_paras = []
                for non_para in task.non_parameters_parsed:
                    action_sim_non_paras.append(Utils.text_similarity(non_para, action.value_text_parsed))
                action_sim_non_para = np.mean(action_sim_non_paras)
            action_vecs.append(np.array([action_sim_non_para]))

            action_sim_para_val = np.zeros(self.n_para_dim)
            if action.is_input:
                for i, para_val in enumerate(task.parameter_values_parsed):
                    if i > self.n_para_dim:
                        break
                    action_sim_para_val[i] = Utils.text_similarity(para_val, action.value_text_parsed)
            action_vecs.append(action_sim_para_val)
        return np.concatenate(action_vecs)

    def get_dom_feature_image(self, task):
        sub_features = []
        if self.use_screenshot:
            sub_features.append(self.get_screenshot_feature(task.state.screenshot))
        if self.use_dom_style:
            sub_features.append(self.get_text_style_feature(task.state))
        if self.use_dom_type:
            sub_features.append(self.get_element_type_feature(task.state))
        if self.use_dom_embed:
            sub_features.append(self.get_text_embedding_feature(task.state))
        if self.use_dom_interacted:
            sub_features.append(self.get_interacted_elements_feature(task.state, task))
        if self.use_sim_non_para:
            sub_features.append(self.get_task_non_para_feature(task.state, task))
        if self.use_sim_para_anno:
            sub_features.append(self.get_task_para_annotation_feature(task.state, task))
        if self.use_sim_para_val:
            sub_features.append(self.get_task_para_value_feature(task.state, task))
        page_feature_image = np.concatenate(sub_features, axis=-1)
        return page_feature_image

    def get_action_feature_image(self, action):
        action_feature_image = np.zeros([self.height, self.width, 1])
        if action.element:
            t, b, l, r = self._get_element_tblr(action.element)
            action_feature_image[t:b + 1, l:r + 1, :] = 1.
        return action_feature_image

    def get_task_feature(self, task):
        sub_features = []
        if self.use_screenshot:
            sub_features.append(self.get_screenshot_feature(task.state.screenshot))
        if self.use_dom_style:
            sub_features.append(self.get_text_style_feature(task.state))
        if self.use_dom_type:
            sub_features.append(self.get_element_type_feature(task.state))
        if self.use_dom_embed:
            sub_features.append(self.get_text_embedding_feature(task.state))
        if self.use_dom_interacted:
            sub_features.append(self.get_interacted_elements_feature(task.state, task))
        if self.use_sim_non_para:
            sub_features.append(self.get_task_non_para_feature(task.state, task))
        if self.use_sim_para_anno:
            sub_features.append(self.get_task_para_annotation_feature(task.state, task))
        if self.use_sim_para_val:
            sub_features.append(self.get_task_para_value_feature(task.state, task))
        task_feature = np.concatenate(sub_features, axis=-1)
        return task_feature

    def get_action_feature(self, task, action):
        if not action or not action.element:
            return np.zeros([self.height, self.width, self.action_dim])
        sub_features = []
        t, b, l, r = self._get_element_tblr(action.element)
        if self.use_action_type:
            action_type_feature = np.zeros([self.height, self.width, len(self.ACTION_TYPES)])
            if action.action_type in self.ACTION_TYPE_MAPPING:
                action_type = self.ACTION_TYPE_MAPPING[action.action_type]
                action_type_index = self.ACTION_TYPES.index(action_type)
                action_type_feature[t:b + 1, l:r + 1, action_type_index] = 1.
            sub_features.append(action_type_feature)
        # print(a_feature.shape)
        action_feature = np.concatenate(sub_features, axis=-1)
        return action_feature

    def get_action_cluster_feature(self, task, actions):
        action_features = [self.get_action_feature(task, action) for action in actions]
        return np.max(action_features, axis=0)

    def get_task_feature_labels(self, task):
        feature_labels = []
        if self.use_screenshot:
            feature_labels.extend(["screenshot"] * self.screenshot_dim)
        if self.use_dom_style:
            feature_labels.append("text style")
        if self.use_dom_type:
            feature_labels.extend(["element type: " + t for t in self.ACTION_TYPES])
        if self.use_dom_embed:
            feature_labels.extend(["text embed"] * self.text_embed_dim)
        if self.use_dom_interacted:
            feature_labels.append("interacted elements")
        if self.use_sim_non_para:
            feature_labels.append("sim_task")
        if self.use_sim_para_anno:
            annotations = [",".join(x) for x in task.parameter_annotations_parsed]
            feature_labels.extend(self._get_parameter_labels(annotations, prefix="sim_anno"))
        if self.use_sim_para_val:
            feature_labels.extend(self._get_parameter_labels(task.parameter_values, prefix="sim_para"))
        return feature_labels

    def get_action_feature_labels(self, task):
        feature_labels = []
        if self.use_action_type:
            feature_labels.extend(["action type: " + t for t in self.ACTION_TYPES])
        if self.use_sim_non_para:
            feature_labels.append("sim_task")
        if self.use_sim_para_val:
            feature_labels.extend(self._get_parameter_labels(task.parameter_values, prefix="sim_para"))
        return feature_labels

    def _get_parameter_labels(self, words, prefix="sim_para"):
        if self.n_para_dim == 1:
            return [prefix]
        parameter_labels = []
        for i in range(self.n_para_dim):
            parameter_labels.append(prefix + ":" + ("" if i >= len(words) else words[i]))
        return parameter_labels

    def get_screenshot_feature(self, image):
        img = image.resize((self.width, self.height))
        image_feature = np.array(img)[..., :3]
        # Convert to gray scale image v
        if self.screenshot_dim == 1:
            image_feature = np.dot(image_feature, [0.299, 0.587, 0.114])
            image_feature = np.expand_dims(image_feature, -1)
        return image_feature / 256

    def get_text_style_feature(self, state):
        def compute_feature(element):
            return min(element.font_size * 0.05, 1.0) if element.own_text else 0

        def merge_features(old_ele_feature, new_ele_feature):
            return np.max([old_ele_feature, new_ele_feature], axis=0)

        feature = np.zeros([self.height, self.width, 1])
        self._render_feature_recursively(feature, merge_features, compute_feature, state.root_element)
        return feature

    def get_element_type_feature(self, state):
        def get_element_type(element):
            action_types = np.zeros(len(self.ACTION_TYPES))
            for env_action_type in element.acceptable_action_types:
                if env_action_type in self.ACTION_TYPE_MAPPING:
                    feature_action_type = self.ACTION_TYPE_MAPPING[env_action_type]
                    action_type_index = self.ACTION_TYPES.index(feature_action_type)
                    action_types[action_type_index] = 1.0
            return action_types

        def merge_features(old_ele_feature, new_ele_feature):
            return np.max([old_ele_feature, new_ele_feature], axis=0)

        feature = np.zeros([self.height, self.width, len(self.ACTION_TYPES)])
        self._render_feature_recursively(feature, merge_features, get_element_type, state.root_element)
        return feature

    def get_text_embedding_feature(self, state):
        def compute_feature(element):
            if element.own_text_parsed:
                return Utils.vec(element.own_text_parsed)[:self.text_embed_dim]
            else:
                return None

        def merge_features(old_ele_feature, new_ele_feature):
            return new_ele_feature if old_ele_feature.all() == 0. \
                else np.mean([new_ele_feature, old_ele_feature], axis=0)

        feature = np.zeros([self.height, self.width, self.text_embed_dim])
        self._render_feature_recursively(feature, merge_features, compute_feature, state.root_element)
        return feature

    def get_interacted_elements_feature(self, state, task):
        interacted_elements, _ = state.interacted_elements(task.action_history)

        def compute_feature(element):
            return 1 if element in interacted_elements else 0

        def merge_features(old_ele_feature, new_ele_feature):
            return np.max([old_ele_feature, new_ele_feature], axis=0)

        feature = np.zeros([self.height, self.width, 1])
        self._render_feature_recursively(feature, merge_features, compute_feature, state.root_element)
        return feature

    def get_task_non_para_feature(self, state, task):
        def compute_feature(element, task):
            if not element.own_text_parsed:
                return None
            return self._get_text_similarity_with_words(
                text=element.own_text_parsed,
                words=task.non_parameters_parsed,
                n_dim=1
            )

        def merge_features(old_ele_feature, new_ele_feature):
            return np.max([old_ele_feature, new_ele_feature], axis=0)

        feature = np.zeros([self.height, self.width, 1])
        self._render_feature_recursively(feature, merge_features, compute_feature, state.root_element, task)
        return feature

    def get_task_para_value_feature(self, state, task):
        # parameter_achieved_scores = self._get_parameter_achieved_scores(task)

        def compute_feature(element, task):
            if not element.own_text_parsed:
                return None
            return self._get_text_similarity_with_words(
                text=element.own_text_parsed,
                words=task.parameter_values_parsed,
                n_dim=self.n_dim_para_sim
                # sub_scores=parameter_achieved_scores
            )

        def merge_features(old_ele_feature, new_ele_feature):
            return np.max([old_ele_feature, new_ele_feature], axis=0)

        feature = np.zeros([self.height, self.width, self.n_dim_para_sim])
        self._render_feature_recursively(feature, merge_features, compute_feature, state.root_element, task)
        return feature

    def get_task_para_annotation_feature(self, state, task):
        def compute_feature(element, task):
            if not element.own_text_parsed:
                return None
            para_similarities = []
            for para_id in task.parameter_ids:
                para_annotations = task.surrounding_words_parsed[para_id]
                para_similarity = self._get_text_similarity_with_words(
                    text=element.own_text_parsed,
                    words=para_annotations,
                    n_dim=self.n_dim_para_sim
                )
                para_similarities.append(para_similarity)
            # if self.n_para_dim == 1:
            return np.max(para_similarities)

        def merge_features(old_ele_feature, new_ele_feature):
            return np.max([old_ele_feature, new_ele_feature], axis=0)

        feature = np.zeros([self.height, self.width, self.n_dim_para_sim])
        self._render_feature_recursively(feature, merge_features, compute_feature, state.root_element, task)
        return feature

    def _render_feature_recursively(self, base_feature, merge_features, compute_feature, element, *args):
        if element.in_window:
            ele_feature = compute_feature(element, *args)
            if ele_feature is not None:
                t, b, l, r = self._get_element_tblr(element)
                old_ele_feature = base_feature[t:b + 1, l:r + 1, :]
                new_ele_feature = np.copy(old_ele_feature)
                new_ele_feature[:, :] = ele_feature
                merged_ele_feature = merge_features(old_ele_feature, new_ele_feature)
                base_feature[t:b + 1, l:r + 1, :] = merged_ele_feature
        for child_ele in element.child_elements:
            self._render_feature_recursively(base_feature, merge_features, compute_feature, child_ele, *args)

    def _get_text_similarity_with_words(self, text, words, n_dim=1, sub_scores=None):
        if not sub_scores:
            sub_scores = [0] * len(words)
        word_similarities = [max(Utils.text_similarity(word, text) - sub_scores[i], 0)
                             for i, word in enumerate(words)]
        if n_dim > 1:
            similarity_array = np.zeros(n_dim)
            if len(words) > 0 and len(text) > 0:
                for i in range(n_dim):
                    if i >= len(words):
                        break
                    similarity_array[i] = word_similarities[i]
            return similarity_array
        else:
            return np.max(word_similarities)

    def _get_element_tblr(self, element):
        bound = element.get_resized_bound(new_width=self.width, new_height=self.height)
        return bound['top'], bound['bottom'], bound['left'], bound['right']

    def show_image(self, fname, inp):
        inp = np.array(inp)
        if inp.shape[-1] == 3:
            # inp = inp.transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            inp = std * inp + mean
            inp = np.clip(inp, 0, 1)
        elif inp.shape[-1] == 1:
            inp = inp.squeeze(-1)
        plt.imsave(fname, inp)

    def plot_feature(self, task, action):
        from mpl_toolkits.mplot3d import Axes3D  # necessary import
        from matplotlib.ticker import MaxNLocator

        task_feature = self.get_task_feature(task)
        task_feature_labels = self.get_task_feature_labels(task)
        action_feature = self.get_action_feature(task, action)
        action_feature_labels = self.get_action_feature_labels(task)

        # create a vertex mesh
        xx, yy = np.meshgrid(np.linspace(1, self.height, self.width), np.linspace(1, self.height, self.width))
        fig = plt.figure()
        plt.subplots_adjust(right=0.8)
        ax = fig.gca(projection=Axes3D.name)

        n_task_dim = len(task_feature_labels)
        n_action_dim = len(action_feature_labels)
        for i in range(0, n_action_dim):
            data = np.copy(action_feature[:, :, i])
            data[0][0] = 1.
            data[0][1] = 0.
            ax.contourf(xx, yy, data, 10, zdir='z', offset=i, cmap=plt.cm.get_cmap("Reds"), alpha=.3)
        for i in range(0, n_task_dim):
            data = np.copy(task_feature[:, :, i])
            data[0][0] = 1.
            data[0][1] = 0.
            if "screenshot" in task_feature_labels[i]:
                ax.contourf(xx, yy, data, 10, zdir='z', offset=n_action_dim, cmap=plt.cm.get_cmap("gray"))
            else:
                ax.contourf(xx, yy, data, 10, zdir='z', offset=n_action_dim + i, cmap=plt.cm.get_cmap("Blues"),
                            alpha=.3)

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.zaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_zlim3d(bottom=0, top=n_action_dim + n_task_dim)
        ax.tick_params(axis='z', labelsize=8)
        ax.zaxis.set_ticks(range(n_action_dim + n_task_dim))
        ax.set_zticklabels(action_feature_labels + task_feature_labels, ha='left')

        ax.set_title("Task: %s\nAction: %s\nURL: %s" % (" ".join(task.query_words), action, task.start_url))
        plt.show()

