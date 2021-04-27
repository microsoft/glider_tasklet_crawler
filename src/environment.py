# -*- coding: utf-8 -*-
import argparse
import hashlib
import json
import logging
import os
import sys
import platform
import random
import re
import time
import copy
import shutil
import traceback
import difflib
import networkx as nx
import numpy as np
import spacy
import functools
from itertools import takewhile
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED
from datetime import datetime
from dateutil import parser as date_parser
from PIL import Image, ImageTk
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support.select import Select
from selenium.common.exceptions import NoSuchElementException

GLOBAL_CONFIGS = {
    "semantic_similarity": False,
    "action_restriction": False,
    "use_annotations": True,
}
REWARD_ITEM_WEIGHTS = {
    "step_count": -1,
    "action_spatial_distance": -2,
    "action_direction": -2,
    # "similarity_with_demo_actions": 0,
    # "similarity_with_demo_surrounding": 0,
    "similarity_with_non_parameters": 5,
    "similarity_with_parameters": 10,
    # "distance_between_parameters": 0,
    "null": 0
}
EXCLUDE_QUERY_WORDS = ["of", "a", "an", "the", "in", "on", "by", "with", "and",
                       "for", "at", "that", "from", "to", "me", "about"]
SPACY_EXCLUDE = ["first", "second", "third"]
DISTANCE_THRESHOLD = 500
UTG_AS_ZIP = True
COMPRESS_METHOD = ZIP_DEFLATED
COMPRESS_LEVEL = 3
# Hosts that need a longer wait time
LONG_WAIT_HOSTS = [
    "academic.microsoft.com",
    "ieeexplore.ieee.org",
    "online.unitconverterpro.com",
    "searchit.libraries.wsu.edu",
    "www.rentcafe.com",
    "www.rent.com",
    "www.joybuy.com",
    "www.kohls.com",
    "www.bing.com",
    "www.google.com",
    "estimatefares.com",
    "ride.guru",
    "www.lyft.com"
]
RESTART_HOSTS = [
    "www.expedia.com",
    "www.bing.com",
    "www.regmovies.com",
    "www.fandango.com",
    "www.atomtickets.com",
    "www.movietickets.com",
    "www.moviefone.com",
    "www.opentable.com",
    "www.uber.com",
    "catalog.swanlibraries.net",
    "dictionary.cambridge.org",
    "www.merriam-webster.com"
]


def lazy_property(func):
    attribute = '_lazy_' + func.__name__

    @property
    @functools.wraps(func)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, func(self))
        return getattr(self, attribute)

    return wrapper


class Utils:
    _instance = None

    def __init__(self):
        self.cache = {}

    @staticmethod
    def _get_instance():
        if Utils._instance is None:
            Utils._instance = Utils()
        return Utils._instance

    @lazy_property
    def _nlp(self):
        nlp = spacy.load("en_core_web_md")
        return nlp

    @staticmethod
    def parse(sentence):
        if not sentence:
            return sentence
        # convert one-way to oneway, g-pound to gpound
        sentence = sentence.replace("-", "")
        tokens = [token.lemma_ for token in Utils.nlp(sentence.lower(), disable=["tokenizer", "parser", "ner"])]
        filtered_tokens = []
        for token in tokens:
            if len(token) > 1 or Utils.is_number(token):
                filtered_tokens.append(token)
        return " ".join(filtered_tokens if len(filtered_tokens) > 0 else tokens)

    @staticmethod
    def md5(text):
        m = hashlib.md5(text.encode("utf-8"))
        return m.hexdigest()

    @staticmethod
    def common_prefix(words):
        return ''.join(c[0] for c in takewhile(lambda x: all(x[0] == y for y in x), zip(*words)))

    @staticmethod
    def parse_time_info(text):
        time_info = []
        try:
            date = date_parser.parse(text)
            today = datetime.today()
            if date.year != today.year:
                time_info.append("year:" + str(date.year))
            if date.month != today.month:
                time_info.append("month:" + str(date.month))
            if date.day != today.day:
                time_info.append("day:" + str(date.day))
            if date.hour != today.hour and date.hour != 0:
                time_info.append("hour:" + str(date.hour))
            if date.minute != today.minute and date.minute != 0:
                time_info.append("minute:" + str(date.minute))
            if date.second != today.second and date.second != 0:
                time_info.append("second:" + str(date.second))
        except Exception:
            pass
        return time_info

    @staticmethod
    def date_similarity(query_text, match_text):
        method_id = "date_similarity(%s,%s)" % (query_text, match_text)
        if method_id not in Utils._get_instance().cache:
            query_time_info = Utils.parse_time_info(query_text)
            if len(query_time_info) == 0:
                return 0
            match_time_info = Utils.parse_time_info(match_text)
            if len(match_time_info) == 0:
                return 0
            query_time_info_set = set(query_time_info)
            match_time_info_set = set(match_time_info)
            if (not query_time_info_set.issuperset(match_time_info_set)) \
                    and (not query_time_info_set.issubset(match_time_info_set)):
                return 0
            similarity = difflib.SequenceMatcher(None, query_time_info, match_time_info).ratio()
            Utils._get_instance().cache[method_id] = similarity
            return similarity
        return Utils._get_instance().cache[method_id]

    @staticmethod
    def words_similarity(query_words, match_words):
        query_words = set(query_words)
        match_words = set(match_words)

        if len(query_words) == 0 or len(match_words) == 0:
            return 0

        common_score = 0.0
        for query_word in query_words:
            similarities = [Utils.word_similarity(query_word, match_word) for match_word in match_words]
            common_score += max(similarities)
        return common_score * 2.0 / (len(query_words) + len(match_words))

    @staticmethod
    def word_similarity(query_word, match_word, enable_semantic_sim=None):
        if enable_semantic_sim is None:
            enable_semantic_sim = GLOBAL_CONFIGS['semantic_similarity']

        if query_word == match_word:
            return 1

        query_word = re.sub("[^0-9A-Za-z]", " ", query_word).strip()
        match_word = re.sub("[^0-9A-Za-z]", " ", match_word).strip()
        if query_word == match_word:
            return 1
        if len(query_word) == 0 or len(match_word) == 0:
            return 0

        # # Dirty workaround to fix synonym matching
        synonym_lists = [
            ["weight", "mass"],
            ["synonym", "thesaurus"],
            ["find", "search"],
            ["bedroom", "bed"],
            ["walk", "foot"],
            ["publication", "pub"],
            ["number", "num"],
            ["destination", "dropoff", "end", "target"],
            ["source", "start", "pickup"],
            ["calculate", "compute", "estimate"],
            ["location", "where", "address"],
            ["route", "direction"]
        ]
        for synonym_list in synonym_lists:
            if query_word in synonym_list and match_word in synonym_list:
                return 1

        # common_prefix = Utils.common_prefix([query_word, match_word])
        # similarity = len(common_prefix) / max(len(query_word), len(match_word))

        if Utils.is_number(query_word) and query_word in match_word.split():
            return 0.8
        if Utils.is_number(match_word) and match_word in query_word.split():
            return 0.8

        similarity = difflib.SequenceMatcher(None, query_word, match_word).ratio()
        if query_word[0] != match_word[0]:
            similarity = similarity * 0.5

        if enable_semantic_sim:
            semantic_similarity = Utils._semantic_similarity(query_word, match_word)
            similarity = max(similarity, semantic_similarity)

        similarity = similarity if similarity > 0.7 else 0
        return similarity

    @staticmethod
    def _semantic_similarity(query_text, match_text):
        semantic_similarity = 0
        query_processed = Utils.nlp(query_text)
        match_processed = Utils.nlp(match_text)
        if query_processed.vector_norm and match_processed.vector_norm:
            semantic_similarity = query_processed.similarity(match_processed)
        if semantic_similarity < 0:
            semantic_similarity = 0
        semantic_similarity **= 0.5
        return semantic_similarity

    @staticmethod
    def number_similarity(query_text, match_text):
        if Utils.is_number(query_text) and Utils.is_number(match_text):
            if abs(float(query_text.replace(",", "")) - float(match_text.replace(",", ""))) < 0.001:
                return 1
        return 0

    @staticmethod
    def text_similarity(query_text, match_text):
        method_id = "text_similarity(%s,%s)" % (query_text, match_text)
        if method_id not in Utils._get_instance().cache:
            # try more advanced string similarity metric
            if len(query_text) == 0 or len(match_text) == 0:
                similarity = 0
            elif query_text == match_text:
                similarity = 1
            elif Utils.is_number(query_text) and Utils.is_number(match_text):
                similarity = Utils.number_similarity(query_text, match_text)
            elif Utils.is_date(query_text) or Utils.is_date(match_text):
                similarity = Utils.date_similarity(query_text, match_text)
            else:
                if len(query_text) > 3 and len(match_text) > 3:
                    similarity = Utils.words_similarity(query_text.split(), match_text.split()) ** 0.5
                else:
                    similarity = Utils.word_similarity(query_text, match_text)
                # date_similarity = Utils.date_similarity(query_text, match_text)
                # similarity = max(similarity, date_similarity)
            similarity = similarity if similarity > 0.5 else 0
            Utils._get_instance().cache[method_id] = similarity
            return similarity
        return Utils._get_instance().cache[method_id]

    @staticmethod
    def weighted_choice(sample_values, reverse=False):
        """
        Choose a sample based on the values.
        Samples with higher/lower values get higher chance to be selected.
        :param sample_values: a dict mapping samples to values, values should be positive
        :param reverse: If set to True, samples with lower values get higher chance to be selected.
        :return: a sample
        """
        if not sample_values:
            return None
        samples, values = list(zip(*sample_values.items()))
        try:
            weights = (1 / (np.array(values) + 0.01)) if reverse else np.array(values)
            weight_sum = np.sum(weights)
            return np.random.choice(samples) if weight_sum == 0 else np.random.choice(samples, p=weights / weight_sum)
        except Exception:
            pass
        return np.random.choice(samples)

    @staticmethod
    def create_fake_state(current_state, action):
        if (not isinstance(current_state, State)) or (not isinstance(action, Action)):
            return None
        fake_state_dict = copy.deepcopy(current_state.state_dict)
        fake_state_screen = current_state.screenshot
        fake_state = State(state_dict=fake_state_dict, screenshot=fake_state_screen)
        target_ele = fake_state.get_element_by_locator(action.element.locator)
        if target_ele is None or not isinstance(target_ele, Element):
            return None
        if action.action_type not in target_ele.acceptable_action_types:
            return None
        if action.action_type == Action.INPUT_TEXT:
            if target_ele.accepts_text(action.value):
                target_ele.ele_dict["value"] = action.value
            return fake_state
        elif action.action_type == Action.SELECT:
            idx_to_select = int(action.value)
            try:
                options = target_ele.ele_dict["actionSet"]["options"]
                option_to_select = options[idx_to_select]
                target_ele.ele_dict["actionSet"]["selectedIndices"] = [idx_to_select]
                target_ele.ele_dict["text"] = option_to_select
                return fake_state
            except:
                return None
        elif action.is_submit:
            return fake_state
        return None

    @staticmethod
    def parse_action_to_triple(action_line):
        if len(action_line) == 3:
            # already a triple
            return action_line
        m = re.match(ACTION_RE, action_line)
        if m:
            return m.group(1), m.group(2), m.group(3)
        return None

    @staticmethod
    def parse_actions_to_triples(action_lines):
        triples = []
        if action_lines is None:
            return triples
        for action_line in action_lines:
            triple = Utils.parse_action_to_triple(action_line)
            triples.append(triple)
        return triples

    @staticmethod
    def force_to_str(text):
        try:
            return str(text)
        except:
            pass
        try:
            return str(text.decode("utf-8"))
        except:
            return text

    @staticmethod
    def get_host(url):
        host_start = url.find("://") + 3
        if host_start < 3:
            host_start = 0
        host_end = url.find("/", host_start)
        if host_end == -1:
            return url[host_start:]
        else:
            return url[host_start:host_end]

    @staticmethod
    def parse_rgb(rgb_str):
        m = re.search("(\d+),\s*(\d+),\s*(\d+)", rgb_str)
        if m:
            return m.group(1), m.group(2), m.group(3)
        else:
            return None

    @staticmethod
    def top_n(sample_values, n, reverse=False):
        """
        get top N samples
        """
        if not sample_values or n == 0:
            return []
        first_value = list(sample_values.values())[0]
        sort_key = lambda x: x[1]
        if isinstance(first_value, tuple) or isinstance(first_value, list):
            sort_key = lambda x: x[1][0]
        sample_value_pairs = sorted(sample_values.items(), key=sort_key, reverse=reverse)
        result = []
        for sample, value in sample_value_pairs:
            if len(result) >= n:
                return result
            result.append(sample)
        return result

    @staticmethod
    def is_number(text):
        try:
            num = float(text.replace(",", ""))
        except:
            return False
        return True

    @staticmethod
    def is_date(text):
        if text is None or Utils.is_number(text):
            return False
        for c in ["-", "/"]:
            if c not in text:
                continue
            if text.startswith(c) or text.endswith(c):
                continue
            s = text.replace(c, "")
            if len(s) > 8:
                continue
            if Utils.is_number(s):
                return True
            ss = s.lower().replace("y", "").replace("m", "").replace("d", "")
            if len(ss) == 0:
                return True
        if len(Utils.parse_time_info(text)) > 0:
            return True
        return False

    @staticmethod
    def get_text_type(text):
        if Utils.is_number(text):
            return "number"
        if Utils.is_date(text):
            return "date"
        return "text"

    @staticmethod
    def get_distance(x1, y1, x2, y2):
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    @staticmethod
    def nlp(text, **kwargs):
        return Utils._get_instance()._nlp(text, **kwargs)

    @staticmethod
    def vec(text):
        text_processed = Utils.nlp(text.lower(), disable=["tokenizer", "parser", "ner"])
        return text_processed.vector

    @staticmethod
    def split_identifier(text):
        matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', text)
        words = []
        for m in matches:
            words.extend(re.split('[-_]', m.group(0)))
        return words


class Element:
    def __init__(self, ele_dict, state):
        self.ele_dict = ele_dict
        self.state = state
        self.id = ele_dict["WebBotID"]
        self.tag_name = ele_dict["tagName"]
        self.xpath = ele_dict["xpath"]
        self.xpath_short = ele_dict["xpathShort"]
        self.locator = ele_dict["locator"]

    @lazy_property
    def type(self):
        if "type" in self.ele_dict:
            return self.ele_dict["type"]
        else:
            return ""

    @lazy_property
    def in_window(self):
        bound = self.ele_dict["bound"]
        return bound["top"] < self.state.window_height and \
               bound["bottom"] > 0 and \
               bound["left"] < self.state.window_width and \
               bound["right"] > 0

    @lazy_property
    def center(self):
        bound = self.ele_dict["bound"]
        center_x = (bound["left"] + bound["right"]) / 2
        center_y = (bound["top"] + bound["bottom"]) / 2
        return center_x, center_y

    @lazy_property
    def position(self):
        bound = self.ele_dict["bound"]
        return bound["left"], bound["top"]

    @lazy_property
    def bound_ltrb(self):
        """
        return element bound coordinates, left, top, right, bottom
        :return:
        """
        bound = self.ele_dict["bound"]
        return bound["left"], bound["top"], bound["right"], bound["bottom"]

    @lazy_property
    def font_size(self):
        if "style" in self.ele_dict and "fontSize" in self.ele_dict["style"]:
            return float(self.ele_dict["style"]["fontSize"])
        return 0

    @lazy_property
    def font_weight(self):
        if "style" in self.ele_dict and "fontWeight" in self.ele_dict["style"]:
            return float(self.ele_dict["style"]["fontWeight"])
        return 0

    @lazy_property
    def has_background_image(self):
        if "style" in self.ele_dict and "hasBgImg" in self.ele_dict["style"]:
            return self.ele_dict["style"]["hasBgImg"]
        return False

    @lazy_property
    def has_border(self):
        if "style" in self.ele_dict and "hasBorder" in self.ele_dict["style"]:
            return self.ele_dict["style"]["hasBorder"]
        return False

    @lazy_property
    def text_color_rgb(self):
        if "style" in self.ele_dict and "color" in self.ele_dict["style"]:
            return Utils.parse_rgb(self.ele_dict["style"]["color"])
        return None

    @lazy_property
    def background_color_rgb(self):
        if "style" in self.ele_dict and "bgColor" in self.ele_dict["style"]:
            return Utils.parse_rgb(self.ele_dict["style"]["bgColor"])
        return None

    @lazy_property
    def is_clickable(self):
        if Action.CLICK in self.acceptable_action_types:
            return True
        if self.parent is None:
            return False
        return self.parent.is_clickable

    @lazy_property
    def dom_id(self):
        self_id = self.ele_dict["domId"] if "domId" in self.ele_dict else None
        if self_id:
            return self_id
        if self.parent is None:
            return ""
        return self.parent.dom_id

    @lazy_property
    def acceptable_action_types(self):
        acceptable_action_types = []
        action_set = self.ele_dict["actionSet"]
        if action_set:
            action_type = action_set["actionType"]
            if action_type in ["click"]:
                acceptable_action_types.append(Action.CLICK)
            elif action_type in ["check"]:
                acceptable_action_types.append(Action.CHECK)
            elif action_type in ["select"]:
                acceptable_action_types.append(Action.SELECT)
            elif action_type in ["setValue"]:
                acceptable_action_types.append(Action.INPUT_TEXT)
                acceptable_action_types.append(Action.PRESS_ENTER)
        return acceptable_action_types

    @lazy_property
    def get_webbot_xpath(self):
        return "//*[@webbotid='%s']" % self.id

    def get_acceptable_actions(self, task=None):
        # Generate possible actions
        acceptable_actions = []
        action_set = self.ele_dict["actionSet"]

        if action_set:
            action_type = action_set["actionType"]
            ele_type = self.ele_dict["type"] if "type" in self.ele_dict else None
            if action_type == "click":
                href = action_set["href"] if "href" in action_set else None
                href_excluded = href and href.startswith("mailto:")
                if ele_type != "reset" and not href_excluded:
                    action = Action(self, Action.CLICK, "")
                    acceptable_actions.append(action)
            elif action_type == "setValue":
                current_value = self.ele_dict["value"] if "value" in self.ele_dict else None
                if task is not None:
                    if current_value in task.query_words:
                        action = Action(self, Action.PRESS_ENTER, "")
                        acceptable_actions.append(action)
                    for i, word in enumerate(task.query_words):
                        annotation = task.query_annotations[i]
                        if annotation:
                            action = Action(self, Action.INPUT_TEXT, word)
                            acceptable_actions.append(action)
                # placeholder = self.ele_dict["placeholder"] if "placeholder" in self.ele_dict else None
                # if placeholder or current_value:
                #     click_action = Action(self, Action.CLICK, "")
                #     acceptable_actions.append(click_action)
            elif action_type == "check":
                checked = action_set["checked"] if "checked" in action_set else False
                if not checked:
                    action = Action(self, Action.CHECK, "")
                    acceptable_actions.append(action)
            elif action_type == "select":
                selected_indices = action_set["selectedIndices"] if "selectedIndices" in action_set else []
                options = action_set["options"] if "options" in action_set else []
                for i in range(len(options)):
                    # if i in selected_indices:
                    #     continue
                    action = Action(self, Action.SELECT, i)
                    acceptable_actions.append(action)
            else:
                raise RuntimeError("Failed to recognize: " + action_type)
        return acceptable_actions

    def accepts_text(self, text):
        if not text:
            return False
        if not Action.INPUT_TEXT in self.acceptable_action_types:
            return False
        ele_type = self.ele_dict["type"] if "type" in self.ele_dict else ""
        if ele_type in ["", "text", "search", "password"]:
            ele_value = self.ele_dict["value"] if "value" in self.ele_dict else ""
            ele_placeholder = self.ele_dict["placeholder"] if "placeholder" in self.ele_dict else ""
            if Utils.is_number(ele_value) or Utils.is_number(ele_placeholder):
                return Utils.is_number(text)
            return True
        elif ele_type in ["button", "checkbox", "file", "hidden", "image", "radio", "reset", "submit", "range"]:
            return False
        elif ele_type in ["number", "range"]:
            return Utils.is_number(text)
        elif ele_type == "date":
            if not re.match(r"\d{4}-\d{2}-\d{2}", text):
                return False
        elif ele_type == "month":
            if not re.match(r"\d{4}-\d{2}", text):
                return False
        elif ele_type == "time":
            if not re.match(r"\d{4}:\d{2}", text):
                return False
        elif ele_type == "week":
            if not re.match(r"\d{4}-W\d{2}", text):
                return False
        elif ele_type == "datetime-local":
            if not re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", text):
                return False
        elif ele_type == "email":
            if not re.match(r"[^@]+@.+", text):
                return False
        return True

    @lazy_property
    def parent(self):
        return self.state.get_parent(self)

    @lazy_property
    def parent_form(self):
        if self.parent is None or self.parent.tag_name == "FORM":
            return self.parent
        return self.parent.parent_form

    @lazy_property
    def form_submit_action(self):
        if self.tag_name != "FORM":
            return None
        for ele in self.all_child_elements:
            if ele.type == "submit":
                return Action(ele, Action.CLICK, "")
        return None

    @lazy_property
    def all_child_elements(self):
        all_child_elements = []
        for ele in self.child_elements:
            all_child_elements.append(ele)
            all_child_elements.extend(ele.all_child_elements)
        return all_child_elements

    @lazy_property
    def child_elements(self):
        return self.state.get_child_elements(self)

    @lazy_property
    def own_text_list(self):
        text_list = []
        if "text" in self.ele_dict and self.ele_dict["text"]:
            text_list.append(Utils.force_to_str(self.ele_dict["text"]).strip())
        if "value" in self.ele_dict and self.ele_dict["value"]:
            text_list.append(Utils.force_to_str(self.ele_dict["value"]).strip())
        if "placeholder" in self.ele_dict and self.ele_dict["placeholder"]:
            text_list.append(Utils.force_to_str(self.ele_dict["placeholder"]).strip())
        if "title" in self.ele_dict and self.ele_dict["title"]:
            text_list.append(Utils.force_to_str(self.ele_dict["title"]).strip())
        if "labelValue" in self.ele_dict and self.ele_dict["labelValue"]:
            text_list.append(Utils.force_to_str(self.ele_dict["labelValue"]).strip())
        if "labelText" in self.ele_dict and self.ele_dict["labelText"]:
            text_list.append(Utils.force_to_str(self.ele_dict["labelText"]).strip())
        if "ariaLabel" in self.ele_dict and self.ele_dict["ariaLabel"]:
            text_list.append(Utils.force_to_str(self.ele_dict["ariaLabel"]).strip())
        # text_list.extend(Utils.split_identifier(self.dom_id))
        return text_list

    @lazy_property
    def own_text_list_parsed(self):
        text_list = []
        for text in self.own_text_list:
            text_list.append(Utils.parse(text))
        return text_list

    @lazy_property
    def own_text(self):
        return self.own_text_list[0] if len(self.own_text_list) > 0 else ""

    @lazy_property
    def own_text_parsed(self):
        return self.own_text_list_parsed[0] if len(self.own_text_list_parsed) > 0 else ""

    @lazy_property
    def own_noninput_text(self):
        input_text = ""
        if Action.INPUT_TEXT in self.acceptable_action_types:
            if "value" in self.ele_dict and self.ele_dict["value"]:
                input_text = self.ele_dict["value"]
        own_text_list = copy.copy(self.own_text_list)
        if input_text in own_text_list:
            own_text_list.remove(input_text)
        return own_text_list[0] if len(own_text_list) > 0 else ""

    @lazy_property
    def own_noninput_text_parsed(self):
        return Utils.parse(self.own_noninput_text)

    @lazy_property
    def inner_text_list(self):
        text_list = list(self.own_text_list)
        if self.tag_name != "SELECT":
            # Ignore text in options, as the text of selected options is already in current element
            for child_ele in self.child_elements:
                text_list.extend(child_ele.inner_text_list)
        return text_list

    @lazy_property
    def inner_text_list_parsed(self):
        text_list = list(self.own_text_list_parsed)
        if self.tag_name != "SELECT":
            for child_ele in self.child_elements:
                text_list.extend(child_ele.inner_text_list_parsed)
        return text_list

    @lazy_property
    def inner_text(self):
        words = [self.own_text]
        if self.tag_name != "SELECT":
            for child_ele in self.child_elements:
                child_inner_text = child_ele.inner_text
                if child_inner_text:
                    words.append(child_inner_text)
        return " ".join(words)

    @lazy_property
    def inner_text_parsed(self):
        words = [self.own_text_parsed]
        if self.tag_name != "SELECT":
            for child_ele in self.child_elements:
                child_inner_text_words = child_ele.inner_text_parsed
                if child_inner_text_words:
                    words.append(child_inner_text_words)
        return " ".join(words)

    def get_neighbour_element(self):
        neighbour_element = None
        shortest_dist = 100
        for ele in self.state.unactionable_elements:
            dist = self.get_shortest_distance(ele)
            if dist < shortest_dist:
                shortest_dist = dist
                neighbour_element = ele
        return neighbour_element

    def max_match_score(self, text, is_input=False):
        """
        compute the maximum matching score between current element and a given text
        """
        scores = [0.0]
        if is_input and Action.INPUT_TEXT in self.acceptable_action_types:
            own_text_original = self.own_text.lower()
            if self.own_text_parsed.startswith(text) or f"({text})" in own_text_original:
                scores.append(1.0)
        word_list = self.inner_text_list_parsed if len(self.inner_text_list_parsed) < 4 else self.own_text_list_parsed
        if len(word_list) < 4:
            for word_parsed in word_list:
                score = Utils.text_similarity(text, word_parsed)
                scores.append(score)
        max_score = max(scores)
        return max_score

    def get_resized_bound(self, new_width, new_height):
        window_w, window_h = self.state.window_width, self.state.window_height
        resize_w, resize_h = float(new_width) / window_w, float(new_height) / window_h
        bound = self.ele_dict["bound"]
        return {
            "top": int(max(0, bound["top"]) * resize_h),
            "bottom": int(min(window_h, bound["bottom"]) * resize_h),
            "left": int(max(0, bound["left"]) * resize_w),
            "right": int(min(window_w, bound["right"]) * resize_w)
        }

    @lazy_property
    def center(self):
        bound = self.ele_dict["bound"]
        center_x = float(bound["left"] + bound["right"]) / 2
        center_y = float(bound["top"] + bound["bottom"]) / 2
        return center_x, center_y

    @lazy_property
    def path_to_root(self):
        path = [self]
        parent_id = self.ele_dict["parent"]
        if parent_id != -1:
            parent_ele = self.state.id2elements[parent_id]
            path.extend(parent_ele.path_to_root)
        return path

    @lazy_property
    def cluster_id(self):
        return re.sub(r"\d+", "?", self.xpath) + self.style_str

    def get_tree_distance(self, ele):
        path1 = self.path_to_root
        path2 = ele.path_to_root
        common_suffix_len = 0
        for i in range(min(len(path1), len(path2))):
            if path1[-i - 1].__str__() != path2[-i - 1].__str__():
                break
            common_suffix_len += 1
        dist = len(path1) + len(path2) - 2 * common_suffix_len
        return dist + 1

    def get_center_distance(self, ele):
        c1 = self.center
        c2 = ele.center
        dist = Utils.get_distance(c1[0], c1[1], c2[0], c2[1])
        return dist

    def get_shortest_distance(self, ele):
        l1, t1, r1, b1 = self.bound_ltrb
        l2, t2, r2, b2 = ele.bound_ltrb
        left = r2 < l1
        right = r1 < l2
        bottom = b2 < t1
        top = b1 < t2
        if top and left:
            return Utils.get_distance(l1, b1, r2, t2) * 2
        elif left and bottom:
            return Utils.get_distance(l1, t1, r2, b2) * 2
        elif bottom and right:
            return Utils.get_distance(r1, t1, l2, b2) * 2
        elif right and top:
            return Utils.get_distance(r1, b1, l2, t2) * 2
        elif left:
            return l1 - r2
        elif right:
            return l2 - r1
        elif bottom:
            return t1 - b2
        elif top:
            return t2 - b1
        else:  # rectangles intersect
            return self.get_center_distance(ele)

    @staticmethod
    def cluster_elements(elements):
        cluster_id_to_elements = {}
        for ele in elements:
            if ele.cluster_id not in cluster_id_to_elements:
                cluster_id_to_elements[ele.cluster_id] = []
            cluster_id_to_elements[ele.cluster_id].append(ele)
        return cluster_id_to_elements

    @lazy_property
    def style_str(self):
        return "FONT:size=%d;weight=%d;color=%s; BACKGROUND:img=%s;border=%s;color=%s;" % (
            self.font_size,
            self.font_weight,
            ",".join(self.text_color_rgb) if self.text_color_rgb else "null",
            "Y" if self.has_background_image else "N",
            "Y" if self.has_border else "N",
            ",".join(self.background_color_rgb) if self.background_color_rgb else "null",
        )

    def __str__(self):
        return "[%s text=\"%s\"]" % (self.tag_name, self.inner_text)


class State:
    def __init__(self, state_dict, screenshot=None):
        self.state_dict = state_dict
        self.screenshot = screenshot

        self.url = state_dict["url"]
        self.host = state_dict["host"]
        self.window_width = state_dict["windowWidth"]
        self.window_height = state_dict["windowHeight"]

        self.possible_actions = []
        self.task = None
        self.finish_action = None
        self.loaded_state_str = None

    def __str__(self):
        return self.url

    def snapshot(self):
        state_dict_copy = copy.deepcopy(self.state_dict)
        return State(state_dict=state_dict_copy, screenshot=self.screenshot)

    @lazy_property
    def elements(self):
        return [Element(state=self, ele_dict=ele_dict) for ele_dict in self.state_dict["elements"]]

    @lazy_property
    def elements_in_window(self):
        elements_in_window = []
        for ele in self.elements:
            if ele.in_window:
                elements_in_window.append(ele)
        return elements_in_window

    @lazy_property
    def unclickable_elements(self):
        elements = []
        for ele in self.elements_in_window:
            if not ele.is_clickable and len(ele.own_text) > 0:
                elements.append(ele)
        return elements

    @lazy_property
    def clickable_elements(self):
        elements = []
        for ele in self.elements_in_window:
            if ele.is_clickable:
                elements.append(ele)
        return elements

    @lazy_property
    def unactionable_elements(self):
        elements = []
        for ele in self.elements_in_window:
            if ele.acceptable_action_types:
                continue
            if not len(ele.own_text):
                continue
            if ele.is_clickable:
                continue
            elements.append(ele)
        return elements

    @lazy_property
    def actionable_elements(self):
        elements = []
        for ele in self.elements_in_window:
            if ele.acceptable_action_types:
                elements.append(ele)
            elif ele.tag_name == "OPTION":
                elements.append(ele)
        return elements

    @lazy_property
    def actionable_text_set(self):
        text_set = set()
        for ele in self.actionable_elements:
            text = ele.own_text_parsed
            # if Action.INPUT_TEXT in ele.acceptable_action_types:
            #     text = ele.own_noninput_text_parsed
            if len(text) == 0 or len(text) > 30:
                continue
            text_set.add(text)
        return text_set

    @lazy_property
    def content_elements(self):
        # return elements that contain content
        # leaf_elements = self.clickable_elements + self.unclickable_elements
        leaf_elements = self.clickable_elements + self.unactionable_elements

        content_elements = []
        clusters = Element.cluster_elements(leaf_elements)
        for cluster_id in clusters:
            cluster_elements = clusters[cluster_id]
            if cluster_elements[0].tag_name != "LABEL" and len(cluster_elements) > 4:
                continue
            content_elements.extend(cluster_elements)
        return content_elements

    @lazy_property
    def id2elements(self):
        return dict([(ele.id, ele) for ele in self.elements])

    @lazy_property
    def root_element(self):
        return self.elements[0]

    @lazy_property
    def text(self):
        return self.root_element.inner_text

    @lazy_property
    def text_in_window(self):
        text = ""
        for ele in self.elements_in_window:
            text += " " + ele.own_text
        return text

    @lazy_property
    def content_text(self):
        text = ""
        for ele in self.content_elements:
            text += " " + ele.own_text
        return text

    @lazy_property
    def image_hash(self):
        import imagehash
        return imagehash.dhash(self.screenshot, hash_size=10).__str__() if self.screenshot else "unknown_image_hash"

    @lazy_property
    def task_specific_hash(self):
        assert self.task is not None
        action_strs = []
        for action in self.possible_actions:
            action_strs.append(action.action_str)
        for text_re in self.task.query_words + self.task.target_text_res:
            action_strs.append("%s:%s" % (text_re, re.search(text_re, self.text_in_window, re.IGNORECASE) is None))
        action_strs = "\n".join(action_strs)
        m = Utils.md5(action_strs)
        return m

    @lazy_property
    def task_independent_hash(self):
        ele_strs = set()
        for ele in self.actionable_elements:
            if not ele:
                continue
            if len(ele.own_text) > 30:
                continue
            ele_action_types = ",".join(ele.acceptable_action_types)
            ele_str = "%s %s %s %s" % (ele_action_types, ele.xpath, ele.own_text, ele.style_str)
            ele_strs.add(ele_str)
        all_ele_str = "\n".join(sorted(ele_strs))
        m = Utils.md5(all_ele_str)
        return m

    @lazy_property
    def action_set_hash(self):
        action_strs = []
        for action in self.possible_actions:
            if not action.element:
                continue
            action_str = "%s %s %s" % (action.action_str, action.element, action.element.style_str)
            action_strs.append(action_str)
        action_strs = "\n".join(action_strs)
        m = Utils.md5(action_strs)
        return m

    @lazy_property
    def state_str(self):
        # return self.task_specific_hash[:10] + self.image_hash[:6]
        if self.loaded_state_str:
            return self.loaded_state_str
        return self.task_independent_hash

    def same_form_elements(self, element):
        ele_form = element.parent_form
        elements = []
        if ele_form:
            submit_elements = []
            for ele in ele_form.all_child_elements:
                if ele.acceptable_action_types:
                    if ele.type == "submit":
                        submit_elements.append(ele)
                    else:
                        elements.append(ele)
            elements.extend(submit_elements)
        else:
            for ele in self.actionable_elements:
                if Action.INPUT_TEXT in ele.acceptable_action_types or \
                        Action.SELECT in ele.acceptable_action_types:
                    elements.append(ele)
        return elements

    def setup(self, task):
        assert isinstance(task, Task)
        # Setup the state
        self.task = task

        for ele in self.elements:
            if not task.in_window_only or ele.in_window:
                self.possible_actions.extend(ele.get_acceptable_actions(task))
        self.finish_action = Action.finish()
        # self.possible_actions.append(self.finish_action)

    def same_as(self, state_):
        return self.state_str == state_.state_str

    @lazy_property
    def possible_action_strs(self):
        return set([action.action_str for action in self.possible_actions])

    def is_page_changed(self, last_state):
        # Whether this state is changed from last state
        return not self.possible_action_strs.issubset(last_state.possible_action_strs)

    def is_error_page(self):
        # Whether current state is an error page
        if len(self.possible_action_strs) == 0:
            return True
        error_messages = [
            "not found",
            "server error"
        ]
        for error_msg in error_messages:
            if re.search(error_msg, self.text_in_window, re.IGNORECASE):
                return True
        return False

    def get_parent(self, element):
        parent_id = element.ele_dict["parent"]
        return self.id2elements[parent_id] if parent_id != -1 else None

    def get_child_elements(self, element):
        child_eles = []
        for child_ele_id in element.ele_dict["children"]:
            child_ele = self.id2elements[child_ele_id]
            child_eles.append(child_ele)
        return child_eles

    def get_common_parent(self, elements):
        paths_to_root = []
        path_to_root = []
        for element in elements:
            path_to_root = [ele.id for ele in element.path_to_root]
            paths_to_root.append(path_to_root)
        for ele_id in path_to_root:
            is_common_parent = True
            for path in paths_to_root:
                if ele_id not in path:
                    is_common_parent = False
                    break
            if is_common_parent:
                return self.id2elements[ele_id]
        return None

    def contains_words(self, words, document=None):
        if document is None:
            document = self.text_in_window
        if not words:
            return True
        for word in words:
            if word not in document:
                return False
        return True

    def url_matches(self, url_res):
        if not url_res:
            return True
        for url_re in url_res:
            if not re.search(url_re, self.url):
                return False
        return True

    def _get_action_match_ratio_matrix(self, query_words):
        match_ratio_matrix = []
        for query_word in query_words:
            action_match_ratios = []
            for action in self.possible_actions:
                action_text = action.element.inner_text if action.action_type == Action.INPUT_TEXT else action.value
                action_match_ratios.append(difflib.SequenceMatcher(None, query_word, action_text).ratio())
            match_ratio_matrix.append(action_match_ratios)
        return np.array(match_ratio_matrix)

    def save(self, state_dir, replace=False, resize=None, file_name=None):
        if not os.path.exists(state_dir):
            os.makedirs(state_dir)
        if file_name is None:
            file_name = self.state_str
        state_json_path = os.path.join(state_dir, file_name + ".json")
        state_image_path = os.path.join(state_dir, file_name + ".png")
        if replace or (not os.path.exists(state_image_path)):
            if self.screenshot:
                screen = self.screenshot.resize(resize, Image.ANTIALIAS) if resize else self.screenshot
                screen.save(state_image_path)
        if replace or (not os.path.exists(state_json_path)):
            json.dump(self.state_dict, open(state_json_path, "w"), indent=1)

    def save_to_zip(self, zip_path, replace=False):
        if not zip_path:
            return False
        try:
            zip_file = ZipFile(zip_path, mode="a", compression=COMPRESS_METHOD)
            state_json_path = "states/" + self.state_str + ".json"
            state_image_path = "states/" + self.state_str + ".png"
            json_exists = True if state_json_path in zip_file.namelist() else False
            image_exists = True if state_image_path in zip_file.namelist() else False
            if replace or (not json_exists):
                state_json_str = json.dumps(self.state_dict, indent=1)
                zip_file.writestr(state_json_path, state_json_str)
            if replace or (not image_exists):
                screen = self.screenshot.resize([240, 240], Image.ANTIALIAS)
                image_file = BytesIO()
                screen.save(image_file, "PNG")
                # Here zf is a zipfile writer
                zip_file.writestr(state_image_path, image_file.getvalue())
            zip_file.close()
            return zip_file
        except:
            traceback.print_exc()
            return None

    @staticmethod
    def load(state_dir, state_str):
        if state_dir and state_str:
            if state_dir.endswith(".zip"):
                return State.load_from_zip(zip_path=state_dir, state_str=state_str)
            state_json_path = os.path.join(state_dir, state_str + ".json")
            state_image_path = os.path.join(state_dir, state_str + ".png")
            state_dict = json.load(open(state_json_path))
            state_image = Image.open(state_image_path)
            state = State(state_dict=state_dict, screenshot=state_image)
            state.loaded_state_str = state_str
            return state
        return None

    @staticmethod
    def load_from_zip(zip_path, state_str):
        try:
            zip_file = ZipFile(zip_path, mode="r")
            state_json_path = "states/" + state_str + ".json"
            state_image_path = "states/" + state_str + ".png"
            state_dict = json.load(zip_file.open(state_json_path))
            state_image = Image.open(zip_file.open(state_image_path))
            state = State(state_dict=state_dict, screenshot=state_image)
            state.loaded_state_str = state_str
            zip_file.close()
            return state
        except:
            # traceback.print_exc()
            pass
        try:
            zip_file = ZipFile(zip_path, mode="r")
            state_json_path = state_str + ".json"
            state_image_path = state_str + ".png"
            state_dict = json.load(zip_file.open(state_json_path))
            state_image = Image.open(zip_file.open(state_image_path))
            state = State(state_dict=state_dict, screenshot=state_image)
            state.loaded_state_str = state_str
            zip_file.close()
            return state
        except:
            # traceback.print_exc()
            return None

    def get_action(self, action_str):
        for action in self.possible_actions:
            if action.action_str == action_str:
                return action
        m = re.match(ACTION_RE, action_str)
        if m:
            action_type, value, target_locator = m.group(1), m.group(2), m.group(3)
            action_element = self.get_element_by_locator(target_locator)
            if action_element:
                return Action(element=action_element, action_type=action_type, value=value)
        return None

    def get_element_by_locator(self, locator):
        locator_xpaths = locator.split(" || ")
        matched_ele = None
        max_match_count = 0
        for ele in self.elements:
            ele_xpaths = ele.locator.split(" || ")
            ele_match_count = sum([(1 if locator_xpath in ele_xpaths else 0) for locator_xpath in locator_xpaths])
            if ele_match_count > max_match_count:
                max_match_count = ele_match_count
                matched_ele = ele
        return matched_ele

    def included_actions(self, actions):
        included_actions = []
        for action in actions:
            if not action.element:
                continue
            action_element = self.get_element_by_locator(action.element.xpath)
            if action_element:
                included_actions.append(action)
        return included_actions

    def interacted_elements(self, actions):
        elements_in_this_state = []
        elements_not_in_this_state = []

        for action in actions:
            if not action.element:
                continue
            action_element = self.get_element_by_locator(action.element.xpath)
            if action_element:
                if action_element not in elements_in_this_state:
                    elements_in_this_state.append(action_element)
            else:
                if action.element not in elements_not_in_this_state:
                    elements_not_in_this_state.append(action.element)
        return elements_in_this_state, elements_not_in_this_state


class Action:
    RESET = "reset"
    FINISH = "finish"

    CLICK = "click"
    CHECK = "check"
    SELECT = "select"
    INPUT_TEXT = "input_text"
    PRESS_ENTER = "press_enter"

    def __init__(self, element, action_type, value):
        self.element = element
        self.action_type = action_type
        self.value = value

    def to_dict(self):
        return {
            "action_type": self.action_type,
            "target": self.element.id,
            "value": self.value
        }

    @staticmethod
    def finish():
        return Action(None, Action.FINISH, "")

    @lazy_property
    def value_text(self):
        if not self.element:
            return ""
        if self.action_type == Action.SELECT:
            selected_index = int(self.value)
            if "actionSet" in self.element.ele_dict \
                    and "options" in self.element.ele_dict["actionSet"] \
                    and len(self.element.ele_dict["actionSet"]["options"]) > selected_index:
                return Utils.force_to_str(self.element.ele_dict["actionSet"]["options"][selected_index])
            else:
                return "ERROR"
        elif self.action_type == Action.INPUT_TEXT:
            return self.value
        elif self.action_type == Action.CLICK:
            return self.element.inner_text
        elif self.action_type == Action.CHECK:
            return self.element.inner_text
        else:
            return self.value

    @lazy_property
    def value_text_parsed(self):
        return Utils.parse(self.value_text)

    @lazy_property
    def surrounding_words(self):
        words = []
        # words.append(self.action_type)
        if self.element is not None:
            words.extend(self.element.inner_text_list)
            # neighbour_ele = self.element.get_neighbour_element()
            # if neighbour_ele:
            #     words.extend(neighbour_ele.own_text_list)
        if self.value_text in words:
            words.remove(self.value_text)
        return words

    @lazy_property
    def surrounding_words_parsed(self):
        words = []
        for word in self.surrounding_words:
            words.append(Utils.parse(word))
        return words

    @lazy_property
    def action_str(self):
        ele_xpath = self.element.xpath_short if self.element else ""
        return "%s #%s# @ %s" % (self.action_type, self.value, ele_xpath)

    @lazy_property
    def norm_action_str(self):
        if self.action_type in [Action.INPUT_TEXT, Action.SELECT]:
            return re.sub(r"#.*#", "#?#", self.action_str)
        else:
            return re.sub(r"\d+", "?", self.action_str)

    @lazy_property
    def is_submit(self):
        if self.action_type == Action.CLICK and self.element and self.element.type == "submit":
            return True
        if self.action_type == Action.PRESS_ENTER:
            return True
        return False

    @lazy_property
    def is_input(self):
        return self.action_type in [Action.INPUT_TEXT, Action.SELECT]

    @staticmethod
    def cluster_actions(actions):
        cluster_id_to_actions = {}
        for action in actions:
            if action.norm_action_str not in cluster_id_to_actions:
                cluster_id_to_actions[action.norm_action_str] = []
            cluster_id_to_actions[action.norm_action_str].append(action)
        return cluster_id_to_actions

    @lazy_property
    def replay_api(self):
        ele_locator = self.element.locator if self.element else ""
        return "%s #%s# @ %s" % (self.action_type, self.value, ele_locator)

    @lazy_property
    def unique_id(self):
        ele_xpath =  self.element.xpath if self.element else ""
        return "%s #%s# @ %s" % (self.action_type, self.value, ele_xpath)

    def match_score(self, action):
        score = Utils.text_similarity(self.value_text_parsed, action.value_text_parsed)
        if self.action_type != action.action_type:
            score *= 0.8
        return score

    def __str__(self):
        return "%s %s @ %s" % (self.action_type, self.value_text, self.element)


class UTG:
    """
    UI transition graph for web pages
    """

    def __init__(self, start_url="", states_dir=None, save_states=False, name="", **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.start_url = start_url
        self.states_dir = states_dir
        self.save_states = save_states
        self.zip_path = None
        self.utg_dir = None
        self.name = name

        self.G = nx.DiGraph()
        self.ineffective_action_strs = set()
        self.action_next_state_count = {}
        self.action_count = 0
        self.start_time = datetime.now()

    def _context_action_str(self, action, state):
        return state.state_str[:4] + ": " + action.action_str

    def add_transition(self, action, old_state, new_state):
        self.add_state(old_state)
        self.add_state(new_state)

        # make sure the states are not None
        if not old_state or not new_state:
            return

        action_str = self._context_action_str(action, old_state)
        self.action_count += 1
        self.add_action_target_state(action_str, new_state.state_str)
        self.G.nodes[old_state.state_str]["tried_action_strs"].append(action_str)

        if old_state.state_str == new_state.state_str:
            self.ineffective_action_strs.add(action_str)
        else:
            if (old_state.state_str, new_state.state_str) not in self.G.edges():
                self.G.add_edge(old_state.state_str, new_state.state_str, actions={})
            self.G[old_state.state_str][new_state.state_str]["actions"][action_str] = self.action_count

    def add_state(self, state):
        if state and (state.state_str not in self.G.nodes()):
            self.G.add_node(state.state_str,
                            url=state.url,
                            action_strs=[self._context_action_str(action, state) for action in state.possible_actions],
                            tried_action_strs=[])
            if self.save_states and self.utg_dir:
                states_dir = os.path.join(self.utg_dir, "states")
                state.save(states_dir)
            if self.save_states and self.zip_path:
                state.save_to_zip(self.zip_path)

    def get_state(self, state_str):
        if self.utg_dir:
            states_dir = os.path.join(self.utg_dir, "states")
            return State.load(states_dir, state_str)
        if self.zip_path:
            return State.load_from_zip(self.zip_path, state_str)

    def add_init_state(self, state):
        self.add_state(state)
        self.add_action_target_state(Action.RESET, state.state_str)

    def get_init_state_str(self):
        next_state_weights = self.get_next_state_weights()
        useless_state_strs = set()
        for state_str in next_state_weights:
            if not self.G[state_str]:
                useless_state_strs.add(state_str)
        for state_str in useless_state_strs:
            next_state_weights.pop(state_str)
        return Utils.weighted_choice(next_state_weights)

    def get_next_state_weights(self, action=None, state=None):
        """
        The next states with weights
        :param action:
        :return:
        """
        action_str = self._context_action_str(action, state) if action else Action.RESET
        return self.action_next_state_count[action_str] if action_str in self.action_next_state_count else None

    def is_ineffective(self, action, state):
        action_str = self._context_action_str(action, state) if action else Action.RESET
        return action_str in self.ineffective_action_strs

    def add_action_target_state(self, action_str, state_str):
        if action_str not in self.action_next_state_count:
            self.action_next_state_count[action_str] = {}
        if state_str in self.action_next_state_count[action_str]:
            self.action_next_state_count[action_str][state_str] += 1
        else:
            self.action_next_state_count[action_str][state_str] = 1

    def get_utg_dict(self):
        utg_nodes = []
        utg_edges = []
        for state_str in self.G.nodes():
            action_strs = self.G.nodes[state_str]["action_strs"]
            tried_action_strs = self.G.nodes[state_str]["tried_action_strs"]
            state_url = self.G.nodes[state_str]["url"]

            state_desc = UTG.list_to_html_table([
                ("url", state_url),
                ("state_str", state_str)
            ])

            utg_node = {
                "id": state_str,
                "label": state_url,
                "state_str": state_str,
                "action_strs": action_strs,
                "tried_action_strs": tried_action_strs,
                "ineffective_action_strs": list(self.ineffective_action_strs.intersection(action_strs)),
                "title": state_desc,
                "shape": "image",
                "image": "states/" + state_str + ".png"
            }

            if state_str in self.action_next_state_count[Action.RESET]:
                utg_node["font"] = "14px Arial red"
            utg_nodes.append(utg_node)

        for state_transition in self.G.edges():
            from_state = state_transition[0]
            to_state = state_transition[1]
            actions = self.G[from_state][to_state]["actions"]
            action_short_descs = []
            action_list = []

            for action_str, action_id in sorted(actions.items(), key=lambda x: x[1]):
                action_short_descs.append((action_id, action_str))
                action_list.append({
                    "action_str": action_str,
                    "action_id": action_id
                })

            utg_edge = {
                "from": from_state,
                "to": to_state,
                "id": from_state + "-->" + to_state,
                "title": UTG.list_to_html_table(action_short_descs),
                "label": ", ".join([str(x["action_id"]) for x in action_list]),
                "actions": action_list
            }

            utg_edges.append(utg_edge)

        utg = {
            "nodes": utg_nodes,
            "edges": utg_edges,

            "num_nodes": len(utg_nodes),
            "num_edges": len(utg_edges),
            "test_date": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "time_spent": (datetime.now() - self.start_time).total_seconds(),

            "start_url": self.start_url,

            "action_count": self.action_count,
            "action_next_state_count": self.action_next_state_count,
            "ineffective_action_strs": list(self.ineffective_action_strs)
        }
        return utg

    def save(self, utg_dir=None):
        """
        Output current UTG to a directory
        """
        if not utg_dir:
            utg_dir = self.utg_dir
        if not utg_dir:
            return
        if not os.path.exists(utg_dir):
            os.makedirs(utg_dir)

        utg_file_path = os.path.join(utg_dir, "utg.js")
        utg_file = open(utg_file_path, "w")
        utg_dict = self.get_utg_dict()
        utg_json = json.dumps(utg_dict, indent=1)
        utg_file.write("var utg = \n")
        utg_file.write(utg_json)
        utg_file.close()

        # Copy HTML/JS files
        utg_index_dst = os.path.join(utg_dir, "index.html")
        utg_stylesheets_dst = os.path.join(utg_dir, "stylesheets")
        if not os.path.exists(utg_index_dst):
            utg_index_src = os.path.join(".", "resources", "utg_visualization", "index.html")
            shutil.copyfile(utg_index_src, utg_index_dst)
            # utg_index_html = open(utg_index_src).read().replace("utg.js", self.name + "utg.js")
            # open(utg_index_dst, "w").write(utg_index_html)
        if not os.path.exists(utg_stylesheets_dst):
            utg_stylesheets_src = os.path.join(".", "resources", "utg_visualization", "stylesheets")
            shutil.copytree(utg_stylesheets_src, utg_stylesheets_dst)

    def save_to_zip(self):
        """
        output current UTG to a zip file
        :return:
        """
        if not self.zip_path:
            return None
        try:
            utg_dict = self.get_utg_dict()
            utg_json_str = "var utg = \n" + json.dumps(utg_dict, indent=1)
            js_file_path = self.zip_path[:-4] + ".js"
            with open(js_file_path, "w") as js_file:
                js_file.write(utg_json_str)
            zip_file = ZipFile(self.zip_path, mode="a", compression=COMPRESS_METHOD)
            zip_file.writestr("utg.js", utg_json_str)
            zip_file.close()
            return zip_file
        except:
            return

    @staticmethod
    def load_from_zip(zip_path):
        if not zip_path:
            return None
        utg_name = os.path.basename(zip_path)[:-len("utg.zip")]
        try:
            zip_file = ZipFile(zip_path, mode="r", compression=COMPRESS_METHOD)
            utg_lines = zip_file.open("utg.js").readlines()[1:]
            utg_body = "\n".join([line.decode() for line in utg_lines])
            utg_dict = json.loads(utg_body)
            utg = UTG.create_utg_from_dict(utg_name, utg_dict)
            utg.zip_path = zip_path
            return utg
        except Exception as e:
            print(e)
            print("No UTG found in %s" % zip_path)
            utg = UTG(name=utg_name)
            utg.zip_path = zip_path
            return utg

    @staticmethod
    def load_from_dir(utg_dir_path):
        if not utg_dir_path:
            return None
        utg_dir_path = os.path.dirname(os.path.join(utg_dir_path, "utg.js"))
        utg_name = os.path.basename(utg_dir_path)[:-len("utg")]
        try:
            utg_js_path = os.path.join(utg_dir_path, "utg.js")
            utg_file = open(utg_js_path)
            utg_body = "".join(utg_file.readlines()[1:])
            utg_file.close()
            utg_dict = json.loads(utg_body)
            utg = UTG.create_utg_from_dict(utg_name, utg_dict)
            utg.utg_dir = utg_dir_path
            return utg
        except Exception as e:
            print(e)
            print("No UTG found in %s" % utg_dir_path)
            utg = UTG(name=utg_name)
            utg.utg_dir = utg_dir_path
            return utg

    @staticmethod
    def load_utgs_from_dir(utgs_dir):
        utgs = []
        if not utgs_dir:
            return utgs
        for root, dirs, files in os.walk(utgs_dir):
            for dir in dirs:
                if dir.endswith("_utg"):
                    dir_path = os.path.join(root, dir)
                    utg = UTG.load_from_dir(dir_path)
                    utgs.append(utg)
            for f in files:
                if f.endswith("_utg.zip"):
                    file_path = os.path.join(root, f)
                    utg = UTG.load_from_zip(file_path)
                    utgs.append(utg)
        return utgs

    @staticmethod
    def create_utg_from_dict(utg_name, utg_dict):
        utg = UTG(name=utg_name, **utg_dict)

        for node in utg_dict["nodes"]:
            state_str = node["state_str"]
            state_url = node["label"]
            action_strs = node["action_strs"]
            tried_action_strs = node["tried_action_strs"]
            utg.G.add_node(state_str, url=state_url, action_strs=action_strs, tried_action_strs=tried_action_strs)

        for edge in utg_dict["edges"]:
            old_state_str = edge["from"]
            new_state_str = edge["to"]
            for action_dict in edge["actions"]:
                action_str = action_dict["action_str"]
                action_id = action_dict["action_id"]
                if (old_state_str, new_state_str) not in utg.G.edges():
                    utg.G.add_edge(old_state_str, new_state_str, actions={})
                utg.G[old_state_str][new_state_str]["actions"][action_str] = action_id

        utg.action_next_state_count = utg_dict["action_next_state_count"]
        utg.ineffective_action_strs.update(utg_dict["ineffective_action_strs"])
        utg.action_count = utg_dict["action_count"]
        return utg

    def get_action_coverages(self, state_or_str, n_steps):
        if not state_or_str or n_steps <= 0:
            return {}
        if isinstance(state_or_str, State):
            return {action: self.get_action_coverage(self._context_action_str(action, state_or_str), n_steps)
                    for action in state_or_str.possible_actions}
        else:
            action_strs = self.G.nodes[state_or_str]["action_strs"] if state_or_str in self.G.nodes else []
            return {action_str: self.get_action_coverage(action_str, n_steps)
                    for action_str in action_strs}

    def get_state_coverage(self, state_str, n_steps):
        action_coverages = self.get_action_coverages(state_str, n_steps)
        if not action_coverages:
            return 1.0, 1.0
        covered = total = 0.0
        for action_covered, action_total in action_coverages.values():
            covered += action_covered
            total += action_total
        return covered, total

    def get_action_coverage(self, action_str, n_steps):
        """
        get number of covered paths and total paths of given action in given number of steps
        :param action_str:
        :param n_steps:
        :return: #covered, #total
        """
        if action_str not in self.action_next_state_count:
            return 0.0, 10.0  # 10 is the estimated number of actions per state
        next_state_count = self.action_next_state_count[action_str]
        next_state_count_sum = sum(next_state_count.values())
        covered = total = 0.0
        for state_str in next_state_count:
            state_weight = float(next_state_count[state_str]) / next_state_count_sum
            state_covered, state_total = self.get_state_coverage(state_str, n_steps - 1)
            covered += state_weight * state_covered
            total += state_weight * state_total
        return covered, total

    @staticmethod
    def list_to_html_table(dict_data):
        table = "<table class=\"table\">\n"
        for (key, value) in dict_data:
            table += "<tr><th>%s</th><td>%s</td></tr>\n" % (key, value)
        table += "</table>"
        return table


class Task:
    def __init__(self, start_url, query_words=None, query_annotations=None, in_window_only=True, included_url_res=None,
                 target_url_res=None, target_text_res=None, target_state_str=None, necessary_actions=None,
                 demonstration=None, states_dir=None, name="", replayable=False, target_url=None, **kwargs):
        self.name = name
        self.logger = logging.getLogger("Task(%s)" % self.name)

        self.query_words = query_words if query_words else []
        self.query_annotations = query_annotations if query_annotations else [""] * len(query_words)
        if len(self.query_words) != len(self.query_annotations):
            raise RuntimeError("The query length doesn't match the query annotation length: " + name)
        self.use_annotations = GLOBAL_CONFIGS["use_annotations"]

        self.start_url = start_url
        self.step_limit = len(self.query_words_parsed) + 3
        self.in_window_only = in_window_only
        self.included_url_res = included_url_res if included_url_res else []
        # self.included_url_res.append(Utils.get_host(start_url))
        self.target_url_res = target_url_res if target_url_res else []
        self.target_text_res = target_text_res if target_text_res else []
        self.target_state_str = target_state_str
        self.target_url = target_url
        self.necessary_actions = necessary_actions
        self.demonstration = demonstration
        self.demo_tasks = None
        self.demo_task = None
        self.states_dir = states_dir
        self.task_str = "%s in %s" % (" ".join(self.query_words), self.start_url)
        self.replayable = replayable
        self.utg = UTG(start_url=start_url, states_dir=self.states_dir, name=name)

        self.__reset_progress()

    def __reset_progress(self):
        self.step = 0
        self.action_history = []
        self.state_history = []
        self.state = None
        self.reward = 0
        self.total_reward = 0
        self.reward_history = []
        self.scores = [0.0] * len(self.score_items)
        self.done = False
        self.target_achieved = False

    @lazy_property
    def query_words_parsed(self):
        words = [Utils.parse(word) for word in self.query_words]
        return words

    @lazy_property
    def query_annotations_parsed(self):
        if not self.use_annotations:
            return [""] * len(self.query_annotations)
        words = [Utils.parse(word) for word in self.query_annotations]
        return words

    @lazy_property
    def all_words_parsed(self):
        words = set(self.query_words_parsed + self.query_annotations_parsed)
        for surrounding_word_list in self.surrounding_words_parsed:
            for word in surrounding_word_list:
                words.add(word)
        if "" in words:
            words.remove("")
        return words

    @lazy_property
    def parameter_ids(self):
        word_ids = []
        for i, word in enumerate(self.query_words_parsed):
            word_annotation = self.query_annotations[i]
            if word_annotation:
                word_ids.append(i)
        return word_ids

    @lazy_property
    def parameter_annotations_parsed(self):
        words = []
        for i in self.parameter_ids:
            words.append(self.surrounding_words_parsed[i])
        return words

    @lazy_property
    def parameter_values(self):
        words = []
        for i in self.parameter_ids:
            words.append(self.query_words[i])
        return words

    @lazy_property
    def parameter_values_parsed(self):
        words = []
        for i in self.parameter_ids:
            words.append(self.query_words_parsed[i])
        return words

    def get_action_parameter_index(self, action):
        # get the index of the action word in the task query
        if action.action_type in [Action.INPUT_TEXT, Action.SELECT]:
            action_value = action.value_text_parsed
            max_sim_idx = self.get_parameter_index(parameter=action_value)
            if max_sim_idx > -1:
                return max_sim_idx
        return -1

    def get_parameter_index(self, parameter):
        max_sim = 0.5
        max_sim_idx = -1
        for i in self.parameter_ids:
            entity_word_parsed = self.query_words_parsed[i]
            similarity = Utils.text_similarity(entity_word_parsed, parameter)
            if similarity > max_sim:
                max_sim = similarity
                max_sim_idx = i
        return max_sim_idx

    def get_parameter_surrounding_word_parsed(self, parameter):
        para_idx = self.get_parameter_index(parameter)
        if para_idx > -1:
            surrounding_words_parsed = self.surrounding_words_parsed[para_idx]
            if isinstance(surrounding_words_parsed, list):
                if len(surrounding_words_parsed) > 0:
                    return surrounding_words_parsed[0]
            else:
                return surrounding_words_parsed
        return ""

    @lazy_property
    def surrounding_words_parsed(self):
        surrounding_words_list = []
        for i, word in enumerate(self.query_words_parsed):
            # surrounding_words_i = set()
            # if i - 1 >= 0:
            #     prefix_word = self.query_words_parsed[i - 1]
            #     surrounding_words_i.add(prefix_word)
            # if i + 1 < len(self.query_words_parsed):
            #     suffix_word = self.query_words_parsed[i + 1]
            #     if Utils.is_number(word) and (suffix_word not in EXCLUDE_QUERY_WORDS):
            #         surrounding_words_i.add(suffix_word)
            # word_annotation = self.query_annotations_parsed[i]
            # if word_annotation:
            #     surrounding_words_i.add(word_annotation)

            # Heuristics added on 10/17/2019, merge surrounding words to one word
            surrounding_words_i = []
            if i - 1 >= 0:
                prefix_word = self.query_words_parsed[i - 1]
                surrounding_words_i.append(prefix_word)
            word_annotation = self.query_annotations_parsed[i]
            if word_annotation and word_annotation not in surrounding_words_i:
                surrounding_words_i.append(word_annotation)
            if i + 1 < len(self.query_words_parsed):
                suffix_word = self.query_words_parsed[i + 1]
                if Utils.is_number(word) and \
                        (suffix_word not in EXCLUDE_QUERY_WORDS) and \
                        (suffix_word not in surrounding_words_i):
                    surrounding_words_i.append(suffix_word)
            # surrounding_words_i = [" ".join(surrounding_words_i)]
            surrounding_words_i = [word for word in surrounding_words_i if word not in EXCLUDE_QUERY_WORDS]
            surrounding_words_list.append(surrounding_words_i)
        return surrounding_words_list

    @lazy_property
    def non_parameters_parsed(self):
        words = set()
        for i, word in enumerate(self.query_words_parsed):
            if word in EXCLUDE_QUERY_WORDS:
                continue
            if self.query_annotations[i]:
                words.add(self.query_annotations_parsed[i])
            else:
                words.add(word)
        if "" in words:
            words.remove("")
        return sorted(words)

    def snapshot(self, step=None):
        task_snapshot = copy.copy(self)
        if step is None or step == self.step:
            task_snapshot.action_history = copy.copy(self.action_history)
            task_snapshot.state_history = copy.copy(self.state_history)
            task_snapshot.reward_history = copy.deepcopy(self.reward_history)
            task_snapshot.scores = copy.copy(self.scores)
        elif 0 <= step < self.step:
            task_snapshot.__reset_progress()
            task_snapshot.step = step
            task_snapshot.state = self.state_history[step]
            task_snapshot.state_history = copy.copy(self.state_history[:step])
            task_snapshot.action_history = copy.copy(self.action_history[:step])
            task_snapshot.reward_history = copy.deepcopy(self.reward_history[:step])
            if len(task_snapshot.reward_history) > 0:
                task_snapshot.reward, task_snapshot.total_reward, task_snapshot.scores = task_snapshot.reward_history[-1]
        else:
            self.logger.warning("Task.snapshot failed with step: " + step)
            task_snapshot = None
        return task_snapshot

    def get_coverage(self):
        return self.utg.get_action_coverage(Action.RESET, self.step_limit + 1)

    def reset(self, state, update_utg=True):
        assert isinstance(state, State)
        self.__reset_progress()
        if update_utg:
            self.utg.add_init_state(state)
        self.state = state
        self._evaluate()

    def update(self, action, new_state, update_utg=True):
        assert isinstance(action, Action)
        if update_utg:
            self.utg.add_transition(action=action, old_state=self.state, new_state=new_state)
        self.action_history.append(action)
        self.state_history.append(self.state)
        if new_state:
            # Ensure self.state not being None
            self.state = new_state
        self._evaluate()
        self.reward_history.append([self.reward, self.total_reward, self.scores])
        self.step += 1
        if new_state is None:
            self.done = True
        return self.reward, self.done

    def get_action_coverages(self):
        step_limit = self.step_limit - len(self.action_history)
        return self.utg.get_action_coverages(self.state, step_limit)

    def get_input_probability(self, input_text):
        if self.state and self.state.actionable_text_set:
            input_text = input_text.lower()
            similarity = max([Utils.text_similarity(input_text, text)
                              for text in self.state.actionable_text_set])
            return 1 - similarity
        else:
            return 1

    @lazy_property
    def score_items(self):
        items = [
            ["step_count"],
            ["action_spatial_distance"],
            ["action_direction"]
            # ["similarity_with_demo_actions"],
            # ["similarity_with_demo_surrounding"]
        ]
        items.append(["similarity_with_non_parameters", self.non_parameters_parsed])
        for i in self.parameter_ids:
            para_val = self.query_words_parsed[i]
            para_anno = self.surrounding_words_parsed[i]
            items.append(["similarity_with_parameters", para_val, para_anno])
        # for i in range(len(entity_words) - 1):
        #     entity_word_i = entity_words[i]
        #     for j in range(i + 1, len(entity_words)):
        #         entity_word_j = entity_words[j]
        #         items.append(["distance_between_parameters", entity_word_i, entity_word_j])
        return items

    @property
    def score_weights(self):
        weights = []
        for reward_item in self.score_items:
            item = reward_item[0]
            if item in REWARD_ITEM_WEIGHTS:
                weights.append(REWARD_ITEM_WEIGHTS[item])
            else:
                weights.append(0)
        return weights

    def query_achieved_scores(self):
        sim_non_para = 0.0
        sim_paras = []
        for i, reward_item in enumerate(self.score_items):
            reward_key = reward_item[0]
            if reward_key in ["similarity_with_non_parameters"]:
                sim_non_para = self.scores[i]
            elif reward_key in ["similarity_with_parameters"]:
                sim_paras.append(self.scores[i])
        return sim_non_para, sim_paras

    def compute_scores(self):
        scores = []

        # elements to compute reward with
        inside_elements, outside_elements = self.state.interacted_elements(self.action_history)
        previous_clicked_elements = []
        previous_input_values = []
        for action in self.action_history:
            if action.element and action.action_type == Action.CLICK and action.element in outside_elements:
                previous_clicked_elements.append(action.element)
            if action.action_type == Action.INPUT_TEXT:
                previous_input_values.append(action.value_text_parsed)
        content_elements = self.state.content_elements
        reward_elements = list(set(inside_elements + content_elements + previous_clicked_elements))

        query_word_matches = {}
        for i, word in enumerate(self.all_words_parsed):
            word_matches = []
            is_input = True if word in previous_input_values else False
            for ele in reward_elements:
                score = ele.max_match_score(word, is_input=is_input)
                # importance of 4 types of elements: input_text < content < clicked < select
                if score > 0.5:
                    if is_input:
                        score *= 0.7  # penalize input value
                    if ele in previous_clicked_elements:
                        score *= 1.3
                    elif Action.SELECT in ele.acceptable_action_types:
                        score *= 1.6
                    word_matches.append((ele, score))
            query_word_matches[word] = word_matches

        for reward_item in self.score_items:
            reward_key = reward_item[0]
            if reward_key in ["step_count"]:
                step_count = 0
                for action in self.action_history:
                    # minimize the number of steps, but encourage submit actions
                    step_count += (0 if action.is_submit else 1)
                scores.append(step_count)
            elif reward_key in ["action_spatial_distance"]:
                action_spatial_distance = 0
                if len(self.action_history) > 1:
                    distances = []
                    for i in range(len(self.action_history) - 1):
                        ele_i = self.action_history[i].element
                        ele_j = self.action_history[i + 1].element
                        state_j = self.state_history[i + 1]
                        if ele_i and ele_j and state_j and state_j.get_element_by_locator(ele_i.xpath):
                            # if the two actions are in the same page
                            distance = ele_i.get_shortest_distance(ele_j)
                            if distance > DISTANCE_THRESHOLD:
                                distances.append(1.0)
                    if distances:
                        action_spatial_distance = np.sum(distances)
                scores.append(action_spatial_distance)
            elif reward_key in ["action_direction"]:
                action_direction = 0
                if len(self.action_history) > 1:
                    for i in range(len(self.action_history) - 1):
                        action_i = self.action_history[i]
                        ele_i = action_i.element
                        if not ele_i:
                            continue
                        word_idx_i = self.get_action_parameter_index(action_i)
                        for j in range(i + 1, len(self.action_history)):
                            action_j = self.action_history[j]
                            ele_j = action_j.element
                            state_j = self.state_history[j]
                            if not ele_j:
                                continue
                            if not state_j.get_element_by_locator(ele_i.xpath):
                                continue
                            word_idx_j = self.get_action_parameter_index(action_j)
                            if 0 <= word_idx_j < word_idx_i:
                                action_direction += 1.0
                            # if ele_i.position != ele_j.position \
                            #         and ele_i.position[0] >= ele_j.position[0] \
                            #         and ele_i.position[1] >= ele_j.position[1]:
                            #     action_direction += 1.0
                scores.append(action_direction)
            elif reward_key in ["similarity_with_demo_actions"]:
                if self.demo_task is None \
                        or len(self.demo_task.action_history) == 0 \
                        or len(self.action_history) == 0:
                    scores.append(0)
                    continue
                match_scores = []
                for demo_action in self.demo_task.action_history:
                    action_match_score = max([demo_action.match_score(action) for action in self.action_history])
                    match_scores.append(action_match_score)
                for self_action in self.action_history:
                    action_match_score = max([self_action.match_score(action) for action in self.action_history])
                    match_scores.append(action_match_score)
                task_len = len(self.demo_task.action_history) + len(self.action_history)
                similarity_with_demo_actions = sum(match_scores) / task_len
                scores.append(similarity_with_demo_actions)
            elif reward_key in ["similarity_with_demo_surrounding"]:
                if not self.demo_task:
                    scores.append(0)
                    continue
                demo_surrounding = self.demo_task.get_entity_surrounding()
                self_surrounding = self.get_entity_surrounding()
                if not demo_surrounding or not self_surrounding:
                    scores.append(0)
                    continue
                match_scores = []
                for demo_entity in demo_surrounding:
                    match_score = max([
                        Utils.text_similarity(demo_entity, self_entity) *
                        Utils.words_similarity(demo_surrounding[demo_entity], self_surrounding[self_entity])
                        for self_entity in self_surrounding])
                    match_scores.append(match_score)
                for self_entity in self_surrounding:
                    match_score = max([
                        Utils.text_similarity(demo_entity, self_entity) *
                        Utils.words_similarity(demo_surrounding[demo_entity], self_surrounding[self_entity])
                        for demo_entity in demo_surrounding])
                    match_scores.append(match_score)
                task_len = len(self.demo_task.action_history) + len(self.action_history)
                similarity_with_demo_surrounding = sum(match_scores) / task_len
                scores.append(similarity_with_demo_surrounding)
            elif reward_key in ["similarity_with_non_parameters"]:
                non_entity_words = reward_item[1]
                match_scores = []
                for word in non_entity_words:
                    similarity = max([0.0] + [score for ele, score in query_word_matches[word] if ele in content_elements])
                    match_scores.append(similarity)
                if not match_scores:
                    match_scores = [0.0]
                similarity_with_non_parameters = np.mean(match_scores)
                scores.append(similarity_with_non_parameters)
            elif reward_key in ["similarity_with_parameters"]:
                para_val = reward_item[1]
                para_annotations = reward_item[2]
                para_scores = [0.0]
                for v1_ele, v1_score in query_word_matches[para_val]:
                    para_scores.append(v1_score)
                    for annotation in para_annotations:
                        for v2_ele, v2_score in query_word_matches[annotation]:
                            dist = 1 if v1_ele == v2_ele else (1 + v1_ele.get_shortest_distance(v2_ele) / 50)
                            para_score = v1_score + v1_score / dist
                            para_scores.append(para_score)
                similarity_with_parameter = max(para_scores)
                scores.append(similarity_with_parameter)
            elif reward_key in ["distance_between_parameters"]:
                entity1 = reward_item[1]
                entity2 = reward_item[2]
                dists = [0.0]
                for v1_ele, v1_score in query_word_matches[entity1]:
                    for v2_ele, v2_score in query_word_matches[entity2]:
                        if v1_ele == v2_ele:
                            continue
                        dist = v1_score * v2_score / (2 + v1_ele.get_shortest_distance(v2_ele) / 50)
                        if v2_ele.center[0] >= v1_ele.center[0] and v2_ele.center[1] >= v1_ele.center[1]:
                            dist *= 1.2
                        dists.append(dist)
                distance_between_parameters = max(dists)
                scores.append(distance_between_parameters)
            else:
                scores.append(0)
        return scores

    def _evaluate(self):
        self.reward, self.done, self.target_achieved = 0, False, False
        last_state = self.state_history[-1] if len(self.action_history) > 0 else None
        last_action = self.action_history[-1] if len(self.action_history) > 0 else None

        if not self.state or not self.state.url_matches(self.included_url_res):
            self.done = True
            self.reward = 0 - self.total_reward
            self.total_reward = 0
            return

        # If last action is not a submit action, clear all accumulated scores
        if last_state and last_action \
                and (not last_action.is_submit):
                # remove below conditions because they will lead to "submit-clear confusion"
                # and (not last_action.action_type == Action.SELECT) \
                # and (last_state.url not in self.state.url):
            self.scores = [0.0] * len(self.score_items)

        current_scores = self.compute_scores()
        self.scores = np.max([self.scores, current_scores], axis=0)
        total_reward = np.sum(np.multiply(self.score_weights, self.scores))

        self.reward = total_reward - self.total_reward
        self.total_reward = total_reward

        n_matches, n_max_matches = self._target_match_ratio()
        if n_matches == n_max_matches and n_max_matches > 0:
            self.target_achieved = True

        if len(self.action_history) >= self.step_limit \
                or (last_action and last_action.action_type == Action.FINISH) \
                or len(self.state.possible_actions) == 0:
            self.done = True
        return

    def get_entity_surrounding(self):
        entity_surrounding = {}
        for action in self.action_history:
            if action.action_type in [Action.INPUT_TEXT, Action.SELECT]:
                entity = action.value_text_parsed
                if len(entity) > 50:
                    continue
                surrounding_words = action.surrounding_words_parsed
                entity_surrounding[entity] = surrounding_words
        return entity_surrounding

    def get_action_usefulness(self, action):
        """
        Evaluate the action's usefulness to this task
        :param action: the action to evaluate
        :return: the match score between the action and this task
        """
        if action.element is None:
            return 0.0

        scores = [0.01]
        for i, word in enumerate(self.all_words_parsed):
            similarity = Utils.text_similarity(word, action.value_text_parsed)
            scores.append(similarity)
        return max(scores)

    def get_preferred_actions(self):
        """
        get the list of preferred actions in the context of current task
        :return:
        """
        restrict_action_space = GLOBAL_CONFIGS['action_restriction']
        w_spatial = REWARD_ITEM_WEIGHTS['action_spatial_distance']
        w_directional = REWARD_ITEM_WEIGHTS['action_direction']

        preferred_actions = []
        visited_state_strs = [state.state_str for state in self.state_history] + [self.state.state_str]
        interacted_action_clusters = set([action.norm_action_str for action in self.action_history])

        inputted_words = []
        pending_input_words = copy.copy(self.query_words)
        for action in self.action_history:
            if action.action_type in [Action.INPUT_TEXT, Action.SELECT]:
                inputted_word = action.value
                inputted_words.append(inputted_word)
                if inputted_word in pending_input_words:
                    pending_input_words.remove(inputted_word)

        interacted_elements, _ = self.state.interacted_elements(self.action_history)

        last_state = self.state_history[-1] if len(self.state_history) > 0 else None
        last_action = self.action_history[-1] if len(self.action_history) > 0 else None

        for action in self.state.possible_actions:
            # TODO: If run on cache, comment following
            if action.norm_action_str in interacted_action_clusters:
                continue

            # if self.utg.is_ineffective(action, self.state):
            #     continue
            #
            # action_ineffective = False
            # next_state_weights = self.utg.get_next_state_weights(action, self.state)
            # if next_state_weights \
            #         and len(next_state_weights) == 1 \
            #         and list(next_state_weights.keys())[0] in visited_state_strs:
            #     total_weight = sum(next_state_weights.values())
            #     max_weight = max(next_state_weights.values())
            #     if float(max_weight) / total_weight > 0.8:
            #         for state_str in next_state_weights:
            #             if next_state_weights[state_str] >= max_weight and state_str in visited_state_strs:
            #                 action_ineffective = True
            #                 break
            # if action_ineffective:
            #     continue

            # Constrain to the same input type
            if action.action_type == Action.INPUT_TEXT:
                current_value = action.element.ele_dict["value"] if "value" in action.element.ele_dict else None
                if action.value == current_value:
                    continue
                if self.get_input_probability(action.value) < 0.8:
                    continue
                if not action.element.accepts_text(action.value):
                    continue
                input_value_type = Utils.get_text_type(action.value)
                if current_value:
                    if current_value in self.query_words:
                        continue
                    current_value_type = Utils.get_text_type(current_value)
                    if input_value_type != current_value_type:
                        continue
                # else:
                #     if "date" in action.element.own_text and input_value_type != "time":
                #         continue

            # Each parameter should be inputted only once.
            # Heuristics added 10/15/2019
            if action.action_type in [Action.INPUT_TEXT, Action.SELECT]:
                if action.value in inputted_words:
                    continue

            # Restrict to the same form
            if w_spatial != 0:  # OR suggested heuristics: if w_spatial is 0, don't restrict action space.
                if last_action is not None \
                        and last_action.element is not None \
                        and last_action.element.parent_form is not None \
                        and not last_action.is_submit:
                    # last action is in a form
                    last_form = last_action.element.parent_form
                    current_form = action.element.parent_form
                    if current_form is None:
                        continue
                    if current_form.xpath != last_form.xpath:
                        continue
                    # if action.action_type == Action.CLICK and not action.is_submit:
                    #     continue
                # make sure press_enter action is taken right after input_text on the same element
                if action.action_type == Action.PRESS_ENTER:
                    if last_action is None or last_action.element is None:
                        continue
                    if action.element.xpath != last_action.element.xpath:
                        continue

            if restrict_action_space and w_directional != 0:
                # If this is not a new state, the next action should be on the right side of or below previous actions
                is_excluded = False
                for interacted_element in interacted_elements:
                    previous_x, previous_y = interacted_element.center
                    x, y = action.element.center
                    if x == previous_x and y == previous_y:
                        continue
                    if x <= previous_x and y <= previous_y:
                        is_excluded = True
                        break
                if is_excluded:
                    continue

            preferred_actions.append(action)
        return preferred_actions

    def get_reward_str(self):
        return "reward:%.2f total:%.2f scores:%s" % \
               (self.reward, self.total_reward, ",".join(["%.2f" % s for s in self.scores]))

    def get_tasklet(self):
        pretty_trace = []
        replay_trace = []
        for i, action in enumerate(self.action_history):
            reward, total_reward, scores = self.reward_history[i]
            reward_str = "reward:%.2f total:%.2f scores:%s" % \
                         (reward, total_reward, ",".join(["%.2f" % s for s in scores]))
            pretty_trace.append("%s, %s" % (action, reward_str))
            replay_trace.append(action.replay_api)
        success = self.target_achieved
        report = "task:%s\n total_reward:%.2f success:%s\n" \
                 " score_items:\n\t%s\n pretty_trace:\n\t%s\n replay_trace:\n\t%s\n" % \
                 (self.task_str,
                  self.total_reward,
                  "Y" if success else "N",
                  "\n\t".join(
                      ["%d. %.1f * %s" % (i, self.score_weights[i], item) for i, item in enumerate(self.score_items)]),
                  "\n\t".join(pretty_trace),
                  "\n\t".join(replay_trace))
        return report

    @staticmethod
    def get_replay_trace_from_tasklet(tasklet):
        replay_trace_offset = tasklet.find("replay_trace:") + 13
        replay_trace_str = tasklet[replay_trace_offset:]
        replay_trace = []
        for line in replay_trace_str.splitlines():
            line = line.strip()
            if len(line) == 0:
                continue
            replay_trace.append(line)
        return replay_trace

    @staticmethod
    def get_total_reward_from_tasklet(tasklet):
        reward_line_re = r"total_reward:(\S+) success:"
        m = re.search(reward_line_re, tasklet)
        if m:
            reward_str = m.group(1)
            return float(reward_str)
        else:
            return None

    def _target_match_ratio(self):
        necessary_action_triples = Utils.parse_actions_to_triples(self.necessary_actions)
        n_max_matches = len(self.target_text_res) + len(self.target_url_res) + len(necessary_action_triples)
        if n_max_matches == 0:
            # target not specified. Use the demonstration actions to identify success
            necessary_action_triples = Utils.parse_actions_to_triples(self.demonstration)
            n_max_matches = len(necessary_action_triples)

        n_matches = 0
        for target_text_re in self.target_text_res:
            if re.search(target_text_re, self.state.text_in_window, re.IGNORECASE) \
                    or target_text_re in self.state.text_in_window:
                n_matches += 1
        for target_url_re in self.target_url_res:
            if re.search(target_url_re, self.state.url, re.IGNORECASE) or target_url_re in self.state.url:
                n_matches += 1
        for action_type, value, target_locator_re in necessary_action_triples:
            last_action_on_target = None
            for action in self.action_history:
                if action.action_type != action_type:
                    continue
                if action.element is None:
                    continue
                if action.element.locator is None:
                    continue
                for target_locator_re_seg in target_locator_re.split(" || "):
                    if target_locator_re_seg in action.element.locator:
                        last_action_on_target = action
                        break
            if last_action_on_target and last_action_on_target.value == value:
                n_matches += 1
        return n_matches, n_max_matches

    def _is_visited(self, state):
        for visited_state in self.state_history:
            if state.same_as(visited_state):
                return True
        return False

    def to_dict(self, as_demo=False):
        if as_demo:
            return {
                "start_url": self.start_url,
                "query_words": self.query_words,
                "query_annotations": self.query_annotations,
                "target_text_res": self.target_text_res,
                "target_url_res": self.target_url_res,
                "target_url": self.target_url,
                "necessary_actions": self.necessary_actions,
                "demonstration": self.demonstration
            }
        else:
            return {
                "start_url": self.start_url,
                "step_limit": self.step_limit,
                "query_words": self.query_words,
                "query_annotations": self.query_annotations,
                "in_window_only": self.in_window_only,
                "included_url_res": self.included_url_res,
                "target_url_res": self.target_url_res,
                "target_text_res": self.target_text_res,
                "target_state_str": self.target_state_str,

                "demonstration": self.demonstration,
                # "coverage": self.get_coverage(),

                "state_history": [state.state_str for state in self.state_history],
                "action_history": [action.replay_api for action in self.action_history],
                "state": self.state.state_str,
                "reward": self.reward,
                "total_reward": self.total_reward,
                "done": self.done,
                "target_achieved": self.target_achieved
            }

    def save(self, task_dir, save_utg=False, overwrite=False, as_demo=False):
        if not task_dir:
            return
        if not os.path.exists(task_dir):
            os.makedirs(task_dir)
        task_file_path = os.path.join(task_dir, self.name + "task.json")
        if os.path.exists(task_file_path) and not overwrite:
            self.name = self.name + "_"
            self.utg.name = self.name
            self.save(task_dir, save_utg)
        else:
            task_json_file = open(os.path.join(task_dir, self.name + "task.json"), "w")
            json.dump(self.to_dict(as_demo=as_demo), task_json_file, indent=2)
            task_json_file.close()
            for state in self.state_history + [self.state]:
                state.save(state_dir=os.path.join(task_dir, "states"))
            if save_utg:
                self.utg.save(task_dir)

    def load_utg(self, task_path):
        task_dir = os.path.dirname(task_path)
        task_name = os.path.basename(task_path)[:-len("task.json")]
        self.utg = UTG.load_from_dir(utg_dir_path=os.path.join(task_dir, task_name + "utg.js"))
        self.utg.states_dir = os.path.join(os.path.dirname(task_path), "states")

    @staticmethod
    def load(task_path, load_utg=False):
        if not task_path.endswith("task.json") or not os.path.exists(task_path):
            return None
        task_dir = os.path.dirname(task_path)
        task_name = os.path.basename(task_path)[:-len("task.json")]
        task_json_file = open(task_path)
        task_dict = json.load(task_json_file)
        task_json_file.close()
        task = Task(name=task_name, **task_dict)
        if load_utg:
            task.utg = UTG.load_from_dir(utg_dir_path=os.path.join(task_dir, task_name + "utg.js"))
            task.utg.states_dir = os.path.join(os.path.dirname(task_path), "states")

        if "state_history" in task_dict and "action_history" in task_dict:
            task.reward = task_dict["reward"]
            task.total_reward = task_dict["total_reward"]
            task.done = task_dict["done"]
            task.target_achieved = task_dict["target_achieved"]
            states_str = os.path.join(task_dir, "states")
            for i in range(len(task_dict["state_history"])):
                state_str = task_dict["state_history"][i]
                action_replay_api = task_dict["action_history"][i]
                state = State.load(state_dir=states_str, state_str=state_str)
                state.setup(task)
                action = state.get_action(action_replay_api)
                task.state_history.append(state)
                task.action_history.append(action)
            task.state = State.load(state_dir=states_str, state_str=task_dict["state"])
            task.state.setup(task)
        return task

    @staticmethod
    def load_tasks(task_path, load_utg=False):
        tasks = []
        if not task_path:
            return tasks
        if os.path.isdir(task_path):
            for file_name in os.listdir(task_path):
                if file_name.endswith("task.json"):
                    task = Task.load(task_path=os.path.join(task_path, file_name), load_utg=load_utg)
                    if task:
                        tasks.append(task)
                if file_name.endswith("taskset.json"):
                    tasks.extend(Task.load_tasks(task_path=os.path.join(task_path, file_name), load_utg=load_utg))
        elif task_path.endswith("task.json"):
            tasks.append(Task.load(task_path=task_path, load_utg=load_utg))
        elif task_path.endswith("taskset.json"):
            taskset_name = os.path.basename(task_path)[:-len("taskset.json")]
            taskset_json_file = open(task_path)
            taskset_dict = json.load(taskset_json_file)
            taskset_json_file.close()
            start_urls = taskset_dict["start_urls"] if "start_urls" in taskset_dict else []
            task_dicts = taskset_dict["tasks"]
            for task_key in task_dicts:
                task_dict = task_dicts[task_key]
                task_urls = [task_dict.pop("start_url")] if "start_url" in task_dict else start_urls
                for task_url in task_urls:
                    task_host = Utils.get_host(url=task_url)
                    task_name = "%s_%s_%s_" % (taskset_name, task_key, task_host)
                    task = Task(name=task_name, start_url=task_url, **task_dict)
                    tasks.append(task)
        return tasks


ACTION_RE = r"(\S+) #(.*)# @ (.+)"
ACTION_DEMO_SAVE = "Shift+S"
ACTION_DEMO_CURRENT = "Shift+C"
ACTION_DEMO_NEXT = "Shift+N"
ACTION_DEMO_QUIT = "Shift+Q"


class ChromeBrowser:
    def __init__(self, wait=1.0, proxy=None, mobile=False, headless=False, restart_reset=False,
                 chrome_path=None, extension_path=None, **kwargs):
        self.logger = logging.getLogger("Browser")
        self.wait = wait
        self.mobile = mobile
        self.headless = headless
        self.restart_reset = restart_reset

        self.window_width = 1200
        self.window_height = 1200
        # Specify window size
        if self.mobile:
            self.window_width = 540
            self.window_height = 960

        webbot_js_path = os.path.abspath(os.path.join(".", "resources", "webbot.js"))
        self.webbot_js = "".join(open(webbot_js_path).readlines())

        self._chrome_options = webdriver.ChromeOptions()

        # Specify window size
        if self.mobile:
            mobile_emulation = {
                "deviceMetrics": {"width": self.window_width, "height": self.window_height},
                "userAgent": "Mozilla/5.0 (Linux; Android 4.2.1; en-us; "
                             "Nexus 5 Build/JOP40D) AppleWebKit/535.19 (KHTML, like Gecko) "
                             "Chrome/18.0.1025.166 Mobile Safari/535.19"
            }
            self._chrome_options.add_experimental_option("mobileEmulation", mobile_emulation)
        else:
            self._chrome_options.add_argument("--window-size=%d,%d" % (self.window_width, self.window_height))
        self._chrome_options.add_argument("--disable-notifications")
        self._chrome_options.add_argument('--no-sandbox')
        self._chrome_options.add_argument('--disable-dev-shm-usage')

        if chrome_path is not None:
            self._chrome_options.binary_location(chrome_path)

        if extension_path is not None:
            print("Loading extension")
            # extension_path = "~/AppData/Local/Google/Chrome/User Data/Default/Extensions/chrome_extension/"
            # self._chrome_options.add_argument("–-load-extension=%s" % extension_path)
            # extension_path = "/mnt/f/Appstract_android/webRL/tests/chrome-extension.crx"
            self._chrome_options.add_extension(extension_path)
        else:
            self.logger.debug("no extension to load ---------------------------------------")

        # Headless chrome doesn"t support extensions
        if headless:
            self._chrome_options.add_argument("--headless")

        # Use proxy
        if proxy:
            self._chrome_options.add_argument("--proxy-server=%s" % proxy)
            self._chrome_options.add_argument("--proxy-bypass-list=localhost")

        self.driver = None
        self.root_window = None
        self.setup()

    def setup(self):
        capabilities = DesiredCapabilities().CHROME
        capabilities["pageLoadStrategy"] = "normal"  # complete
        # capabilities["pageLoadStrategy"] = "eager"  #  interactive, NOTE: not supported
        # capabilities["pageLoadStrategy"] = "none"
        capabilities["loggingPrefs"] = {"browser": "ALL"}
        system = platform.system()
        if "Microsoft" in platform.release():
            chromedriver_path = os.path.abspath(os.path.join(".", "resources", "chromedriver_win32.exe"))
        elif system == "Windows":
            chromedriver_path = os.path.abspath(os.path.join(".", "resources", "chromedriver_win32.exe"))
        elif system == "Linux":
            chromedriver_path = os.path.abspath(os.path.join(".", "resources", "chromedriver_linux64"))
        elif system == "Darwin":
            chromedriver_path = os.path.abspath(os.path.join(".", "resources", "chromedriver_max64"))
        else:
            self.logger.warning("Unsupported system: %s" % system)
            sys.exit(-1)

        self.driver = webdriver.Chrome(
            chrome_options=self._chrome_options,
            desired_capabilities=capabilities,
            executable_path=chromedriver_path)
        self.driver.implicitly_wait(10)
        self.driver.set_page_load_timeout(10)
        self.root_window = self.driver.current_window_handle
        self._resize_window()

        #test for extension
        #self.driver.get('https://bing.com')
        #try:
        #    time.sleep(3)
        #    header = self.driver.find_element_by_id('installed')
        #    print('Success! :-)')
        #except NoSuchElementException:
        #    print('Failure! :-(')
        #finally:
        #    self.driver.quit()


    def _resize_window(self):
        if self.mobile:
            return
        cmd = "return [window.outerWidth-window.innerWidth, window.outerHeight-window.innerHeight];"
        padding = self.driver.execute_script(cmd)
        expected_window_size = [self.window_width + padding[0], self.window_height + padding[1]]
        self.driver.set_window_size(width=expected_window_size[0], height=expected_window_size[1])

    def _check_webbot_present(self):
        cmd = "return window.ylController != undefined;"
        try:
            if self.driver.execute_script(cmd):
                return True
        except:
            pass
        self.driver.execute_script(self.webbot_js)
        # time.sleep(2)
        is_present = self.driver.execute_script(cmd)
        assert is_present
        return True

    def _execute_script(self, cmd, *args):
        self._check_webbot_present()
        return self.driver.execute_script(cmd, args)

    def _take_screenshot(self):
        screenshot = self.driver.get_screenshot_as_png()
        screenshot = Image.open(BytesIO(screenshot))
        return screenshot

    def _zoom_page(self, ratio=0.9):
        script = "document.body.style.zoom=%s;" % ratio
        return self.driver.execute_script(script)

    def _get_state_from_json(self, state_json):
        try:
            state_dict = json.loads(state_json)
            screenshot = self._take_screenshot()
            state = State(state_dict=state_dict, screenshot=screenshot)
            return state
        except Exception:
            self.logger.warning("_get_state_from_json failed.")
        return None

    def get_current_state(self):
        try:
            cmd = "return window.ylController.getStateJson();"
            state_json = self._execute_script(cmd)
            state = self._get_state_from_json(state_json)
            return state
        except TimeoutException:
            self.logger.warning("get_current_state timeout")
            return None

    def perform_action(self, action):
        target_locator = action.element.locator if action.element else None
        return self.perform_action_by_locator(action.action_type, action.value, target_locator)

    def _filter_actions(self, current_state, actions):
        filtered_actions = actions
        return filtered_actions

    def locate_element(self, locator):
        xpaths = locator.split(" || ")
        for xpath in xpaths:
            try:
                target_element = self.driver.find_element_by_xpath(xpath)
                if isinstance(target_element, WebElement):
                    return target_element
            except Exception:
                pass
        self.logger.warning("Unable to locate element: %s" % locator)

    def perform_action_by_locator(self, action_type, value, target_locator):
        success = True
        try:
            if action_type == Action.FINISH:
                return success
            target_element = self.locate_element(target_locator)
            xpath = target_locator.split(" || ")[0]

            if action_type == Action.CLICK:
                try:
                    target_element.click()
                except:
                    self.logger.warning("Selenium failed to perform click, using JS instead. %s" % xpath)
                    action_dict = {
                        "action_type": "click",
                        "target_locator": xpath,
                        "value": value
                    }
                    self._execute_script("window.ylController.performActionJson(arguments[0]);",
                                         json.dumps(action_dict))
            elif action_type == Action.CHECK:
                try:
                    target_element.click()
                except:
                    self.logger.warning("Selenium failed to perform check, using JS instead. %s" % xpath)
                    action_dict = {
                        "action_type": "check",
                        "target_locator": xpath,
                        "value": value
                    }
                    self._execute_script("window.ylController.performActionJson(arguments[0]);",
                                         json.dumps(action_dict))
            elif action_type == Action.SELECT:
                try:
                    selected_index = int(value)
                    target_select = Select(target_element)
                    target_select.select_by_index(selected_index)
                except:
                    self.logger.warning("Selenium failed to perform select, using JS instead. %s" % xpath)
                    action_dict = {
                        "action_type": "check",
                        "target_locator": xpath,
                        "value": value
                    }
                    self._execute_script("window.ylController.performActionJson(arguments[0]);",
                                         json.dumps(action_dict))
            elif action_type == Action.INPUT_TEXT:
                try:
                    target_element.clear()
                    time.sleep(0.5)
                    target_element.send_keys(value)
                except:
                    self.logger.warning("Selenium failed to perform input, using JS instead. %s" % xpath)
                    action_dict = {
                        "action_type": "setValue",
                        "target_locator": xpath,
                        "value": value
                    }
                    self._execute_script("window.ylController.performActionJson(arguments[0]);",
                                         json.dumps(action_dict))
            elif action_type == Action.PRESS_ENTER:
                target_element.send_keys(Keys.ENTER)
            else:
                self.logger.warning("Cannot perform action %s" % action_type)
            time.sleep(self.wait)
            if Utils.get_host(self.driver.current_url) in LONG_WAIT_HOSTS:
                time.sleep(3)
        except TimeoutException:
            self.logger.warning("perform_action_by_locator timeout")
            pass
        except Exception as e:
            self.logger.warning("perform_action_by_locator failed: %s, locator: %s" %
                                (str(e).splitlines()[0], target_locator))
            success = False
        self.driver.switch_to.window(self.driver.window_handles[-1])
        return success

    def start_log_server(self):
        import threading
        if sys.version_info[0] == 2:
            from BaseHTTPServer import SimpleHTTPRequestHandler, HTTPServer
        else:
            from http.server import SimpleHTTPRequestHandler, HTTPServer

        class MyHandler(SimpleHTTPRequestHandler):
            def do_POST(self):
                content_length = int(self.headers['Content-Length'])  # Gets the size of data
                post_data = self.rfile.read(content_length)  # Gets the data itself
                log_line = post_data.decode("utf-8") if isinstance(post_data, bytes) else str(post_data)
                outer.log_lines.append(log_line)
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(b"OK")

        outer = self
        self.log_lines = []
        self.log_server = HTTPServer(("", 7336), MyHandler)

        def start_log_server():
            self.log_server.serve_forever()

        self.log_server_thread = threading.Thread(target=start_log_server)
        self.log_server_thread.setDaemon(True)
        self.log_server_thread.start()
        time.sleep(1)

    def stop_log_server(self):
        if hasattr(self, "log_server") and self.log_server:
            self.log_server.shutdown()
            self.log_server_thread.join(0)

    def get_log(self, through_console=False):
        if through_console:
            log_lines = []
            log_entries = self.driver.get_log("browser")
            for log_entry in log_entries:
                message = log_entry["message"]
                m = re.match(r"console-api \d+:\d+ \"WebBot (.+)\"", message)
                if m:
                    log_lines.append(m.group(1).replace('\\"', '\"'))
            return log_lines
        if hasattr(self, "log_lines"):
            r = self.log_lines.copy()
            self.log_lines.clear()
            return r
        return []

    def reset(self, url, restart=False):
        if url.startswith("miniwob"):
            rel_path = os.path.join("resources", "miniwob", url[len("miniwob_"):])
            url = "file:///" + os.path.abspath(rel_path)
        if self.restart_reset or restart or Utils.get_host(url) in RESTART_HOSTS:
            self.close()
            self.setup()
        else:
            try:
                self.driver.delete_all_cookies()
                self.driver.execute_script("window.localStorage.clear();")
                self.driver.execute_script("window.sessionStorage.clear();")
            except Exception:
                pass
                # self.logger.warning("Failed to clear cookie, session, or localStorage (ignored).")
            try:
                if self.root_window not in self.driver.window_handles:
                    self.root_window = self.driver.window_handles[0]
                for window_handle in self.driver.window_handles:
                    if window_handle == self.root_window:
                        continue
                    self.driver.switch_to.window(window_handle)
                    self.driver.close()
                self.driver.switch_to.window(self.root_window)
            except Exception:
                self.logger.warning("Failed to switch window, restarting browser.")
                self.close()
                self.setup()
        try:
            self._resize_window()
            self.driver.get(url)
            time.sleep(self.wait)
            url_host = Utils.get_host(url)
            if url_host in LONG_WAIT_HOSTS:
                time.sleep(3)
            if "demo.openmrs.org" in url_host:
                script = '''
                    document.getElementById("username").value = "admin";
                    document.getElementById("password").value = "Admin123"
                    document.getElementById("Pharmacy").click();
                    document.getElementById("loginButton").click();'''
                self._execute_script(script)
                time.sleep(self.wait)
            return True
        except Exception as e:
            self.logger.warning("Failed to resize window or open url: %s" % e)
            return False

    def close(self):
        if self.driver:
            self.driver.quit()


class CacheBrowser:
    def __init__(self, utgs, name=None, **kwargs):
        self.name = name if name else ""
        self.logger = logging.getLogger("CacheBrowser(%s)" % self.name)
        self.window_width = 1200
        self.window_height = 1200
        self._state_cache = {}

        self.utgs = utgs
        self.current_utg = None
        self.current_state_str = None
        self.use_fake_state = False

    def _get_state(self, state_str):
        if state_str not in self._state_cache:
            state = self.current_utg.get_state(state_str)
            self._state_cache[state_str] = state
        return self._state_cache[state_str]

    def get_current_state(self):
        return self._get_state(self.current_state_str)

    def perform_action(self, action):
        if action.action_type == Action.FINISH:
            return True
        current_state = self.get_current_state()
        next_state_weights = self.current_utg.get_next_state_weights(action, current_state)
        if next_state_weights:
            self.current_state_str = Utils.weighted_choice(next_state_weights)
            return True
        # elif action.action_type in [Action.SELECT, Action.INPUT_TEXT]:
        if self.use_fake_state:
            fake_state = Utils.create_fake_state(current_state, action)
            if fake_state is not None:
                fake_state_str = fake_state.state_str
                if fake_state_str not in self.current_utg.G.nodes:
                    self._state_cache[fake_state_str] = fake_state
                    self.logger.warning("perform_action: fake state unseen")
                else:
                    self.logger.warning("perform_action: fake state matched actual state")
                self.current_state_str = fake_state_str
                return True
        self.logger.warning("perform_action: cache miss with action: %s", action)
        self.current_state_str = self.current_utg.get_init_state_str()
        return False

    def _filter_actions(self, current_state, actions):
        filtered_actions = []
        for action in actions:
            if self.current_utg.get_next_state_weights(action, current_state):
                filtered_actions.append(action)
        return filtered_actions

    def reset(self, url, **kwargs):
        self.current_utg = None
        for utg in self.utgs:
            if utg.start_url == url:
                self.current_utg = utg
                break
        if self.current_utg is None:
            self.logger.warning("reset: cache miss with url: %s" % url)
            return False
        self.current_state_str = self.current_utg.get_init_state_str()
        return True

    def close(self):
        pass


class TaskCanvas:
    def __init__(self, title=None, state_size=None):
        self.title = title if title else "TaskCanvas"
        self.padding = 50

        if state_size is None:
            state_size = [256, 256]
        self.state_width, self.state_height = state_size

        canvas_w = self.state_width * 2 + self.padding
        canvas_h = self.state_height + self.padding

        if sys.version_info[0] < 3:
            import Tkinter as tkinter
        else:
            import tkinter
        self.tk = tkinter.Tk()
        self.tk.geometry("%dx%d+0+0" % (canvas_w, canvas_h))
        self.tk.title(self.title)
        self.canvas = tkinter.Canvas(self.tk, bg="white", width=canvas_w, height=canvas_h)
        self.canvas.pack()
        self.__canvas_tmp_objs = []

    def render_task(self, task):
        assert isinstance(task, Task)
        try:
            c = self.canvas
            for obj in self.__canvas_tmp_objs:
                c.delete(obj)
            if len(task.state_history) > 0:
                state1 = task.state_history[-1]
                screen = state1.screenshot.resize([self.state_width, self.state_height], Image.ANTIALIAS)
                img1 = ImageTk.PhotoImage(screen)
                img1_ = c.create_image(0, 0, anchor="nw", image=img1)
                self.__canvas_tmp_objs.append(img1_)
            if task.state:
                state2 = task.state
                screen = state2.screenshot.resize([self.state_width, self.state_height], Image.ANTIALIAS)
                img2 = ImageTk.PhotoImage(screen)
                img2_ = c.create_image(self.state_width + self.padding, 0, anchor="nw", image=img2)
                self.__canvas_tmp_objs.append(img2_)
            if len(task.action_history) > 0:
                action = task.action_history[-1]
                text1 = c.create_text(self.state_width + self.padding / 2,
                                      self.state_height + self.padding / 2,
                                      anchor="center",
                                      text="task:%s \t action:%s \t reward:%d" %
                                           (task.name, action.action_type, task.reward))
                self.__canvas_tmp_objs.append(text1)
                arrow1 = c.create_line(self.state_width, self.state_height / 2,
                                       self.state_height + self.padding, self.state_height / 2,
                                       arrow="last", width=4.0)
                self.__canvas_tmp_objs.append(arrow1)
                if action.element:
                    ele_bound = action.element.get_resized_bound(new_width=self.state_width,
                                                                 new_height=self.state_height)
                    bbox1 = c.create_rectangle(ele_bound["left"], ele_bound["top"],
                                               ele_bound["right"], ele_bound["bottom"],
                                               outline="red", width=2.0)
                    self.__canvas_tmp_objs.append(bbox1)
            self.tk.update()
        except:
            traceback.print_exc()

    def destroy(self):
        self.tk.destroy()


class WebBotEnv:
    def __init__(self, browser, tasks, visualize=False, **kwargs):
        self.logger = logging.getLogger("WebBotEnv")

        self.browser = browser
        self.tasks = tasks
        self.visualize = visualize
        self.canvas = TaskCanvas() if self.visualize else None
        self.current_task = None

    def render(self):
        if self.visualize and self.canvas:
            self.canvas.render_task(self.current_task)
        # self.logger.info(f" number of actions in current state: {len(self.current_task.state.possible_actions)}")

    def step(self, action):
        """
        Take an action in current environment.
        :param action: the action to take
        :return: new_state, reward, done
        """
        state = None
        try:
            # action_success = self.browser.perform_action(action)
            # state = self.get_state() if action_success else None
            success = self.browser.perform_action(action)
            state = self.get_state() if success else None
        except Exception:
            self.logger.warning("step failed: %s" % action)
            traceback.print_exc()
        reward, done = self.current_task.update(action, state)
        return state, reward, done

    def explore(self, n_episodes=50, output_dir=None, policy="random", save_utg=False):
        try:
            if output_dir and save_utg:
                for task in self.tasks:
                    task.utg.states_dir = os.path.join(output_dir, "states")
            for episode in range(1, n_episodes + 1):
                self.reset()
                task = self.current_task.snapshot()
                self.logger.info("Episode %d/%d, task: %s" % (episode, n_episodes, task.task_str))
                while True:
                    if task.done:
                        break
                    self.render()
                    preferred_actions = task.get_preferred_actions()
                    possible_actions = task.state.possible_actions
                    if len(possible_actions) == 0:
                        break
                    if len(preferred_actions) == 0:
                        preferred_actions = possible_actions
                    if policy == "similarity":
                        action_scores = {}
                        for action in preferred_actions:
                            action_scores[action] = task.get_action_usefulness(action)
                        action = Utils.weighted_choice(action_scores)
                    elif policy == "random":
                        action = random.choice(possible_actions)
                    elif policy == "random_restricted":
                        action = random.choice(preferred_actions)
                    elif policy == "demo_biased":
                        actions_in_demo = []
                        for action in preferred_actions:
                            if action.replay_api in task.demonstration:
                                actions_in_demo.append(action)
                        rand = random.uniform(0, 1)
                        if len(actions_in_demo) > 0 and rand < 0.5:
                            action = random.choice(actions_in_demo)
                        else:
                            action = random.choice(preferred_actions)
                    else:
                        action = random.choice(possible_actions)
                    self.step(action)
                    task_ = self.current_task.snapshot()
                    task = task_
                    self.logger.info("\tExplore, action:%s, %s" % (action, task.get_reward_str()))
                if output_dir and episode % 50 == 0:
                    task.save(task_dir=output_dir, save_utg=save_utg, overwrite=True)
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt.")
            pass
        except Exception as e:
            self.logger.warning("gen_utg failed: %s" % e)
            traceback.print_exc()

    def get_state(self):
        try:
            state = self.browser.get_current_state()
            if state:
                state.setup(self.current_task)
            return state
        except Exception as e:
            self.logger.error("get_state failed with error: %s" % e)
        return None

    def reset(self, new_task=None):
        # Switch to a random task
        self.current_task = new_task if new_task else random.choice(self.tasks)
        self.browser.reset(self.current_task.start_url)

        retry_count = 0
        while True:
            state = self.get_state()
            if state and len(state.possible_actions) > 0:
                break
            retry_count += 1
            if retry_count == 2:
                self.logger.warning("reset failed, restarting browser: %s." % self.current_task.start_url)
                self.browser.reset(self.current_task.start_url, restart=True)
                continue
            if retry_count > 3:
                self.logger.warning("reset failed: %s." % self.current_task.start_url)
                break
        self.current_task.reset(state)
        return state

    def _listen_actions(self):
        def parse_action_lines(lines):
            lines_parsed = []
            last_line, last_action_type, last_target_locator = None, None, None
            last_output_ele = None
            for line in lines:
                m = re.match(ACTION_RE, line)
                if m:
                    action_type, value, target_locator = m.group(1), m.group(2), m.group(3)
                    if last_action_type == Action.INPUT_TEXT and \
                            (action_type != last_action_type or target_locator != last_target_locator):
                        lines_parsed.append(last_line)
                    if action_type == Action.FINISH:
                        if target_locator == last_output_ele:
                            lines_parsed.append(line)
                            last_output_ele = None
                        else:
                            last_output_ele = target_locator
                    if action_type != Action.INPUT_TEXT and action_type != Action.FINISH:
                        lines_parsed.append(line)
                    last_line, last_action_type, last_target_locator = line, action_type, target_locator
                else:
                    self.logger.warning("parse_action_lines failed: %s" % line)
            if last_action_type == Action.INPUT_TEXT:
                lines_parsed.append(last_line)

            filtered_action_lines = []
            for i, action_line in enumerate(lines_parsed):
                if i > 0:
                    m = re.match(ACTION_RE, action_line)
                    action_type, value, target_locator = m.group(1), m.group(2), m.group(3)
                    if action_type == Action.INPUT_TEXT:
                        last_action_line = filtered_action_lines[-1]
                        m = re.match(ACTION_RE, last_action_line)
                        last_action_type, last_value, last_target_locator = m.group(1), m.group(2), m.group(3)
                        if last_action_type == Action.CLICK and last_target_locator == target_locator:
                            filtered_action_lines.pop(len(filtered_action_lines) - 1)
                filtered_action_lines.append(action_line)
            return filtered_action_lines

        log_lines = []
        self.browser.get_log()
        while True:
            try:
                for log_line in self.browser.get_log():
                    print("Action captured: ", log_line)
                    if log_line in [ACTION_DEMO_SAVE, ACTION_DEMO_CURRENT, ACTION_DEMO_NEXT, ACTION_DEMO_QUIT]:
                        return log_line, parse_action_lines(log_lines)
                    else:
                        log_lines.append(log_line)
                if self.browser.driver.current_window_handle != self.browser.driver.window_handles[-1]:
                    self.browser.driver.switch_to.window(self.browser.driver.window_handles[-1])
                self.browser._execute_script("window.ylController.listenActions();")
                time.sleep(0.5)
            except Exception:
                pass

    def demonstrate(self, output_dir=None, save_utg=False, skip_existing=True, verify_after_demo=True):
        self.browser.start_log_server()
        for task in self.tasks:
            if skip_existing and os.path.exists(os.path.join(output_dir, task.name + "task.json")):
                continue
            while True:
                self.reset(new_task=task)
                print("Demonstrating task %s: %s" % (task.name, task.task_str))
                print("\t%s: \tSave the demonstration of current task" % ACTION_DEMO_SAVE)
                print("\t%s: \tRe-demonstrate current task" % ACTION_DEMO_CURRENT)
                print("\t%s: \tDemonstrate next task" % ACTION_DEMO_NEXT)
                print("\t%s: \tQuit demonstration" % ACTION_DEMO_QUIT)
                print("\tDouble click: \tSelect output")
                demo_control, action_lines = self._listen_actions()
                if demo_control in [ACTION_DEMO_SAVE]:
                    task.demonstration = action_lines
                    task.target_url = task.state.url
                    task.save(output_dir, save_utg=save_utg, as_demo=True)
                    if verify_after_demo:
                        print("verifying the demonstration by replaying ...")
                        reply_succeeded = self._replay_actions(task, action_lines=action_lines)
                        if reply_succeeded:
                            print("replay succeeded")
                    continue
                elif demo_control in [ACTION_DEMO_CURRENT]:
                    continue
                elif demo_control in [ACTION_DEMO_NEXT, ACTION_DEMO_QUIT]:
                    break
                else:
                    self.logger.warning("Unknown demo_control: " + demo_control)
            if demo_control in [ACTION_DEMO_QUIT]:
                break
        self.browser.stop_log_server()
        self.logger.info("Done demonstrating tasks.")

    def _replay_actions(self, task, action_lines):
        success = True
        try:
            self.reset(new_task=task)
        except Exception as e:
            self.logger.error("replay failed during resetting: %s" % e)
            traceback.print_exc()
            return False
        for line in action_lines:
            try:
                self.render()
                m = re.match(ACTION_RE, line)
                action_type, value, target_locator = m.group(1), m.group(2), m.group(3)
                target_ele = task.state.get_element_by_locator(target_locator)
                action = Action(element=target_ele, action_type=action_type, value=value)
                state, reward, done = self.step(action)
                self.logger.info("\tReplay, action:%s, %s" % (action, task.get_reward_str()))
            except Exception as e:
                self.logger.error("replay failed during executing %s: %s" % (line, e))
                success = False
                traceback.print_exc()
                continue
            if state is None:
                success = False
        task.target_state_str = task.state.state_str
        self.logger.info("final url is %s" % task.state.url)
        return success

    def analyze_task_complexity(self, task, action_lines):
        success = True
        try:
            self.reset(new_task=task)
        except Exception as e:
            self.logger.error("replay failed during resetting: %s" % e)
            traceback.print_exc()
            return False

        num_actions_list = []
        num_sim_actions_list = []
        num_inputs_list = []
        num_words_list = []
        num_elements_list = []
        set_urls = set()
        from form_agent import FormManager
        for line in action_lines + [None]:
            try:
                self.render()
                state = task.state
                if not isinstance(state, State):
                    continue
                words = state.text.split()
                num_words_list.append(len(words))
                num_actions_list.append(len(state.possible_actions))
                num_sim_actions_list.append(len(task.get_preferred_actions()))
                num_elements_list.append(len(state.elements_in_window))
                input_candidates = FormManager.extract_input_candidates(task)
                num_inputs_list.append(len(input_candidates))
                set_urls.add(state.url)

                if line is None:
                    continue
                m = re.match(ACTION_RE, line)
                action_type, value, target_locator = m.group(1), m.group(2), m.group(3)
                target_ele = task.state.get_element_by_locator(target_locator)
                action = Action(element=target_ele, action_type=action_type, value=value)
                state, reward, done = self.step(action)
                self.logger.info("\tReplay, action:%s, %s" % (action, task.get_reward_str()))
            except Exception as e:
                self.logger.error("replay failed during executing %s: %s" % (line, e))
                success = False
                traceback.print_exc()
                continue
            if state is None:
                success = False

        num_actions = np.mean(num_actions_list)
        num_sim_actions = np.mean(num_sim_actions_list)
        num_elements = np.mean(num_elements_list)
        num_inputs = np.mean(num_inputs_list)
        num_words = np.mean(num_words_list)
        num_pages = len(set_urls)
        num_demo_steps = len(action_lines)
        num_demo_click_steps = len([line for line in action_lines if line.startswith('click')])
        num_demo_input_steps = len([line for line in action_lines if line.startswith('input_text') or line.startswith('select')])
        num_parameters = len(task.parameter_ids)
        num_task_words = len(task.query_words)
        task.target_state_str = task.state.state_str
        self.logger.info("final url is %s" % task.state.url)
        success_tag = 'Y' if success else 'N'
        self.logger.info("Task complexity head task_name "
                         "num_actions num_sim_actions num_inputs num_words num_elements "
                         "num_pages num_demo_steps num_demo_click_steps num_demo_input_steps "
                         "num_parameters num_task_words success_tag")
        self.logger.info(f"Task complexity row {task.name} "
                         f"{num_actions} {num_sim_actions} {num_inputs} {num_words} {num_elements} "
                         f"{num_pages} {num_demo_steps} {num_demo_click_steps} {num_demo_input_steps} "
                         f"{num_parameters} {num_task_words} {success_tag}")
        return success


    def _generalize(self, task, action_lines):
        # TODO implement this
        replay_success = self._replay_actions(task, action_lines)
        if not replay_success:
            return False

        state_explored_action_lines = {}
        state_effective_actions = {}

        def is_fully_explored():
            for state_str in state_explored_action_lines:
                tried_actions = state_explored_action_lines[state_str]
                for action_line in action_lines:
                    if action_line not in tried_actions:
                        return False
            return True

        def get_unexplored_action(state):
            for action_line in action_lines:
                if action_line in state_explored_action_lines[state.state_str]:
                    continue
                state_explored_action_lines[state.state_str].add(action_line)
                m = re.match(ACTION_RE, action_line)
                action_type, value, target_locator = m.group(1), m.group(2), m.group(3)
                element = state.get_element_by_locator(target_locator)
                if element:
                    return Action(element=element, action_type=action_type, value=value)
            return None

        while not is_fully_explored():
            self.reset(new_task=task)
            while True:
                if task.state.state_str not in state_explored_action_lines:
                    state_explored_action_lines[task.state.state_str] = set()
                if task.state.state_str not in state_effective_actions:
                    state_effective_actions[task.state.state_str] = set()
                action = get_unexplored_action(task.state)
                if not action:
                    action = random.choice(state_effective_actions[task.state.state_str])
                state, reward, done = self.step(action=action)
                if state:
                    state_effective_actions[task.state.state_str].add(action)
                self.logger.info("\tGeneralize, action:%s, reward:%.2f, done:%s" %
                                 (action, task.reward, task.done))
        return True

    def replay(self, replay_source=None):
        for task in self.tasks:
            self.logger.info("Replaying task: %s" % task.task_str)
            if replay_source == "demonstration" or replay_source is None:
                action_lines = task.demonstration
            elif replay_source == "history":
                action_lines = task.action_history
            else:
                action_lines = replay_source
            if action_lines:
                reply_succeeded = self._replay_actions(task, action_lines=action_lines)
                if reply_succeeded:
                    self.logger.info("Done replaying, total_reward %.2f" % task.total_reward)
            else:
                self.logger.warning("Cannot replay task: %s" % task.task_str)

    def post_process(self, task, tasklet):
        self.logger.info("Postprocessing tasklet:\n%s" % tasklet)
        original_replay_trace = Task.get_replay_trace_from_tasklet(tasklet)
        original_reward = Task.get_total_reward_from_tasklet(tasklet)
        original_tasklet = tasklet
        replay_trace = copy.copy(original_replay_trace)
        self._replay_actions(task, replay_trace)
        if original_reward is None:
            original_reward = task.total_reward
        original_final_screen = task.state.screenshot

        # try remove negative-reward actions
        negative_reward_actions = []
        if len(task.action_history) == len(task.reward_history):
            for i, action in enumerate(task.action_history):
                action_reward = task.reward_history[i][0]
                if i == len(replay_trace) and action.is_submit:
                    continue
                if action_reward <= 0:
                    negative_reward_actions.append(action)

        for action in negative_reward_actions:
            new_replay_trace = copy.copy(replay_trace)
            if action.replay_api not in new_replay_trace:
                continue
            new_replay_trace.remove(action.replay_api)
            self.logger.info("checking whether action is removable: %s" % action)
            replay_success = self._replay_actions(task, new_replay_trace)
            if replay_success and task.total_reward >= original_reward:
                # this action is removable
                self.logger.info("removable: %s" % action)
                replay_trace = new_replay_trace
            else:
                # this action is not removable
                self.logger.info("not removable: %s" % action)

        if len(negative_reward_actions) > 0:
            self._replay_actions(task, replay_trace)

        # try append form submission action
        if len(task.action_history) > 0:
            last_action = task.action_history[-1]
            form_submit_action = None
            if last_action.element and last_action.element.parent_form and not last_action.is_submit:
                form_submit_action = last_action.element.parent_form.form_submit_action
            if form_submit_action:
                self.logger.info("append action: %s" % form_submit_action)
                replay_trace.append(form_submit_action.replay_api)
            elif last_action.action_type == Action.INPUT_TEXT:
                press_enter_action = Action(last_action.element, Action.PRESS_ENTER, "")
                self.logger.info("append action: %s" % press_enter_action)
                replay_trace.append(press_enter_action.replay_api)
            self._replay_actions(task, replay_trace)
        if task.total_reward >= original_reward:
            final_screen = task.state.screenshot
            return task.get_tasklet(), task.total_reward, final_screen
        else:
            return original_tasklet, original_reward, original_final_screen

    def destroy(self):
        if self.canvas:
            self.canvas.destroy()


def parse_args():
    """
    Parse command line input
    :return:
    """
    parser = argparse.ArgumentParser(description="Start WebBot to test websites.")

    # Task definitions
    parser.add_argument("-start_url", action="store", dest="start_url", default=None,
                        help="The url to start with, ex. https://www.google.com/.")
    parser.add_argument("-query", action="store", dest="query", default="",
                        help="The task query, ex. \"search\"")
    parser.add_argument("-parameter", action="append", dest="parameters", default=[],
                        help="The parameter key:value pair, ex. \"word:microsoft\"")
    parser.add_argument("-included_url_re", action="append", dest="included_url_res", default=[],
                        help="Stop (fail) current episode if the url doesn't match any included_url_re")
    parser.add_argument("-target_url_re", action="append", dest="target_url_res", default=[],
                        help="Stop (success) current episode if the url matches all target_url_re")
    parser.add_argument("-target_text_re", action="append", dest="target_text_res", default=[],
                        help="Stop (success) current episode if the text matches all target_text_re")
    parser.add_argument("-task_path", action="store", dest="task_path", default=None,
                        help="Path to *task.json file or directory.")

    # Browser settings
    parser.add_argument("-wait", action="store", dest="wait", type=float, default=0.1,
                        help="Minimum time to wait after each action, in seconds.")
    parser.add_argument("-proxy", action="store", dest="proxy", type=str, default=None,
                        help="IP:PORT proxy to use.")
    parser.add_argument("-mobile", action="store_true", dest="mobile", default=False,
                        help="Test in mobile mode.")
    parser.add_argument("-headless", action="store_true", dest="headless", default=False,
                        help="Run browser in headless mode (do not start browser GUI).")
    parser.add_argument("-extension_path", action="store", dest="extension_path", default=None,
                        help="Path to extension .crx file. Run browser with a predefined extension installed."
                             "It does not work if headless is set to True.")

    # Environment settings
    parser.add_argument("-demonstrate", action="store_true", dest="demonstrate", default=False,
                        help="Demonstrate the tasks.")
    parser.add_argument("-replay", action="store", dest="replay_source", default=None,
                        help="Replay the tasks. Argument value can be \"demonstration\" or \"history\"")
    parser.add_argument("-explore", action="store_true", dest="explore", default=False,
                        help="Explore the tasks.")
    parser.add_argument("-visualize", action="store_true", dest="visualize", default=False,
                        help="Visualize state transitions.")
    parser.add_argument("-output_dir", action="store", dest="output_dir", default=None,
                        help="The directory to save utg.")
    parser.add_argument("-save_utg", action="store_true", dest="save_utg", default=False,
                        help="Save the UI transition graph and states.")

    args, unknown = parser.parse_known_args()
    return args


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    args = parse_args()
    print(args)

    arg_browser = ChromeBrowser(
        wait=args.wait,
        proxy=args.proxy,
        mobile=args.mobile,
        headless=args.headless,
        extension_path = args.extension_path
    )

    tasks = []

    if args.start_url:
        arg_task = Task(
            start_url=args.start_url,
            included_url_res=args.included_url_res,
            target_url_res=args.target_url_res,
            target_text_res=args.target_text_res,
            query_words=args.query.split()
        )
        tasks.append(arg_task)

    tasks.extend(Task.load_tasks(args.task_path))

    if args.save_utg and args.output_dir:
        for task in tasks:
            task.utg.states_dir = os.path.join(args.output_dir, "states")

    # cache_browser = CacheBrowser(
    #     utgs=[task.utg for task in tasks]
    # )

    arg_env = WebBotEnv(
        tasks=tasks,
        browser=arg_browser,
        visualize=args.visualize
    )

    if args.demonstrate:
        arg_env.demonstrate(output_dir=args.output_dir, save_utg=args.save_utg)
    if args.replay_source:
        arg_env.replay(replay_source=args.replay_source)
    if args.explore:
        arg_env.explore(output_dir=args.output_dir, save_utg=args.save_utg)


if __name__ == "__main__":
    main()
