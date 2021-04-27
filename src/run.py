# -*- coding: utf-8 -*-
import argparse
import json
import logging
import os
import random
import re
import shutil
from datetime import datetime

ADBLOCK_PROXY = "http://localhost:8118"  # Optionally set up a proxy to block the ads
from environment import REWARD_ITEM_WEIGHTS, GLOBAL_CONFIGS


def parse_args():
    """
    Parse command line input
    :return:
    """
    parser = argparse.ArgumentParser(description="Run the experiments.")

    parser.add_argument("--data_dir", action="store", dest="data_dir", type=str, default='../data/',
                        help="The dir of webbot code and all input, output data.")
    parser.add_argument("--log_dir", action="store", dest="log_dir", type=str, default=None,
                        help="The dir of logs.")
    parser.add_argument("--model_dir", action="store", dest="model_dir", type=str, default=None,
                        help="The dir of models.")
    parser.add_argument("--tag", action="store", dest="tag", type=str, default="T",
                        help="A tag string to identify the job category.")

    parser.add_argument("--task_dir", action="store", dest="task_dir", type=str, required=False, default="tasks",
                        help="The relative path of tasks from data_dir.")
    parser.add_argument("--config_path", action="store", dest="config_path", type=str, required=True,
                        help="The relative path from data_dir to model config file.")
    parser.add_argument("--phases", action="store", dest="phases", type=str, required=False, default="crawl",
                        help="The phases to run. Could be crawl,replay")
    parser.add_argument("--algorithm", action="store", dest="algorithm", default="q_table",
                        help="The algorithm to use, could be q_table, dqn, hill_climbing, monte_carlo")

    parser.add_argument("--wait", action="store", dest="wait", type=float, default=0.1,
                        help="The time to wait after taking an action.")
    parser.add_argument("--proxy", action="store", dest="proxy", default=None,
                        help="Run browser with a proxy (HOST:PORT).")
    parser.add_argument("--headless", action="store_true", dest="headless", default=False,
                        help="Run browser in headless mode.")
    parser.add_argument("--save_reports", action="store_true", dest="save_reports", default=False,
                        help="save reports.")
    parser.add_argument("--restart_reset", action="store_true", dest="restart_reset", default=False,
                        help="Restart the browser in each episode.")
    parser.add_argument("--use_cache", action="store_true", dest="use_cache", default=False,
                        help="Use cache for testing (quick but less accurate).")
    parser.add_argument("--chrome_path", action="store", dest="chrome_path", default=None,
                        help="Path to the chrome binary.")
    parser.add_argument("--extension_path", action="store", dest="extension_path", default=None,
                        help="Path to extension .crx file. Run browser with a predefined extension installed. "
                             "It does not work if headless is set to True.")

    parser.add_argument("--semantic_similarity", action="store_true", dest="semantic_similarity", default=False,
                        help="enable semantic similarity")
    parser.add_argument("--action_restriction", action="store_true", dest="action_restriction", default=False,
                        help="enable action restriction")
    parser.add_argument("--skip_postprocessing", action="store_true", dest="skip_postprocess", default=False,
                        help="skip postprocessing")
    parser.add_argument("--disable_text_embed", action="store_true", dest="disable_text_embed", default=False,
                        help="disable text embedding")
    parser.add_argument("--num_train_tasks", action="store", dest="num_train_tasks",
                        type=int, required=False, default=-1,
                        help="Number of tasks used to train the exploration model.")
    parser.add_argument("--explore_policy", action="store", dest="explore_policy",
                        type=str, required=False, default=None,
                        help="Exploration policy of the agent, could be random or similarity.")
    parser.add_argument("--explore_rate", action="store", dest="explore_rate",
                        type=float, required=False, default=None,
                        help="Exploration rate, could be 0.0 - 1.0.")
    parser.add_argument("--n_test_episodes", action="store", dest="n_test_episodes",
                        type=int, required=False, default=-1,
                        help="Number of testing episodes.")
    parser.add_argument("--eval_model", action="store", dest="eval_model",
                        type=str, required=False, default=None,
                        help="Evaluate the exploration model. intra_website, intra_domain, inter_domain")
    parser.add_argument("--reward_weights", action="store", dest="reward_weights",
                        type=str, required=False, default="",
                        help="Reward item weights")

    args, unknown = parser.parse_known_args()
    return args


def sync_annotations_to_demos():
    from environment import Task
    processed_task_names = set()
    for task in Task.load_tasks("tasks/"):
        print(task.name)
        demo_task_path = os.path.join("demonstrations", task.name + "task.json")
        demo_task = Task.load(demo_task_path)
        if demo_task is None:
            continue

        demonstration = []
        for action_line in demo_task.demonstration:
            segs = action_line.split(" || ")
            if len(segs) == 3:
                print(action_line)
                action_line = " || ".join(segs[:2])
                print(action_line)
            demonstration.append(action_line)

        demo_task_dict = {
            "start_url": task.start_url,
            "query_words": task.query_words,
            "query_annotations": task.query_annotations,
            "demonstration": demonstration,
            "target_text_res": task.target_text_res
        }
        output_path = os.path.join("demonstrations", task.name + "task.json")
        json.dump(demo_task_dict, open(output_path, "w"), indent=2)
        processed_task_names.add(task.name)

    for task in Task.load_tasks("demonstrations"):
        if task.name not in processed_task_names:
            print("not processed: %s" % task.name)


def verify_demos():
    from environment import Task
    from environment import ACTION_RE, Action
    for task in Task.load_tasks("tasks/"):
        if len(task.query_words) != len(task.query_annotations):
            print(task.name, "query annotation mismatch!")
        filtered_action_lines = []
        for i, action_line in enumerate(task.demonstration):
            if i > 0:
                m = re.match(ACTION_RE, action_line)
                action_type, value, target_locator = m.group(1), m.group(2), m.group(3)
                if action_type == Action.INPUT_TEXT:
                    last_action_line = filtered_action_lines[-1]
                    m = re.match(ACTION_RE, last_action_line)
                    last_action_type, last_value, last_target_locator = m.group(1), m.group(2), m.group(3)
                    if last_action_type == Action.CLICK and last_target_locator == target_locator:
                        print(task.name, "redundant click action")
                        filtered_action_lines.pop(len(filtered_action_lines) - 1)
            filtered_action_lines.append(action_line)
        demo_task_dict = {
            "start_url": task.start_url,
            "query_words": task.query_words,
            "query_annotations": task.query_annotations,
            "demonstration": filtered_action_lines,
            "target_text_res": task.target_text_res
        }
        output_path = os.path.join("demonstrations", task.name + "task.json")
        json.dump(demo_task_dict, open(output_path, "w"), indent=2)


def task_matches(task_label, task_label_res):
    if not task_label_res:
        return False
    for task_label_re in task_label_res:
        if re.search(task_label_re, task_label):
            return True
    return False


def load_tasks(train_task_res, test_task_res, args):
    """
    Load tasks from task_dir
    :param train_task_res: regex to match train tasks
    :param test_task_res: regex to match test tasks
    :param args: command line arguments
    :return: train_tasks, test_tasks
    """
    from environment import Task
    if args.task_dir.endswith("task.json"):
        task_paths = args.task_dir.split(",")
        test_tasks = []
        for task_path in task_paths:
            try:
                task_path = os.path.join(args.data_dir, task_path)
                test_task = Task.load(task_path, load_utg=False)
                if test_task is not None:
                    test_tasks.append(test_task)
            except:
                import traceback
                traceback.print_exc()
                continue
        return [], test_tasks

    all_tasks = []
    train_tasks = []
    test_tasks = []

    if args.task_dir == "tasks/*":
        task_dir = os.path.join(args.data_dir, "tasks")
        for file_name in os.listdir(task_dir):
            if not file_name.endswith("task.json"):
                pass
            try:
                task_path = os.path.join(task_dir, file_name)
                task = Task.load(task_path, load_utg=False)
                if task is None:
                    continue
                all_tasks.append(task)
                test_tasks.append(task)
            except:
                logging.warning("Failed to load: %s" % file_name)
                import traceback
                traceback.print_exc()
                continue
        return [], test_tasks

    # from generate_philly_jobs import PARAMETER_VARIATION_TASKS
    # test_task_names = PARAMETER_VARIATION_TASKS.splitlines()

    task_dir = os.path.join(args.data_dir, args.task_dir)
    for file_name in os.listdir(task_dir):
        if not file_name.endswith("task.json"):
            pass
        try:
            task_path = os.path.join(task_dir, file_name)
            task = Task.load(task_path, load_utg=False)
            if task is None:
                continue
            # Exclude tasks without a demonstration
            if not task.demonstration:
                continue
            all_tasks.append(task)

            # if file_name in test_task_names:
            #     test_tasks.append(task)

            if task_matches(task_path, test_task_res):
                test_tasks.append(task)
            if task_matches(task_path, train_task_res):
                if task.demonstration:
                    train_tasks.append(task)
        except:
            logging.warning("Failed to load: %s" % file_name)
            import traceback
            traceback.print_exc()
            continue

    query2tasks = {}
    for task in all_tasks:
        query = " ".join(task.query_words)
        if query not in query2tasks:
            query2tasks[query] = [task]
        else:
            query2tasks[query].append(task)
    for query in query2tasks.keys():
        tasks = query2tasks[query]
        demo_tasks = set()
        for task in tasks:
            if task.demonstration:
                demo_tasks.add(task)
        for task in tasks:
            demo_tasks_cp = list(demo_tasks)
            if task in demo_tasks_cp:
                demo_tasks_cp.remove(task)
            task.demo_tasks = demo_tasks_cp
            if len(demo_tasks_cp) > 0:
                task.demo_task = random.choice(demo_tasks_cp)
    return train_tasks, test_tasks


def train_and_test(train_tasks, test_tasks, args, model_param, output_dir=None):
    from agent import TestQTable, TestDQN
    from model import Qda2pModel
    from environment import Task
    from environment import ChromeBrowser, CacheBrowser, WebBotEnv, Utils, UTG

    # Training on other tasks is currently not supported
    # logging.info("Train on %d tasks:\n" % len(train_tasks) + "\n".join([task.task_str for task in train_tasks]))
    logging.info("Test on %d tasks:\n" % len(test_tasks) + "\n".join(task.task_str for task in test_tasks))

    if args.disable_text_embed:
        model_param["disable_text_embed"] = args.disable_text_embed

    supervised_model = None
    if train_tasks and "train" in args.phases:
        if len(train_tasks) > args.num_train_tasks > 0:
            train_tasks = random.sample(train_tasks, args.num_train_tasks)
        if args.use_cache:
            utgs = UTG.load_utgs_from_dir("output/utg_zips")
            browser = CacheBrowser(utgs=utgs)
        else:
            browser = ChromeBrowser(
                wait=args.wait, headless=args.headless, proxy=args.proxy,
                restart_reset=args.restart_reset,
                chrome_path=args.chrome_path,
                extension_path=args.extension_path
            )
        n_train_episodes = model_param["n_train_episodes"] if "n_train_episodes" in model_param else 200
        model_param["n_episodes"] = n_train_episodes

        supervised_model = Qda2pModel(
            data_dir=args.data_dir,
            log_dir=args.log_dir,
            model_dir=args.model_dir,
            **model_param
        )
        supervised_model.train(tasks=train_tasks, browser=browser)
        browser.close()
    if test_tasks and "replay" in args.phases:
        if args.use_cache:
            utgs = UTG.load_utgs_from_dir("output/utg_zips")
            browser = CacheBrowser(utgs=utgs)
        else:
            browser = ChromeBrowser(
                wait=args.wait, headless=args.headless, proxy=args.proxy,
                restart_reset=args.restart_reset,
                chrome_path=args.chrome_path,
                extension_path=args.extension_path
            )
        for test_task in test_tasks:
            env = WebBotEnv(tasks=[test_task], browser=browser)
            env.replay()
            replay_tasklet = test_task.get_tasklet()
            logging.info(" replay finished:\n%s" % replay_tasklet)

    if "eval_reward" in args.phases:
        from environment import State, Action, ACTION_RE
        import traceback
        import numpy as np

        # utgs = UTG.load_utgs_from_dir("output/pv_utg_zips")
        # browser = CacheBrowser(utgs=utgs)

        trace_dir = args.task_dir
        states_dir = os.path.join(trace_dir, "states")
        results_dir = os.path.join(trace_dir, "results")
        if not os.path.exists(states_dir):
            states_dir = os.path.join(trace_dir, "states.zip")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        def load_action(state, action_line):
            m = re.match(ACTION_RE, action_line)
            action_type, value, target_locator = m.group(1), m.group(2), m.group(3)
            target_ele = state.get_element_by_locator(target_locator)
            action = Action(element=target_ele, action_type=action_type, value=value)
            return action

        def compute_reward(task, trace_lines):
            # logging.info(f"compute_reward starts at {datetime.now()}")
            states = []
            actions = []
            # browser.reset(task.start_url)
            state_action_lines = [(line[:(line.find(": "))], line[(line.find(": ") + 2):])
                                  for line in trace_lines]
            current_state_str, action_line = state_action_lines[0]
            current_state = State.load(states_dir, current_state_str)
            actions.append("RESET")
            states.append(current_state)
            task.reset(current_state, update_utg=False)
            last_action = load_action(current_state, action_line)
            actions.append(action_line)
            end_reached = False
            correct_rewards = [0]
            incorrect_rewards = [task.total_reward]
            for state_str, action_line in state_action_lines[1:]:
                current_state = State.load(states_dir, state_str)
                states.append(current_state)
                task.update(last_action, current_state, update_utg=False)
                if task.target_achieved:
                    correct_rewards.append(task.total_reward)
                else:
                    incorrect_rewards.append(task.total_reward)
                if action_line == "END":
                    end_reached = True
                    break
                else:
                    last_action = load_action(current_state, action_line)
            max_correct_reward = max(correct_rewards)
            max_incorrect_reward = max(incorrect_rewards)
            logging.info(f"  task got correct reward {max_correct_reward:6.3f}"
                         f" and incorrect reward {max_incorrect_reward:3.3f}: {task.name}")
            return max_correct_reward, max_incorrect_reward

        def compute_rewards(task, traces):
            correct_rewards = []
            incorrect_rewards = []
            new_traces = []
            for trace in traces:
                if len(trace) == 0:
                    continue
                try:
                    correct_reward, incorrect_reward = compute_reward(task, trace.splitlines())
                    correct_rewards.append(correct_reward)
                    incorrect_rewards.append(incorrect_reward)
                except:
                    # traceback.print_exc()
                    # logging.warning(f"compute_reward failed for {task.task_str}:\n{trace}")
                    pass
            return correct_rewards, incorrect_rewards

        all_traces = {}
        for task_name in os.listdir(trace_dir):
            if not task_name.endswith(".json"):
                continue
            task_file_path = os.path.join(trace_dir, task_name)
            trace_file = open(task_file_path)
            try:
                trace_dict = json.load(trace_file)
            except:
                logging.warning(f"unable to load task: {task_name}")
                continue
            correct_traces = trace_dict["correct_traces"]
            incorrect_traces = trace_dict["incorrect_traces"]
            if len(correct_traces) > 3:
                trace2len = {trace: len(trace.splitlines()) for trace in correct_traces}
                correct_traces = Utils.top_n(trace2len, 3)
            num_incorrect_traces = len(correct_traces) * 10
            if len(incorrect_traces) > num_incorrect_traces:
                incorrect_traces = incorrect_traces[:num_incorrect_traces]

            correct_trace_str = "\n\n".join(correct_traces)
            filtered_incorrect_traces = []
            for trace in incorrect_traces:
                if trace not in correct_trace_str:
                    filtered_incorrect_traces.append(trace)
            incorrect_traces = filtered_incorrect_traces

            if len(correct_traces) > 0 and len(incorrect_traces) > 0:
                all_traces[task_name] = {}
                all_traces[task_name]["correct_traces"] = correct_traces
                all_traces[task_name]["incorrect_traces"] = incorrect_traces

        def get_reward_results(tag):
            logging.info(f"evaluating reward function for configuration: {tag}")
            all_rewards = {}
            for task_name in all_traces:
                logging.info(f" computing reward for {task_name}")
                task_file_path = os.path.join(trace_dir, task_name)
                test_task = Task.load(task_file_path)
                correct_traces = all_traces[task_name]["correct_traces"]
                incorrect_traces = all_traces[task_name]["incorrect_traces"]
                if len(correct_traces) == 0 or len(incorrect_traces) == 0:
                    continue
                correct_rewards, incorrect_rewards = compute_rewards(test_task, correct_traces + incorrect_traces)
                if len(correct_rewards) > 0 and len(incorrect_rewards) > 0:
                    all_rewards[task_name] = {}
                    all_rewards[task_name]["correct_rewards"] = correct_rewards
                    all_rewards[task_name]["incorrect_rewards"] = incorrect_rewards

                    logging.info(test_task.task_str)
                    logging.info(f"correct: {len(correct_traces)} "
                                 f"max: {np.max(correct_rewards)}")
                    logging.info(f"incorrect: {len(incorrect_traces)} "
                                 f"max: {np.max(incorrect_rewards)}")
            result_file_path = os.path.join(results_dir, f"{tag}.json")
            with open(result_file_path, "w") as f:
                json.dump(all_rewards, f, indent=2)

        get_reward_results(args.reward_weights)

    if test_tasks and "collect_pv_traces" in args.phases:
        if args.use_cache:
            utgs = UTG.load_utgs_from_dir("output/utg_zips")
            browser = CacheBrowser(utgs=utgs)
        else:
            browser = ChromeBrowser(
                wait=args.wait, headless=args.headless, proxy=args.proxy,
                restart_reset=args.restart_reset,
                chrome_path=args.chrome_path,
                extension_path=args.extension_path
            )

        pv_trace_dir = os.path.join("output", "pv_traces")
        pv_states_dir = os.path.join(pv_trace_dir, "states")
        if not os.path.exists(pv_trace_dir):
            os.mkdir(pv_trace_dir)
        if not os.path.exists(pv_states_dir):
            os.mkdir(pv_states_dir)

        def get_replay_trace(task, states_dir=None):
            replay_trace = []
            for i, action in enumerate(task.action_history):
                state = task.state_history[i]
                replay_trace.append(f"{state.state_str}: {action.replay_api}")
                if states_dir:
                    state.save(states_dir)
            replay_trace.append(f"{task.state.state_str}: END")
            if states_dir:
                task.state.save(states_dir)
            return "\n".join(replay_trace)

        def explore_task(n_episodes, policy, states_dir=None):
            for episode in range(1, n_episodes + 1):
                env.reset()
                task = env.current_task.snapshot()
                target_achieved = False
                while True:
                    if task.done:
                        break
                    env.render()
                    preferred_actions = env.browser._filter_actions(task.state, task.get_preferred_actions())
                    possible_actions = env.browser._filter_actions(task.state, task.state.possible_actions)
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
                    env.step(action)
                    if task.target_achieved:
                        target_achieved = True
                    task_ = env.current_task.snapshot()
                    task = task_
                replay_trace = get_replay_trace(task, states_dir)
                if target_achieved:
                    correct_traces.append(replay_trace)
                if not target_achieved:
                    incorrect_traces.append(replay_trace)

        task_traces = {}
        for test_task in test_tasks:
            assert isinstance(test_task, Task)
            if not test_task.replayable:
                continue
            task_file_path = os.path.join(pv_trace_dir, test_task.name + "task.json")
            if os.path.exists(task_file_path):
                continue
            try:
                correct_traces = []
                incorrect_traces = []

                env = WebBotEnv(tasks=[test_task], browser=browser)
                env.replay()
                if test_task.target_achieved:
                    correct_traces.append(get_replay_trace(test_task, pv_states_dir))
                explore_task(10, "similarity", pv_states_dir)
                explore_task(10, "random", pv_states_dir)
                explore_task(10, "random_restricted", pv_states_dir)
                explore_task(10, "demo_biased", pv_states_dir)
                GLOBAL_CONFIGS["semantic_similarity"] = True
                explore_task(10, "similarity", pv_states_dir)
                GLOBAL_CONFIGS["semantic_similarity"] = False
                correct_traces = list(set(correct_traces))
                incorrect_traces = list(set(incorrect_traces))
                task_traces[test_task] = [correct_traces, incorrect_traces]
                task_dict = test_task.to_dict(as_demo=True)
                task_dict["correct_traces"] = correct_traces
                task_dict["incorrect_traces"] = incorrect_traces
                with open(task_file_path, "w") as task_file:
                    json.dump(task_dict, task_file, indent=2)
                logging.info(f" collected {len(correct_traces)} correct traces and"
                             f" {len(incorrect_traces)} incorrect traces"
                             f" in task {test_task.name}")
            except:
                logging.info(f" failed to collect traces in task {test_task.name}")

    if test_tasks and "crawl" in args.phases:
        if args.use_cache:
            utgs = UTG.load_utgs_from_dir("output/utg_zips")
            browser = CacheBrowser(utgs=utgs)
        else:
            browser = ChromeBrowser(
                wait=args.wait, headless=args.headless, proxy=args.proxy,
                restart_reset=args.restart_reset,
                chrome_path=args.chrome_path,
                extension_path=args.extension_path
            )
        n_test_episodes = model_param["n_test_episodes"] if "n_test_episodes" in model_param else 100

        model_param["n_episodes"] = n_test_episodes
        if args.explore_policy:
            model_param["explore_policy"] = args.explore_policy
        if args.explore_rate:
            model_param["explore_rate"] = args.explore_rate

        test_results = []
        for test_task in test_tasks:
            # if not test_task.demonstration:
            #     continue
            env = WebBotEnv(tasks=[test_task], browser=browser)
            env.replay()
            demo_tasklet = test_task.get_tasklet()
            logging.info(" demonstration:\n%s" % demo_tasklet)

            if args.algorithm == "dqn":
                agent = TestDQN(
                    data_dir=args.data_dir,
                    log_dir=args.log_dir,
                    model_dir=args.model_dir,
                    supervised_model=supervised_model,
                    **model_param
                )
            elif args.algorithm == "hill_climbing":
                from baseline_agents import HillClimbing
                agent = HillClimbing(
                    data_dir=args.data_dir,
                    log_dir=args.log_dir,
                    model_dir=args.model_dir,
                    **model_param
                )
            elif args.algorithm == "monte_carlo":
                from baseline_agents import MonteCarlo
                agent = MonteCarlo(
                    data_dir=args.data_dir,
                    log_dir=args.log_dir,
                    model_dir=args.model_dir,
                    **model_param
                )
            elif args.algorithm == "random":
                model_param["explore_policy"] = "full_rand"
                model_param["explore_rate"] = 1.0
                agent = TestDQN(
                    data_dir=args.data_dir,
                    log_dir=args.log_dir,
                    model_dir=args.model_dir,
                    supervised_model=supervised_model,
                    **model_param
                )
            else:
                agent = TestQTable(
                    data_dir=args.data_dir,
                    log_dir=args.log_dir,
                    model_dir=args.model_dir,
                    supervised_model=supervised_model,
                    **model_param
                )

            found_tasklets = agent.train(tasks=[test_task], browser=browser)

            top_tasklets = Utils.top_n(found_tasklets, 10, reverse=True)
            post_processed_tasklets = {}
            for i, tasklet in enumerate(top_tasklets):
                original_total_reward, episode, original_final_screen = found_tasklets[tasklet]
                tasklet, total_reward, final_screen = env.post_process(test_task, tasklet)
                if total_reward > original_total_reward:
                    logging.debug("post-processing improved the total reward.")
                post_processed_tasklets[tasklet] = (total_reward, episode, final_screen)
            top_tasklets = Utils.top_n(post_processed_tasklets, 5, reverse=True)
            test_results.append(top_tasklets)

            task_output_dir = None
            if output_dir:
                task_output_dir = os.path.join(output_dir, test_task.name)
                if not os.path.exists(task_output_dir):
                    os.mkdir(task_output_dir)
            test_report = "\n" + "=" * 50 + "\n demonstration:\n%s\n" % demo_tasklet
            for i, tasklet in enumerate(top_tasklets):
                total_reward, episode, final_screen = post_processed_tasklets[tasklet]
                test_report += "-" * 50 + "\n tasklet-%d (episode %d):\n%s\n" % (i, episode, tasklet)
                if task_output_dir:
                    try:
                        final_screen_path = os.path.join(task_output_dir, "final_state_tasklet-%d.png" % i)
                        final_screen.save(final_screen_path)
                    except:
                        pass
            logging.info(test_report)
            if task_output_dir:
                try:
                    task_report_path = os.path.join(task_output_dir, "report.txt")
                    with open(task_report_path, "w") as f:
                        f.write(test_report)
                except:
                    pass

        result_lines = []
        success_ids = []
        for i in range(len(test_results)):
            top_tasklets = test_results[i]
            test_task = test_tasks[i]
            succeed_id = -1
            for i, tasklet in enumerate(top_tasklets):
                if "success:Y" in tasklet:
                    succeed_id = i
            success_ids.append(succeed_id)
            result_lines.append("success:{:s} tasklet:{:d} #episodes:{:3d}  #demo_steps:{:2d}  task:{:60s} {:s}".format(
                "Y" if succeed_id >= 0 else "N",
                succeed_id,
                -1,  # TODO get the episode number
                len(test_task.demonstration),
                test_task.name,
                test_task.task_str)
            )

        success_count = sum([(1 if succeed_id > 0 else 0) for succeed_id in success_ids])
        success_rate = (float(success_count) / len(test_results)) if len(test_results) > 0 else 0
        result_lines.append("Success rate: %.3f" % success_rate)
        overall_report = "Result:\n" + "\n".join(result_lines)
        logging.info(overall_report)
        if output_dir:
            try:
                overall_report_path = os.path.join(output_dir, "overall.txt")
                with open(overall_report_path, "w") as f:
                    f.write(overall_report)
            except:
                pass

    if test_tasks and "analyze_task_complexity" in args.phases:
        browser = ChromeBrowser(
            wait=args.wait, headless=args.headless, proxy=args.proxy,
            restart_reset=args.restart_reset,
            chrome_path=args.chrome_path,
            extension_path=args.extension_path
        )
        for test_task in test_tasks:
            env = WebBotEnv(browser=browser, tasks=[test_task])
            env.analyze_task_complexity(test_task, test_task.demonstration)

    if test_tasks and "build_cache" in args.phases:
        from environment import UTG
        browser = ChromeBrowser(wait=args.wait, headless=args.headless, proxy=args.proxy,
                                restart_reset=args.restart_reset,
                                chrome_path=args.chrome_path,
                                extension_path=args.extension_path)
        for test_task in test_tasks:
            task_category = test_task.name.split("_")[0]
            task_host = Utils.get_host(test_task.start_url)
            logging.info("building cache for %s" % test_task.task_str)
            env = WebBotEnv(tasks=[test_task], browser=browser)

            # save UTG to dir
            utg_dir_path = os.path.join("output", "utgs", "%s_%s_utg" % (task_category, task_host))
            test_task.utg = UTG.load_from_dir(utg_dir_path)
            test_task.utg.save_states = True
            test_task.utg.start_url = test_task.start_url

            logging.info("replaying the demonstration")
            env.replay()
            test_task.utg.save()

            logging.info("exploring other paths: demo_biased strategy")
            env.explore(policy="demo_biased", n_episodes=50)
            test_task.utg.save()

            logging.info("exploring other paths: similarity strategy")
            env.explore(policy="similarity", n_episodes=50)
            test_task.utg.save()

            logging.info("exploring other paths: random strategy")
            env.explore(policy="random", n_episodes=50)
            test_task.utg.save()

            utg_zip_path = os.path.join("output", "utg_zips", "%s_%s_utg" % (task_category, task_host))
            if not os.path.exists(os.path.dirname(utg_zip_path)):
                os.makedirs(os.path.dirname(utg_zip_path))
            shutil.make_archive(base_name=utg_zip_path, format="zip", root_dir=utg_dir_path)

            logging.info("done building cache for %s" % test_task.task_str)
            logging.info("UTG saved to %s" % utg_zip_path)
            # pv_utg_zip_dir = os.path.join("output", "pv_utg_zips")
            # shutil.copy(utg_zip_path, pv_utg_zip_dir)
            env.destroy()


def main():
    args = parse_args()
    config_name = os.path.basename(args.config_path) if args.config_path else ""
    taskset_name = os.path.basename(args.task_dir) if args.task_dir else ""
    job_result_name = "result-{}-{}-{}".format(args.tag, config_name, taskset_name)

    log_handlers = [logging.StreamHandler()]
    job_output_dir = os.path.join("output", job_result_name)
    log_path = os.path.join(job_output_dir, f"{args.tag}_log.txt")
    log_file_handler = logging.FileHandler(log_path, encoding="utf-8", delay=True)
    if args.save_reports:
        if not os.path.exists(job_output_dir):
            os.makedirs(job_output_dir)
        log_handlers.append(log_file_handler)
    else:
        job_output_dir = None

    logging.basicConfig(
        level=logging.INFO,
        handlers=log_handlers,
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
    )

    if os.path.exists(args.config_path):
        model_param = json.load(open(args.config_path))
    else:
        model_param = {
            "feature_cache_size": 5000,
            "n_backup_episodes": 50,
            "n_train_episodes": 200,
            "n_test_episodes": 100
        }
    if args.n_test_episodes > 0:
        model_param["n_test_episodes"] = args.n_test_episodes

    start_time = datetime.now()
    logging.info(args)
    logging.info("Job started")

    GLOBAL_CONFIGS['semantic_similarity'] = args.semantic_similarity
    GLOBAL_CONFIGS['action_restriction'] = args.action_restriction

    for para_config in args.reward_weights.split(","):
        try:
            para_value = float(para_config[3:]) if len(para_config) > 3 else 0
        except:
            para_value = 0
        if para_config.startswith("w1="):
            REWARD_ITEM_WEIGHTS["action_spatial_distance"] = para_value
        if para_config.startswith("w2="):
            REWARD_ITEM_WEIGHTS["action_direction"] = para_value
        if para_config.startswith("w3="):
            REWARD_ITEM_WEIGHTS["similarity_with_non_parameters"] = para_value
        if para_config.startswith("w4="):
            REWARD_ITEM_WEIGHTS["similarity_with_parameters"] = para_value
        if para_config.startswith("semantic"):
            GLOBAL_CONFIGS['semantic_similarity'] = True
        if para_config.startswith("annotation"):
            GLOBAL_CONFIGS['use_annotations'] = False

    train_tasks, test_tasks = load_tasks(
        train_task_res=model_param["train_tasks"] if "train_tasks" in model_param else [],
        test_task_res=model_param["test_tasks"] if "test_tasks" in model_param else [],
        args=args
    )
    train_and_test(
        train_tasks=train_tasks,
        test_tasks=test_tasks,
        args=args,
        model_param=model_param,
        output_dir=job_output_dir
    )

    total_seconds = (datetime.now() - start_time).total_seconds()
    logging.info("Job finished. total_seconds: %.3f." % total_seconds)


if __name__ == "__main__":
    main()
