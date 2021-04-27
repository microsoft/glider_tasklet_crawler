# Glider: A reinforcement learning approach to extract UI scripts from websites

This repository contains the code and data for the models and experiments described in "Glider: A reinforcement learning approach to extract UI scripts from websites" by Yuanchun Li and Oriana Riva, which was accepted at the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2021).

## About

Glider is a tool to extract UI scripts from the web. Given a website URL (e.g., http://www.unitconversion.org/) and a task description (e.g., “convert length 7333 from inch to foot”), Glider explores the website's UI to identify a sequence of UI actions (click, type, select, etc.) to complete the given task.

## In this repository

- `data` directory contains the sample tasks we created for the experiments.
    - Each file is a json file containing the website URL, query, query annotations, etc. To define your own task, simply create a similar task file and fill in the `start_url`, `query_words` and `query_annotations` fields.
    - For each task we have created a demonstration useful to verify whether the extracted UI script is correct. For some tasks, we also include several fields (e.g., `target_url`, `target_text_res`, etc.) used to more reliably verify whether the task completed correctly. However, both the demonstration and target matching rules are not the only indicators of task completion because there might be other ways to complete the task.
- `src` directory contains the code of Glider, including the scripts to crawl tasklets, randomly explore websites, record/replay demonstrations, etc. 
- `config` directory contains sample configuration files to run the tests.

## Setup

- Install Python 3.6
- Install the Chrome browser and web driver
    - Simply install the Chrome web browser application. On a Linux machine, install chromium with `apt install chromium-browser`
    - Download the Chrome web driver from https://chromedriver.chromium.org/downloads and place it in the `src/resources` directory.
        - The web drivers should be named as:
            - `src/resources/chromedriver_linux64`
            - `src/resources/chromedriver_mac64`
            - `src/resources/chromedriver_win32.exe`
- Install required Python packages
    - `pip install -r requirements.txt`
    - `python -m spacy download en_core_web_lg`

## Usage

To crawl web automation scripts:

1. Add the task definitions into the `data/tasks` directory.
2. Edit the configuration file in the `configs` directory. In the config file, the `test_tasks` field is used to specify the task descriptions saved in the `data/tasks` directory.
3. Start crawling using the `run.py` file. An example command line is:
```
cd glider/src
python run.py --config_path ../configs/unit_conversion.json
```


If you use any of the materials in this repository, please cite the following paper.

```
@inproceedings{glider:sigir21,
  title = {Glider: A reinforcement learning approach to extract {UI} scripts from websites},
  author = {Yuanchun Li and Oriana Riva},
  booktitle = {44th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2021)},
  year = {2021},
}
```
