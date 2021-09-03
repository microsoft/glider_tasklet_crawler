# Glider: A reinforcement learning approach to extract UI scripts from websites

This repository contains the code and data for the models and experiments described in "Glider: A reinforcement learning approach to extract UI scripts from websites" by Yuanchun Li and Oriana Riva, which was accepted at the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2021).

If you use any of the materials in this repository, please cite the following paper.

``` bibtex
@inproceedings{glider:sigir21,
  title = {Glider: A reinforcement learning approach to extract {UI} scripts from websites},
  author = {Yuanchun Li and Oriana Riva},
  booktitle = {44th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2021)},
  year = {2021},
  month = jul,
  address = "Online"
  doi = {10.1145/3404835.3462905}
}
```

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
    - Download the [Chrome web driver](https://chromedriver.chromium.org/downloads) and place it in the `src/resources` directory.
        - The web drivers should be named as:
            - `src/resources/chromedriver_linux64`
            - `src/resources/chromedriver_mac64`
            - `src/resources/chromedriver_win32.exe`
- Install required Python packages
    - `pip install -r requirements.txt`
    - `python -m spacy download en_core_web_lg`

## Usage

To crawl web automation scripts:

1. Add the task definitions in the `data/tasks` directory.
2. Edit the configuration file in the `configs` directory. In the configuration file, the `test_tasks` field is used to specify the task descriptions saved in the `data/tasks` directory.
3. Start crawling using the `run.py` file. An example command line is:

``` bash
cd glider/src
python run.py --config_path ../configs/unit_conversion.json
``` 

## Disclaimer

The code and dataset in this repository are intended to be used for research purposes. Microsoft takes no responsibility for what users use this tool for or for any damages caused from using this code. By downloading and using this software, you agree that you take full responsibility for any damages and liability.

Query examples in the dataset that appear to be attributed to a user or related user contact information are for illustration only and are fictitious. No real association is intended or inferred.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
