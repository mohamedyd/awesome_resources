# Datasets, Pretrained Models, and ML Problems

This page contains plenty of resources, including pretrained models, datasets, and ML problems for learning purposes. 

* [Large Language Models](#large-language-models) 
* [ML & DL](#ml--dl) 
* [Data-Centric AI](#data-centric-ai)
* [Datasets](#datasets) 



## Large Language Models

* [H2O LLM studio by H2O.ai](https://github.com/h2oai/h2o-llmstudio): no-code toolkit to finetune large language models for multiple downstream applications. It's equipped with multiple finetuning techniques, such as Low-Rank Adaptation (LoRA) and Reinforecement Learning (RL). It also allows to track and evaluate finetuning experiments using Neptune toolkit and import/export models with Hugging Face

* [Open LLMs](https://github.com/eugeneyan/open-llms): a list of LLMs (Large Language Models) are all licensed for commercial use (e.g., Apache 2.0, MIT, OpenRAIL-M).

* [List of Open Sourced Fine-Tuned Large Language Models (LLM)](https://medium.com/geekculture/list-of-open-sourced-fine-tuned-large-language-models-llm-8d95a2e0dc76): An incomplete list of open-sourced fine-tuned Large Language Models (LLM) you can run locally on your computer. You can have most up and running for less than $100, including training dataset and GPU costs. 

* [Scikit-LLM: Sklearn Meets Large Language Models](https://github.com/iryna-kondr/scikit-llm): Seamlessly integrate powerful language models like ChatGPT into scikit-learn for enhanced text analysis tasks.

* [StarCoder](https://huggingface.co/bigcode/starcoder): StarCoder is the first large model (15B) which is both high performance (beating the like of PaLM, LLaMa, CodeGen or OpenAI code-crushman-001 on code generation) and also trained only on carefully vetted data.

* [Practical Guides for Large Language Models ](https://github.com/Mooler0410/LLMsPracticalGuide): A curated (still actively updated) list of practical guide resources of LLMs

* [Chat with Open Large Language Models](https://chat.lmsys.org/): The Chatbot Arena by FastChat enables sending a query to two LLMs in parallel and getting the answers side by side. The Chatbot Arena currently supports nine open-source LLMs, such as StableLM, LLaMA, Dolly, Alpaca, etc.

* [StructGPT](https://github.com/RUCAIBox/StructGPT): A general framework for Large Language Model to Reason on Structured Data

* [Auto-GPT](https://github.com/Significant-Gravitas/Auto-GPT): Auto-GPT is an experimental open-source application showcasing the capabilities of the GPT-4 language model. Its main features are (1) Internet access for searches and information gathering, (2) Long-Term and Short-Term memory management, (3) GPT-4 instances for text generation, (4) Access to popular websites and platforms, and (5) File storage and summarization with GPT-3.5.(article: [Intro to Auto-GPT](https://autogpt.net/autogpt-step-by-step-full-setup-guide/), Videos: [Install Auto-GPT Locally](https://www.youtube.com/watch?app=desktop&v=0m0AbdoFLq4), [Create Your Personal AI Assistant](https://www.youtube.com/watch?app=desktop&v=jn8n212l3PQ)) Note: some practitioners reported that Auto-GPT fails to carry out 80% of the assigned tasks (date: April 17, 2023).

* [H2O LLM Studio](https://github.com/h2oai/h2o-llmstudio): a framework and no-code GUI designed for fine-tuning state-of-the-art large language models (LLMs). [Colab](https://colab.research.google.com/drive/1-OYccyTvmfa3r7cAquw8sioFFPJcn4R9?usp=sharing&pli=1&authuser=1#scrollTo=a5WqLjn4-chc)

* [HELM](https://crfm.stanford.edu/helm/latest/?): Holistic Evaluation of Language Models (HELM) is a living benchmark that aims to improve the transparency of language models. HELM evaluates 36 LLM models on the same scenarios with the same adaptation strategy (e.g., prompting), allowing for controlled comparisons. 

* [Free Dolly: Introducing the World's First Truly Open Instruction-Tuned LLM](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm): Databricks are open-sourcing the entirety of Dolly 2.0, including the training code, the dataset, and the model weights, all suitable for commercial use. This means that any organization can create, own, and customize powerful LLMs that can talk to people, without paying for API access or sharing data with third parties.

* [LLaMA](https://github.com/facebookresearch/llama): LLaMA is a collection of foundation language models ranging from 7B to 65B parameters. The models have been trained on trillions of tokens, and show that it is possible to train state-of-the-art models using publicly available datasets exclusively, without resorting to proprietary and inaccessible datasets. More details can be found in this [blog post](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/).

* [𝐋𝐚𝐧𝐠𝐜𝐡𝐚𝐢𝐧 101: 𝐄𝐚𝐬𝐢𝐞𝐬𝐭 𝐰𝐚𝐲 𝐭𝐨 𝐛𝐮𝐢𝐥𝐝 𝐋𝐋𝐌 𝐚𝐩𝐩𝐬](https://docs.langchain.com/docs/): LangChain is a framework for developing applications powered by language models. Currrent use cases are Personal assistants, Question answering over database(s), Chatbots, Querying tabular data, Interacting with APIs, Model Evaluation. [Git Repo](https://lnkd.in/dSp5n3Sa), [Cookbook by Gregory Kamradt(Easy way to get started)](https://lnkd.in/dqQGMW5u), [Youtube Tutorials](https://lnkd.in/dh3rGuch)

* [FastChat](https://github.com/lm-sys/FastChat): An open platform for training, serving, and evaluating large language model based chatbots. The platform includes the Vicuña - an open-source chatbot based on Python created by a team of researchers from UC Berkeley, CMU, Stanford, and UC San Diego.

* [GPT4All](https://github.com/nomic-ai/gpt4all): Demo, data and code to train an assistant-style large language model with ~800k GPT-3.5-Turbo Generations based on LLaMa

* [Hugging Face](https://huggingface.co/): A list of datasets and pretrained-models for different ML applications


## ML & DL

* [Phoenix](https://github.com/Arize-ai/phoenix): Phoenix provides MLOps insights at lightning speed with zero-config observability for model drift, performance, and data quality. Phoenix is notebook-first python library that leverages embeddings to uncover problematic cohorts of your LLM, CV, NLP and tabular models.

* [Ludwig](https://github.com/ludwig-ai/ludwig): Ludwig is a declarative machine learning framework that makes it easy to define machine learning pipelines using a simple and flexible data-driven configuration system. Ludwig will build an end-to-end machine learning pipeline automatically, using whatever is explicitly specified in the configuration, while falling back to smart defaults for any parameters that are not.

* [Sktime](https://github.com/sktime/sktime): it is a library for time series analysis in Python. It provides a unified interface for multiple time series learning tasks. Currently, this includes time series classification, regression, clustering, annotation and forecasting. It comes with time series algorithms and scikit-learn compatible tools to build, tune and validate time series models.

* [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor): Tensor2Tensor is a library of deep learning models and datasets designed to make deep learning more accessible and accelerate ML research.

* [Keras Code Example](https://keras.io/examples/): A list of hosted Jupyter notebooks which provide focused demonstrations of vertical deep learning workflows.

* [285+ Machine Learning Projects with Python](https://medium.com/coders-camp/230-machine-learning-projects-with-python-5d0c7abf8265): This article will introduce you to over 290 ML projects solved and explained using Python.

* [Workspaces on FloydHub](https://blog.floydhub.com/workspaces/): A new cloud IDE for deep learning, powered by FloydHub GPUs

* [AI Use Cases by DataRobot](https://pathfinder.datarobot.com/use-cases?industry=manufacturing): Hundreds of AI use cases captured by DataRobost's experts from real-world experiences with customers.

* [Computer Vision Templates](https://roboflow.com/templates): Connect computer vision models to your business logic with our pre-made templates.

* [Darts -- Time Series Made Easy in Python](https://github.com/unit8co/darts): Darts is a Python library for user-friendly forecasting and anomaly detection on time series. It contains a variety of models, from classics such as ARIMA to deep neural networks. 

* [OpenMMLab](https://github.com/open-mmlab): OpenMMLab builds the most influential open-source computer vision algorithm system in the deep learning era. Based on PyTorch, OpenMMLab develops MMEngine to provide universal training and evaluation engine, and MMCV to provide neural network operators and data transforms, which serves as a foundation of the whole project. 

* [Roboflow/Notebooks](https://github.com/roboflow/notebooks): This repository contains Jupyter Notebooks that have been featured in their blog posts and YouTube videos.

* [Machine learning system design pattern](https://mercari.github.io/ml-system-design-pattern/): This repository contains system design patterns for training, serving and operation of machine learning systems in production. 

* [TowardsAI tutorials](https://github.com/towardsai/tutorials): All tutorials, articles, and books listed in this repository are property of Towards AI, Inc.

* [roboflow/notebooks](https://github.com/roboflow/notebooks): This repository contains Jupyter Notebooks about computer vision models that roboflow has featured in their blog posts and YouTube videos.

* [SAM: Segment Anything Model](https://github.com/facebookresearch/segment-anything): This model produces high quality object masks from input prompts such as points or boxes, and it can be used to generate masks for all objects in an image.


## Data-Centric AI

* [Awesome-Data-Centric-AI](https://github.com/daochenzha/data-centric-AI): A curated, but incomplete, list of data-centric AI resources. Data-centric AI is an emerging field that focuses on engineering data to improve AI systems with enhanced data quality and quantity.

* [Data-Centric AI](https://github.com/HazyResearch/data-centric-ai): We're collecting (an admittedly opinionated) list of resources and progress made in data-centric AI, with exciting directions past, present and future.


## Datasets

* [Auctus: A Dataset Search Engine for Data Discovery and Augmentation](https://auctus.vida-nyu.org/): open-source dataset search engine that
has been designed to support data discovery and augmentation. It supports a rich set of discovery queries: in addition to keyword-based search,users can specify spatial and temporal queries, data integration queries (i.e., searching for datasets that can be concatenated to or joined with a query dataset), and they can also pose complex queries that combine multiple constraints.

* [32 datasets to uplift your skills in data science](https://datasciencedojo.com/blog/datasets-data-science-skills/): An archive of 32 datasets to use to practice and improve your skills as a data scientist. The data sets are categorized according to varying difficulty levels to be suitable for everyone.

* [Awesome Public Datasets](https://github.com/awesomedata/awesome-public-datasets): This repository lists topic-centric public data sources in high quality collected from blogs, answers, and user responses. 

* [OpenML Data Repositories](https://openml.github.io/OpenML/Data-Repositories/): This is a list of public dataset repositories

* [AWS Free Datasets](https://aws.amazon.com/marketplace/search/results?trk=8384929b-0eb1-4af3-8996-07aa409646bc&sc_channel=el&FULFILLMENT_OPTION_TYPE=DATA_EXCHANGE&CONTRACT_TYPE=OPEN_DATA_LICENSES&filters=FULFILLMENT_OPTION_TYPE%2CCONTRACT_TYPE): A list of dataset provided by Amazon AWS.
