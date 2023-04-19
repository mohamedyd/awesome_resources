# Datasets, Pretrained Models, and ML Problems

| |
|-|
| [Large Language Models](#large-language-models) |
| [ML & DL](#ml--dl) |
| [Data-Centric AI](#data-centric-ai) |
| [Datasets](#datasets) |



## Large Language Models

* [Auto-GPT](https://github.com/Significant-Gravitas/Auto-GPT): Auto-GPT is an experimental open-source application showcasing the capabilities of the GPT-4 language model. Its main features are (1) Internet access for searches and information gathering, (2) Long-Term and Short-Term memory management, (3) GPT-4 instances for text generation, (4) Access to popular websites and platforms, and (5) File storage and summarization with GPT-3.5.(article: [Intro to Auto-GPT](https://autogpt.net/autogpt-step-by-step-full-setup-guide/), Videos: [Install Auto-GPT Locally](https://www.youtube.com/watch?app=desktop&v=0m0AbdoFLq4), [Create Your Personal AI Assistant](https://www.youtube.com/watch?app=desktop&v=jn8n212l3PQ)) Note: some practitioners reported that Auto-GPT fails to carry out 80% of the assigned tasks (date: April 17, 2023).

* [Free Dolly: Introducing the World's First Truly Open Instruction-Tuned LLM](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm): Databricks are open-sourcing the entirety of Dolly 2.0, including the training code, the dataset, and the model weights, all suitable for commercial use. This means that any organization can create, own, and customize powerful LLMs that can talk to people, without paying for API access or sharing data with third parties.

* [LLaMA](https://github.com/facebookresearch/llama): LLaMA is a collection of foundation language models ranging from 7B to 65B parameters. The models have been trained on trillions of tokens, and show that it is possible to train state-of-the-art models using publicly available datasets exclusively, without resorting to proprietary and inaccessible datasets. More details can be found in this [blog post](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/).

* [𝐋𝐚𝐧𝐠𝐜𝐡𝐚𝐢𝐧 101: 𝐄𝐚𝐬𝐢𝐞𝐬𝐭 𝐰𝐚𝐲 𝐭𝐨 𝐛𝐮𝐢𝐥𝐝 𝐋𝐋𝐌 𝐚𝐩𝐩𝐬](https://docs.langchain.com/docs/): LangChain is a framework for developing applications powered by language models. Currrent use cases are Personal assistants, Question answering over database(s), Chatbots, Querying tabular data, Interacting with APIs, Model Evaluation. [Git Repo](https://lnkd.in/dSp5n3Sa), [Cookbook by Gregory Kamradt(Easy way to get started)](https://lnkd.in/dqQGMW5u), [Youtube Tutorials](https://lnkd.in/dh3rGuch)

* [GPT4All](https://github.com/nomic-ai/gpt4all): Demo, data and code to train an assistant-style large language model with ~800k GPT-3.5-Turbo Generations based on LLaMa

* [Hugging Face](https://huggingface.co/): A list of datasets and pretrained-models for different ML applications


## ML & DL

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