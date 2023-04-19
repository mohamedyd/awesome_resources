# Table of Contents

This repository comprises plenty of resources related to machine learning (ML), machine learning operations (MLOps), python, explainable AI (XAI), reinforcement learning (RL), and other learning resources. 

| | | | | |
|-|-|-|-|-|
| Machine Learning      | [ML Books](ml-books.md) | [ML Articles](ml-blogs-articles.md) | [ML Courses](ml-courses.md)   | [Datasets & ML Models](#datasets-pretrained-models-and-ml-problems) |
| MLOps     		| [MLOps Books](#mlops-books) | [MLOps Blogs & Articles](#mlops-blogs-and-articles) | [Courses](#mlops-courses)      | [MLOps OSS Tools](#mlops-oss-tools) |
| Python		| [Books](#programming-books) | [Language-related Repos](#language-related-repos) | [Articles](#python-articles--notes) | [AI-related Repos](#ai-related-repos) | 
| Explainable AI | [XAI Books](#xai-books) | [XAI Repos & Tools](#xai-repos--tools) | | |
| Reinforcement Learning | [RL Books](#rl-books) | [RL Courses](#rl-courses) | [RL Articles](#rl-articles) | |
| Miscellaneous Resources | [Learning Platforms](#learning-platforms) | [Courses](#courses) | [Books](#useful-books) | [Useful Repositories & Tools](#repos--tools) |


# Machine Learning

## Datasets, Pretrained Models, and ML Problems

* [Auto-GPT](https://github.com/Significant-Gravitas/Auto-GPT): Auto-GPT is an experimental open-source application showcasing the capabilities of the GPT-4 language model. Its main features are (1) Internet access for searches and information gathering, (2) Long-Term and Short-Term memory management, (3) GPT-4 instances for text generation, (4) Access to popular websites and platforms, and (5) File storage and summarization with GPT-3.5.(article: [Intro to Auto-GPT](https://autogpt.net/autogpt-step-by-step-full-setup-guide/), Videos: [Install Auto-GPT Locally](https://www.youtube.com/watch?app=desktop&v=0m0AbdoFLq4), [Create Your Personal AI Assistant](https://www.youtube.com/watch?app=desktop&v=jn8n212l3PQ)) Note: some practitioners reported that Auto-GPT fails to carry out 80% of the assigned tasks (date: April 17, 2023).

* [Free Dolly: Introducing the World's First Truly Open Instruction-Tuned LLM](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm): Databricks are open-sourcing the entirety of Dolly 2.0, including the training code, the dataset, and the model weights, all suitable for commercial use. This means that any organization can create, own, and customize powerful LLMs that can talk to people, without paying for API access or sharing data with third parties.

* [Auctus: A Dataset Search Engine for Data Discovery and Augmentation](https://auctus.vida-nyu.org/): open-source dataset search engine that
has been designed to support data discovery and augmentation. It supports a rich set of discovery queries: in addition to keyword-based search,users can specify spatial and temporal queries, data integration queries (i.e., searching for datasets that can be concatenated to or joined with a query dataset), and they can also pose complex queries that combine multiple constraints.

* [LLaMA](https://github.com/facebookresearch/llama): LLaMA is a collection of foundation language models ranging from 7B to 65B parameters. The models have been trained on trillions of tokens, and show that it is possible to train state-of-the-art models using publicly available datasets exclusively, without resorting to proprietary and inaccessible datasets. More details can be found in this [blog post](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/).

* [Sktime](https://github.com/sktime/sktime): it is a library for time series analysis in Python. It provides a unified interface for multiple time series learning tasks. Currently, this includes time series classification, regression, clustering, annotation and forecasting. It comes with time series algorithms and scikit-learn compatible tools to build, tune and validate time series models.

* [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor): Tensor2Tensor is a library of deep learning models and datasets designed to make deep learning more accessible and accelerate ML research.

* [32 datasets to uplift your skills in data science](https://datasciencedojo.com/blog/datasets-data-science-skills/): An archive of 32 datasets to use to practice and improve your skills as a data scientist. The data sets are categorized according to varying difficulty levels to be suitable for everyone.

* [Keras Code Example](https://keras.io/examples/): A list of hosted Jupyter notebooks which provide focused demonstrations of vertical deep learning workflows.

* [Hugging Face](https://huggingface.co/): A list of datasets and pretrained-models for different ML applications

* [Awesome Public Datasets](https://github.com/awesomedata/awesome-public-datasets): This repository lists topic-centric public data sources in high quality collected from blogs, answers, and user responses. 

* [285+ Machine Learning Projects with Python](https://medium.com/coders-camp/230-machine-learning-projects-with-python-5d0c7abf8265): This article will introduce you to over 290 ML projects solved and explained using Python.

* [AWS Free Datasets](https://aws.amazon.com/marketplace/search/results?trk=8384929b-0eb1-4af3-8996-07aa409646bc&sc_channel=el&FULFILLMENT_OPTION_TYPE=DATA_EXCHANGE&CONTRACT_TYPE=OPEN_DATA_LICENSES&filters=FULFILLMENT_OPTION_TYPE%2CCONTRACT_TYPE): A list of dataset provided by Amazon AWS.

* [Workspaces on FloydHub](https://blog.floydhub.com/workspaces/): A new cloud IDE for deep learning, powered by FloydHub GPUs

* [OpenML Data Repositories](https://openml.github.io/OpenML/Data-Repositories/): This is a list of public dataset repositories

* [AI Use Cases by DataRobot](https://pathfinder.datarobot.com/use-cases?industry=manufacturing): Hundreds of AI use cases captured by DataRobost's experts from real-world experiences with customers.

* [Computer Vision Templates](https://roboflow.com/templates): Connect computer vision models to your business logic with our pre-made templates.

* [Darts -- Time Series Made Easy in Python](https://github.com/unit8co/darts): Darts is a Python library for user-friendly forecasting and anomaly detection on time series. It contains a variety of models, from classics such as ARIMA to deep neural networks. 

* [OpenMMLab](https://github.com/open-mmlab): OpenMMLab builds the most influential open-source computer vision algorithm system in the deep learning era. Based on PyTorch, OpenMMLab develops MMEngine to provide universal training and evaluation engine, and MMCV to provide neural network operators and data transforms, which serves as a foundation of the whole project. 

* [Roboflow/Notebooks](https://github.com/roboflow/notebooks): This repository contains Jupyter Notebooks that have been featured in their blog posts and YouTube videos.

* [Machine learning system design pattern](https://mercari.github.io/ml-system-design-pattern/): This repository contains system design patterns for training, serving and operation of machine learning systems in production. 

* [TowardsAI tutorials](https://github.com/towardsai/tutorials): All tutorials, articles, and books listed in this repository are property of Towards AI, Inc.

* [roboflow/notebooks](https://github.com/roboflow/notebooks): This repository contains Jupyter Notebooks about computer vision models that roboflow has featured in their blog posts and YouTube videos.

* [GPT4All](https://github.com/nomic-ai/gpt4all): Demo, data and code to train an assistant-style large language model with ~800k GPT-3.5-Turbo Generations based on LLaMa

* [Awesome-Data-Centric-AI](https://github.com/daochenzha/data-centric-AI): A curated, but incomplete, list of data-centric AI resources. Data-centric AI is an emerging field that focuses on engineering data to improve AI systems with enhanced data quality and quantity.

* [SAM: Segment Anything Model](https://github.com/facebookresearch/segment-anything): This model produces high quality object masks from input prompts such as points or boxes, and it can be used to generate masks for all objects in an image.

* [𝐋𝐚𝐧𝐠𝐜𝐡𝐚𝐢𝐧 101: 𝐄𝐚𝐬𝐢𝐞𝐬𝐭 𝐰𝐚𝐲 𝐭𝐨 𝐛𝐮𝐢𝐥𝐝 𝐋𝐋𝐌 𝐚𝐩𝐩𝐬](https://docs.langchain.com/docs/): LangChain is a framework for developing applications powered by language models. Currrent use cases are Personal assistants, Question answering over database(s), Chatbots, Querying tabular data, Interacting with APIs, Model Evaluation. [Git Repo](https://lnkd.in/dSp5n3Sa), [Cookbook by Gregory Kamradt(Easy way to get started)](https://lnkd.in/dqQGMW5u), [Youtube Tutorials](https://lnkd.in/dh3rGuch)



# MLOps

## MLOps Books

* [Designing Machine Learning Systems, Chip Huyen](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/): Why does 9/10 models never make it to production? Read this book and compare to how ML is done at most companies, then you know… Highly recommended!

* [Machine Learning Engineering by Andriy Burkov](http://www.mlebook.com/wiki/doku.php?id=start): This book is available as a read-first-buy-later principle.

* [𝗕𝘂𝗶𝗹𝗱𝗶𝗻𝗴 𝗠𝗮𝗰𝗵𝗶𝗻𝗲 𝗟𝗲𝗮𝗿𝗻𝗶𝗻𝗴 𝗣𝗶𝗽𝗲𝗹𝗶𝗻𝗲𝘀, Hannes Hapke & Catherine Nelson](https://www.oreilly.com/library/view/building-machine-learning/9781492053187/?_gl=1*1rzym6j*_ga*MTM4OTgzMTk4MC4xNjczMjEzNzY2*_ga_092EL089CH*MTY3MzQ2NjIyMi4yLjEuMTY3MzQ3MDk0Ni4yOS4wLjA): I bought this book specifically for the chapters on TensorFlow Serving, well worth it! Other chapters are however also interesting to get some insights to the TFX ecosystem.

* [Deep Learning in Production](https://www.amazon.com/gp/product/6180033773/ref=as_li_tl?ie=UTF8&tag=aisummer-20&camp=1789&creative=9325&linkCode=as2&creativeASIN=6180033773&linkId=fb7484e19fc03923fa2ec3012e5629cd): Build, train, deploy, scale and maintain deep learning models. Understand ML infrastructure and MLOps using hands-on examples.

* [𝐁𝐞𝐠𝐢𝐧𝐧𝐢𝐧𝐠 𝐌𝐋𝐎𝐩𝐬 𝐰𝐢𝐭𝐡 𝐌𝐋𝐅𝐥𝐨𝐰](https://lnkd.in/dgiKNgfX): This book guides you through the process of integrating MLOps principles into existing or future projects using MLFlow, operationalizing your models, and deploying them in AWS SageMaker, Google Cloud, and Microsoft Azure.

* [MLOps Engineering at Scale, Carl Osipov](https://www.oreilly.com/library/view/mlops-engineering-at/9781617297762/): This book teaches you how to implement efficient machine learning systems using pre-built services from AWS and other cloud vendors. This easy-to-follow book guides you step-by-step as you set up your serverless ML infrastructure, even if you’ve never used a cloud platform before.
 
* [𝗠𝗮𝗰𝗵𝗶𝗻𝗲 𝗟𝗲𝗮𝗿𝗻𝗶𝗻𝗴 𝗗𝗲𝘀𝗶𝗴𝗻 𝗣𝗮𝘁𝘁𝗲𝗿𝗻𝘀, Valliappa Lakshmanan, Sara Robinson, Michael Munn](https://www.oreilly.com/library/view/machine-learning-design/9781098115777/): An excellent introduction to ML modeling and feature representation for all kinds on different data modalities. I regularly use it as inspiration for data preprocessing and feature engineering.
 
* [Practical Deep Learning at Scale with MLflow, Yong Liu, Dr. Matei Zaharia](https://www.oreilly.com/library/view/practical-deep-learning/9781803241333/): This book focuses on deep learning models and MLflow to develop practical business AI solutions at scale. It also ship deep learning pipelines from experimentation to production with provenance tracking.
 
* [Practical MLOps, Noah Gift & Alfredo Deza](https://www.oreilly.com/library/view/practical-mlops/9781098103002/): The perfect high-level overview of MLOps, buy this book if you do not know where to start or what you want to read about.
 
* 𝗘𝗻𝗴𝗶𝗻𝗲𝗲𝗿𝗶𝗻𝗴 𝗠𝗟𝗢𝗽𝘀, Emmanuel Raj: I can’t put my finger on what this book is about, but there are a lot of golden nuggets on project architecture and software engineering aspects that are needed to create a nice and agile ML lifecycle.
 
* 𝗘𝗳𝗳𝗲𝗰𝘁𝗶𝘃𝗲 𝗗𝗮𝘁𝗮 𝗦𝗰𝗶𝗲𝗻𝗰𝗲 𝗜𝗻𝗳𝗿𝗮𝘀𝘁𝗿𝘂𝗰𝘁𝘂𝗿𝗲, Ville Tuulos: Very nice and easy read suitable for the late nights when sleeping is impossible. Explains the reasoning behind the ML pipelines architectures and is worth reading regardless of if you are interested in Metaflow or not.
 
* 𝗗𝗮𝘁𝗮 𝗦𝗰𝗶𝗲𝗻𝗰𝗲 𝗼𝗻 𝗔𝗪𝗦, Chris Fregly & Antje Barth: Some parts of the book are very nice but some parts feel like it is only code blocks stacked onto code blocks with no explanation of the ”why”. I use it as inspiration on how to use the AWS APIs.
 
* 𝗟𝗲𝗮𝗿𝗻 𝗔𝗺𝗮𝘇𝗼𝗻 𝗦𝗮𝗴𝗲𝗠𝗮𝗸𝗲𝗿, Julien SIMON: Haven’t gotten to finish this one yet, it will keep me occupied for some more weeks. Well written and much better of explaining why things are done in a specific way and what options there are.



## MLOps Blogs and Articles

* [applied-ml](https://github.com/eugeneyan/applied-ml): Curated papers, articles, and blogs on data science & machine learning in production

* [An MLOps story: how Wayflyer creates ML model cards](https://www.evidentlyai.com/blog/how-wayflyer-creates-ml-model-cards): Keep reading to learn more about: ML applications at Wayflyer, how they create ML model cards using open-source tools, what each ML model card contains, how they incorporate data quality and drift analysis, why the model card is a great tool to adopt early in the MLOps journey, and even what Stable Diffusion has to do with it!

* [MLOps 101: The Foundation for Your AI Strategy](https://www.datarobot.com/mlops-101/): This articile nicely introduces the concept of MLOps and its main pillars, including model deployment, model monitoring, model lifecycle management, and production model governance.

* [Machine learning in production by Chip Huyen](https://huyenchip.com/archive/): A set of tutorials, interviews, and courses.

* [Machine Learning Operations](https://ml-ops.org/): This site is very useful to understand the MLOps principles. It provides several articles about interesting topics, such as MLOps stack canvas, State of MLOps (Tools & Frameworks), End-to-End ML Workflow Lifecycle, and Three Levels of ML-based Software.

* [MLOps Guide](https://mlops-guide.github.io/): This site is intended to be a MLOps Guide to help projects and companies to build more reliable MLOps environment. This guide should contemplate the theory behind MLOps and an implementation that should fit for most use cases. 

* [Stanford MLSys Seminar Series](https://mlsys.stanford.edu/): This seminar series takes a look at the frontier of machine learning systems, and how machine learning changes the modern programming stack.

* [Awesome-mlops](https://github.com/visenger/awesome-mlops#mlops-books): An awesome list of references for MLOps, including books recommendation, scientific papers, blogs, courses, and talks about different MLOps topics.

* [Awesome-mlops-tools](https://github.com/kelvins/awesome-mlops#data-processing): A curated list of awesome MLOps tools 

* [MLOps.toys](https://mlops.toys/): A curated list of MLOps projects, tools and resources.

* [Awesome Production Machine Learning](https://github.com/EthicalML/awesome-production-machine-learning): This repository contains a curated list of awesome open source libraries that will help you deploy, monitor, version, scale and secure your production machine learning.

* [Made With ML](https://madewithml.com/): Learn how to combine machine learning with software engineering to develop, deploy & maintain production ML applications. It includes an MLOps course which focuses on both theory and hands-on, It includes a hands-on explanation of concepts and tools such as Testing Data, Code and Model, Optuna's Hyperparameter Tuning, REST API, CI/CD using Github Actions, Feature Store and Monitoring.

* [MLOps Community](https://mlops.community/): Learn about feature stores, machine learning monitoring, and metadata storage and management.

* [Practitioners guide to MLOps: A framework for continuous delivery and automation of machine learning, Google Cloud](https://github.com/mohamedyd/awesome_resources/files/10401885/practitioners_guide_to_mlops_whitepaper.pdf)

* [Data Science Topics](https://github.com/khuyentran1401/Data-science): A collection of useful data science topics along with articles and videos. The repo includes articles about the following topics: MLOps, Testing, Machine Learning, Natural Language Processing, Computer Vision, Time Series, Feature Engineering, Visualization, Productive Tools, Python, linear algebra, statistics, VSCode, among others.

* [Ultimate MLOps Learning Roadmap with Free Learning Resources In 2023](https://pub.towardsai.net/ultimate-mlops-learning-roadmap-with-free-learning-resources-in-2023-3ba7664cb1e9): A comprehensive guide to learning about MLOps, including the key concepts and skills that you need to master.

* [Launching Cookiecutter-MLOps](https://dagshub.com/blog/cookiecutter-mlops-a-production-focused-project-template/): An open-source project scaffolding for machine learning and #MLOps, built on top of Cookiecutter Data Science, and inspired by Shreya Shankar’s amazing tweetstorm.

* [End to End Machine Learning Pipeline With MLOps Tools (MLFlow+DVC+Flask+Heroku+EvidentlyAI+Github Actions)](https://medium.com/@shanakachathuranga/end-to-end-machine-learning-pipeline-with-mlops-tools-mlflow-dvc-flask-heroku-evidentlyai-github-c38b5233778c): This article will show you how to automate the entire machine learning lifecycle with the MLOps tools.

* [A simple solution for monitoring ML systems](https://www.jeremyjordan.me/ml-monitoring/): This blog post aims to provide a simple, open-source solution for monitoring ML systems. 

* [Applied ML](https://github.com/eugeneyan/applied-ml#data-quality): A curated papers, articles, and blogs on data science & ML in production

* [What is a Feature Store? by Feast](https://feast.dev/blog/what-is-a-feature-store/): An article introducing feature stores and their role in MLOps pipelines. 

* [The Base Camp for your Ascent in ML](https://databasecamp.de/en/homepage): In this blog, you will find all the topics to help you master the expedition into the world of Artificial Intelligence.

* [Deep Learning Blogs: 10 Weeks MLOps course](https://www.ravirajag.dev/blog): The goal of the series is to understand the basics of MLOps (model building, monitoring, configurations, testing, packaging, deployment, cicd).

* [A tutorial on building ML and data monitoring dashboards with Evidently and Streamlit](https://www.evidentlyai.com/blog/ml-model-monitoring-dashboard-tutorial): In this tutorial, you will go through an example of creating and customizing an ML monitoring dashboard using the two open-source libraries. 



## MLOps Courses

* [mlops-zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp): Teach practical aspects of productionizing ML services — from collecting requirements to model deployment and monitoring.

* [CS 329S: Machine Learning Systems Design](https://stanford-cs329s.github.io/):  This course aims to provide an iterative framework for developing real-world machine learning systems that are deployable, reliable, and scalable.

* [Full Stack Deep Learning](https://fall2019.fullstackdeeplearning.com/): This course helps you bridge the gap from training machine learning models to deploying AI systems in the real world. An updated version of the course can be found (here)[https://fullstackdeeplearning.com/course/2022/].

* [Machine Learning Model Deployment by Srivatsan Srinivasan](https://www.youtube.com/playlist?list=PL3N9eeOlCrP5PlN1jwOB3jVZE6nYTVswk): This playlist covers techniques and approaches for model deployment on to production. This playlist has right theory and hands on methods required to build production ready models 

* [End to End ML Lifecycle walkthrough by Srivatsan Srinivasan](https://www.youtube.com/watch?v=gc1DGuzJPDA&list=PL3N9eeOlCrP6Y73-dOA5Meso7Dv7qYiUU): The course covers ML lifecycle End-to-EndML, starting from problem identification till model deployment and model monitoring. It provides high level overview of different phases within ML lifecycle while Srivatsan has other video in this channel detailing out individual component.

* [Machine Learning Engineering for Production (MLOps), Andrew Ng, Youtube](https://www.youtube.com/playlist?list=PLkDaE6sCZn6GMoA0wbpJLi3t34Gd8l0aK): Through this course, you will learn about the ML lifecycle and how to deploy ML models. 

* [CS 329S: Machine Learning Systems Design](https://stanford-cs329s.github.io/syllabus.html?r=152co):  This course aims to provide an iterative framework for developing real-world machine learning systems that are deployable, reliable, and scalable.

* [Deep Reinforcement Learning: CS 285 Fall 2020](https://www.youtube.com/playlist?list=PL_iWQOsE6TfURIIhCrlt-wj9ByIVpbfGc): Lectures for UC Berkeley CS 285: Deep Reinforcement Learning.

* [MLOps 8 weeks course](https://github.com/mohamedyd/MLOps-Basics): The goal of the series is to understand the basics of MLOps like model building, monitoring, configurations, testing, packaging, deployment, cicd, etc.

* [Airflow Tutorial for Beginners](https://www.youtube.com/watch?v=K9AnJ9_ZAXE&list=PLwFJcsJ61oujAqYpMp1kdUBcPG0sE0QMT): This 2-hour Airflow Tutorial for Beginners course combines theory explanation and practical demos to help you get started quickly as an absolute beginner. You don’t need any prerequisite to start this course, but basic python knowledge is recommended. 

* [MLOps (Machine Learning Operations) Fundamentals -- Google Cloud (Free)](https://www.coursera.org/lecture/mlops-fundamentals/data-scientists-pain-points-ePAtg): This course introduces participants to MLOps tools and best practices for deploying, evaluating, monitoring and operating production ML systems on Google Cloud.  

* [AI Workflow: AI in Production](https://www.coursera.org/learn/ibm-ai-workflow-ai-production): This course focuses on models in production at a hypothetical streaming media company.  There is an introduction to IBM Watson Machine Learning.  You will build your own API in a Docker container and learn how to manage containers with Kubernetes. 


## MLOps OSS Tools

* [ClearML](https://clear.ml/): with ClearML, you can easily develop, orchestrate, and automate ML workflows at scale. The free plan enables the following tools: experiment management, model repository, artifacts, dataset versioning, pipelines, agent orchestration, CI/CD automation, reports.
 


# Python 

## Python Books

* [Python Crash Course](https://www.amazon.de/-/en/Eric-Matthes-dp-1718502702/dp/1718502702/ref=dp_ob_title_bk):  A Hands-On, Project-Based Introduction to Programming Book by Eric Matthes. This is a comprehensive introduction to Python programming that emphasizes hands-on learning and project-based approach.

* [Clean Code: A Handbook of Agile Software Craftsmanship by Robert C. Martin](https://www.amazon.de/-/en/Robert-Martin/dp/0132350882): The book emphasizes the importance of writing code that is easy to read, understand, and modify, and provides practical guidelines and best practices for achieving this goal. The book covers topics such as naming, functions, comments, formatting, error handling, testing, and refactoring, as well as higher-level concepts such as software design and architecture.

* [Software Engineering for Data Scientists](https://www.manning.com/books/software-engineering-for-data-scientists): The book covers topics such as project planning, version control, testing, documentation, and collaboration, as well as specific tools and techniques for data analysis and modeling.

* [Python Data Science Handbook](https://github.com/jakevdp/PythonDataScienceHandbook): This repository contains the entire Python Data Science Handbook, in the form of (free!) Jupyter notebooks. The book introduces the core libraries essential for working with data in Python: particularly IPython, NumPy, Pandas, Matplotlib, Scikit-Learn, and related packages. 

* [Full Speed Python](https://github.com/joaoventura/full-speed-python): This repo contains a book for self-learners and aims to teach Python using a practical approach. Includes exercises that play a crucial role in bringing Python learners up to speed with Python’s syntax in a short period.

* [Clean Code in Python by Mariano Anaya](https://www.oreilly.com/library/view/clean-code-in/9781788835831/): This book will help you to become a better Python programmer by covering topics like Design Patterns, Software Architecture, Decorators, and more.

* [Python for Data Science: The Ultimate Step-by-Step Guide to Python Programming DF 2023](https://www.reviewslearningfree.com/2023/03/python-for-data-science-ultimate-step.html): This book is here to guide you through the field of data science from the very beginning. You will learn the fundamental skills and tools to support your learning process. If you’re a beginner, this is the book to help you easily understand the basics of data science.

* [Python for Data Analysis, 3E](https://wesmckinney.com/book/): After reading the book, you'll have a solid understanding of the Python data analysis ecosystem, including libraries like NumPy, pandas, Matplotlib and Jupyter.


## Python Repos

* [Complete Python 3 Bootcamp](https://github.com/Pierian-Data/Complete-Python-3-Bootcamp): Course Files for Complete Python 3 Bootcamp Course on Udemy

* [python-patterns](https://github.com/faif/python-patterns): A collection of design patterns and idioms in Python.

* [Python-For-Data-Professionals](https://github.com/hemansnation/Python-For-Data-Professionals): This course is for Data Professionals so that they can learn Python in 12 days for starting their data journey.

* [9 Best GitHub repositories to learn python.pdf, Himanshu Ramchandani](https://github.com/mohamedyd/awesome_resources/files/10401452/9.Best.GitHub.repos.to.learn.python.pdf)

* [Clean-Code-Notes](https://github.com/JuanCrg90/Clean-Code-Notes): This repo contains useful notes from Clean Code book that will help with writing quality code. The repo is well summarised that could be used as a refresher or a quick guide for junior engineers.

* [LeetCode Questions & Answers](https://github.com/ThinamXx/ML..Interview..Preparation): This repo targets preparing you for the machine learning interviews.

* [Playground and Cheatsheet for Learning Python](https://github.com/trekhleb/learn-python): This is a collection of Python scripts that are split by topics and contain code examples with explanations, different use cases and links to further readings.

* [The Algorithms - Python](https://github.com/TheAlgorithms/Python): All algorithms implemented in Python - for education

* [Awesome Python](https://github.com/vinta/awesome-python): Collection of Python frameworks, libraries, tools, books, newsletters, podcasts, and web series dedicated to making Python easier for everyone. The repo has listed over 90 different categories for individual projects or topics including admin panels, data validations, computer vision, algorithms and design patterns, ChatOps tools, and so much more.

* [Project-based-learning](https://github.com/practical-tutorials/project-based-learning#python): A list of programming tutorials in which aspiring software developers learn how to build an application from scratch. These tutorials are divided into different primary programming languages. Tutorials may involve multiple technologies and languages.

* [Python_reference](https://github.com/rasbt/python_reference): The content of this repo covers: Python tips and tutorials, Algorithms and Data Structures, Plotting and Visualization, Data Science, and more. 


## Python Articles & Notes

* [12 Python Decorators to Take Your Code to the Next Level](https://towardsdatascience.com/12-python-decorators-to-take-your-code-to-the-next-level-a910a1ab3e99): This post is a documented list of 12 helpful decorators I regularly use in my projects to extend my code with extra functionalities.


## AI-related Repos

* [the incredible PyTorch](https://github.com/ritchieng/the-incredible-pytorch): This repo is a curated list of tutorials, projects, libraries, videos, papers, books and anything related to the incredible PyTorch.

* [* Efficient Python for Data Scientists](https://github.com/youssefHosni/Efficient-Python-for-Data-Scientists): Learn how to write efficient python code as a data scientist


# Explainable AI

## XAI Books

* [A Guide for Making Black Box Models Explainable (Free ebook)](https://christophm.github.io/interpretable-ml-book/): the focus of the book is on model-agnostic methods for interpreting black box models such as feature importance and accumulated local effects, and explaining individual predictions with Shapley values and LIME. In addition, the book presents methods specific to deep neural networks.

## XAI Repos & Tools

* [ PiML-Toolbox](https://github.com/SelfExplainML/PiML-Toolbox): PiML is a new Python toolbox for interpretable ML model development and validation. Through low-code interface and high-code APIs, PiML supports a growing list of inherently interpretable ML models.



# Reinforcement Learning

## RL Books


* (Introduction to Reinforcement Learning, Free-ebook)[http://incompleteideas.net/book/the-book-2nd.html]: The book is divided into three parts. Part I defines the reinforcement learning problem in terms of Markov decision processes. Part II provides basic solution methods: dynamic programming, Monte Carlo methods, and temporal-difference learning. Part III presents a unified view of the solution methods and incorporates artificial neural networks, eligibility traces, and planning; the two final chapters present case studies and consider the future of reinforcement learning.

## RL Courses

* [Foundations of Deep RL -- 6-lecture series by Pieter Abbeel](https://www.youtube.com/playlist?list=PLwRJQ4m4UJjNymuBM9RdmB3Z9N5-0IlY0)

* [Introduction to reinforcement learning — UCL & Deep Mind by David Silver](https://www.davidsilver.uk/teaching/): This course provides an introduction to the RL framework and covers key algorithms, including value and policy iteration, Q-learning, and SARSA, as well as more recent developments such as deep RL and actor-critic methods.

* [CS234: Reinforcement Learning — Stanford University by Emma Brunskill](https://web.stanford.edu/class/cs234/): This class will provide a solid introduction to the field of reinforcement learning and students will learn about the core challenges and approaches, including generalization and exploration ([Videos](https://www.youtube.com/playlist?list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u)).

* [CS885: Reinforcement Learning — University of Waterloo by Pascal Poupart](https://cs.uwaterloo.ca/~ppoupart/teaching/cs885-spring20/index.html): The course introduces students to the design of algorithms that enable machines to learn based on reinforcements. At the end of the course, students should have the ability to: (1) Model tasks as reinforcement learning problems, (2) Identify suitable algorithms and apply them to different reinforcement learning problems, and (3) Design new reinforcement learning algorithms ([Videos](https://www.youtube.com/playlist?list=PLdAoL1zKcqTXFJniO3Tqqn6xMBBL07EDc)). 

* [CS 285 — Deep Reinforcement Learning- UC Berkeley by Sergey Levine](http://rail.eecs.berkeley.edu/deeprlcourse/): The topics covered in this course are: Introduction to Reinforcement Learning, Policy Gradients, Actor-Critic Algorithms, Value Function Methods, Deep RL with Q-Functions, Advanced Policy Gradients, Optimal Control and Planning, Model-Based Reinforcement Learning, Offline Reinforcement Learning, Reinforcement Learning Theory Basics, Variational Inference and Generative Models, The connection between Inference and Control, Inverse Reinforcement Learning, Meta-Learning and Transfer Learning ([Videos 2022](https://www.youtube.com/playlist?list=PL_iWQOsE6TfX7MaC6C3HcdOf1g337dlC9), [Fall 2019](https://www.youtube.com/playlist?list=PLkFD6_40KJIwhWJpGazJ9VSj9CFMkb79A), [Fall 2020](https://www.youtube.com/playlist?list=PL_iWQOsE6TfURIIhCrlt-wj9ByIVpbfGc))

* [Reinforcement Learning Course at ASU, Spring, 2021 by Dimitri Bertsekas](http://web.mit.edu/dimitrib/www/RLbook.html): The primary emphasis of the course is to encourage graduate research in reinforcement learning through directed reading and interactions with the instructor over zoom. [Videos](https://www.youtube.com/playlist?list=PLmH30BG15SIp79JRJ-MVF12uvB1qPtPzn)


## RL Articles

* [Deep Reinforcement Learning Explained](https://torres.ai/deep-reinforcement-learning-explained-series/): This is a relaxed introductory series with a practical approach that tries to cover the basic concepts in Reinforcement Learning and Deep Learning to begin in the area of Deep Reinforcement Learning.



# Miscellaneous Resources
Each of these resources include plenty of books, courses, articles, and other learning materials in different AI topics. 

## Courses

* [Computer Science courses with video lectures](https://github.com/youssefHosni/cs-video-courses): This repo includes plenty of courses in different CS areas.

* [FCIH OCW](https://www.youtube.com/user/FCIHOCW): Aset of courses, by Dr Waleed Yousef, about different topics in Computer Science.

* [Software engineering Youtube CS106A - Stanford](https://www.youtube.com/watch?v=KkMDCCdjyW8&list=PL84A56BC7F4A1F852): CS106A is an Introduction to the engineering of computer applications emphasizing modern software engineering principles: object-oriented design, decomposition, encapsulation, abstraction, and testing. Uses the Java programming language. Emphasis is on good programming style and the built-in facilities of the Java language.  

* [Big Data Courses (Arabic)](https://github.com/ahmedsami76/AraBigData): long videos covering various topics, such as [Docker](https://github.com/ahmedsami76/AraBigData/blob/main/Docker.ipynb), Git, and Hadoop. [[Youtube Channel](https://www.youtube.com/@bigdata4756)]


## Learning Platforms

* [PyData](https://www.youtube.com/@PyDataTV/about): PyData provides a forum for the international community of users and developers of data analysis tools to share ideas and learn from each other. The global PyData network promotes discussion of best practices, new approaches, and emerging technologies for data management, processing, analytics, and visualization. PyData communities approach data science using many languages, including (but not limited to) Python, Julia, and R.

* [Data-Science-Educational-Resoruces, Youssef Hosni](https://github.com/youssefHosni/Data-Science-Educational-Resoruces): The repository covers all the data science theoretical and practical skills, e.g., data collection, data prepration, computer vision, MLOps, etc.

* [Exploring the art of AI one concept at a time, Aman Chadha](https://aman.ai/): The website covers Distilled AI (courses), Research, Primers, coding, and recommendation lists. 

* [AIMultiple: AI Use Cases & Tools to Grow Your Business](https://research.aimultiple.com/): Plenty of articles about AI use cases and tools. 

* [Learn GIT branching](https://learngitbranching.js.org/): An interactive website for teaching how to deal with Git repositories. 

* [Git Notes for Professionals](https://drive.google.com/file/d/1HlqW-u-K7ThAqpXYGoUVHukbW5dC3a6p/view?usp=share_link): This is an unofficial free book created for educational purposes

* [HPI tele-TASK](https://www.tele-task.de/): Plenty of unseful courses on different domains provided by Hasso-Plattner Institute. You can also find OpenHPI tutorials on [Youtube](https://www.youtube.com/user/xhakil26/playlists).   

* [Reviews learning free](https://www.reviewslearningfree.com/): This website offers a collection of valuable and valuable courses and books on the science of artificial intelligence, especially machine learning.

* [Stanford ML courses](https://explorecourses.stanford.edu/search?view=catalog&filter-coursestatus-Active=on&page=0&catalog=&academicYear=&q=machine+learning&collapse=): plenty of ML courses provided by Stanford


## Useful Books

* [Free Programming Books](https://github.com/EbookFoundation/free-programming-books/blob/main/books/free-programming-books-langs.md): A comperhensive list of free books to learn all programming languages.

* [100+ Best Free Data Science Books for Beginners and Experts](https://www.theinsaneapp.com/2020/12/free-data-science-books-pdf.html): All the books listed are open sourced and are in a mixed order.

* [Software Engineering at Google](https://abseil.io/resources/swe-book/html/toc.html): This book is about the engineering practices utilized at Google to make their codebase sustainable and healthy.


## Repos & Tools

* [ChatPDF](https://www.chatpdf.com/): ChatPDF is a tool that enables users to interact with their PDF documents as if it were a human. 

* [Perplexity AI](https://www.perplexity.ai/): Ask anything tool (similar to chatGPT). It leverages LLMs with scientific papers, blogs, and other resources to provide precise and compact answers. It enables follow up questions, but it does not store the questions/answers, like in the case of chatGPT.

* [Source code for books published by Apress](https://github.com/Apress)

* [𝗔𝗻 𝗔𝗜 𝗖𝗼𝗱𝗶𝗻𝗴 𝗔𝘀𝘀𝗶𝘀𝘁𝗮𝗻𝘁 𝗳𝗼𝗿 𝗗𝗮𝘁𝗮 𝗦𝗰𝗶𝗲𝗻𝘁𝗶𝘀𝘁𝘀](https://codesquire.ai/):  AI code writing assistant for data scientists, engineers, and analysts. Get code completions and suggestions as you type (Chrome extension). 

* [Awesome Open Source](https://awesomeopensource.com/): Find And compare open source projects in different fields and programming languages, e.g., python, machine learning, datasets, deployment, data preprocessing, user interfance, operating systems, and blockchain.   

* [Connected Papers](https://www.connectedpapers.com/): Connected Papers allows you to easily navigate the landscape of papers on a given theme. It also ranks the papers that are most relevant to a given input paper. It's excellent for compiling a 'reading list' when researching a particular topic.

* [Semantic Scholar](https://lnkd.in/dKkGMeXw) Finding & Synthesizing Literature. Get access to 200 million research papers, discover links between topics, get recommendations based on recent searches and generate summaries.


## Articles

* [What we look for in a resume](https://huyenchip.com/2023/01/24/what-we-look-for-in-a-candidate.html): A set of tips for creating a professional resume and cover letter.
