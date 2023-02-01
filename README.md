# Table of Contents

This repository includes plenty of resources related to different domains. 

| | | | | |
|-|-|-|-|-|
| General Resources | [Courses](#courses) | [Learning Platforms](#learning-platforms) | [Books](#books) | |
| Reinforcement Learning | [courses](#rl-courses) | | | |
| Explainable AI | [Books](#xai-books) | | | |
| MLOps     		| [Books](#mlops-books) | [Blogs & Articles](#mlops-blogs-and-articles) | [Courses](#mlops-courses)      | |
| Machine Learning      | [Books](#ml-books) | [Articles](#ml-articles) | [Courses](#ml-courses)   | [Datasets, Pretrained Models, & ML Problems](#datasets-pretrained-models-and-ml-problems) |
| Python		| [Language-related Repos](#language-related-repos) | [AI-related Repos](#ai-related-repos) | | |

# General Resources
Each of these resources include plenty of books, courses, articles, and other learning materials in different AI topics. 

## Courses

* [Computer Science courses with video lectures](https://github.com/youssefHosni/cs-video-courses): This repo includes plenty of courses in different CS areas.

## Learning Platforms

* [Data-Science-Educational-Resoruces, Youssef Hosni](https://github.com/youssefHosni/Data-Science-Educational-Resoruces): The repository covers all the data science theoretical and practical skills, e.g., data collection, data prepration, computer vision, MLOps, etc.

* [Exploring the art of AI one concept at a time, Aman Chadha](https://aman.ai/): The website covers Distilled AI (courses), Research, Primers, coding, and recommendation lists. 

## Books

[Software Engineering at Google](https://abseil.io/resources/swe-book/html/toc.html): This book is about the engineering practices utilized at Google to make their codebase sustainable and healthy.


# Explainable AI

## XAI Books

* [A Guide for Making Black Box Models Explainable (Free ebook)](https://christophm.github.io/interpretable-ml-book/): he focus of the book is on model-agnostic methods for interpreting black box models such as feature importance and accumulated local effects, and explaining individual predictions with Shapley values and LIME. In addition, the book presents methods specific to deep neural networks.


# MLOps

## MLOps Books

* 𝗗𝗲𝘀𝗶𝗴𝗻𝗶𝗻𝗴 𝗠𝗮𝗰𝗵𝗶𝗻𝗲 𝗟𝗲𝗮𝗿𝗻𝗶𝗻𝗴 𝗦𝘆𝘀𝘁𝗲𝗺𝘀, Chip Huyen: Why does 9/10 models never make it to production? Read this book and compare to how ML is done at most companies, then you know… Highly recommended!

* [𝗕𝘂𝗶𝗹𝗱𝗶𝗻𝗴 𝗠𝗮𝗰𝗵𝗶𝗻𝗲 𝗟𝗲𝗮𝗿𝗻𝗶𝗻𝗴 𝗣𝗶𝗽𝗲𝗹𝗶𝗻𝗲𝘀, Hannes Hapke & Catherine Nelson](https://www.oreilly.com/library/view/building-machine-learning/9781492053187/?_gl=1*1rzym6j*_ga*MTM4OTgzMTk4MC4xNjczMjEzNzY2*_ga_092EL089CH*MTY3MzQ2NjIyMi4yLjEuMTY3MzQ3MDk0Ni4yOS4wLjA): I bought this book specifically for the chapters on TensorFlow Serving, well worth it! Other chapters are however also interesting to get some insights to the TFX ecosystem.

* [𝐁𝐞𝐠𝐢𝐧𝐧𝐢𝐧𝐠 𝐌𝐋𝐎𝐩𝐬 𝐰𝐢𝐭𝐡 𝐌𝐋𝐅𝐥𝐨𝐰](https://lnkd.in/dgiKNgfX): This book guides you through the process of integrating MLOps principles into existing or future projects using MLFlow, operationalizing your models, and deploying them in AWS SageMaker, Google Cloud, and Microsoft Azure.

* [MLOps Engineering at Scale, Carl Osipov](https://www.oreilly.com/library/view/mlops-engineering-at/9781617297762/): This book teaches you how to implement efficient machine learning systems using pre-built services from AWS and other cloud vendors. This easy-to-follow book guides you step-by-step as you set up your serverless ML infrastructure, even if you’ve never used a cloud platform before.
 
* [𝗠𝗮𝗰𝗵𝗶𝗻𝗲 𝗟𝗲𝗮𝗿𝗻𝗶𝗻𝗴 𝗗𝗲𝘀𝗶𝗴𝗻 𝗣𝗮𝘁𝘁𝗲𝗿𝗻𝘀, Valliappa Lakshmanan, Sara Robinson, Michael Munn](https://www.oreilly.com/library/view/machine-learning-design/9781098115777/): An excellent introduction to ML modeling and feature representation for all kinds on different data modalities. I regularly use it as inspiration for data preprocessing and feature engineering.
 
*  [Practical Deep Learning at Scale with MLflow, Yong Liu, Dr. Matei Zaharia](https://www.oreilly.com/library/view/practical-deep-learning/9781803241333/): This book focuses on deep learning models and MLflow to develop practical business AI solutions at scale. It also ship deep learning pipelines from experimentation to production with provenance tracking.
 
* 𝗣𝗿𝗮𝗰𝘁𝗶𝗰𝗮𝗹 𝗠𝗟𝗢𝗽𝘀, Noah Gift & Alfredo Deza: The perfect high-level overview of MLOps, buy this book if you do not know where to start or what you want to read about.
 
* 𝗘𝗻𝗴𝗶𝗻𝗲𝗲𝗿𝗶𝗻𝗴 𝗠𝗟𝗢𝗽𝘀, Emmanuel Raj: I can’t put my finger on what this book is about, but there are a lot of golden nuggets on project architecture and software engineering aspects that are needed to create a nice and agile ML lifecycle.
 
* 𝗘𝗳𝗳𝗲𝗰𝘁𝗶𝘃𝗲 𝗗𝗮𝘁𝗮 𝗦𝗰𝗶𝗲𝗻𝗰𝗲 𝗜𝗻𝗳𝗿𝗮𝘀𝘁𝗿𝘂𝗰𝘁𝘂𝗿𝗲, Ville Tuulos: Very nice and easy read suitable for the late nights when sleeping is impossible. Explains the reasoning behind the ML pipelines architectures and is worth reading regardless of if you are interested in Metaflow or not.
 
* 𝗗𝗮𝘁𝗮 𝗦𝗰𝗶𝗲𝗻𝗰𝗲 𝗼𝗻 𝗔𝗪𝗦, Chris Fregly & Antje Barth: Some parts of the book are very nice but some parts feel like it is only code blocks stacked onto code blocks with no explanation of the ”why”. I use it as inspiration on how to use the AWS APIs.
 
* 𝗟𝗲𝗮𝗿𝗻 𝗔𝗺𝗮𝘇𝗼𝗻 𝗦𝗮𝗴𝗲𝗠𝗮𝗸𝗲𝗿, Julien SIMON: Haven’t gotten to finish this one yet, it will keep me occupied for some more weeks. Well written and much better of explaining why things are done in a specific way and what options there are.


## MLOps Blogs and Articles

* [Machine Learning Operations](https://ml-ops.org/): This site is very useful to understand the MLOps principles. It provides several articles about interesting topics, such as MLOps stack canvas, State of MLOps (Tools & Frameworks), End-to-End ML Workflow Lifecycle, and Three Levels of ML-based Software.

* [MLOps Guide](https://mlops-guide.github.io/): This site is intended to be a MLOps Guide to help projects and companies to build more reliable MLOps environment. This guide should contemplate the theory behind MLOps and an implementation that should fit for most use cases. 

* [Stanford MLSys Seminar Series](https://mlsys.stanford.edu/): This seminar series takes a look at the frontier of machine learning systems, and how machine learning changes the modern programming stack.

* [Awesome-mlops](https://github.com/visenger/awesome-mlops#mlops-books): An awesome list of references for MLOps, including books recommendation, scientific papers, blogs, courses, and talks about different MLOps topics.

* [Awesome-mlops-tools](https://github.com/kelvins/awesome-mlops#data-processing): A curated list of awesome MLOps tools 

* [MLOps.toys](https://mlops.toys/): A curated list of MLOps projects, tools and resources.

* [Awesome Production Machine Learning](https://github.com/EthicalML/awesome-production-machine-learning): This repository contains a curated list of awesome open source libraries that will help you deploy, monitor, version, scale and secure your production machine learning.

* [Made With ML](https://madewithml.com/): Learn how to combine machine learning with software engineering to develop, deploy & maintain production ML applications.

* [MLOps Community](https://mlops.community/): Learn about feature stores, machine learning monitoring, and metadata storage and management.

* [Practitioners guide to MLOps: A framework for continuous delivery and automation of machine learning, Google Cloud](https://github.com/mohamedyd/awesome_resources/files/10401885/practitioners_guide_to_mlops_whitepaper.pdf)

* [Data Science Topics](https://github.com/khuyentran1401/Data-science): A collection of useful data science topics along with articles and videos. The repo includes articles about the following topics: MLOps, Testing, Machine Learning, Natural Language Processing, Computer Vision, Time Series, Feature Engineering, Visualization, Productive Tools, Python, linear algebra, statistics, VSCode, among others.

* [Ultimate MLOps Learning Roadmap with Free Learning Resources In 2023](https://pub.towardsai.net/ultimate-mlops-learning-roadmap-with-free-learning-resources-in-2023-3ba7664cb1e9): A comprehensive guide to learning about MLOps, including the key concepts and skills that you need to master.

* [Launching Cookiecutter-MLOps](https://dagshub.com/blog/cookiecutter-mlops-a-production-focused-project-template/): An open-source project scaffolding for machine learning and #MLOps, built on top of Cookiecutter Data Science, and inspired by Shreya Shankar’s amazing tweetstorm.

* [End to End Machine Learning Pipeline With MLOps Tools (MLFlow+DVC+Flask+Heroku+EvidentlyAI+Github Actions)](https://medium.com/@shanakachathuranga/end-to-end-machine-learning-pipeline-with-mlops-tools-mlflow-dvc-flask-heroku-evidentlyai-github-c38b5233778c): This article will show you how to automate the entire machine learning lifecycle with the MLOps tools.

* [A simple solution for monitoring ML systems](https://www.jeremyjordan.me/ml-monitoring/): This blog post aims to provide a simple, open-source solution for monitoring ML systems. 

* [Applied ML](https://github.com/eugeneyan/applied-ml#data-quality): A curated papers, articles, and blogs on data science & ML in production


## MLOps Courses

* [Machine Learning Engineering for Production (MLOps), Andrew Ng, Youtube](https://www.youtube.com/playlist?list=PLkDaE6sCZn6GMoA0wbpJLi3t34Gd8l0aK): Through this course, you will learn about the ML lifecycle and how to deploy ML models. 

* (CS 329S: Machine Learning Systems Design)[https://stanford-cs329s.github.io/syllabus.html?r=152co]:  This course aims to provide an iterative framework for developing real-world machine learning systems that are deployable, reliable, and scalable.



# Machine Learning

## ML Books

* 𝑼𝒏𝒅𝒆𝒓𝒔𝒕𝒂𝒏𝒅𝒊𝒏𝒈 𝑩𝒖𝒔𝒔𝒊𝒏𝒆𝒔 𝑷𝒓𝒐𝒃𝒍𝒆𝒎
	* [𝐃𝐚𝐭𝐚 𝐒𝐜𝐢𝐞𝐧𝐜𝐞 𝐟𝐨𝐫 𝐁𝐮𝐬𝐢𝐧𝐞𝐬𝐬](https://lnkd.in/dmaY5zwG): This book walks you through the “data-analytic thinking” necessary for extracting useful knowledge and business value from the data you collect.
	* [𝐀𝐩𝐩𝐫𝐨𝐚𝐜𝐡𝐢𝐧𝐠 (𝐀𝐥𝐦𝐨𝐬𝐭) 𝐀𝐧𝐲 𝐌𝐚𝐜𝐡𝐢𝐧𝐞 𝐋𝐞𝐚𝐫𝐧𝐢𝐧𝐠 𝐏𝐫𝐨𝐛𝐥𝐞𝐦](https://lnkd.in/dYE5Jsn3): This book is for people who have some theoretical knowledge of ML and deep learning and want to dive into applied ML. The book does not explain the algorithms but is more oriented toward how and what should you use to solve ML and deep learning problems.

* 𝑫𝒂𝒕𝒂 𝑪𝒐𝒍𝒍𝒆𝒄𝒕𝒊𝒐𝒏
	* [𝐖𝐞𝐛 𝐒𝐜𝐫𝐚𝐩𝐢𝐧𝐠 𝐰𝐢𝐭𝐡 𝐏𝐲𝐭𝐡𝐨𝐧: 𝐂𝐨𝐥𝐥𝐞𝐜𝐭𝐢𝐧𝐠 𝐃𝐚𝐭𝐚 𝐟𝐫𝐨𝐦 𝐭𝐡𝐞 𝐌𝐨𝐝𝐞𝐫𝐧 𝐖𝐞𝐛](https://lnkd.in/d5Mq9ZXC): This practical book not only introduces you to web scraping but also serves as a comprehensive guide to scraping almost every type of data from the modern web.
	* [𝐒𝐐𝐋 𝐟𝐨𝐫 𝐃𝐚𝐭𝐚 𝐒𝐜𝐢𝐞𝐧𝐜𝐞](https://lnkd.in/dXnBVhZG): This textbook explains SQL within the context of data science and introduces the different parts of SQL as they are needed for the tasks usually carried out during data analysis, e.g., data loading, cleaning, and pre-processing.

* 𝑫𝒂𝒕𝒂 Exploration & 𝑷𝒓𝒆𝒑𝒂𝒓𝒂𝒕𝒊𝒐𝒏
	* [𝐁𝐚𝐝 𝐃𝐚𝐭𝐚](https://lnkd.in/dgbd-uiJ): This book is a collection of essays by 19 ML practitioners and is full of useful nuggets on data preparation and management.
	* [𝐏𝐲𝐭𝐡𝐨𝐧 𝐅𝐞𝐚𝐭𝐮𝐫𝐞 𝐄𝐧𝐠𝐢𝐧𝐞𝐞𝐫𝐢𝐧𝐠 𝐂𝐨𝐨𝐤𝐛𝐨𝐨𝐤](https://lnkd.in/d9XNqMD5): This cookbook takes the struggle out of feature engineering by showing you how to use open-source Python libraries to accelerate the process via a plethora of practical, hands-on recipes.
	* [𝐏𝐫𝐚𝐜𝐭𝐢𝐜𝐚𝐥 𝐒𝐭𝐚𝐭𝐢𝐬𝐭𝐢𝐜𝐬 𝐅𝐨𝐫 𝐃𝐚𝐭𝐚 𝐒𝐜𝐢𝐞𝐧𝐭𝐢𝐬𝐭𝐬](https://lnkd.in/d7bWPst8): This book covers a wide variety of statistical procedures used in data science. It dives into more complex statistical concepts with a more mathematical explanation compared to the other books.
	* [𝐇𝐚𝐧𝐝𝐬-𝐎𝐧 𝐄𝐱𝐩𝐥𝐨𝐫𝐚𝐭𝐨𝐫𝐲 𝐃𝐚𝐭𝐚 𝐀𝐧𝐚𝐥𝐲𝐬𝐢𝐬 𝐰𝐢𝐭𝐡 𝐏𝐲𝐭𝐡𝐨𝐧](https://lnkd.in/dY_-WbS2): This book will help you gain practical knowledge of the main pillars of EDA — data cleaning, data preparation, data exploration, and data visualization

* 𝑴𝒐𝒅𝒆𝒍𝒊𝒏𝒈
	* [𝐇𝐚𝐧𝐝𝐬-𝐎𝐧 𝐌𝐚𝐜𝐡𝐢𝐧𝐞 𝐋𝐞𝐚𝐫𝐧𝐢𝐧𝐠 𝐰𝐢𝐭𝐡 𝐒𝐜𝐢𝐤𝐢𝐭-𝐋𝐞𝐚𝐫𝐧 𝐚𝐧𝐝 𝐓𝐞𝐧𝐬𝐨𝐫𝐅𝐥𝐨𝐰, 3rd Edition](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/): The book has a unique approach. It usually starts with a high-level description of different ML concepts to give a general idea; then you go through hands-on coding with Python libraries. It breaks down the algorithms in a nice way that balances the needed theory and mathematics, and applied concepts.
	* [Deep Learning with TensorFlow and Keras - Third Edition Deep Learning with TensorFlow and Keras - Third Edition](https://www.oreilly.com/library/view/deep-learning-with/9781803232911/): This book teaches you how to build cutting edge machine and deep learning systems for the lab, production, and mobile devices.
	* [Neural Networks and Deep Learning, Free online Deep Learning](http://neuralnetworksanddeeplearning.com/index.html): The book is to help you master the core concepts of neural networks, including modern techniques for deep learning. After working through the book you will have written code that uses neural networks and deep learning to solve complex pattern recognition problems. And you will have a foundation to use neural networks and deep learning to attack problems of your own devising.
	* [Probabilistic Machine Learning: An Introduction, Free ebook](https://probml.github.io/pml-book/book1.html): This book offers a detailed and up-to-date introduction to ML (including deep learning) through the unifying lens of probabilistic modeling and Bayesian decision theory. The book covers mathematical background (including linear algebra and optimization), basic supervised learning (including linear and logistic regression and deep neural networks), as well as more advanced topics (including transfer learning and unsupervised learning). 

* 𝑫𝒂𝒕𝒂 𝑺𝒕𝒐𝒓𝒚𝒕𝒆𝒍𝒍𝒊𝒏𝒈
	* [𝐒𝐭𝐨𝐫𝐲𝐭𝐞𝐥𝐥𝐢𝐧𝐠 𝐰𝐢𝐭𝐡 𝐃𝐚𝐭𝐚](https://lnkd.in/dzrQg7VQ): This book teaches you the fundamentals of data visualization and how to communicate effectively with data.
 	* [𝐈𝐧𝐭𝐞𝐫𝐚𝐜𝐭𝐢𝐯𝐞 𝐃𝐚𝐬𝐡𝐛𝐨𝐚𝐫𝐝𝐬 𝐚𝐧𝐝 𝐃𝐚𝐭𝐚 𝐀𝐩𝐩𝐬 𝐰𝐢𝐭𝐡 𝐏𝐥𝐨𝐭𝐥𝐲 𝐚𝐧𝐝 𝐃𝐚𝐬𝐡](https://lnkd.in/dfVDeVjY): This book helps you to develop complete data apps and interactive dashboards without JavaScript. Through this book, you will be able to explore the functionalities of Dash for visualizing data in different ways.


## ML Articles

* [From Zero to AI Research Scientist Full Resources Guide](https://github.com/ahmedbahaaeldin/From-0-to-Research-Scientist-resources-guide): A guide for anyone with basic programming knowledge or a computer science background interested in becoming a Research Scientist with 🎯 on Deep Learning and NLP.

* [Papers With Code: The latest in Machine Learning](https://paperswithcode.com/): A list of machine learning papers, their datasets, and their source code.

* [labml.ai Annotated PyTorch Paper Implementations](https://nn.labml.ai/): A collection of simple PyTorch implementations of neural networks and related algorithms. These implementations are documented with explanations, and the website renders these as side-by-side formatted notes.

* [Your Guide to Data Quality Management](https://www.scnsoft.com/blog/guide-to-data-quality-management): tips on how a company can measure and improve the quality of their data.

* [Deep Learning Tuning Playbook, Google & Harvard](https://github.com/google-research/tuning_playbook): If you have ever wondered about the process of hyperparameter tuning, this guide is for you. 

* [Transformer models: an introduction and catalog — 2023 Edition](https://amatriain.net/blog/transformer-models-an-introduction-and-catalog-2d1e9039f376/): A Catalog of Transformer AI models.

## ML Courses

* [openHPI Self-paced Courses](https://open.hpi.de/courses?q=&channel=&lang=&topic=Big+Data+and+AI&level=): plenty of courses in English and German related to ML, knowledge graphs, big data management, operating systems, and programming. 

* [Applied Machine Learning (Cornell Tech CS 5787, Fall 2020)](https://www.youtube.com/playlist?list=PL2UML_KCiC0UlY7iCQDSiGDMovaupqc83): Starting from the very basics, the course covers all of the most important ML algorithms and how to apply them in practice. The slides are also Jupyter notebooks with programmatically generated figures (GitHub)[https://github.com/kuleshov/cornell-cs5785-2020-applied-ml]. 

* [Machine Learning Course Notes](https://github.com/dair-ai/ML-Course-Notes): This repository shares videos and lecture notes on several topics related to ML, NLP, Transformers, and Deep Reinforcement Learning.

* [𝐈𝐧𝐭𝐫𝐨𝐝𝐮𝐜𝐭𝐢𝐨𝐧 𝐭𝐨 𝐌𝐚𝐜𝐡𝐢𝐧𝐞 𝐋𝐞𝐚𝐫𝐧𝐢𝐧𝐠 — 𝐔𝐂 𝐁𝐞𝐫𝐤𝐞𝐥𝐞𝐲](https://lnkd.in/dChzX6dZ)

* [𝐈𝐧𝐭𝐫𝐨𝐝𝐮𝐜𝐭𝐢𝐨𝐧 𝐭𝐨 𝐌𝐚𝐜𝐡𝐢𝐧𝐞 𝐋𝐞𝐚𝐫𝐧𝐢𝐧𝐠 — 𝐂𝐚𝐫𝐧𝐞𝐠𝐢𝐞 𝐌𝐞𝐥𝐥𝐨𝐧 𝐔𝐧𝐢𝐯𝐞𝐫𝐬𝐢𝐭𝐲](https://lnkd.in/dH8ktatw)

* [𝐌𝐚𝐜𝐡𝐢𝐧𝐞 𝐋𝐞𝐚𝐫𝐧𝐢𝐧𝐠 —  𝐒𝐭𝐚𝐧𝐟𝐨𝐫𝐝 𝐔𝐧𝐢𝐯𝐞𝐫𝐬𝐢𝐭𝐲](https://lnkd.in/d4FzSKpJ)

* [𝐌𝐚𝐜𝐡𝐢𝐧𝐞 𝐋𝐞𝐚𝐫𝐧𝐢𝐧𝐠 & 𝐃𝐚𝐭𝐚 𝐌𝐢𝐧𝐢𝐧𝐠 — 𝐂𝐚𝐥𝐭𝐞𝐜𝐡](https://lnkd.in/dUhbEyBx)

* [𝐋𝐞𝐚𝐫𝐧𝐢𝐧𝐠 𝐟𝐫𝐨𝐦 𝐃𝐚𝐭𝐚 — 𝐂𝐚𝐥𝐭𝐞𝐜𝐡](https://lnkd.in/d4zZZJ5h)

* [𝐌𝐚𝐜𝐡𝐢𝐧𝐞 𝐋𝐞𝐚𝐫𝐧𝐢𝐧𝐠 𝐟𝐨𝐫 𝐈𝐧𝐭𝐞𝐥𝐥𝐢𝐠𝐞𝐧𝐭 𝐒𝐲𝐬𝐭𝐞𝐦𝐬  —  𝐂𝐨𝐫𝐧𝐞𝐥𝐥 𝐔𝐧𝐢𝐯𝐞𝐫𝐬𝐢𝐭𝐲](https://lnkd.in/dtSjQ22i)

* [𝐋𝐚𝐫𝐠𝐞 𝐒𝐜𝐚𝐥𝐞 𝐌𝐚𝐜𝐡𝐢𝐧𝐞 𝐋𝐞𝐚𝐫𝐧𝐢𝐧𝐠  — 𝐔𝐧𝐢𝐯𝐞𝐫𝐬𝐢𝐭𝐲 𝐨𝐟 𝐓𝐨𝐫𝐨𝐧𝐭𝐨](https://lnkd.in/dv8-7EFE)

* [𝐌𝐚𝐜𝐡𝐢𝐧𝐞 𝐋𝐞𝐚𝐫𝐧𝐢𝐧𝐠 𝐰𝐢𝐭𝐡 𝐋𝐚𝐫𝐠𝐞 𝐃𝐚𝐭𝐚𝐬𝐞𝐭𝐬— 𝐂𝐚𝐫𝐧𝐞𝐠𝐢𝐞 𝐌𝐞𝐥𝐥𝐨𝐧 𝐔𝐧𝐢𝐯𝐞𝐫𝐬𝐢𝐭𝐲](https://lnkd.in/dGaN3p53)

* [𝐅𝐨𝐮𝐧𝐝𝐚𝐭𝐢𝐨𝐧𝐬 𝐨𝐟 𝐌𝐚𝐜𝐡𝐢𝐧𝐞 𝐋𝐞𝐚𝐫𝐧𝐢𝐧𝐠 𝐚𝐧𝐝 𝐒𝐭𝐚𝐭𝐢𝐬𝐭𝐢𝐜𝐚𝐥 𝐈𝐧𝐟𝐞𝐫𝐞𝐧𝐜𝐞 — 𝐂𝐚𝐥𝐭𝐞𝐜𝐡](https://lnkd.in/d9J9iksZ)

* [𝐒𝐭𝐚𝐭𝐢𝐬𝐭𝐢𝐜𝐚𝐥 𝐋𝐞𝐚𝐫𝐧𝐢𝐧𝐠 — 𝐒𝐭𝐚𝐧𝐟𝐨𝐫𝐝 𝐔𝐧𝐢𝐯𝐞𝐫𝐬𝐢𝐭𝐲](https://lnkd.in/dKttsi_3)

* [Rasa Algorithm Whiteboard - Transformers & Attention (Large Language Models)](https://www.youtube.com/watch?v=yGTUuEx3GkA&list=PL75e0qA87dlG-za8eLI6t0_Pbxafk-cxb&index=10): This Youtube course provides an explanation of the attention mechanisms.

* [Neural networks for NLP (CMU CS 11-747, Spring 2021), Carnegie Mellon University](https://www.phontron.com/class/nn4nlp2021/schedule.html):  It covers major topics in neural networks modeling and training
	- Language modeling and training tricks
	- Neural networks building
	- Recurrent neural networks
	- Attention mechanism and efficiency tricks
	- Contextual word representations
	- Debugging neural networks
	- Model interpretation
	- Trees and Graphs
	- Reinforcement learning for structured prediction
	- Knowledge bases with neural networks
	- Adversarial methods and advanced search algorithms

* [Learn Computer Vision From Top Universities](https://medium.com/mlearning-ai/learn-computer-vision-from-top-universities-bb6019be74d2): the article provides courses in several domains, including image and signal processing, computer vision, machine learning and deep learning for computer vision, programming for Computer Vision, and Photogrammetry.

* [Learn AI from Top Universities Through these 10 Courses](https://pub.towardsai.net/learn-ai-from-top-universities-through-these-10-courses-13e7a8d3957b): This article introduces ten courses from top universities that will help you to have a better understanding of different concepts in AI and ML.


## Datasets, Pretrained Models, and ML Problems

* [32 datasets to uplift your skills in data science](https://datasciencedojo.com/blog/datasets-data-science-skills/): An archive of 32 datasets to use to practice and improve your skills as a data scientist. The data sets are categorized according to varying difficulty levels to be suitable for everyone.

* [Keras Code Example](https://keras.io/examples/): A list of hosted Jupyter notebooks which provide focused demonstrations of vertical deep learning workflows.

* [Hugging Face](https://huggingface.co/): A list of datasets and pretrained-models for different ML applications

* [Awesome Public Datasets](https://github.com/awesomedata/awesome-public-datasets): This repository lists topic-centric public data sources in high quality collected from blogs, answers, and user responses. 

* [285+ Machine Learning Projects with Python](https://medium.com/coders-camp/230-machine-learning-projects-with-python-5d0c7abf8265): This article will introduce you to over 290 ML projects solved and explained using Python.

* [AWS Free Datasets](https://aws.amazon.com/marketplace/search/results?trk=8384929b-0eb1-4af3-8996-07aa409646bc&sc_channel=el&FULFILLMENT_OPTION_TYPE=DATA_EXCHANGE&CONTRACT_TYPE=OPEN_DATA_LICENSES&filters=FULFILLMENT_OPTION_TYPE%2CCONTRACT_TYPE): A list of dataset provided by Amazon AWS.



# Reinforcement Learning

## RL Books

* (Introduction to Reinforcement Learning, Free-ebook)[http://incompleteideas.net/book/the-book-2nd.html]: The book is divided into three parts. Part I defines the reinforcement learning problem in terms of Markov decision processes. Part II provides basic solution methods: dynamic programming, Monte Carlo methods, and temporal-difference learning. Part III presents a unified view of the solution methods and incorporates artificial neural networks, eligibility traces, and planning; the two final chapters present case studies and consider the future of reinforcement learning.

## RL Courses

* [Learn Reinforcement Learning from Top Universities](https://pub.towardsai.net/learn-reinforcement-learning-from-top-universities-54f4fb1eab57): This article provides an overview of some of the best RL programs available, including what you can expect to learn and how you can get started.

## RL Articles

* [Deep Reinforcement Learning Explained](https://torres.ai/deep-reinforcement-learning-explained-series/): This is a relaxed introductory series with a practical approach that tries to cover the basic concepts in Reinforcement Learning and Deep Learning to begin in the area of Deep Reinforcement Learning.




# Python 

## Language-related Repos

* [9 Best GitHub repositories to learn python.pdf, Himanshu Ramchandani](https://github.com/mohamedyd/awesome_resources/files/10401452/9.Best.GitHub.repos.to.learn.python.pdf)

* [Clean-Code-Notes](https://github.com/JuanCrg90/Clean-Code-Notes): This repo contains useful notes from Clean Code book that will help with writing quality code. The repo is well summarised that could be used as a refresher or a quick guide for junior engineers.

* [LeetCode Questions & Answers](https://github.com/ThinamXx/ML..Interview..Preparation): This repo targets preparing you for the machine learning interviews.

## AI-related Repos

* [the incredible PyTorch](https://github.com/ritchieng/the-incredible-pytorch): This repo is a curated list of tutorials, projects, libraries, videos, papers, books and anything related to the incredible PyTorch.



# Unorganized Resources

* Source code for books published by Apress: https://github.com/Apress
* End to End ML Overview		https://bit.ly/3xl9kDk
* Meta-learning and multi-task	Stanford	https://bit.ly/2S8aCBT
* RL 	https://bit.ly/3ve28ry
* Data Engineering Specialization Coursera	https://bit.ly/3v9FxMf				
* MLOps Fundamentals		Coursera	https://bit.ly/2QokZ3G				
* ML Deployment	https://bit.ly/319mOmV				
* Production ML Systems		Google		https://bit.ly/3liWOP6				
* MLOps Fundamentals		Google		https://bit.ly/30RqGbZ				
* AI Workflow: AI in Production	IBM		https://bit.ly/30QFlnU				
* Software engineering		Youtube		CS106A - Stanford			
* Smart Data Discovery Lux	Online		https://cutt.ly/IcO7Ijx
* ML		https://machinelearningmastery.com/start-here/  
* Matplotlib			Online 		https://bit.ly/3rS9pLY				
* From-0-to-Research-Scientist	Github		https://bit.ly/3sQZZ3E				
* ML system design patterns	Github		https://bit.ly/3rRQe4L					
* Statistical ML			Youtube		https://bit.ly/2SH6NUz				

* The Base Camp for your Ascent in Machine Learning (ML simply explained)
https://databasecamp.de/en/homepage


* 𝗔𝗻 𝗔𝗜 𝗖𝗼𝗱𝗶𝗻𝗴 𝗔𝘀𝘀𝗶𝘀𝘁𝗮𝗻𝘁 𝗳𝗼𝗿 𝗗𝗮𝘁𝗮 𝗦𝗰𝗶𝗲𝗻𝘁𝗶𝘀𝘁𝘀! 
https://codesquire.ai/


* comprehensive list of MLOps resources
https://github.com/khuyentran1401/Data-science#mlops


* Efficient Python for Data Scientists
https://github.com/youssefHosni/Efficient-Python-for-Data-Scientists

* Github crash course
https://lnkd.in/dCHHf_Ax

Kimia Lab: Machine Intelligence
https://www.youtube.com/playlist?list=PLZWvneBOrhoEWyByqPli18AScodr_MzEK



Deep Learning Drizzle
https://deep-learning-drizzle.github.io/?fbclid=IwAR0I_rx1toiamVdLpveDSaTOVUrb8zi0RjbBhGcslObwTrflR-ajDtoa7vs

ML Course
http://ciml.info 

Introduction to Machine Learning, Deep learning using Pytorch
https://sebastianraschka.com/blog/2021/ml-course.html, https://www.youtube.com/c/SebastianRaschka/playlists

Deep Learning course at VU university of Amsterdam
https://dlvu.github.io/

TowardsAI tutorials:
https://github.com/towardsai/tutorials



Machine learning Harvard Free Course
https://cs50.harvard.edu/ai/2020/

stanford course on ML:
https://cs229.stanford.edu/syllabus-summer2020.html


Stanford CS330: Multi-Task and Meta-Learning, 2019
https://www.youtube.com/watch?v=0rZtSwNOTQo&list=PLoROMvodv4rMC6zfYmnD7UG3LVvwaITY5&index=1

MLOps resources:

https://www.ravirajag.dev/blog (10 Weeks course)

https://ml-ops.org/ (articles)

Reading Notes and Slides from Chip Huyen's CS 392S(Machine Learning Systems Design) (https://stanford-cs329s.github.io/)

💹Taking Made with ML course by Goku Mohandas - the course focuses on both theory and hands-on, It includes a hands-on explanation of concepts and tools such as Testing Data, Code and Model, Optuna's Hyperparameter Tuning, REST API, CI/CD using Github Actions, Feature Store and Monitoring (https://madewithml.com/#mlops)

💹Last but Important - Learning about any Cloud system and deploying a project with Docker on Cloud. It is more important since, in most real-world systems, you will be deploying projects in the cloud, and having beforehand experience will give you an edge when you work on a real-time project for the first time

📍Full Stack Deep Learning Course (https://fullstackdeeplearning.com/)

MLOps 8 weeks course
https://github.com/mohamedyd/MLOps-Basics

* MLOps - tools, best practices, and case studies (Chip Huyen): https://huyenchip.com/mlops/

-------------------------------------------------------------------

Variational autoencoder

https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73
https://medium.com/mlearning-ai/generating-artificial-tabular-data-with-an-encoder-decoder-5e4de9b4d02e
https://colab.research.google.com/github/lschmiddey/fastpages_/blob/master/_notebooks/2021-03-14-tabular-data-variational-autoencoder.ipynb#scrollTo=a5HpJEuuZLf5

Deep Learning online book (free):
http://neuralnetworksanddeeplearning.com/

Microsoft ML for beginners:
https://github.com/microsoft/ML-For-Beginners

Knowledge Graphs:
	- https://web.stanford.edu/class/cs520/
Web semantics: https://www.youtube.com/user/xhakil26/playlists
other courses: https://www.tele-task.de/

Intro to database systems: https://www.youtube.com/playlist?list=PLSE8ODhjZXjbohkNBWQs_otTrBTrjyohi

ML interview book: https://huyenchip.com/ml-interviews-book/

Knowledge graph course: https://web.stanford.edu/class/cs520/ 



https://www.youtube.com/watch?v=5DCS9LE-8rE

https://github.com/RDFLib/rdflib

https://www.youtube.com/watch?v=p9ZE4tzQn70


https://christophm.github.io/interpretable-ml-book/


Tubingen Machine Learning https://www.youtube.com/channel/UCupmCsCA5CFXmm31PkUhEbA

https://thodrek.github.io/CS839_spring18/


Dr Waleed Yousef
Computer science basics
https://www.youtube.com/user/FCIHOCW

lots of interesting topics: https://www.youtube.com/channel/UCUcpVoi5KkJmnE3bvEhHR0Q

https://stanford.edu/~shervine/teaching/cs-229/

https://christophm.github.io/interpretable-ml-book/

PythonDataScienceHandbook: https://github.com/jakevdp/PythonDataScienceHandbook

Introduction to Probability for Data Science: https://probability4datascience.com/

Git
https://learngitbranching.js.org/

Feature store: https://feast.dev/blog/what-is-a-feature-store/


Extra resources about MLOPS:
- Deep Learning in Production (https://amzn.to/3oa50Vj)
- Made With ML (https://madewithml.com/)
- Full Stack Deep Learning (https://lnkd.in/eaRwfjXT)
- Awesome MLOPs repo (https://lnkd.in/eUPvz-2)


https://research.aimultiple.com/




ITI Matreial in Arabic
https://drive.google.com/drive/folders/1B5ELjvlDSq4hvghOcaV2ssAv5pokECTA
