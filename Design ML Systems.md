In this document, a summary of the main ideas in the [Desiging ML Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) book by Chip Huyen is provided. I highly recommend buying this interesting book. 

| | | 
|-|-|
|Chapter | Title |
| 1 | [Overview of Machine Learning Systems](#overview-of-machine-learning-systems) |
| 2 | [Introduction to Machine Learning Systems Design](#introduction-to-machine-learning-systems-design) |
| 3 | [Data Engineering Fundamentals](#data-engineering-fundamentals) |
| 4 | [Training Data](#training-data) |
| 5 | [Feature Engineering](#feature-engineering) |
| 6 | [Model Development and Offline Evaluation](#model-development-and-offline-evaluation) |
| 7 | [Model Deployment and Prediction Service](#model-deployment-and-prediction-service) |
| 8 | [Data Distribution Shifts and Monitoring](#data-distribution-shifts-and-monitoring) |
| 9 | [Continual Learning and Test in Production](#continual-learning-and-test-in-production) |
| 10 | [Infrastructure and Tooling for MLOps](#infrastructure-and-tooling-for-mlops) |
| 11 |  [The Human Side of Machine Learning](#the-human-side-of-machine-learning) |


# 1 Overview of Machine Learning Systems

* ML: learn complex patterns from existing data and use these patterns to make predictions on unseen data
* Zero-shot and continual learning are two approaches to learn with small data or no data
* ML shine if: the problem is repetitive, cost of wrong prediction is cheap, it's at scale, patterns constantly changing
* Key differences between ML in research and in production

| | | |
|-|-|-|
|| Research | Production |
| Requiremetns | State-of-the-art model performance on benchmark datasets | Different stakeholders have different requirements |
| Computational priority | Fast training, high throughput | Fast inference, low latency |
| Data | Static | Constantly shifting |
| Fairness | Often not a focus | Must be considered |
| Interppretability | Often not a focus| Must be considered | 
* Typical stakeholders involve ML engineers, sales team, product team, infrastructure engineers, and managers
* ML Systems versus traditional Software

| | | |
|-|-|-|
||Traditional Software| ML Systems|
| System development | Code and data are separated | The system is part code, part data, and part artifacts |
| Data | Constant | continously changing leading to faster development and deployment cycles |
| Testing | Focus on testing and code versioning | testing and versioning code and data (challenging task) |


# 2 Introduction to Machine Learning Systems Design

* Most ML systems should have characteristics: reliability, scalability, maintainability, adaptability
* It's good practice to decouple objectives to simplify the model development & maintenance (p. 41-43) 

# 3 Data Engineering Fundamentals

* First-party data: data that your company collects about your customers. Second-party data: data collected by another company about their own customers, and made this data available to you. Third-party data: companies collect data on the public who are not their direct customers. 
* Data formats: JSON, row-Major (CSV), column-major (Parquet)  
* For row-major formats, accessing data by rows much faster than by columns
* Data Models: relational data, NoSQL (document model, e.g., JSON, XML, BSON, and graph model), 
* A good practice is to normalize (1NF, 2NF) relational databases (spread them across multiple relations) to reduce redundancy and improve integration
* SQL is declarative language (specify the output and the computer figures out the steps to retrieve this output), Python is imperative language (user defines the steps)
* Query optimizer: it examines all possible ways to execute a query and select the fastest one to do so.
* Declarative ML: user declares the features' schema and the task, the system then find the best model
* Structured data follows predefined schema, while unstructured data, e.g., text, images, audio, does not adhere to schema, but may contains intrinsic patterns. 
* Online transactional processing (OLTP), e.g., ride-sharing service, involves users, low latency & high availability required, may follow ACID principles (Atomicity, consistency, isolation, durability), row-major.
* Online Analytical processing (OLAP), aggregating data in columns
* ETL: extract data from multiple sources, transform it into desired format, load it into the target destination, e.g., data warehouse, database, or ML model. 
* Modes of data flow: data passing through databases, data passing through services, data passing through real-time transport, e.g., Kafka. 

# 4 Training Data

* Understanding sampling methods (non-probability, random sampling) --> avoid potential sampling biases
* Non-probability sampling: convenience, snowball, judgement, quota sampling (biases typically occur)
Probability sampling: simple random sampling, stratified sampling, weighted sampling, reservoir sampling (streaming data), importance sampling
* Hand labels are expensive, raise privacy concerns, slow, and difficult to update
* Label multiplicity (disagreement among annotators), data lineage (keep track of origin of data sample & labels)
* Natural labels (ground truth labels): labels from the user feedback
* Feedback loop length: time it takes from when a prediction is served until when the feedback on it is received.
* Handling lack of labels: Weak-supervision (e.g., Snorkel), semi-supervision, transfer learning, active learning, data augmentation
* Self-training: use predictions on small labeled data as new labels to enhance the training data
* Perturbation-based method (data augmentation): small perturbations to a sample shouldn't change its label
* Active learning: label the samples most helpful to the model selected based on some metrics or heuristics
	* Such as uncertainty measurement, query by committee (disagreement among multiple candidate models)
* Comment: it could be interesting to compare between Active learning and Data Valuation for active data acquisition
* Class imbalance: caused by biases in the sampling process or labeling errors
* Sensitivity to class imbalance increases with the complexity of the predictive task
	* Linearly-separable problems are unaffected by all levels of class imbalance
	* Deep neural networks performed much better on imbalanced data than shallower neural networks
* Handling class imbalance: choosing the right metrics, data-level methods, algorithmic-level methods
* Data-level methods: 
	* undersampling (Tomek Links; risk of losing important information) or 
	* oversampling (SMOTE; risk of overfitting)
	* Advanced methods: Two-phase learning, dynamic sampling
* Algorithmic methods:
	* Cost-sensitive learning: modify loss function to consider the varying costs of majority and minority classes
	* Class-balanced loss: assign weight of each class proportional to the number of samples in that class
    * Focal loss: higher weight assigned to samples with low probability of being right
* Data Augmentation (DA)
    * Types of DA: label-preserving transformation (e.g., rotating images), perturbation (adding random noise), data synthesis
    * In NLP, templates is a cheap way to synthesis training data
    * In computer vision, the mixup method synthesis data via combining existing examples 

# 5 Feature Engineering

* missing values: Missing at random (MAR), missing completely at random (MCAR), missing not at random (MNAR)
    * MNAR: reason is because of the true value itself (e.g., respondants do not like to disclose their income)
	* MAR: reason is not related to the value itself, but to another observed variable (e.g., females do not like to disclose their age)
	* MCAR: no pattern in when the value is missing (rare cases)
* Deletion: cause loss of information, create biases (especially with MAR)
* Imputation: you risk injecting your own bias into and adding noise, or worse, data leakage.
* Standard scaling assumes that the data follows normal distribution
* Feature scaling: if the features follow skewed distribution, use log transformation to mitigate the skewness (this does not work well for all cases)
* Scaling requires global statistics, hence it can be source of data leakage during the inference; or the statistics will be less accurate if the new data during inference deviates from the original data
* Hashing trick: solve the problem of having categorical features with changing categories (collision may occur)
* Feature crossing: combining two or more features to generate new features to model nonlinear relationships
* Embedding: vector represents a piece of data. Word embeddings is the most common use of embeddings
* Causes of data leakage: splitting time-correlated data randomly instead of by time, scaling before splitting, filling in missing data with statistics from the test split, poor handling of data duplication before splitting, group leakage, leakage from data generation process
* Addressing data leakage: ablation study, measuring predictive power of each feature, monitor new features added
More features does not mean better model performance, reasons: more opportunities for data leakage, may cause overfitting, increase memory required, increase inference latency, technical debts (updating data becomes a cumbersome task)

# 6 Model Development and Offline Evaluation

* Classical ML models are still used in various applications, e.g., recommender systems & classification tasks with strict latency requirements.
* Six tips for model selection: avoid the state-of-the-art trap, start with the simplest model, avoid human biases in selecting models, evaluate good performance now versus good performance later, evaluate trade-offs (false positive vs. false negatives, compute requirement vs. accuracy, etc.), understand model's assumptions.
* Ensembles are less favored in production because they are more complex to deploy and harder to maintain. However, they are being used when a small performance boost leads to huge financial gain.
* There are three ways to create ensembles: bagging (bootstrap aggregation, Random forest), boosting, & stacking
* Bagging: sample with replacement to create different datasets, called bootstraps, and train a model on each of these bootstraps. The final prediction is obtained by majority vote (classification) or the average (regression)
* Bagging improves unstable models, such as neural networks, and decision trees. 
* Boosting: convert weak learners to strong one. Each learner is trained on the same set of samples, but the samples are weighted differently among iterations. Hence, future learners focus more on the samples that previous weak learners misclassified. 
* Stacking: training base learners on the training data, then create a meta-learner that combines the output of base learners. The meta-learner can be majority vote or average vote. 
* Experiment tracking: the process of tracking the progress and results of an experiment
	* Tools such as MLflow and Weights & Biases
* Versioning: logging all the details of an experiment to reproduce or compare the results with other experiments.
	* Tools such as DVC and Deltalake
* Data versioning is challenging: (1) data size is often large, (2) dataset might not fit into a local machine, (3) there is a confusion of what a Diff is, (4) not clear how to resolve merge conflicts, (5) regulations, GDPR, makes it complicated to version users data
* Debugging ML models: (1) ML models fail silently, (2) it may become slow to validate whether the bug has been fixed, (3) debugging ML models is hard due to their cross-functional complexity. 
* What makes  a ML model to fail: (1) Theoretical constraints, (2) poor implementation of the model, (3) poor choice of hyperparameters, (4) data problems, (5) poor choice of features
* Best practices for training and debugging ML models: (start simple and gradually add more components), (2) overfit a single batch, (3) set a random seed
* Distributed training: Gradient checkpointing (approach to train using less memory footprint), data parallelism (distribute data and aggregate weights), model parallelism (distribute model layers)
* Soft autoML: hyperparameter tuning
* Hard autoML: architecture search and learned optimizer
* Four phases on ML model development: (1) before ML (try solving the problem without ML), simplest ML models, (3) optimizing simples models, (4) complex models
* Evaluation metrics, by themselves, mean little. When evaluating your model, it's essential to know the baseline you're evaluating it against. These baselines can be either: (1) random baseline, (2) simple heuristic, (3) zero rule baseline (always predict the most common class), (4) Human baseline, or (5) existing solutions
* Evaluation methods: 
	* perturbation tests: test model on noisy input rather than clean data
	* invariance tests: detect biases via altering or removing sensitive information from the features
	* directional expectation tests: if the output changes in the opposite expected direction --> model not learning 
	* model calibration: Platt scaling method used to calibrate your model
	* confidence measurement: measuring confidence about each prediction (certainty threshold)
    * slice-based evaluation: useful to detect Simpson's paradox 

# 7 Model Deployment and Prediction Service

* The hard parts of deploying ML models:
	* Make them available to millions of users with a latency of milliseconds and 99% uptime
	* Setting up infrastructure to notify the right person when something goes wrong
	* Being able to figure out what went wrong
	* Seamlessly deploying updates to fix what's wrong
* Serialization: converting ML model into a format that can be used by other application. 
	* Two parts of ML model are exported together: model architecture and model's parameter values
* Online prediction (on-demand)
	* Generating predictions as soon as requests for these predictions arrive
	* It uses either batch features or combination of streaming and batch features
* Batch prediction
	* Predictions are generated periodically or whenever triggered
	* Predictions are stored in SQL tables or in-memory database, and retrieved as needed
* Hybrid approach is to precompute predictions for popular queries, then generate predictions online for less popular queries. 
* Model compression: low-rank factorization, knowledge distillation, pruning, and quantization. 
	* Low rank factorization: replacing high-dimensional tensors with lower-dimensional tensors
	* Knowledge distillation: train a student (small) model to mimic a larger model (teacher)
	* Pruning: remove redundant and uncritical nodes from a neural network or finding parameters least useful to prediction and set them to zero
	* Quantization: use fewer bits to represent the model parameters (most common method)
* Pros of edge deployment: reasonable cost, no rely on Internet connection, no worry about network latency, appealing when handling sensitive data  
* Instead of targeting new compilers and libraries for every new hardware backend, it makes sense to create a middleman to bridge frameworks, e.g., PyTorch and TensorFlow, and platforms, e.g., CPUs, GPUs, and TPUs.  
* Lowering: a process in which the compiler generates a series of high- and low-level intermediate representations before generating the code native to a hardware backend. 
* Even if ML models work fine in development, they can be slow when deployed in production
* Models can be optimized locally or globally
	* Local: optimizing an operator or a set of operators of the model (reducing memory access or parallelization)
	* Global: optimizing the entire computation graph end-to-end
* AutoTVM is a ML-based compiler stack used to automatically optimize ML models 
* WebAssembly is an open standard that allows you to run executable programs in browsers. After building ML models, we can compile them to WebAssembly to get an executable file used with JavaScript (it is slower than running models natively on devices, e.g., Android or iOS apps).

# 8 Data Distribution Shifts and Monitoring

* Software system failure (failures that would have happened to non-ML systems): dependency failures, deployment failures, hardware failures, downtime or craching. 
* ML-specific failures: 
	* data collection and processing problems, e.g., sampling biases and class imbalance (covered in Chapter 4)
	* poor hyperparameters (Chapter 6)
	* production data differing from training data
		* if the training and production data come from different distributions, the model can not generalize well.
		* data distribution shifts happen all the time, suddenly, gradually, or seasonally 
		* Due to complexity of ML systems and poor deployment practices, a large percentage of data shifts are caused by internal (human) errors
			* bugs in the data pipeline
			* missing values incorrectly imputed
			* inconsistencies between the features extracted during training and inference
			* features standardized using statistics from the wrong subset of data
			* wrong model version
			* bugs in the app interface that force users to change their behaviors  
	* edge cases
		* they are data samples so extreme that they can cause the model to make catastrophic mistakes, e.g., [self-driving cars](https://rodneybrooks.com/edge-cases-for-self-driving-cars/)
	* degenerate feedback loop
		* it happens when the predictions themselves influence the next iterations of the model
		* at best, it can cause a model to perform sub-optimally. At worest, it can magnify biases embedded in data
		* Two techniques can be used to correct degenerate feedback loop, including: randomization and the use of positional features
* Data distribution shifts
	* A phenomenon is supervised learning when the data a model works with changes over time, causing the model performance to deteriorate.
	* Features can be seen as a set of samples from the joint distribution P(X, Y) and the ML usually models P(Y|X).
	* The joint distribution P(X, Y) can be decomposed into P(Y|X)P(X) or into P(X|Y)P(Y)
	* Types of data distribution shifts:
		* Covariate shift: when P(X) changes, but P(Y|X) remains the same
			* Can be caused by biases during the data selection process or becuase the training data is artificially altered to make it easier for the ML model to learn (e.g., addressing class imbalance). 
			* In production, covariate shifts typically occur becuase of major changes in the environment or in the way your application us used. 
		* Label shift (aka prior shift and target shift): when P(Y) changes, but P(X|Y) remains the same
			* The prior probability of the classes changes between the training and test data
			* If the training data has a different class balance than the test data, this can cause a shift in the prior probabilities.
		* Concept drift (aka posterior shift): when P(Y|X) changes, but P(X) remains the same
			* You can think of this as "same input, different output". They are cyclic or seasonal, e.g., ride sharing or flight tickets.
		* Other types of changes in the real world
			* feature change: new features added, older features are removed, or the set of all possible values of a feature changes
			* label scheme change: the set of possible values for Y change (both P(Y) and P(X|Y) change), e.g., new classes added.  
	* Detecting data distribution shifts:
		* The easiest way is to monitor the performance of the ML models, however this solution assumes the availability of ground truth of the model's predictions, which is not often the case. 
		* Recently, there have been research to detect label shifts without knowing the ground truth e.g., [Black Box Shift Estimation (BBSE)](https://arxiv.org/abs/1802.03916). However, in industry, drift detection efforts focus on detecting changes in the input distributions. 
		* Statistical methods
			*  [TensorFlow Extended's built-in data validation tools](https://www.tensorflow.org/tfx/data_validation/get_started) (Mean, median, variance)
				* There is no guarantee that there's no shift, if those metrics are similar
			* Two-sample hypothesis test: check whether the difference between two populations is statistically significant (If this is the case, the probability that the difference is a random fluctuation due to sampling variability is very low, and the difference is caused by the fact that the two populations comes from two different distributions).
				* If you are able to detect the difference from a relatively small sample, then it is a serious difference. If you need a huge number of samples to detect, then the difference is probably not worth worrying about. 
				* Examples: 
					* The Kolmogorov-Smirnov (KS) test: nonparametric test (no parameters & assumptions about the distribution, can be used with only one-dimensional data). It can be computationally expensive and produce too many false positives ([Reference](https://www.tensorflow.org/tfx/data_validation/get_started)).
					* Kullback-Leibler (KL) divergence (It quantifies the amount of information lost when approximating one distribution with another)
					* Chi-Squared test for categorical features
					* Least-squares density difference (LSDD)
					* Maximum Mean Discrepancy (MMD)
				* Shifts can be either spatial (shifts happen across access points, e.g., app is used on a different type of devices) or temporal (shifts that happen over time; we should treat input data as time-series data).
				* For temporal shifts, the time scale window of the data we look at affects the shifts we can detect.
					* the shorter the window, the faster you will be able to detect changes. However, too short windows can lead to false positives. 
			* Other techniques to detect shifts:
				* [Subset scanning](https://arxiv.org/pdf/2111.14938v2.pdf) (fast and efficient, unsupervised, batch data)
				* Causal forests (supervised, detect concept drift in real-time)
				* [Failing Loudly](https://github.com/steverab/failing-loudly/blob/master/shift_detector.py): An Empirical Study of Methods for Detecting Dataset Shift (link)
				* XAI methods (e.g., [clustering SAPH values](https://openreview.net/pdf?id=P_ImdNqo7S), [explanations and two-sample test](https://github.com/DFKI-NLP/xai-shift-detection))
				* Bayesian Changepoint Detection (probabilistic) 
				* Kernel Changepoint Detection (KSWIN (Kernel-based Sliding Window) & KED (Kernel-based Estimation of Drift))
				* Unsupervised Change Detection algorithm (difference-based, clustering)
				* ECD (Ensemble Changepoint Detection) algorithm and the E-Divisive algorithm
	* Addressing Data Distribution Shifts
		* Training ML models with massive datasets
		* Adapting trained models to a target distribution (domain adaptation)
		* Retraining models using labeled data from the target distribution
			* Stateless retraining (Everytime training the entire model from scratch)
			* Stateful training (continual learning): the model continues training on new data
				* Example: river (open source) library for detecting drifts and continual learning (GitHub link)
		* Ensemble learning with model weighting
* Monitoring and observability
	* Monitoring refers to the act of tracking, measuring, and loggin different metrics that can help us determine when someting goes wrong.
	* Observability (instrumentation) means setting up our system in a way that gives us visibility into our system to help us investigate what went wrong, e.g., adding timers to your functions, counting NaNs, tracking how features are transformed. 
	* Metrics to be monitored:
		* operational metrics, e.g., availability, latency, throughput, CPU/GPU utilization, memory utilization, etc.
		* ML-specific metrics
			* Model's accuracy related metrics (requires user feedback or an oracle for labeling)
			* Prediction, e.g., using two-sample test
			* Features (validation using open-source libraries: Deequ and Great Expectations) or two-sample tests
			* Minor changes might not harm the model performance / changes might be caused by the preparation pipeline
			* Raw inputs (can be difficult due to collecting data from different sources)
	* Monitoring toolbox: logs, dashboards, and alerts
		* logs: recording events produced at runtime. To make it easy to search the logs, distributed tracing is used.
			* distributed tracing: giveing each process an ID, when something goes wrong, the error message shows that ID. It is also important to record the necessary metadata: time when the event happens, the ervices where it happens, the function that is called, the user associated with the process, etc. 
			* the number of logs can grow very large quickly. Therefore, there have been tools to help companies manage and analyze their logs. ML has also being used to detect anomalistic events in the logs. 
		* dashboards: useful for revealing relationships between metrics, making monitoring accessible to nonengineers. 
			* dashboard rot: a phenomenon where execssive metrics on a dashboard can be counterproductive. 
			* alters: it consists of:
				* an alert policy: this describes the condition for an alert
				* a notification channels: who is to be notified when the condition is met (it is important to direct the alerts to the right persons to mitigate the alert fatigue)
				* a description of the alert: detailed description will help the notified person understand the problem.

# 9 Continual Learning and Test in Production

* Companies that employ continual learning in production update their models in micro-batches. The updated model should not be deployed until it is been evaluated.
* stateful training requires less data, less compute power, and faster convergence time than stateless retraining. With stateful training, it is possible to avoid storing data (save costs and eliminate privacy concerns).
* Some companies tend to use both stateful training and stateless retraining (occasionally).
* Continual learning is about setting up infrastrcture in a way that allows data scientists to update ML models whenever needed. 
* Why Continual learning? 
	* Combat data distribution shifts, adapt to rare events, overcome the cold start problem (dealing with new users/situations, where historical data does not represent those users/situations). 
* Continual learning challenges:
	* fresh data access challenge
		* getting fresh data can be slow if the data is pulled from a warehouse, comes from multiple sources. Getting data directly from real-time transports, e.g., Kafka and Kinesis, can be much faster. 
		* best candidates for continual learning are tasks where you can get natual labels with short feedback loops (e.g., dynamic pricing).
		* Label computation: the process of searching the logs to extract labels (can be costly when delaing with large number of logs, much faster to leverage stream processing to extract labels from the real-time transports directly).
	* evaluation challenge (ensuring that the updates are good enough (performance and safety) to be deployed)
		* evaluation is necessary: with continual learning, the models more susceptible to coordinated manipulation and adversarial attacks.
		* evaluation takes time, which might be bottleneck for model update frequency. 
	* algorithm challenge
		* tree-based and matrix-based models (e.g., collaborative filtering model) require the entire dataset to be updated.
		* Scaling features require statistics computed over the entire dataset, which can be challenging in continal learning.
			* solution is to rely on running statistics: incrementally compute these statistics as new data arrives.
* Four stages of continual learning
	* Stage 1: Maunal, stateless retraining (Your priority is to develop ML models to solve as many business problems as possible)
	* Stage 2: Automated retraining (Your priority is to maintain and improve existing models)
		* three factors define the feasibility of writing scripts to automate the workflow
			* scheduler: tool that handles task scheduling. 
			* data availablility and accessibility (most people's time might be spent here)
			* model store: automatically version and store all artifacts needed to reproduce the ML models (e.g., MLflow).
		* feature reuse (log and wait): using the features extracted by the prediction service to retrain the models (save computation & allows for consistency between prediction and training)
	* Stage 3: Automated stateful training
		* the main thing needed here is a way to track data and model lineage (no existing model store has this model lineage capacity)
		* model is updated based on fixed schedule set out by developers (this task is not straightforward)
	* Stage 4: continual learning
		* model is updated automatically whenever data distributions shift and the model's performance plummets.
		* in this stage, you need a mechanism to trigger model updates, where the trigger can be:
			* time-based, performance-based, volume-based (i.e., size of labeled data), drift-based
		* Monitoring is needed in this stage, where it is not hard to detect changes, but to determine which of these changes matter.
* How often to update your ML models
	* to answer this question, we need first to figure out how much gain your model will get from being updated with fresh data.
	* the more gain your model can get, the more frequently it should be retrained.
	* To measure value of data freshness, train your model on data from different time windows in the past and evaluate it on the data from today to see how the performace changes. 
	* Model iteration (adding new features or changing model architecture) versus data iteration. Currently, there is no theoritical foundation which tell which approach is better for a certain situation. Hence, you have to do experiments to find out. 
* Test in production
	* Shadow deployment (safest way, but expensive since it relies on two co-existing prediction services)
		* deploy the candidate model in parallel with the existing model
		* Use both models to serve the incoming requests, but only supply the users with predictions from the existing model
		* log the predictions from the candidate model for analysis purposes 
		* replace the existing model with the candidate model, only if the candidate model's predictions are satisfactory
	* A/B testing (requires randomness and sufficient number of samples)
		* deploy the candidate model alongside the existing model.
		* divide the incoming traffic between these two models
		* monitor and analyze the predictions and user feedback to determine whether the difference in the two models' performance is statistically significant (hypothesis testing, e.g., two-sample tests, are used). 
	* Canary release
		* deploy the candidate model (canary) alongside the existing model.
		* a portion of the traffic routed to the candidate model
		* if its performance is satisfactory, increase the traffic to the canndidat model. If not, abort the canary and route all traffic back to the existing model.
		* stop when either the canary serves all traffic or when the canary is aborted
	* Interleaving experiments 
		* serve users with predictions from the candidate and the exiting models.
		* interleaving identifies the best model with smaller sample size compared to traditional A/B testing.
	* Bandit (more data-efficient than A/B testing, but more difficult to implement than A/B testing)
		* multi-armed bandits are algorithms that balance between exploration and exploitation in reinforcement learning problems.
		* Bandits allow you to determine how to route traffic to each model for prediction to determine the best model.
		* Bandits require three things
			* your model must be able to make online predictions
			* preferably short feedback loops
			* a mechanism to collect feedback, calculate and keep track of each model's performance, and to route requests to different models based on their current performance.


# 10 Infrastructure and Tooling for MLOps (TBD)

* infrastructure requirement depends on the number of developed applications and how specialized the applications are
	* One simple ML app: No infra ineeded
	* Multiple common apps, e.g., fraud detection, recommenders, price optimization: Generalized infra
	* Serving millions requests/hour, e.g., Facebook and Googel: Highly specialized infra
* an infrastructure compose the following layers:
	* storage and compute: where data is collected, stored, and processed for generating features and training ML models.
		* can be hard drive disk (HDD), solid state disk (SSD), Amazon S3, distributed over multiple locations
		* compute unit is characterized by two metrics: how much memory it has and how fast it runs ana operation (common metric: FLOPS)
		* [MLPerf](https://www.nvidia.com/en-us/data-center/resources/mlperf-benchmarks/): benchmark for hardware vendors to measure thier hardware performance (e.g., how long it takes to run ResNet-50 on the ImageNet dataset)
		* AWS uses the concept of vCPUs (virtual CPUs), practically can be thought of as a half physical core.
	* resource management: scheule and orchestrate workloads to improve resources utilization, e.g., Airflow, Kubeflow, and Metaflow 
	* ML platform: model stores, feature stores, and monitoring tools (e.g., SageMaker and MLflow) 
	* development environment: where code is written and experiments are run. (code versioning and experiment tracking)


# 11  The Human Side of Machine Learning

* It is difficult for data scientists to equally focus on model development and on model deployment. 
* However, there are several drawbacks for splitting the team into ML engineers and DevOps engineers
	* Communication and coordination overhead
	* Debugging challenges: when trying to figure out where are the failures come from
	* Finger-pointing: each team might think it's another team's responsibility to fix the failures
	* Narrow context: no one has visibility into the entire process to optimize it
* A solution to this dilemma is to allow data scientists to own the entire process end-to-end without having to worry about the infrastructure
    * What if I can tell the tool, "Here where I store my data (S3), here are the steps to run my code (featurization, modeling), here's where my code should run (EC2 instances, serverless stuff like AWS Batch, Functions, etc.), here's what my code needs to run at each step (dependencies)," and then this tool manages all the infrastructure stuff for the data scientist.