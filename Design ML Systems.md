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
||||
|-|-|-|
|| Research | Production |
| Requiremetns | State-of-the-art model performance on benchmark datasets | Different stakeholders have different requirements |
| Computational priority | Fast training, high throughput | Fast inference, low latency |
| Data | Static | Constantly shifting |
| Fairness | Often not a focus | Must be considered |
| Interppretability | Often not a focus| Must be considered | 
* Typical stakeholders involve ML engineers, sales team, product team, infrastructure engineers, and managers
* ML Systems versus traditional Software
||||
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
* Handling lack of labels: Weak-supervision, semi-supervision, transfer learning, active learning, data augmentation
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
    * Types of DA: label-preserving transformation, perturbation (adding random noise), data synthesis
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

# 9 Continual Learning and Test in Production (TBD)

# 10 Infrastructure and Tooling for MLOps (TBD)

# 11  The Human Side of Machine Learning

* It is difficult for data scientists to equally focus on model development and on model deployment. 
* However, there are several drawbacks for splitting the team into ML engineers and DevOps engineers
	* Communication and coordination overhead
	* Debugging challenges: when trying to figure out where are the failures come from
	* Finger-pointing: each team might think it's another team's responsibility to fix the failures
	* Narrow context: no one has visibility into the entire process to optimize it
* A solution to this dilemma is to allow data scientists to own the entire process end-to-end without having to worry about the infrastructure
    * What if I can tell the tool, "Here where I store my data (S3), here are the steps to run my code (featurization, modeling), here's where my code should run (EC2 instances, serverless stuff like AWS Batch, Functions, etc.), here's what my code needs to run at each step (dependencies)," and then this tool manages all the infrastructure stuff for the data scientist.