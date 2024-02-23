| Chapter | Topic     | Notes      |
|---------|-----------|------------|
| 1       | Understanding Large Language Models   |  
- Large language models (LLMs) are deep neural network models for Natural Language Processing (NLP).
- LLMs can process and generate text in ways that appear coherent and contextually relevant.
- LLMs are trained on vast quantities of text data, capturing deeper contextual information and subtleties of human language.
- Unlike NLP models designed for specific tasks, LMs demonstrate a broader proficiency across a wide range of NLP tasks, such as text translation, sentiment analysis, and question answering.
- The success of LLMs is attributed to the [transformer architecture](https://arxiv.org/abs/1706.03762) and the vast amounts of data they are trained on.
- The "large" in large language model refers to the model's size in terms of parameters and the immense dataset on which it's trained.
- LLM models often have tens or even hundreds of billions of parameters, the adjustable weights in the network optimized during training to predict the next word in a sequence.
- Existing LLMs can be adapted and finetuned to outperform general LLMs, as demonstrated by [teams from Google Research and Google DeepMind](https://arxiv.org/abs/2305.09617) and a team at Bloomberg via [BloombergGPT](https://arxiv.org/abs/2303.17564).
- The creation of an LLM includes pretraining and finetuning.
   - Pretraining: LLM is trained on large, diverse unlabeled text data to develop a broad understanding of language. This phase leverages self-supervised learning.
   - Finetuning: pretrained LLM is then trained on a narrower labeled dataset more specific to particular tasks or domains. The two most common categories for finetuning LLMs are:
     - Instruction-finetuning: the labeled dataset consists of instruction and answer pairs.
     - Classification finetuning: the labeled dataset consists of texts and associated class labels. |
