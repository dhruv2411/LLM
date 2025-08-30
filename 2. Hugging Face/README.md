**1.fine-tuning-pipeline-with-hugging-face.ipynb**
**Project Description**

This project focuses on preparing the IMDb movie review dataset for natural language processing (NLP) tasks, specifically sentiment analysis using transformer-based models like BERT.

**Objective:**

To load and preprocess the IMDb movie reviews dataset by converting raw text data into tokenized numerical inputs that are compatible with transformer models.

**Key Steps:**

Dataset Loading:
The IMDb dataset is downloaded and loaded using the Hugging Face datasets library. This dataset contains 50,000 movie reviews labeled with sentiment (positive or negative) split into training and testing sets.

Tokenization:
Each movie review text is tokenized using the BERT tokenizer (specifically distilbert-base-uncased). Tokenization converts raw text into input tokens (integers) that the model can understand. The text is padded or truncated to a fixed length to standardize input size.

Data Preparation:
The tokenized dataset now includes:

input_ids: token identifiers for each word piece,

attention_mask: masks to tell the model which tokens are actual words and which are padding,

token_type_ids: segment indicators (mostly zeros here as reviews are single segments),

and the original labels for sentiment classification.

**Outcome:**

The processed dataset is ready for training a transformer-based sentiment classification model. This preprocessing pipeline ensures that the raw text is efficiently and consistently transformed into the format required by models like BERT or DistilBERT.

**2.text-generation-with-hugging-face.ipynb**
**Project Description**

This project implements a web-based text generation application using the GPT-2 transformer model, enabling users to generate coherent text continuations based on their input prompts.

**Objective:**

To create an interactive interface where users can input a custom prompt and receive AI-generated text completions using a pre-trained GPT-2 model from Hugging Face’s Transformers library.

**Key Steps:**

Model Loading:
The GPT-2 model and tokenizer are loaded via the Hugging Face pipeline API configured for text generation.

Text Generation:
The user’s input prompt is processed by GPT-2 to generate a continuation, with a maximum length of 50 tokens. The generation respects padding and truncation settings to ensure consistent output length.

Interface Creation:
A Gradio interface is created to provide a simple web UI. It includes a text input box for the prompt and displays the generated output text, making it accessible for users without coding experience.

**Outcome:**

An easy-to-use web application where users can interact with a powerful language model to explore creative writing, brainstorming, or AI-assisted text generation.


**3.x-shot-learning.ipynb**
**Project Description**

This project demonstrates the use of GPT-2 transformer model for natural language generation tasks using few-shot and zero-shot learning approaches. It leverages Hugging Face’s Transformers pipeline to generate text based on user-defined prompts with minimal or no fine-tuning.

**Objective:**

To explore how GPT-2 can perform text generation for tasks like summarization and translation by providing a few examples (few-shot learning) or no examples (zero-shot learning) in the prompt, showcasing the model’s flexibility in understanding instructions and generating relevant output.

**Key Steps:**

**Few-Shot Learning:**

A prompt containing input-output pairs (examples) is provided to GPT-2.

The model continues the pattern to generate an output for a new input, attempting to mimic the example style.

This approach helps GPT-2 learn the task context on-the-fly through the prompt without retraining.

**Zero-Shot Learning:**

A plain instruction prompt without examples is given, such as a translation request.

GPT-2 attempts to perform the task based solely on its pre-trained knowledge.

This tests the model’s ability to generalize and understand natural language instructions directly.

**Implementation Details:**

GPT-2 is loaded using Hugging Face’s pipeline API for text generation.

Prompts are carefully crafted to specify the desired task and inputs.

Outputs are generated with controlled max token length and handling for truncation and padding.

Model runs on CPU with warnings about device setups noted but non-blocking.

**Outcome:**

The project highlights GPT-2’s capabilities in performing language tasks with minimal guidance, illustrating both the strengths and limitations of few-shot and zero-shot learning using large language models.

**4.hugging_face_-nlp-examples.ipynb**
**Sentiment Analysis**

Uses the default DistilBERT model fine-tuned on the SST-2 dataset to classify text as positive or negative sentiment. Your example sentence "I loved Star Wars so much!" was classified as positive with very high confidence.

**Grammatical Correctness Classification**

Uses a DistilBERT model fine-tuned on the CoLA dataset to determine if a sentence is grammatically acceptable or unacceptable. The input "I will walk to home when I went through the bus." was classified as unacceptable grammar.

**Grammar Correction**

Uses a T5-based sequence-to-sequence model trained for grammar correction to rewrite grammatically incorrect sentences into proper English. For example, the input "I will walk to home when I went through the bus." was corrected to "I will walk home when I go through the bus."

**Question Answering**

Uses a DistilBERT model fine-tuned on SQuAD to extract answers to questions given a context paragraph. Given the context "My name is Merve and I live in İstanbul." and question "Where do I live?", it correctly extracted "İstanbul" as the answer.

**Translation**

Uses the Helsinki-NLP MarianMT models to translate text between English and Spanish. For example, "Hello, How are you?" translated to "Hola, ¿cómo estás?", and the reverse translation of "Hola, ¿cómo estás?" returned "Hey, how are you?"

**Summarization**

Uses the DistilBART model fine-tuned on the CNN/DailyMail dataset to generate a concise summary of longer text. The paragraph about Paris was summarized, highlighting its status as the capital of France and its population proportion within the Paris Region.
