# A basic Chatbot implemented using Python

A basic bot created to understand NLP and machine learning techniques involved in training a chatbot.

## Table of Contents

1. [Project Description](#project-description)
2. [Getting Started](#getting-started)
   - [Chatbot Training and Model Creation](#chatbot-training-and-model-creation)
      - [Prerequisites](#prerequisites)
   - [Running the Chat User Interface](#running-the-chat-user-interface)
3. [Key Modules and Components](#key-modules-and-components)
4. [Conclusion](#conclusion)

## Project Description

The concept of chatbot technology dates back to the 1960s, marked by the creation of the pioneering chatbot named [ELIZA](https://en.wikipedia.org/wiki/ELIZA) by MIT professor Joseph Weizenbaum. ELIZA, an early natural language processing program, employed pattern-matching and substitution techniques to simulate human-like conversations. Eventually, a JavaScript version of the original emerged.

Over time, the evolution of chatbots gave rise to more advanced iterations such as [Jabberwacky](https://en.wikipedia.org/wiki/Jabberwacky), [Dr. Sbaitso](https://en.wikipedia.org/wiki/Dr._Sbaitso), [A.L.I.C.E. (Artificial Linguistic Internet Computer Entity)](https://en.wikipedia.org/wiki/Artificial_Linguistic_Internet_Computer_Entity), and, most recently, the formidable AI chatbot [Chat GPT](https://openai.com/blog/chatgpt). Early chatbots relied on contextual pattern matching and various natural language processing methods. Later developments incorporated AI functionalities, including voice-operated systems like Dr. Sbaitso and the now-familiar Siri, Google Assistant, and Alexa.

As technology progressed, chatbots began integrating different NLP and AI models to facilitate seamless user data exchange. Chatbots' applications and use cases have multiplied significantly, reflecting their growing importance in various domains. Looking ahead, we can anticipate the development of even more advanced versions of chatbots aimed at further simplifying and improving our daily lives.

Nonetheless, comprehending the fundamental workings of a chatbot remains essential, and this project aims to delve into precisely that. The project's objective is to thoroughly investigate this aspect. The chatbot is created using Python, incorporating NLP and machine learning techniques. It is trained on a dataset sourced from Kaggle, which can be customized to suit specific responses or use cases for the chatbot as 
preferred.

## Getting Started

These instructions will help you set up and run the chatbot on your local machine.
1. **Chatbot Training and Model Creation**
2. **Run the application GUI**

## Chatbot Training and Model Creation

### Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.x
- Required Python packages:
  - nltk
  - keras
  - numpy

Download the NLTK data (needed for natural language processing):
```python
import nltk
nltk.download('wordnet')
```

You can install the required Python packages using `pip`:

```bash
pip install nltk keras numpy

```
The dataset used for the [intents.json ](https://www.kaggle.com/datasets/niraliivaghani/chatbot-dataset/data) is found on Kaggle.

Ensure you have the JSON file containing the chatbot intents. You can specify the path to the JSON file in the code:

```python
intents = json.loads(open("path/to/your/intents.json").read())

```
You can customize the chatbot's behavior by modifying the intents JSON file. Each intent should have patterns and responses defined.

```json
"intents": [
  {
    "tag": "greeting",
    "patterns": ["Hello", "Hi", "Hey"],
    "responses": ["Hello! How can I help you?", "Hi there!", "Hey! What can I assist you with?"]
  },
  {
    "tag": "farewell",
    "patterns": ["Goodbye", "See you later", "Bye"],
    "responses": ["Goodbye! Have a great day!", "See you later!", "Bye!"]
  }
]
```

Run the Python script to train the chatbot model:

```bash
python train_bot.py

```
The `train_bot.py` script performs the following essential tasks:

1. **Tokenization**: It tokenizes patterns from the provided intents JSON file, breaking down sentences into individual words or tokens.

2. **Lemmatization and Preprocessing**: The script lemmatizes and preprocesses words, reducing them to their base forms and applying necessary text preprocessing techniques.

3. **Training Data Creation**: It creates training data with bag-of-words representations. This representation helps convert text data into numerical format suitable for machine learning.

4. **Neural Network Model**: The script defines and compiles a neural network model using Keras. This model is designed to predict the appropriate intent based on user input.

5. **Model Training**: It fits the neural network model to the training data, allowing the model to learn from the provided intents and patterns.

By executing this script, you train a chatbot model capable of understanding and responding to user queries effectively.

## Running the Chat User Interface

1. Ensure that you have trained the chatbot model as described earlier in this README.

Before you begin executing the GUI, ensure you have met the following requirements:

- Python 3.x
- Required Python packages:
  - nltk
  - keras
  - numpy
  - Pillow

You can install these packages using pip with the following command:
```bash
pip install nltk keras numpy pillow
```

2. To run the GUI, execute the following Python script:

   ```bash
   python application.py
   ```
   
The GUI window will appear, allowing you to start conversations with the chatbot.

### Using the Chatbot
 - The GUI provides a text input area at the bottom of the window. You can type your message there.

 - Press the "Send" button.

 - Your messages will appear in blue, and the chatbot's responses will appear in green in the chat window.

 - You can continue the conversation by typing new messages and sending them to the chatbot.

 - To exit the chat, simply close the GUI window.

This user-friendly interface allows you to interact with the chatbot seamlessly and see its responses in real-time. The GUI component of this chatbot application is built using the tkinter library. It can also be customised using other libraries like PyQt, PySide, wxPython, etc.

## Key Modules and Components 

This section provides an overview of the essential modules and components used in the chatbot code, explaining their roles and significance in the application.

### NLTK (Natural Language Toolkit)
A Bag-of-Words (BoW) model is employed in the code and requires the following:

- **Tokenization**: NLTK's tokenizer (`nltk.word_tokenize()`) breaks down sentences or phrases into their individual components, known as tokens. For example, If the user input is "Hi there!", it is converted into a list of tokens or words, "["Hi", "there", "!"]" and "!" is categorized as a punctuation mark. Tokenizer plays a crucial role in preprocessing user input and chatbot responses and it is the first step in constructing the BoW model. 


- **Lemmatization**: The [WordNetLemmatizer](https://www.nltk.org/_modules/nltk/stem/wordnet.html) from NLTK is employed to lemmatize words. Lemmatization reduces words to their base or dictionary forms, ensuring consistent word representations for the chatbot's understanding. In the code, irrelevant tokens, such as punctuation, are excluded, while the remaining words are organized into a unique and sorted vocabulary to enhance text processing.

- **Bag of Words**: The bag-of-words model is method of feature extraction to preprocess the text by converting it into numeric format also known as vectors. For the bot to understand, we need to transform data into a numeric format and therefore to train the model using a neural network algorithm, it is important that we convert the vocabulary to numbers first which is achieved by this model. For example, if we have the voacbulary from the last step as: 

```css
vocabulary = ["a", "and", "are", "chatbot", "hello", "hi", "how", "i", "is", "it", "name", "there", "what", "who", "you", "your"] 
```
The BoW representation for an input sentence "Hi there" would be: 
```css
BoW = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
```

### Training Model Architecture

The chatbot is trained using Keras [Sequential Neural Network Model](https://keras.io/guides/sequential_model/). [Keras](https://keras.io/) is an open-source deep learning library that provides a user-friendly interface for building and training neural network models.

The following model is defined:
 - The input layer has neurons equal to the vocabulary size of the BoW model.
 - First layer with 128 neurons using the ReLU (Rectified Linear Unit) activation function.
 - A dropout layer is added with a rate of 0.5. It helps to prevent overfitting by randomly deactivating a fraction of neurons during training.
 - The second layer has 64 neurons with ReLU activation after which another dropout layer is added.
 - The output layer has as many neurons as there are unique intent classes and uses the softmax activation function to predict the intent class probabilities.


### Model Compilation using Stochastic Gradient Descent(SGD) with Nesterov momentum: 
The [Stochastic Gradient Descent (SGD)](https://keras.io/api/optimizers/sgd/) optimizer is used with Nesterov momentum to train the neural network model. Nesterov Momentum builds upon the traditional momentum-based SGD. It calculates the gradient not at the current position of the model parameters but at a position "looked ahead" in the direction of the momentum reducing the momentum-induced oscillations, making it better at navigating sharp turns and achieving faster convergence. SGD optimizer adjusts the model's weights to minimize the loss function during training. The loss function (categorical cross-entropy), and metrics (accuracy) are defined to configure how the model will be trained and evaluated.

### Model Training

After training  with 200 epochs (training iterations) and a batch size of 5, the model is saved to a file named 'chatbot_model.h5' which can later be loaded and used for making predictions in the chatbot application.

## Conclusion
This simple project demonstrates a comprehensive overview of the code, covering essential concepts such as tokenization, lemmatization, and the creation of a Bag-of-Words (BoW) model. It has also delved into the training process of a neural network using Stochastic Gradient Descent with Nesterov momentum. These foundational concepts and techniques are pivotal for crafting a chatbot that can interpret user input and deliver meaningful responses. This code can be further enhanced to meet unique requirements. The project is an attempt to understand and explore the underlying methods in the functioning of chatbot.

## References

1. NLTK Documentation. [https://www.nltk.org/](https://www.nltk.org/)
2. Keras Documentation. [https://keras.io/](https://keras.io/)
3. Stochastic Gradient Descent. [https://scikit-learn.org/stable/modules/sgd.html#:~:text=Stochastic%20Gradient%20Descent%20(SGD)%20is,Vector%20Machines%20and%20Logistic%20Regression.]
4. Dataset. [https://www.kaggle.com/datasets/niraliivaghani/chatbot-dataset/data]




