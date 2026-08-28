# Module 5: Language Modelling
### N-grams, Hidden Markov Models, RNNs, and LSTMs

This guide breaks down four core ideas in Natural Language Processing (NLP) in plain language, aimed at first-year students who are new to the topic. Each section explains the "what," the "why," and includes simple code examples with explanations.

---

## 1. N-grams: Teaching a Computer to Guess the Next Word

### What is an N-gram?

Imagine you're texting a friend, and your phone suggests the next word before you finish typing. That's essentially what an N-gram model does — it looks at a chunk of words and tries to predict what comes next.

An **N-gram** is just a sequence of N words (or other items) taken in order from a piece of text. "N" is simply a number you choose:

- **Unigram (N=1):** a single word, like `"Natural"`
- **Bigram (N=2):** two words in a row, like `"Natural Language"`
- **Trigram (N=3):** three words in a row, like `"Natural Language Processing"`

Think of it like a sliding window moving across a sentence, capturing a few words at a time.

### Why do we care about N-grams?

N-grams help computers understand which words tend to appear together. This is useful for:

- **Predicting text** (like autocomplete on your phone)
- **Translating languages** (understanding word context)
- **Speech recognition** (guessing which words were spoken based on what usually comes next)
- **Generating text** (writing sentences that sound natural)

### How do you build N-grams in Python?

Here's a simple way to break a sentence into unigrams, bigrams, and trigrams using the `nltk` library:

```python
from nltk import ngrams
import nltk
nltk.download('punkt')

text = "Natural Language Processing is a fascinating field of study."
tokens = nltk.word_tokenize(text)  # splits text into individual words

def generate_ngrams(tokens, n):
    return [' '.join(grams) for grams in ngrams(tokens, n)]

unigrams = generate_ngrams(tokens, 1)
bigrams = generate_ngrams(tokens, 2)
trigrams = generate_ngrams(tokens, 3)
```

**What's happening here, step by step:**
1. `nltk.word_tokenize(text)` splits the sentence into a list of individual words (this is called *tokenizing*).
2. The `generate_ngrams` function slides a window of size `n` across the list of words and joins each group into a string.
3. We call this function three times with `n=1`, `n=2`, and `n=3` to get unigrams, bigrams, and trigrams.

### How does an N-gram model predict the next word?

The model estimates a **probability** — basically, "given the last word (or few words), how likely is the next word to be X?"

For example, imagine we counted a big pile of text and found:
- The word "language" appears 200 times.
- The phrase "language processing" appears 50 times.

Then the probability of "processing" following "language" is:

```
P(processing | language) = 50 / 200 = 0.25 (or 25%)
```

In other words, 25% of the time that "language" appears, it's followed by "processing."

To build this kind of model in Python, you count how often each word follows another, then divide by the total to turn counts into probabilities:

```python
from collections import defaultdict
from nltk.util import ngrams

def train_bigram_model(tokenized_corpus):
    model = defaultdict(lambda: defaultdict(lambda: 0))
    # Step 1: Count how often each word follows another
    for sentence in tokenized_corpus:
        for w1, w2 in ngrams(sentence, 2):
            model[w1][w2] += 1
    # Step 2: Turn counts into probabilities
    for w1 in model:
        total_count = float(sum(model[w1].values()))
        for w2 in model[w1]:
            model[w1][w2] /= total_count
    return model
```

### What are the downsides of N-grams?

N-grams are simple and easy to understand, but they have real limitations:

| Problem | In Plain English |
|---|---|
| **Sparsity** | As N gets bigger, there are SO many possible word combinations that most of them never even show up in your training text, so you can't estimate their probability well. |
| **Limited memory** | A bigram model only looks at the *one* previous word. It can't "remember" what happened several words back, so it misses the bigger picture of a sentence. |
| **Storage cost** | The more word combinations you track, the more memory you need to store all those probabilities. |
| **No real understanding** | N-grams just count word patterns — they don't understand what words actually *mean*. |

Despite these downsides, N-grams are a great starting point and laid the groundwork for the more advanced models covered next.

---

## 2. Hidden Markov Models (HMMs): Guessing What You Can't See

### What is an HMM?

Imagine you're trying to figure out someone's mood just by watching their actions, without them telling you directly. You see clues (like "smiling" or "sighing"), and from those clues, you *guess* their hidden mood.

That's exactly the idea behind a **Hidden Markov Model (HMM)**. It's a statistical tool used when:
- There's a sequence of things you *can* observe (like words in a sentence).
- Behind those observations, there's a sequence of *hidden* states you can't directly see (like the grammatical role of each word — noun, verb, etc.).

HMMs are widely used for:
- **Part-of-speech tagging** (labeling each word as a noun, verb, adjective, etc.)
- **Named entity recognition** (finding names of people, places, organizations in text)
- **Speech recognition** (converting sound into text)

### The Building Blocks of an HMM

An HMM has five key ingredients:

1. **States** — The hidden things you're trying to figure out (e.g., "Noun" or "Verb").
2. **Observations** — The things you can actually see (e.g., the words themselves).
3. **Transition Probabilities** — How likely you are to move from one hidden state to another (e.g., how often a noun is followed by a verb).
4. **Emission Probabilities** — How likely a particular hidden state is to produce a particular observation (e.g., how likely the word "run" is to come from the "Verb" state).
5. **Initial Probabilities** — How likely each state is to be the *first* one in the sequence.

**Quick analogy:** Think of weather forecasting. The hidden states might be "Sunny" or "Rainy" (you can't directly measure the "state" of the sky's mood), but the observations are things like "people carrying umbrellas." Transition probabilities tell you how likely it is to go from a sunny day to a rainy day; emission probabilities tell you how likely people are to carry umbrellas on a sunny vs. rainy day.

### The Three Big Questions HMMs Answer

1. **Evaluation Problem:** *"How likely is it that this HMM produced this particular sequence of observations?"*
   - Solved using the **Forward Algorithm**, which efficiently adds up the probabilities across all possible hidden state sequences.

2. **Decoding Problem:** *"Given what I observed, what's the most likely sequence of hidden states?"*
   - Solved using the **Viterbi Algorithm**, which works step-by-step, always keeping track of the *most probable* path so far, and finally backtracks to reconstruct the best overall sequence.

3. **Learning Problem:** *"I don't know the transition/emission probabilities yet — how do I figure them out from data?"*
   - Solved using the **Baum-Welch Algorithm** (a specific version of the Expectation-Maximization or "EM" algorithm), which repeatedly refines its guesses about the probabilities until they stop changing much.

### HMMs in Python

Here's a simplified example using the `hmmlearn` library to predict whether each word in a sentence is a noun or verb:

```python
import numpy as np
from hmmlearn import hmm

# The hidden states we're trying to guess
states = ["Noun", "Verb"]

# The words we can actually observe
observations = ["I", "run", "to", "the", "store"]

# How likely is it to move from one state to another?
transition_probability = np.array([
    [0.7, 0.3],  # From Noun -> [Noun, Verb]
    [0.4, 0.6]   # From Verb -> [Noun, Verb]
])

# How likely is each word, given the state?
emission_probability = np.array([
    [0.2, 0.3, 0.2, 0.1, 0.2],  # From Noun
    [0.1, 0.6, 0.1, 0.1, 0.1]   # From Verb
])

# How likely is each state to start the sentence?
start_probability = np.array([0.6, 0.4])

# Build and configure the model
model = hmm.MultinomialHMM(n_components=2)
model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability

# Turn the sentence into numbers the model understands
observation_sequence = np.array([0, 1, 2, 3, 4]).reshape(-1, 1)

# Use the Viterbi algorithm to guess the hidden states
logprob, hidden_states = model.decode(observation_sequence, algorithm="viterbi")

print("Words:", [observations[i] for i in observation_sequence.flatten()])
print("Predicted tags:", [states[i] for i in hidden_states])
```

**What this code does, in plain terms:**
- We tell the model what states exist ("Noun", "Verb") and what words we might see.
- We give it some starting probabilities (guesses, essentially).
- We feed it a sentence ("I run to the store") turned into numbers.
- The model uses the Viterbi algorithm to output its best guess for the grammatical role of each word.

The expected output would tag "I" as a Noun, "run" as a Verb, and so on — matching how we'd naturally read the sentence.

---

## 3. Recurrent Neural Networks (RNNs): Giving Computers a Memory

### What makes RNNs different?

A regular neural network looks at one input and produces one output — it has no memory of what came before. But language isn't like that: understanding a sentence often depends on the words that came earlier.

A **Recurrent Neural Network (RNN)** solves this by adding a loop: at each step, the network looks at the current input *and* remembers something about the previous steps. This "memory" is called the **hidden state**.

**Simple analogy:** Reading a sentence word by word while keeping a mental note of what you've read so far — that mental note is like the hidden state.

Mathematically, the hidden state at time `t` depends on the current input and the previous hidden state:

```
h_t = f(W · x_t + U · h_(t-1) + b)
```

In plain English: *"My current understanding = some function of what I'm seeing right now, PLUS what I remembered from before."*

### Where are RNNs useful?

- Understanding and generating text
- Speech recognition
- Video analysis
- Financial forecasting (or anything involving a time-ordered sequence)

### The Problems RNNs Run Into

RNNs sound great in theory, but they have some real headaches in practice:

1. **Vanishing Gradients** — During training, the "error signal" used to update the network can shrink to almost nothing as it travels back through many time steps. This makes it hard for the network to learn from things that happened far in the past.
2. **Exploding Gradients** — The opposite problem: sometimes that error signal grows enormously instead, making training unstable.
3. **Trouble with Long-Term Memory** — Because of the above issues, RNNs often struggle to remember information from many steps ago, even though that's exactly what they were designed to do.
4. **Slow to Train** — Since each step depends on the one before it, you can't easily parallelize the computation, so training takes a while.
5. **Tricky to Train Well** — Getting all the settings right (initial weights, activation functions, etc.) takes care and experimentation.
6. **Overfitting** — Like most deep learning models, RNNs can "memorize" training data instead of learning general patterns, especially with small datasets.

**Common fixes:** gradient clipping (capping how large gradient updates can get), dropout (randomly ignoring some neurons during training to prevent memorization), and — most importantly — switching to more advanced architectures like **LSTMs** (covered next).

### Building a Simple RNN in Python

Here's a tiny example that trains an RNN to predict the next *character* in the word "hello world":

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.utils import to_categorical

text = "hello world"

# Turn each unique character into a number
chars = sorted(set(text))
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

# Build training examples: 3 characters -> predict the 4th
sequence_length = 3
X, y = [], []
for i in range(len(text) - sequence_length):
    X.append([char_to_idx[c] for c in text[i:i + sequence_length]])
    y.append(char_to_idx[text[i + sequence_length]])

X = np.array(X).reshape((len(X), sequence_length, 1))
y = to_categorical(y, num_classes=len(chars))

# Build a simple RNN model
model = Sequential()
model.add(SimpleRNN(50, input_shape=(sequence_length, 1)))
model.add(Dense(len(chars), activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy')

model.fit(X, y, epochs=200, verbose=0)
```

**Breaking it down:**
1. We turn each letter into a number, since neural networks work with numbers, not letters.
2. We slide a window of 3 characters across the text, and for each window, the "answer" is the character that comes right after.
3. We build a small RNN with 50 "memory units," followed by a layer that predicts probabilities for each possible next character.
4. We train it — over many rounds ("epochs") — to get better at predicting the next character.

After training, feeding it `"hel"` should let it predict something close to `"lo w"`, continuing the pattern from "hello world."

---

## 4. Long Short-Term Memory Networks (LSTMs): A Better Memory

### Why do we need LSTMs?

RNNs *try* to remember the past, but as we just saw, they're bad at holding onto information for very long due to vanishing/exploding gradients. **LSTMs** were specifically designed to fix this.

**Analogy:** If a regular RNN's memory is like a whiteboard that gets erased a little more each time you write on it, an LSTM is like a notebook with a smart assistant deciding exactly what to write down, what to cross out, and what to read back — giving it much better long-term memory.

### The Key Parts of an LSTM

An LSTM cell has a few extra components compared to a simple RNN:

- **Cell State (Cₜ):** This is the "long-term memory" that flows through the network, largely unchanged unless something updates it.
- **Hidden State (hₜ):** This is the "short-term memory" — it's what gets used to produce the output at each time step.
- **Forget Gate:** Decides what old information to throw away.
- **Input Gate:** Decides what new information to add to the memory.
- **Output Gate:** Decides what part of the memory to reveal as output right now.

Think of these three gates like a filing cabinet with a strict assistant:
- The **forget gate** throws out old files you no longer need.
- The **input gate** decides which new files are worth keeping.
- The **output gate** decides what to hand you right now, based on what's in the cabinet.

Each gate uses a **sigmoid function**, which squashes values between 0 and 1 — think of it as a dial that ranges from "completely block this information" (0) to "let it all through" (1).

### How does this fix the RNN's problem?

Because the cell state has a more direct path through time (with gates carefully controlling what changes), gradients can flow backward more easily during training. This means LSTMs are much better at learning dependencies that span many steps — like remembering the subject of a sentence all the way at the end of a long paragraph.

### Building a Simple LSTM in Python

The code looks almost identical to the RNN example — that's the beauty of these libraries, they let you swap in a more powerful layer with minimal changes:

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.utils import to_categorical

text = "hello world"
chars = sorted(set(text))
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

sequence_length = 3
X, y = [], []
for i in range(len(text) - sequence_length):
    X.append([char_to_idx[c] for c in text[i:i + sequence_length]])
    y.append(char_to_idx[text[i + sequence_length]])

X = np.array(X).reshape((len(X), sequence_length, 1))
y = to_categorical(y, num_classes=len(chars))

# The only real difference from the RNN example: we use LSTM instead of SimpleRNN
model = Sequential()
model.add(LSTM(50, input_shape=(sequence_length, 1)))
model.add(Dense(len(chars), activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy')

model.fit(X, y, epochs=200, verbose=0)
```

The only structural change from the RNN example is swapping `SimpleRNN(50, ...)` for `LSTM(50, ...)` — but under the hood, the LSTM is doing a lot more work to manage its memory using those forget/input/output gates.

### Real-World Uses of LSTMs

LSTMs are used all over the place because they're good at handling sequences with long-range dependencies:

- **Text generation** — writing new sentences in a certain style
- **Machine translation** — like Google Translate
- **Speech recognition** — powering assistants like Siri or Alexa
- **Time series prediction** — forecasting stock prices or weather
- **Sentiment analysis** — figuring out if a review is positive or negative
- **Video analysis** — understanding sequences of video frames
- **Handwriting recognition** — converting handwriting into digital text
- **Healthcare predictions** — predicting how a disease might progress
- **Music generation** — composing new music in a certain style
- **Anomaly detection** — spotting unusual patterns, like fraud

---

## Putting It All Together: How These Models Relate

Think of this as an evolution, where each model tries to fix the weaknesses of the one before it:

| Model | Main Idea | Key Weakness | What Fixed It |
|---|---|---|---|
| **N-grams** | Predict the next word using counts of recent word combinations | Only remembers a few words back; can't understand meaning | RNNs added the idea of a continuously updated "memory" |
| **HMMs** | Use hidden states + observations to model sequences probabilistically | Assumes the next state only depends on the current state (a simplifying assumption); doesn't capture deep patterns | Neural network-based models (RNNs) that learn richer patterns from data |
| **RNNs** | Use a hidden state passed step-by-step to "remember" previous inputs | Struggles with long-term memory due to vanishing/exploding gradients | LSTMs added gates to control memory more precisely |
| **LSTMs** | Use gates (forget, input, output) and a dedicated cell state to manage memory over long sequences | More complex and slower to train than simple RNNs | (Later architectures like Transformers built on these ideas further) |

Understanding this progression — from simple word-counting (N-grams), to probabilistic state-guessing (HMMs), to networks with memory (RNNs), to networks with *smart* memory (LSTMs) — gives you a strong foundation for understanding modern NLP and the more advanced models (like Transformers) that came after.
