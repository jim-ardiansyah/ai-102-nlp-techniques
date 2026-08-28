# Module 4: Feature Engineering for NLP 

Computers are great with numbers, but they don't understand words the way we do. Before a computer can do anything useful with text — like figuring out if a review is positive or negative, or answering a question — we first have to turn words into numbers.

This guide walks through four ways of doing that, from the simplest to the most advanced:

1. **Bag of Words** — just count the words
2. **TF-IDF** — count the words, but smarter
3. **Word Embeddings (Word2Vec & GloVe)** — give words "meaning"
4. **BERT** — understand words *in context*

Think of it like learning to cook: Bag of Words is a microwave dinner (fast, simple, gets the job done). BERT is a five-course meal (takes more time and skill, but tastes way better). Let's go through each one.

---

## 1. Bag of Words (BoW): Just Count the Words

### The Big Idea

Imagine you dump all the words from a sentence into a bag, shake it up, and don't care about the order anymore — just how many times each word shows up. That's Bag of Words.

It **throws away grammar and word order** and only keeps track of **which words appear and how often**. It sounds too simple to work, but it's a great starting point and still used today.

### How It Works (3 Simple Steps)

**Step 1: Split the text into words (Tokenizing)**

Take a sentence and break it into individual words.

> "Natural language processing is fun"
> → `["Natural", "language", "processing", "is", "fun"]`

**Step 2: Make a list of every unique word (Building a Vocabulary)**

Look at *all* your sentences together and make one big list of every different word that shows up (no duplicates).

Say we have two sentences:
- "Natural language processing is fun"
- "Language models are important in NLP"

Our vocabulary (usually lowercased) is:
```
["natural", "language", "processing", "is", "fun", "models", "are", "important", "in", "nlp"]
```

**Step 3: Turn each sentence into a row of numbers (Vectorizing)**

For each sentence, count how many times each vocabulary word appears, in the same order as the vocabulary list.

| word → | natural | language | processing | is | fun | models | are | important | in | nlp |
|---|---|---|---|---|---|---|---|---|---|---|
| **Sentence 1** | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| **Sentence 2** | 0 | 1 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 |

That's it! Each sentence is now a list of numbers a computer can work with.

### Trying It in Python

```python
from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "Natural language processing is fun",
    "Language models are important in NLP"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)

print("Vocabulary:", vectorizer.get_feature_names_out())
print("Word counts:\n", X.toarray())
```

`CountVectorizer` does all three steps for you automatically — it builds the vocabulary and counts the words in one line of code.

### 👍 Why People Like It
- **Super easy to understand** — no complicated math.
- **Fast** — works well for small and medium amounts of text.
- **Good starting point** — a simple baseline to compare fancier methods against.

### 👎 Where It Falls Short
- **Word order doesn't matter** — "dog bites man" and "man bites dog" look identical to Bag of Words. That's a problem, since these mean very different things!
- **Big vocabulary = big vectors** — with thousands of unique words, each sentence turns into a huge row of mostly zeros.
- **Mostly zeros (sparsity)** — wastes memory and computing power.

### A Quick Real Example: Sorting Text into Categories

You can use these word-count vectors to train a simple classifier (like a spam filter). Give the computer some labeled examples ("this is about NLP" vs. "this is about AI"), and a simple algorithm called Naive Bayes can learn to sort new sentences into the right category — often with very high accuracy on easy examples.

---

## 2. TF-IDF: Counting Words, But Smarter

### The Problem With Just Counting

Bag of Words treats every word equally. But words like "the," "is," and "and" show up *everywhere* and don't tell you much about what a document is really about. Meanwhile, a rare word like "photosynthesis" tells you a lot. TF-IDF fixes this by giving important, rare words a bigger score and common, boring words a smaller score.

### The Two Ingredients

**TF (Term Frequency)** — how often a word shows up *in this one document*. Simple.

$$TF = \frac{\text{times the word appears in this document}}{\text{total words in this document}}$$

**IDF (Inverse Document Frequency)** — how *rare* a word is across *all* your documents. If a word shows up in almost every document (like "the"), it gets a low score. If it only shows up in a few documents, it gets a high score.

$$IDF = \log\left(\frac{\text{total number of documents}}{\text{number of documents that contain the word}}\right)$$

**Put them together:**

$$TF\text{-}IDF = TF \times IDF$$

**In plain English:** *"How often does this word show up here, multiplied by how rare it is everywhere else?"* A word gets a high TF-IDF score only if it's common in this one document but uncommon everywhere else.

### Trying It in Python

```python
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "Natural language processing is fun",
    "Language models are important in NLP",
    "I enjoy learning about artificial intelligence",
    "Machine learning and NLP are closely related",
    "Deep learning is a subset of machine learning"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

print("Vocabulary:", vectorizer.get_feature_names_out())
print("TF-IDF scores:\n", X.toarray())
```

This works almost exactly like `CountVectorizer`, except instead of raw counts, you get weighted importance scores.

### Bag of Words vs. TF-IDF, Side by Side

| | Bag of Words | TF-IDF |
|---|---|---|
| What it tracks | Raw word counts | Weighted importance |
| Treats "the" and "important-word" | The same | Differently — downweights "the" |
| Best for | Quick and simple tasks | Search engines, document comparison, classification |

### 👍 Why People Like It
- Highlights the words that actually matter in a document.
- Reduces "noise" from common filler words.
- Works well for search engines, spam filters, and grouping similar documents.

### 👎 Where It Falls Short
- Still doesn't understand word *meaning* — it just does clever counting.
- Still creates big, mostly-empty (sparse) vectors for large vocabularies.
- Doesn't know that "happy" and "joyful" mean similar things.

---

## 3. Word Embeddings: Giving Words "Meaning"

### The Big Idea

Here's the limitation of everything above: to a computer, "happy" and "joyful" are just two completely different strings of letters — even though we know they mean almost the same thing. Word embeddings fix this.

Instead of counting words, embeddings place every word at a **point in space** (imagine a map, but with maybe 100 dimensions instead of 2). Words with similar meanings end up **close together** on this map. Words like "king" and "queen" land near each other. "Pizza" and "car" land far apart.

Because words are now points in space, you can do surprisingly cool math with them:

$$\vec{king} - \vec{man} + \vec{woman} \approx \vec{queen}$$

That equation is basically the computer "understanding" that the relationship between "king" and "man" is similar to the relationship between "queen" and "woman" (both are about gender).

### Two Popular Ways to Build These Maps

**Word2Vec (built by Google)**

Word2Vec learns by playing a fill-in-the-blank game with lots of text. It has two modes:

- **CBOW (Continuous Bag of Words):** Given the words around a blank, guess the missing word.
  *"The ___ sat on the mat"* → probably "cat"
- **Skip-Gram:** The opposite — given one word, guess what words are likely to appear around it.

By playing this game millions of times over huge amounts of text, the model learns which words tend to appear in similar situations — and gives them similar vector positions.

**GloVe (built at Stanford)**

GloVe takes a different approach: instead of playing fill-in-the-blank, it counts how often every pair of words appears together *anywhere* in a huge collection of text, then uses that big counting table to place words on the map. It captures more of the "big picture" patterns across the whole text collection.

**Rule of thumb:** Word2Vec learns from a small window around each word (fast, good with huge datasets). GloVe learns from the whole corpus's statistics at once (often captures broader patterns).

### Trying It in Python

```python
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
nltk.download('punkt')

text = ("Natural language processing is fun and exciting. Language "
        "models are important in NLP. I enjoy learning about artificial "
        "intelligence. Machine learning and NLP are closely related. "
        "Deep learning is a subset of machine learning.")

sentences = sent_tokenize(text)
tokenized_sentences = [word_tokenize(s) for s in sentences]

# Train your own mini Word2Vec model
model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, sg=1, min_count=1)

# See the vector for "language"
print(model.wv['language'])

# Find words with similar meaning
print(model.wv.most_similar('language'))
```

Or, instead of training your own, you can download embeddings someone else already trained on huge amounts of text (like all of Wikipedia):

```python
import gensim.downloader as api

glove_model = api.load("glove-wiki-gigaword-100")
print(glove_model.most_similar('language'))
# → ['languages', 'linguistic', 'bilingual', 'translation', ...]
```

Using someone else's pre-trained embeddings is called **transfer learning** — you get the benefit of training on billions of words without having to do that training yourself.

### 👍 Why People Like It
- **Captures meaning**, not just spelling — similar words get similar vectors.
- **Compact** — usually only 100–300 numbers per word, much smaller than Bag of Words vectors.
- **Reusable** — pre-trained embeddings save huge amounts of time.

### 👎 Where It Falls Short
- **One word = one vector, always.** The word "bank" gets the *same* vector whether you mean a riverbank or a savings account. That's a problem, because words can mean different things in different sentences.
- **Unknown words are a mystery** — if a word wasn't in the training data, the model has no vector for it at all.

---

## 4. BERT: Understanding Words *in Context*

### The Big Idea

Remember the "bank" problem? BERT solves it. Instead of giving every word one fixed vector forever, BERT looks at the **entire sentence** and generates a *different* vector for a word depending on how it's being used.

> "The **bank** can guarantee deposits will remain safe." → *financial institution*
> "We sat by the river**bank** and watched the water." → *edge of a river*

BERT reads the whole sentence — words *before and after* — at the same time, which is why it's called **bidirectional**. Older models could only read left-to-right (like reading a book), so they never got to "see the ending" before deciding what an earlier word meant.

### How BERT Learns

BERT gets trained by playing two games on massive amounts of text (like all of Wikipedia):

**Game 1: Fill in the blank (Masked Language Modeling)**

Take a sentence, hide ("mask") a random word, and make BERT guess it using everything else in the sentence.

> "The quick brown [MASK] jumps over the lazy dog." → BERT guesses "fox"

To guess correctly, BERT has to actually understand how the sentence fits together — not just memorize word order.

**Game 2: Does this sentence come next? (Next Sentence Prediction)**

Show BERT two sentences and ask: "does the second one logically follow the first?" This teaches BERT how sentences relate to each other, which helps with tasks like answering questions or summarizing.

### From "Pre-trained" to "Fine-tuned"

Training BERT from scratch takes enormous amounts of data and computing power — way more than any of us can do at home. The good news: someone already did it for you.

- **Pre-training** = BERT already learned general language skills from a massive amount of text (done once, by Google).
- **Fine-tuning** = you take that already-smart model and give it a smaller, specific task (like "is this email spam?") using your own, much smaller dataset. BERT adjusts slightly to specialize in your task while keeping everything it already knows about language.

This is like hiring someone who's already fluent in English and just needs a quick training session for your specific job — much faster than teaching them English from scratch.

### Trying It in Python

```python
from transformers import BertTokenizer, BertModel
import torch

# Load a version of BERT that's already been pre-trained
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

text = "Natural Language Processing is fascinating."
inputs = tokenizer(text, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)

# A vector that represents the whole sentence
sentence_vector = outputs.last_hidden_state[:, 0, :]
print(sentence_vector)
```

### 👍 Why People Like It
- **Understands context** — the same word gets different vectors depending on its meaning in that sentence.
- **State-of-the-art results** — as of when this material was written, BERT-style models led the pack on many NLP benchmarks (answering questions, sentiment analysis, and more).
- **Reusable and adaptable** — thanks to fine-tuning, you don't need millions of examples to get great results on your own task.

### 👎 Where It Falls Short
- **Needs serious computing power** — training or even running BERT well usually requires a GPU, not just a regular laptop.
- **More complicated** — lots of moving parts (attention layers, tokenizers, fine-tuning steps) means a steeper learning curve than Bag of Words or TF-IDF.

---

## Putting It All Together

| Method | Basic Idea | Understands Meaning? | Understands Context? | How Hard to Use? |
|---|---|---|---|---|
| **Bag of Words** | Count the words | ❌ No | ❌ No | 🟢 Very easy |
| **TF-IDF** | Count words, weighted by importance | ❌ No | ❌ No | 🟢 Easy |
| **Word2Vec / GloVe** | Place words on a "meaning map" | ✅ Yes | ❌ No (one vector per word) | 🟡 Medium |
| **BERT** | Understand words based on the full sentence | ✅ Yes | ✅ Yes | 🔴 Advanced |

**The big picture:** each method is built to fix the previous one's biggest weakness.
- Bag of Words ignores importance → **TF-IDF** fixes that.
- TF-IDF still doesn't understand meaning → **Word embeddings** fix that.
- Embeddings give every word only one fixed meaning → **BERT** fixes that by reading the whole sentence.

As a beginner, a good learning path is: start by trying Bag of Words and TF-IDF yourself (they're quick to code and easy to understand), then move on to pre-trained Word2Vec/GloVe embeddings, and once you're comfortable, experiment with BERT using Hugging Face's `transformers` library.
