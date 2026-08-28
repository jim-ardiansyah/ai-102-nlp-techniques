# Module 2: Getting Started with Natural Language Processing

Welcome! This module answers three big questions:

1. **What is NLP, actually?**
2. **Why should you care about it?**
3. **Why do we use Python to build it?**

Don't stress about memorizing every detail. By the end, you should just have a clear mental picture of what this field is about — the code will make a lot more sense once we start writing it together.

---

## 1. What Is Natural Language Processing?

### The Simple Definition

**Natural Language Processing (NLP)** is a branch of AI focused on one big question:

> *How do we get computers to work with human language?*

"Working with language" can mean a lot of things — reading it, figuring out what it means, or even generating brand-new language of its own.

### Why This Is Actually a Hard Problem

Computers are amazing at math and rules. Give a computer `2 + 2` and it will never mess that up. But human language isn't like that at all — it's messy, full of exceptions, slang, and double meanings, and it depends heavily on context.

Think about the difference between these two:
- `2 + 2` → always equals 4, no matter who asks or when.
- "That's just great" → could be genuine praise *or* total sarcasm, depending on tone and situation.

**NLP is the toolkit that bridges this gap** — taking something as flexible and slippery as everyday speech and turning it into something a machine can actually work with.

### You're Already Using NLP Every Day

You probably use NLP dozens of times a day without even noticing:

- **Typing a question into a search engine** — NLP helps the engine figure out what you *actually* mean, not just match exact words.
- **Google Translate** — NLP converts one language into another.
- **Siri or Alexa** — NLP understands what you're asking and responds sensibly.
- **Companies reading reviews** — NLP figures out if customers are happy or upset (called **sentiment analysis**).
- **Auto-summarizing a long article** — NLP shrinks it down to the key points.

None of this is easy under the hood — but that's exactly why it's such an interesting field to study.

### The Four Levels of "Understanding" Text

Imagine a computer climbing a staircase to fully understand a sentence. Each step builds on the one below it:

| Level | What It Figures Out | Simple Example |
|---|---|---|
| **1. Text Processing** | Cleaning and breaking text into pieces | Splitting "I'm running" into "I'm" and "running" |
| **2. Syntactic Analysis** | Grammar — which words are nouns, verbs, etc. | Knowing "dog" is a noun and "runs" is a verb |
| **3. Semantic Analysis** | Meaning — what words mean *in context* | Knowing "bank" means money here, not a river |
| **4. Pragmatic Analysis** | Intent — what the person *really* wants | Knowing "It's cold in here" is a hint to close the window |

Level 1 just asks "what are the words?" By Level 4, we're asking "what is this person actually trying to tell me?" That's a huge jump — and it's why NLP is genuinely difficult.

### A Quick Tour of Real NLP Applications

| Application | What It Does |
|---|---|
| **Search Engines** | Understand what you're really asking for |
| **Machine Translation** | Convert meaning between languages, not just words |
| **Chatbots & Virtual Assistants** | Understand requests and respond sensibly |
| **Sentiment Analysis** | Figure out how customers feel from reviews/posts |
| **Text Summarization** | Shrink long documents into key points |
| **Spam Detection** | Filter junk mail automatically |
| **Speech Recognition** | Turn spoken words into written text |
| **Recommendation Systems** | Suggest shows/products based on what you write and browse |
| **Healthcare** | Analyze clinical notes to support patient care |
| **Legal Tech** | Speed up contract and case-law review |

**Notice the pattern:** in every single one of these, messy human text gets turned into something a machine can actually act on. That's the whole game.

### Why Any of This Matters

Every day, humans produce a mind-boggling amount of text — emails, reviews, tweets, medical notes, legal contracts. Historically, all of that was locked away, readable only by a human, one document at a time. NLP is what lets us process it *at scale*.

**Some concrete examples:**
- A hospital can scan thousands of clinical notes to spot patterns in patient care.
- A law firm can review contracts in a fraction of the time it used to take.
- A business can find out, almost instantly, whether customers are happy or frustrated.

To be clear — NLP doesn't replace human judgment. It just makes the process of extracting insight from language dramatically faster.

### A First Taste of Code: Splitting Text Into Pieces

One of the very first steps in almost any NLP project is **tokenization** — breaking a sentence down into individual pieces called **tokens** (usually words or punctuation marks). Think of it like cutting up a sentence into individual puzzle pieces before you can start assembling anything with them.

```python
import nltk
nltk.download('punkt')  # downloads the tokenizer's data
from nltk.tokenize import word_tokenize

text = "Natural Language Processing (NLP) enables machines to understand human language."
tokens = word_tokenize(text)
print(tokens)
```

**Line by line:**
- `import nltk` — brings in NLTK, our toolbox for language tasks.
- `nltk.download('punkt')` — downloads a small pre-trained model that knows how to split text into sentences and words.
- `from nltk.tokenize import word_tokenize` — grabs the specific tool for splitting text into words.
- `word_tokenize(text)` — runs it on our sentence and gives us back a list of pieces.

**Output:**
```
['Natural', 'Language', 'Processing', '(', 'NLP', ')', 'enables', 'machines', 'to', 'understand', 'human', 'language', '.']
```

Notice that punctuation marks like `(`, `)`, and `.` become their own separate tokens.

**Why this matters:** almost every later step — counting words, tagging grammar, feeding text into a machine learning model — needs text broken into these clean pieces *first*. Tokenization is the small but essential first domino in the chain.

### The Honest Challenges of NLP

NLP isn't smooth sailing. Here are the recurring headaches you'll see throughout this course — understanding *why* they're hard is just as important as knowing they exist.

| Challenge | Why It's Hard | Example |
|---|---|---|
| **Ambiguity** | One word, multiple meanings | "bank" = money *or* riverbank |
| **Context** | Meaning shifts based on nearby words | "bat" flew away vs. he swung the "bat" |
| **Language Diversity** | Grammar/vocabulary differ hugely across languages | A model trained on English struggles with Chinese or Arabic |
| **Idioms** | Phrases don't mean what the words literally say | "kick the bucket" ≠ actually kicking a bucket |
| **Sarcasm & Irony** | Tone flips the literal meaning | "Oh great, another traffic jam" (said unhappily) |
| **Named Entity Recognition** | Spotting names of people/places/orgs is trickier than it looks | Inconsistent capitalization confuses models |
| **Sentiment Analysis** | Emotion is nuanced, and can be mixed within one sentence | "The food was great but the service was awful" |
| **Domain Knowledge** | Different fields use very different vocabulary | Medical language ≠ everyday language |
| **Scale** | Real systems need to process huge amounts of text, fast | Millions of tweets per hour |
| **Fairness** | Training data can carry hidden biases | A biased dataset produces a biased model |

Keep this list in the back of your mind — we'll come back to many of these challenges as the course goes on. They're exactly why NLP is still such an active research area today.

---

## 2. Why NLP Matters — and Where We See It in Action

### Five Reasons NLP Is a Big Deal

**1. Enhanced Communication**
NLP lets people interact with technology using ordinary language instead of memorizing rigid commands. That's a huge accessibility win — more people, regardless of technical background, can comfortably use modern technology.

**2. Automating Repetitive Work**
Tedious tasks — sorting emails, filtering spam, routing customer questions — can now be handled automatically, freeing people up for more creative, higher-value work.

**3. Accessibility**
Speech recognition and text-to-speech open doors for people with visual or hearing impairments. Screen readers, real-time transcription, and voice commands all rely on NLP.

**4. Making Sense of Data**
We generate a staggering amount of text every day. NLP turns that sea of unstructured text into organized, structured insight we can actually use.

**5. Personalization**
By analyzing what you write, search for, and interact with, NLP helps platforms tailor content to *you* — like a streaming service recommending shows, or a store suggesting products based on your reviews and browsing habits.

### Applications, With Code You Can Actually Run

**Search Engines**

When you type "best restaurants near me," NLP figures out you want highly-rated places close to you — not just any page that happens to contain those exact words.

**Machine Translation**

Good translation has to preserve grammar, tone, and cultural nuance — not just swap words one-for-one.

```python
from translate import Translator

translator = Translator(to_lang="es")
translation = translator.translate("How are you?")
print(translation)  # ¿Cómo estás?
```

We create a `Translator` aimed at Spanish (`"es"`), then call `.translate()` on our English phrase — the library returns a natural-sounding Spanish equivalent.

**Chatbots and Virtual Assistants**

When you say "play some music" to a voice assistant, NLP decodes what you *mean*, matches it to an action, and the assistant carries it out. It feels effortless, but several layers of language understanding are working together behind the scenes.

**Sentiment Analysis**

Businesses use this to figure out — automatically, at scale — whether feedback is positive, negative, or neutral.

```python
from textblob import TextBlob

text = "I love this product! It's amazing."
blob = TextBlob(text)
sentiment = blob.sentiment
print(sentiment)  # Sentiment(polarity=0.65, subjectivity=0.6)
```

This gives us two numbers:
- **Polarity** ranges from -1 (very negative) to 1 (very positive). Here, 0.65 tells us this is clearly a positive sentence.
- **Subjectivity** ranges from 0 (pure fact) to 1 (pure opinion). Here, 0.6 tells us this is a fairly personal, opinion-based statement.

**Text Summarization**

Long documents can be automatically shrunk down to their most important sentences.

```python
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

text = """
Natural Language Processing (NLP) is a fascinating field at the
intersection of computer science, artificial intelligence, and
linguistics. It enables machines to understand, interpret, and
generate human language, opening up applications from chatbots
and translation to sentiment analysis and beyond.
"""

parser = PlaintextParser.from_string(text, Tokenizer("english"))
summarizer = LsaSummarizer()
summary = summarizer(parser.document, 2)  # keep 2 sentences

for sentence in summary:
    print(sentence)
```

We wrap our text in a parser, then hand it to an `LsaSummarizer`, asking for a two-sentence summary. Rather than just cutting the text short, it picks out the sentences that best represent the whole passage.

**Healthcare, Legal Tech, and E-Commerce**

- **Healthcare** — NLP helps extract useful information from clinical notes and predict patient outcomes.
- **Legal Tech** — speeds up contract review and flags compliance risks buried in dense legal text.
- **E-Commerce** — powers product recommendations, smarter search, customer-service chatbots, and mining reviews for recurring complaints or praise.

### A Worked Example: Scoring Customer Reviews Automatically

Let's tie these ideas together with something realistic: scanning a handful of reviews and scoring their sentiment.

```python
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

reviews = [
    "This product is fantastic! It exceeded my expectations.",
    "Not worth the price. I'm disappointed with the quality.",
    "Good value for money. Will buy again.",
]

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

for review in reviews:
    sentiment = sia.polarity_scores(review)
    print(f"Review: {review}")
    print(f"Sentiment: {sentiment}")
```

We use NLTK's **VADER** tool, which is specifically tuned for short, informal text like reviews and social posts. For each review, `polarity_scores()` returns four numbers:
- `neg`, `neu`, `pos` — how negative, neutral, and positive the text is.
- `compound` — an overall score summarizing the sentiment in a single value from -1 to 1.

A glowing review lands with a high positive and compound score; a critical review skews negative; a mixed review like "Good value for money. Will buy again." lands somewhere balanced but still leans positive overall.

Imagine running this across thousands of reviews instead of just three — that's the real power here.

---

## 3. Why We'll Be Using Python

Every field needs a good workshop full of tools — for NLP, Python is that workshop. Let's talk about why, then get your environment set up.

### What Makes Python a Great Fit for NLP

- **Readable and Simple** — Python's clean syntax lets you focus on the *ideas* behind an algorithm instead of fighting with the language itself.
- **Rich Libraries** — tools like NLTK, spaCy, and gensim come with ready-made functions and pre-trained models, so you're rarely starting from scratch.
- **A Large Community** — tons of tutorials, documentation, and forums to lean on when you get stuck.
- **Strong Machine Learning Integration** — Python connects smoothly with TensorFlow, PyTorch, and scikit-learn, so you can move from basic text processing all the way to full machine learning models without switching languages.

### Meet the Core Libraries

Think of these four libraries as different tools in a toolbox — each one is best suited for a different job.

**🔧 NLTK — the Natural Language Toolkit**

One of the oldest, most complete NLP libraries. Great for *learning*, because it shows you the building blocks (tokenizing, stemming, lemmatizing) very transparently.

```python
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

text = "Natural Language Processing with Python is fun!"
tokens = word_tokenize(text)
print(tokens)
```

**Output:** `['Natural', 'Language', 'Processing', 'with', 'Python', 'is', 'fun', '!']`

**⚡ spaCy — built for speed and real-world use**

Where NLTK is great for learning fundamentals, spaCy is built for production — fast, efficient, and comes with strong pre-trained models. Let's use it for **Named Entity Recognition (NER)** — automatically spotting names of people, places, and organizations.

```python
import spacy

nlp = spacy.load("en_core_web_sm")
text = "Apple is looking at buying U.K. startup for 1 billion."
doc = nlp(text)

for ent in doc.ents:
    print(ent.text, ent.label_)
```

We load a small pre-trained English model, run it over our sentence, then loop through `doc.ents` — the entities spaCy found, each labeled with its type.

**Output:**
```
Apple   → ORG    (an organization)
U.K.    → GPE    (a geopolitical entity)
1 billion → MONEY
```

Notice spaCy correctly tells apart the company, the country, and the dollar amount — all from one plain, un-annotated sentence.

**🧩 gensim — topic modeling and word meaning**

gensim shines when you want to understand relationships between words and documents — for example, training a **Word2Vec** model, which represents each word as a list of numbers capturing its meaning based on the words it tends to appear near.

```python
from gensim.models import Word2Vec

sentences = [
    ["natural", "language", "processing"],
    ["python", "is", "a", "powerful", "language"],
    ["text", "processing", "with", "gensim"],
]

model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
vector = model.wv['language']
print(vector)
```

**What each setting means:**
- `vector_size` — how many numbers represent each word (more numbers can capture richer meaning, at a computational cost).
- `window` — how many neighboring words count as "context."
- `min_count` — ignores words that are too rare.
- `workers` — how many CPU threads to use during training.

The result: a 100-number vector for the word "language" — words used in similar contexts end up with similar-looking vectors. This kind of representation powers text classification, clustering similar documents, recommendation systems, and measuring how semantically similar two pieces of text are.

**📊 scikit-learn — turning text into predictions**

A general-purpose machine learning library that pairs beautifully with NLP once your text has been turned into numbers.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

texts = ["I love this product", "This is the worst experience",
         "Absolutely fantastic!", "Not good at all"]
labels = [1, 0, 1, 0]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

classifier = MultinomialNB()
classifier.fit(X, labels)

new_text = ["I hate this"]
X_new = vectorizer.transform(new_text)
prediction = classifier.predict(X_new)
print(prediction)  # [0]
```

**The flow:**
1. `CountVectorizer` turns our sentences into a table of word counts.
2. We train a `MultinomialNB` classifier on that table, alongside our labels (1 = positive, 0 = negative).
3. We transform a brand-new sentence the same way and ask the trained classifier to predict its sentiment.

Result: `[0]` — correctly predicted as negative.

### Setting Up Your Own Environment

We’ll use Google Colab for our Python programming in this course, so you don’t need to install Python or any additional software on your computer.

However, if you prefer to run Python on your local machine, you can follow the instructions below to install and set up the necessary tools. Running Python locally is completely optional—Google Colab is recommended for this course because it provides an easy setup and allows you to start coding right away.

**Step 1 — Install Python**

Download the latest version from [python.org/downloads](https://python.org/downloads) for your operating system. During installation, check the box that adds Python to your system PATH — this lets you run Python from the command line. Confirm it worked:

```bash
python --version
```

**Step 2 — Create a Virtual Environment**

Think of a virtual environment as a separate, clean toolbox for each project — it keeps each project's libraries from interfering with each other. This saves a lot of headaches later.

```bash
python -m venv nlp_env

# On Windows:
nlp_env\Scripts\activate

# On macOS/Linux:
source nlp_env/bin/activate
```

**Step 3 — Install the Libraries We'll Use**

```bash
pip install nltk spacy gensim scikit-learn
```

**Step 4 — Download the Language Resources**

A few libraries need extra data files before they'll work:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
```

```bash
python -m spacy download en_core_web_sm
```

**Step 5 — Confirm Everything Works**

Save this as `test_nlp.py` and run it. If each library prints sensible output, your environment is ready.

```python
import nltk
from nltk.tokenize import word_tokenize
import spacy
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('punkt')
text = "Natural Language Processing with Python is fun!"
tokens = word_tokenize(text)
print("NLTK Tokens:", tokens)

nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
print("SpaCy Tokens:", [token.text for token in doc])

sentences = [["natural", "language", "processing"], ["python", "is", "fun"]]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
print("Word2Vec Vocabulary:", list(model.wv.index_to_key))

vectorizer = CountVectorizer()
X = vectorizer.fit_transform([text])
print("CountVectorizer Feature Names:", vectorizer.get_feature_names_out())
```

### Putting It All Together: A Mini Pipeline

Let's close by combining several tools into one small end-to-end pipeline: text processing → feature extraction → classification, all chained together. This is a preview of how real NLP projects are structured.

```python
import nltk
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords

nltk.download('stopwords')

texts = [
    "I love this product! It's amazing.",
    "This is the worst experience I've ever had.",
    "Absolutely fantastic! Highly recommend.",
    "Not good at all. Very disappointing."
]
labels = [1, 0, 1, 0]

nlp = spacy.load("en_core_web_sm")

def spacy_tokenizer(sentence):
    doc = nlp(sentence)
    return [token.text for token in doc]

stop_words = set(stopwords.words('english'))

pipeline = Pipeline([
    ('vectorizer', CountVectorizer(tokenizer=spacy_tokenizer, stop_words=stop_words)),
    ('classifier', MultinomialNB())
])

pipeline.fit(texts, labels)

new_text = ["I hate this product"]
prediction = pipeline.predict(new_text)
print(prediction)  # [0]
```

**Each piece plays a specific role, like an assembly line:**

| Step | Tool | Job |
|---|---|---|
| 1 | spaCy | Tokenizes the text (splits it into words) |
| 2 | NLTK stopwords | Filters out common filler words |
| 3 | `CountVectorizer` | Turns the cleaned text into numbers |
| 4 | `MultinomialNB` | Learns to classify sentiment from those numbers |

Wrapping it all in a `Pipeline` means we can train and predict with just two clean method calls — `fit()` and `predict()` — instead of juggling every step by hand.

---

## Wrapping Up

**The big picture:** NLP is the field devoted to helping machines understand and generate human language, and it already touches nearly every corner of modern technology — from search engines to healthcare to customer service.

**Why Python?** Because it's approachable, well-supported, and backed by a rich ecosystem of libraries, each good at a different job:

| Library | Best For |
|---|---|
| **NLTK** | Learning the fundamentals |
| **spaCy** | Speed and production use |
| **gensim** | Word meaning and topic modeling |
| **scikit-learn** | Turning text into predictions |

In our next session, we'll roll up our sleeves and start building with these tools ourselves. Make sure your environment is set up beforehand so we can dive straight into the hands-on work. See you then!
