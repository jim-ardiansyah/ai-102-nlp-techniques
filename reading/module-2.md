# Chapter 1: Getting Started with Natural Language Processing
*Lecture Notes — Module 1*

Welcome, everyone. In this first set of notes, I want to give you a solid, friendly introduction to Natural Language Processing — or "NLP" for short. We'll cover three things: what NLP actually is, why it matters so much in the world around us, and why Python has become the go-to language for building NLP projects. Don't worry about memorizing every detail on your first read. My goal here is for you to walk away with a clear mental picture of the field before we start writing code together.

## 1.1 What Is Natural Language Processing?

Let's start with a simple definition. Natural Language Processing is a branch of artificial intelligence concerned with one big question: how can we get computers to work with human language? "Working with" language can mean many things — reading it, understanding what it means, and even producing new language of its own.

Think about it this way. Computers are naturally good with numbers and rigid rules, but human language is messy, full of exceptions, and often depends on context. NLP is the toolkit we use to bridge that gap — to take something as flexible as everyday speech or writing and make it usable for a machine.

You already use NLP every day, probably without noticing it. When you type a question into a search engine, NLP helps that engine figure out what you actually mean. When you use Google Translate, NLP is doing the heavy lifting of converting one language into another. Voice assistants like Siri or Alexa rely on NLP to understand your spoken requests and respond sensibly. Companies also use NLP to read through thousands of customer reviews and figure out whether people are happy or upset — this is called sentiment analysis. And when a tool automatically shrinks a long article down into a short summary, that's NLP too.

Of course, none of this is easy for a machine. Human language is ambiguous, deeply tied to context, and incredibly diverse across cultures. Building systems that handle these challenges well requires thoughtful design and clever algorithms, which is exactly what we'll be building toward this semester.

### 1.1.1 Defining the Scope of NLP

NLP is often described as sitting at the intersection of three fields: linguistics, computer science, and artificial intelligence. To make sense of such a broad field, it helps to break it down into a few layers of analysis. Think of these as different "levels of understanding" a computer must climb through to truly grasp a piece of text.

- **Text Processing** — the groundwork. This includes tokenization (splitting text into individual words or pieces), stemming and lemmatization (reducing words down to their root or base form), and cleaning up unwanted symbols or filler words.
- **Syntactic Analysis** — understanding grammar. This step figures out the structure of a sentence: which words are nouns, which are verbs, and how they relate to one another.
- **Semantic Analysis** — understanding meaning. Here the system tries to figure out what words actually mean given their context, and what role each word plays relative to the main action of the sentence.
- **Pragmatic Analysis** — understanding intent. This is the deepest and hardest level: figuring out what a speaker really means, taking into account context, culture, and situation — not just the literal words.

You can think of these four levels as a staircase. Each step builds on the one below it, moving from "What are the words?" all the way up to "What is this person actually trying to tell me?"

### 1.1.2 A Quick Tour of NLP Applications

Let's look at a few real-world applications so this feels less abstract.

- **Search Engines**: help interpret what you're really asking for, not just matching keywords.
- **Machine Translation**: services like Google Translate convert meaning — not just words — between languages.
- **Chatbots and Virtual Assistants**: Siri, Alexa, and similar tools understand spoken or typed requests and generate a sensible reply.
- **Sentiment Analysis**: businesses scan reviews and social media posts to understand how customers feel.
- **Text Summarization**: long documents get condensed into short, digestible summaries — useful in law, research, and journalism.
- **Spam Detection**: email providers use NLP to recognize and filter out junk mail.
- **Speech Recognition**: spoken words get converted into written text for transcription and voice control.
- **Recommendation Systems**: platforms like Netflix or Amazon read your reviews and history to suggest what you might like next.
- **Healthcare**: clinical notes and research papers get analyzed to support better patient care.
- **Legal Tech**: law firms use NLP to speed up the review of contracts and case law.

Notice the pattern across all of these examples: somewhere, unstructured human text is being turned into something a machine can act on.

### 1.1.3 Why Should You Care About NLP?

Here's the big picture. NLP matters because it turns an enormous, mostly untapped resource — the world's text and speech — into something we can actually analyze and use. Every day, we produce staggering amounts of language: emails, reviews, articles, tweets, medical notes, legal contracts. Most of this data used to be locked away, readable only by humans, one document at a time. NLP gives us the tools to process it at scale.

This matters in very concrete ways. A hospital can scan thousands of clinical notes to spot patterns in patient care. A law firm can review contracts in a fraction of the time it used to take. A business can understand, almost instantly, whether its customers are happy or frustrated. None of this replaces human judgment — but it dramatically speeds up the process of getting useful insight out of language.

### 1.1.4 A First Taste of Code: Tokenization

Let's ground this with a small, concrete example. One of the very first steps in almost any NLP pipeline is tokenization — breaking a sentence down into individual pieces, called tokens. These tokens are usually words or punctuation marks.

Here's how you might do this in Python using a popular library called NLTK (short for the Natural Language Toolkit):

```python
import nltk
nltk.download('punkt')  # download the tokenizer's data
from nltk.tokenize import word_tokenize

text = "Natural Language Processing (NLP) enables machines to understand human language."
tokens = word_tokenize(text)
print(tokens)
```

Let's walk through what's happening, one line at a time:

- `import nltk` — brings in the NLTK library, our toolbox for language tasks.
- `nltk.download('punkt')` — downloads a small pre-trained model that knows how to split text into sentences and words.
- `from nltk.tokenize import word_tokenize` — pulls in the specific function we need for word-level tokenization.
- `word_tokenize(text)` — runs the tokenizer on our sample sentence and hands back a list of tokens.

> Running this code produces: `['Natural', 'Language', 'Processing', '(', 'NLP', ')', 'enables', 'machines', 'to', 'understand', 'human', 'language', '.']` — notice that punctuation marks become their own tokens too.

Why does this matter? Because almost every later step in an NLP pipeline — counting word frequencies, tagging parts of speech, feeding text into a machine learning model — needs text broken into these clean, discrete pieces first. Tokenization is the humble but essential first step.

### 1.1.5 The Honest Challenges of NLP

I don't want to give you the impression this is all smooth sailing — NLP is genuinely hard, and it's worth understanding why up front, because these challenges will come up again and again this semester.

- **Ambiguity**: a single word can mean different things. "Bank" could mean a financial institution or a riverbank.
- **Context**: the same word shifts meaning depending on what surrounds it — think about how differently "bat" is used in "the bat flew away" versus "he swung the bat."
- **Language Diversity**: grammar and vocabulary vary enormously across languages, so a model trained on English rarely works well on Chinese or Arabic without real adaptation.
- **Idioms**: phrases like "kick the bucket" don't mean what their individual words suggest.
- **Sarcasm and Irony**: a sentence like "Oh great, another traffic jam" is negative in tone despite using a positive word.
- **Named Entity Recognition**: correctly spotting names of people, places, and organizations is trickier than it sounds, especially with inconsistent capitalization.
- **Sentiment Analysis**: human emotion is nuanced — a review might express mixed or conflicting feelings in the same sentence.
- **Domain Knowledge**: medical, legal, and everyday language all use very different vocabularies, so models often need to be tailored to a specific domain.
- **Scale**: real-world systems need to process huge volumes of text quickly, sometimes in real time.
- **Fairness**: training data can carry hidden biases, and those biases can end up baked into the model's behavior — something we always need to watch for.

We'll return to many of these challenges throughout the course as we build increasingly capable tools. For now, just keep them in the back of your mind — they're the reason NLP is such an active and evolving research area.

## 1.2 Why NLP Matters — and Where We See It in Action

Now that you know what NLP is, let's spend some time appreciating just how much it shapes our daily lives — and look at a handful of hands-on examples so the ideas stick.

### 1.2.1 Five Reasons NLP Is a Big Deal

**Enhanced Communication**
NLP lets people interact with technology using ordinary, everyday language instead of memorizing rigid commands. That might sound small, but it's a huge accessibility win — it means more people, regardless of technical background, can use modern technology comfortably.

**Automating Repetitive Work**
A lot of tedious, repetitive tasks — sorting emails, filtering spam, routing customer questions — can now be handled automatically. This frees up people to spend their time on more creative, higher-value work, while machines take care of the routine parts.

**Accessibility**
Speech recognition and text-to-speech technology open doors for people with visual or hearing impairments. Real-time transcription, screen readers, and voice commands all rely on NLP to make digital spaces more inclusive.

**Making Sense of Data**
We generate a staggering amount of text every day — social posts, reviews, business documents, medical records. NLP is what lets us turn that sea of unstructured text into organized, structured insight that we can actually act on.

**Personalization**
By analyzing what you write, search for, and interact with, NLP helps platforms tailor content specifically to you — think of how a streaming service recommends shows, or how an online store suggests products based on your past reviews and browsing habits.

### 1.2.2 Applications, With Examples You Can Run

**Search Engines**
When you type "best restaurants near me" into a search engine, NLP is working behind the scenes to figure out that you want highly-rated places close to your current location — not just pages that happen to contain those exact words.

**Machine Translation**
Translation tools have to do more than swap words — they need to preserve grammar, tone, and cultural nuance. Here's a simple example using Python's `translate` library:

```python
from translate import Translator

translator = Translator(to_lang="es")
translation = translator.translate("How are you?")
print(translation)  # ¿Cómo estás?
```

We create a `Translator` object aimed at Spanish ("es"), then call `.translate()` on our English phrase. The library takes care of returning a natural-sounding Spanish equivalent.

**Chatbots and Virtual Assistants**
When you say "play some music" to a voice assistant, NLP decodes your intent, matches it to an action, and the assistant carries it out. It feels effortless to us, but underneath, several layers of language understanding are working together.

**Sentiment Analysis**
Businesses use sentiment analysis to understand whether feedback is positive, negative, or neutral, at scale. Here's an example using the TextBlob library:

```python
from textblob import TextBlob

text = "I love this product! It's amazing."
blob = TextBlob(text)
sentiment = blob.sentiment
print(sentiment)  # Sentiment(polarity=0.65, subjectivity=0.6)
```

The result gives us two numbers. Polarity ranges from -1 (very negative) to 1 (very positive) — 0.65 here tells us the sentence is clearly positive. Subjectivity ranges from 0 (purely factual) to 1 (purely opinion) — 0.6 tells us this is a fairly personal, opinion-based statement.

**Text Summarization**
Long documents can be automatically condensed to their most important sentences. Here's an example using the `sumy` library and a technique called Latent Semantic Analysis (LSA), which identifies the most representative sentences in a document based on patterns in word usage:

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

We first wrap our text in a parser, then hand it to an `LsaSummarizer`, asking for a two-sentence summary. The summarizer picks out the most representative sentences rather than simply cutting the text short.

**Healthcare, Legal Tech, and E-Commerce**
In healthcare, NLP helps clinicians extract useful information from clinical notes and predict patient outcomes from historical data. In the legal field, it speeds up contract review and helps flag compliance risks buried in dense legal text. In e-commerce, it powers product recommendations, smarter search, and customer-service chatbots — while also mining reviews for recurring complaints or praise that a business can act on.

### 1.2.3 A Worked Example: Analyzing Customer Reviews

Let's tie several of these ideas together with a small, realistic example: scanning a handful of customer reviews and scoring their sentiment automatically.

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

Here we use NLTK's VADER tool, which is specifically tuned for short, informal text like reviews and social posts. For each review, `polarity_scores()` returns four numbers: how negative (`neg`), neutral (`neu`), and positive (`pos`) the text is, plus an overall `compound` score that summarizes the sentiment in a single value from -1 to 1.

> A glowing review lands with a high positive and compound score; a critical review skews negative; a mixed review like "Good value for money. Will buy again." lands somewhere balanced but still leans positive overall.

This little script is a great example of how much insight a business can extract automatically — imagine running this across thousands of reviews instead of just three.

## 1.3 Why We'll Be Using Python

Every field needs a workshop full of good tools, and for NLP, Python is that workshop. Let's talk about why, and then get your environment ready so we can start writing real code together.

### 1.3.1 What Makes Python a Great Fit for NLP

- **Readable and Simple**: Python's clean syntax means you can focus on the ideas behind an NLP algorithm instead of fighting with the language itself.
- **Rich Libraries**: Tools like NLTK, spaCy, and gensim come with ready-made functions and pre-trained models, so you're rarely starting from scratch.
- **A Large Community**: because so many people use Python for NLP, you'll find plenty of tutorials, documentation, and forums to lean on when you get stuck.
- **Strong Machine Learning Integration**: Python connects smoothly with libraries like TensorFlow, PyTorch, and scikit-learn, so you can move from basic text processing to full machine learning models without switching languages.

### 1.3.2 The Core Libraries We'll Use

**NLTK — the Natural Language Toolkit**
NLTK is one of the oldest and most complete NLP libraries in Python. It's especially good for learning, since it exposes the building blocks — tokenizing, stemming, lemmatizing, and more — in a very transparent way.

```python
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

text = "Natural Language Processing with Python is fun!"
tokens = word_tokenize(text)
print(tokens)
```

> Output: `['Natural', 'Language', 'Processing', 'with', 'Python', 'is', 'fun', '!']` — each word and punctuation mark becomes its own token.

**spaCy — built for speed and real-world use**
Where NLTK is great for learning the fundamentals, spaCy is built for production: it's fast, efficient, and ships with strong pre-trained models. Let's use it for Named Entity Recognition, or NER — the task of automatically spotting names of people, places, and organizations in text.

```python
import spacy

nlp = spacy.load("en_core_web_sm")
text = "Apple is looking at buying U.K. startup for 1 billion."
doc = nlp(text)

for ent in doc.ents:
    print(ent.text, ent.label_)
```

We load a small pre-trained English model, run it over our sentence, and then loop through `doc.ents` — the entities spaCy found. Each one comes with a label telling us what kind of entity it is.

> Output: Apple → `ORG` (an organization), U.K. → `GPE` (a geopolitical entity), 1 billion → `MONEY`. Notice spaCy correctly separates the company from the country and the dollar figure — all from a single, un-annotated sentence.

**gensim — topic modeling and word meaning**
gensim shines when you want to understand relationships between words and documents at scale — for example, training a Word2Vec model, which represents each word as a vector of numbers capturing its meaning based on the company it keeps.

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

A few parameters worth understanding: `vector_size` controls how many numbers represent each word (more numbers can capture richer meaning, at a computational cost); `window` controls how many neighboring words are considered as context; `min_count` filters out rare words; and `workers` controls how many CPU threads are used during training. The result is a 100-number vector for the word "language" — words used in similar contexts end up with similar vectors.

This kind of representation is genuinely useful: it underlies text classification, clustering similar documents, building recommendation systems, and measuring how semantically similar two pieces of text are.

**scikit-learn — turning text into predictions**
scikit-learn is a general-purpose machine learning library, and it pairs beautifully with NLP once you've turned text into numbers. Here's a small text classification example using a Naive Bayes classifier:

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

Here's the flow: `CountVectorizer` turns our sentences into a table of word counts that a machine learning model can actually work with. We train a Naive Bayes classifier on that table alongside our labels (1 for positive, 0 for negative). Finally, we transform a brand-new sentence the same way and ask the trained classifier to predict its sentiment — in this case, correctly guessing negative.

### 1.3.3 Setting Up Your Own Environment

Before our next session, please set up Python on your own machine so you're ready to follow along with hands-on exercises.

**Step 1 — Install Python**
Download the latest version from python.org/downloads for your operating system. During installation, be sure to check the box that adds Python to your system PATH — this lets you run Python from the command line. You can confirm it worked by opening a terminal and typing:

```bash
python --version
```

**Step 2 — Create a Virtual Environment**
A virtual environment keeps each project's dependencies separate, which will save you a lot of headaches later. Create and activate one like this:

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
A few of these libraries need extra data files before they'll work:

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
Save the following as `test_nlp.py` and run it. If each library prints sensible output, your environment is ready to go.

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

### 1.3.4 Putting It All Together: A Mini Pipeline

Let's close this section by combining several tools into one small end-to-end pipeline — text processing, feature extraction, and classification, chained together. This is a preview of how real NLP projects are structured.

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

Notice how each piece plays a specific role: spaCy handles the tokenizing, NLTK's stopword list filters out common filler words, `CountVectorizer` turns the cleaned text into numbers, and `MultinomialNB` learns to classify sentiment from those numbers. Wrapping it all in a `Pipeline` means we can train and predict with just two clean method calls — `fit()` and `predict()` — instead of juggling every step by hand.

## Wrapping Up

Let's recap. NLP is the field devoted to helping machines understand and generate human language, and it already touches nearly every corner of modern technology — from search engines to healthcare to customer service. Python has become our tool of choice for this work because it's approachable, well-supported, and backed by a rich ecosystem of libraries: NLTK for the fundamentals, spaCy for speed and production use, gensim for word meaning and topic modeling, and scikit-learn for turning text into predictions.

In our next session, we'll roll up our sleeves and start building with these tools ourselves. Make sure your environment is set up beforehand so we can dive straight into the hands-on work. See you then!
