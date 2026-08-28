# Module 3: Basic Text Processing in NLP


---

## Why Do We Even Need This?

Computers are amazing at math, but they're terrible at reading. If you hand a computer the sentence *"Running, runners run — RUN!"* it has no idea that "Running," "runners," and "RUN" are all related to the same idea: to run.

**Text preprocessing** is the "cleaning and organizing" step we do *before* any real analysis happens — like washing and chopping vegetables before you cook. If you skip it, everything downstream gets messier, slower, and less accurate.

In this module, we'll cover:
1. **Understanding text data** — why it's messy in the first place
2. **Text cleaning** — removing junk words and standardizing word forms
3. **Regular expressions** — finding patterns in text (like emails or dates)
4. **Tokenization** — breaking text into bite-sized pieces

---

## 1. Understanding Text Data

### Why Text Is "Unstructured"

Think about the difference between a spreadsheet and a diary entry.

A spreadsheet is **structured**: every row is a record, every column has a clear meaning (Name, Age, Score). A computer can instantly sort, filter, and calculate with it.

A diary entry — or a tweet, an email, a product review — is **unstructured**. It doesn't come in neat rows and columns. It has slang, typos, punctuation, sentence fragments, and multiple ways to say the same thing ("great," "awesome," "10/10," "🔥"). Before a computer can work with this, we need to turn it into something structured.

That's where NLP (Natural Language Processing) preprocessing comes in — it's the translation step between "how humans write" and "what computers can calculate with."

### An Example Sentence We'll Use Throughout

> "Natural Language Processing (NLP) enables computers to understand human language."

Notice this one sentence already has:
- **Capital letters** (Natural, Language, Processing, NLP)
- **Punctuation** (parentheses, a period)
- **A mix of word types** (nouns like "computers," verbs like "understand")

Every one of these details needs to be handled before we can analyze the text properly.

### Peeking Inside a Piece of Text

Before cleaning anything, it often helps to just *look* at your data — how long is it? How many words? What characters show up? Here's how to check that in Python:

```python
# Sample text
text = "Natural Language Processing (NLP) enables computers to understand human language."

print("Original Text:")
print(text)

# How many characters total?
print("\nLength of the text:", len(text))

# What unique characters appear?
unique_characters = set(text)
print("\nUnique characters:", unique_characters)

# How many words?
words = text.split()
print("\nNumber of words:", len(words))
print("\nWords in the text:")
print(words)
```

**What's happening here, step by step:**
1. We save our sentence in a variable called `text`.
2. `len(text)` counts *every character* — letters, spaces, and punctuation all count.
3. `set(text)` collects every unique character (no duplicates) — a quick way to spot which symbols show up.
4. `text.split()` breaks the sentence into a list of words, cutting wherever there's a space.

**Output:**
```
Length of the text: 77
Number of words: 10

Words in the text:
['Natural', 'Language', 'Processing', '(NLP)', 'enables', 'computers', 'to', 'understand', 'human', 'language.']
```

Notice `'(NLP)'` and `'language.'` still have punctuation stuck to them — `.split()` only breaks on spaces, not punctuation. That's exactly the kind of messiness we'll clean up next.

### Why We Bother Preprocessing (The 3 Big Reasons)

**Reason 1: Noise Reduction — throw out the junk**

A lot of what's in raw text doesn't actually help us understand the *meaning*. Think of it like static on the radio — we want to turn it down so the actual signal comes through clearly.

- **Remove punctuation** — commas, periods, and parentheses rarely change the meaning we're trying to extract.
- **Remove stop words** — tiny, extremely common words like "the," "is," "and," "in" appear constantly but carry almost no unique meaning.
- **Remove other junk** — numbers, HTML tags, special characters, etc., if they're not useful for your task.

**Reason 2: Standardization — make similar things look the same**

If your computer treats "Apple," "apple," and "APPLE" as three totally different words, it will miss the fact they mean the same thing. Standardizing fixes this:

- **Lowercasing** — turn everything lowercase, so "Apple" and "apple" match.
- **Stemming** — chop words down to a rough "root" (e.g., "running" → "run").
- **Lemmatization** — a smarter version of stemming that finds the actual dictionary form of a word, using grammar rules (e.g., "better" → "good").

**Reason 3: Feature Extraction — turn words into numbers**

Eventually we need actual numbers a computer can crunch. This involves:

- **Tokenization** — splitting text into individual pieces (words, sentences, etc.)
- **Vectorization** — turning those pieces into numbers (e.g., Bag of Words, TF-IDF)
- **Embeddings** — turning words into numbers that also capture *meaning* (e.g., Word2Vec, GloVe, BERT)

*(We cover vectorization and embeddings in more depth in Module 3's companion notes — this module focuses on getting text ready for that step.)*

### Common Headaches When Working With Text

Here are the recurring challenges you'll run into, explained simply:

| Challenge | What It Means | Everyday Example |
|---|---|---|
| **Ambiguity** | One word, multiple meanings | "bank" = a riverbank *or* a place to keep your money |
| **Variability** | Same idea, wildly different writing styles | A tweet ("omg so good!!") vs. a formal essay |
| **Noisy Data** | Junk mixed in with useful content | HTML tags, typos, random numbers |
| **High Dimensionality** | Huge vocabulary = huge, unwieldy data | 50,000 unique words = 50,000 "columns" to deal with |
| **Sentiment & Subjectivity** | Opinions and emotions are hard to measure | "Not bad" actually means "pretty good" |
| **Context Dependency** | Meaning shifts based on surrounding words | "bank" means something different next to "river" vs. "money" |
| **Language Diversity** | Thousands of languages, each with their own rules | Some languages don't even use spaces between words! |
| **Sarcasm & Irony** | Tone that flips the literal meaning | "Oh great, ANOTHER meeting" (said unhappily) |

Modern tools like BERT and GPT-style models handle many of these better than older methods, because they pay attention to the surrounding context — but even they still struggle with sarcasm and irony sometimes, just like people occasionally do!

### Hands-On Example: A First Pass at Cleaning Text

Let's do the simplest possible cleanup: lowercase everything, strip out punctuation, and split it into words.

```python
import string

# Sample text
text = "Natural Language Processing (NLP) enables computers to understand human language."

# Step 1: Make everything lowercase
text = text.lower()
print("Lowercased Text:")
print(text)

# Step 2: Strip out punctuation
text = text.translate(str.maketrans('', '', string.punctuation))
print("\nText without Punctuation:")
print(text)

# Step 3: Split into individual words (tokens)
tokens = text.split()
print("\nTokens:")
print(tokens)
```

**What each step does:**
1. `.lower()` makes "Natural" and "natural" identical to the computer.
2. `str.maketrans('', '', string.punctuation)` builds a little "translation table" that maps every punctuation character to nothing, and `.translate()` applies it — effectively deleting all punctuation.
3. `.split()` breaks the now-clean text into a list of separate words.

**Output:**
```
Tokens:
['natural', 'language', 'processing', 'nlp', 'enables', 'computers', 'to', 'understand', 'human', 'language']
```

Compare this to our earlier output — now "(NLP)" is just "nlp," and "language." lost its period. Much cleaner!

---

## 2. Text Cleaning: Removing Junk Words and Standardizing Word Forms

This section covers three of the most common cleaning techniques, and how they're different:

| Technique | What it does | Example |
|---|---|---|
| **Stop word removal** | Deletes common, low-meaning words | "the," "is," "in," "and" → gone |
| **Stemming** | Chops words down to a rough root (fast but crude) | "running," "runner" → "run" |
| **Lemmatization** | Finds the true dictionary form (smarter, uses grammar) | "better" → "good" |

### 2.1 Stop Word Removal

**The idea:** words like "the," "a," "is," and "and" show up *everywhere* in English but don't tell you much about what a specific piece of text is actually about. Removing them lets your analysis focus on the words that matter.

**Why bother:**
1. **Less data to process** — a smaller, more focused set of words.
2. **Faster processing** — algorithms have fewer things to look at.
3. **Better accuracy** — less "noise" confusing the model.

```python
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

text = "Natural Language Processing enables computers to understand human language."
tokens = text.split()

# Build a set of English stop words
stop_words = set(stopwords.words('english'))

# Keep only the words that are NOT stop words
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

print("Original Tokens:")
print(tokens)
print("\nFiltered Tokens:")
print(filtered_tokens)
```

**Output:**
```
Filtered Tokens:
['Natural', 'Language', 'Processing', 'enables', 'computers', 'understand', 'human', 'language.']
```

Notice "to" disappeared — it's a stop word that doesn't add much meaning here.

### 2.2 Stemming: The Quick-and-Dirty Way to Simplify Words

**The idea:** chop off the end of a word to get something close to its "root," so different forms of the same word count as one.

Think of stemming like a lawnmower — fast and effective, but not always precise. It just follows simple rules (like "if a word ends in -ing, cut it off") without actually knowing what the word means.

**Why bother:**
1. **Fewer unique words to deal with** overall.
2. **Groups similar words together** — "running," "runner," "runs" all become "run."
3. **Saves memory and processing time.**

The most common stemming tool is the **Porter Stemmer**, created back in 1980 and still widely used today.

```python
from nltk.stem import PorterStemmer

text = "Natural Language Processing enables computers to understand human language."
tokens = text.split()

stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in tokens]

print("Original Tokens:")
print(tokens)
print("\nStemmed Tokens:")
print(stemmed_tokens)
```

**Output:**
```
Stemmed Tokens:
['natur', 'languag', 'process', 'enabl', 'comput', 'to', 'understand', 'human', 'languag.']
```

Notice the results aren't always real words! "Natural" became "natur" and "language" became "languag." That's the trade-off — stemming is fast, but crude.

**Where stemming can go wrong:**
- **Overstemming** — cutting off too much, losing meaning. ("university" → "univers")
- **Understemming** — related words end up with *different* stems when they shouldn't. ("organization" and "organizing" might not match)
- **No context awareness** — stemming can't tell that "bank" means something different in "river bank" vs. "savings bank" — it just applies the same rule no matter what.

### 2.3 Lemmatization: The Smarter Way to Simplify Words

**The idea:** instead of just chopping letters off, lemmatization looks up the *actual dictionary form* of a word, taking grammar into account. It's slower than stemming, but far more accurate.

Where stemming is a lawnmower, lemmatization is more like a skilled gardener who knows exactly which part of the plant to trim.

**Why bother:**
1. **More accurate results** — "better" correctly becomes "good," not some chopped-up fragment.
2. **Cleaner text analysis** — helpful for classification, search, and sentiment analysis.
3. **Smarter search results** — search for "running" and also find results with "run" or "runs."

```python
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

text = "Natural Language Processing enables computers to understand human language."
tokens = text.split()

lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]

print("Original Tokens:")
print(tokens)
print("\nLemmatized Tokens:")
print(lemmatized_tokens)
```

**Output:**
```
Lemmatized Tokens:
['Natural', 'Language', 'Processing', 'enables', 'computer', 'to', 'understand', 'human', 'language.']
```

Notice "computers" correctly became "computer" — a real word, unlike what stemming gave us.

### 2.4 Putting It All Together

In practice, you'll usually combine several of these steps in a row: lowercase → remove punctuation → tokenize → remove stop words → stem/lemmatize.

```python
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
import string
nltk.download('stopwords')
nltk.download('wordnet')

text = "Natural Language Processing enables computers to understand human language."

# 1. Lowercase
text = text.lower()

# 2. Remove punctuation
text = text.translate(str.maketrans('', '', string.punctuation))

# 3. Tokenize
tokens = text.split()

# 4. Remove stop words
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word not in stop_words]

# 5. Stem and lemmatize
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
processed_tokens = [lemmatizer.lemmatize(stemmer.stem(word)) for word in filtered_tokens]

print("Filtered Tokens (Stop Words Removed):")
print(filtered_tokens)
print("\nProcessed Tokens (Stemmed and Lemmatized):")
print(processed_tokens)
```

**Quick recap:**
- **Stop words** → common filler words we throw away.
- **Stemming** → fast, rough word-shortening.
- **Lemmatization** → slower, smarter, grammar-aware word-shortening.

---

## 3. Regular Expressions: Finding Patterns in Text

### What Is a Regular Expression?

A **regular expression** (or "regex") is like a search query with superpowers. Instead of searching for one exact word, you describe a *pattern* — like "any sequence of digits that looks like a phone number" — and the computer finds every match.

It's a bit like using wildcards when searching for files (`*.jpg` finds every image file), except regex patterns can be far more precise and powerful.

In Python, we use the built-in `re` module.

```python
import re

text = "The quick brown fox jumps over the lazy dog."

# Look for the word "fox"
pattern = r"fox"
match = re.search(pattern, text)

if match:
    print("Match found:", match.group())
else:
    print("No match found.")
```

**Output:** `Match found: fox`

**What's happening:**
1. We import Python's `re` module — this gives us regex tools.
2. `pattern = r"fox"` — the `r` before the quotes means "treat this as a raw string" (so backslashes aren't misinterpreted).
3. `re.search()` scans the text for that pattern.
4. If it finds something, we get a "match object" — `.group()` shows us the actual matched text.

### The Building Blocks (Cheat Sheet)

You don't need to memorize all of these — just know they exist and look them up when you need them.

| Symbol | What It Matches | Simple Way to Remember |
|---|---|---|
| `.` | Any single character | "any letter/symbol here" |
| `^` | Start of the text | "must begin with this" |
| `$` | End of the text | "must end with this" |
| `*` | Zero or more of the thing before it | "maybe none, maybe a bunch" |
| `+` | One or more of the thing before it | "at least one" |
| `?` | Zero or one (optional) | "might or might not be there" |
| `[]` | A set of allowed characters | "any one of these" |
| `\d` | Any digit (0–9) | "a number" |
| `\w` | Any letter, digit, or underscore | "a word character" |
| `\s` | Any whitespace (space, tab, newline) | "a gap" |
| `\|` | OR | "this pattern OR that one" |
| `()` | Groups part of the pattern together | "treat this chunk as one unit" |

### Real Examples You'll Actually Use

**Finding email addresses:**

```python
import re

text = "Please contact us at support@example.com or sales@example.com for further information."
pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
emails = re.findall(pattern, text)

print("Extracted Email Addresses:")
print(emails)
```
**Output:** `['support@example.com', 'sales@example.com']`

*How the pattern works, piece by piece:*
- `[A-Za-z0-9._%+-]+` → the part before the `@` (letters, numbers, dots, etc.)
- `@` → the literal @ symbol
- `[A-Za-z0-9.-]+` → the domain name
- `\.[A-Z|a-z]{2,}` → a dot followed by at least 2 letters (like ".com" or ".org")

**Finding phone numbers:**

```python
import re

text = "Contact us at (123) 456-7890 or (987) 654-3210."
pattern = r"\(\d{3}\) \d{3}-\d{4}"
phone_numbers = re.findall(pattern, text)

print(phone_numbers)
```
**Output:** `['(123) 456-7890', '(987) 654-3210']`

**Swapping out words:**

```python
import re

text = "The quick brown fox jumps over the lazy dog. The fox is clever."
new_text = re.sub(r"fox", "cat", text)

print(new_text)
```
**Output:** `The quick brown cat jumps over the lazy dog. The cat is clever.`

`re.sub(pattern, replacement, text)` finds every match of `pattern` and swaps it out for `replacement`.

**Finding dates in two different formats at once:**

```python
import re

text = "The event is scheduled for 2022-08-15. Another event is on 15/08/2022."
pattern = r"\b(?:\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})\b"
dates = re.findall(pattern, text)

print(dates)
```
**Output:** `['2022-08-15', '15/08/2022']`

The `|` here means "match either this pattern or that pattern" — handy when your data isn't formatted consistently.

**Pulling out hashtags from social media posts:**

```python
import re

text = "Loving the new features of this product! #excited #newrelease #tech"
hashtags = re.findall(r"#\w+", text)

print(hashtags)
```
**Output:** `['#excited', '#newrelease', '#tech']`

### Where You'll Use Regex in Real Life

- **Cleaning data** — stripping out HTML tags, extra whitespace, or unwanted symbols.
- **Validating input** — checking that a user actually typed a real-looking email or phone number.
- **Extracting information** — pulling hashtags, mentions, or URLs out of social media posts.
- **Text search** — quickly finding all instances of a pattern in a huge document.

---

## 4. Tokenization: Breaking Text Into Bite-Sized Pieces

### What Is Tokenization?

**Tokenization** means splitting a wall of text into smaller pieces called **tokens** — usually words, sentences, or even individual characters. It's the very first step that turns unstructured text into something structured enough to work with.

Think of it like cutting a loaf of bread into slices before you can make a sandwich — you can't do much with the whole loaf at once.

### Why It Matters

1. **Simplification** — big blocks of text become small, manageable pieces.
2. **Standardization** — creates a consistent way of representing text.
3. **Feature extraction** — those individual tokens become the building blocks for turning text into numbers later.

### The Three Main Types

| Type | Splits Into | Useful For |
|---|---|---|
| **Word Tokenization** | Individual words | Classification, tagging parts of speech, finding names |
| **Sentence Tokenization** | Individual sentences | Summarization, translation, topic detection |
| **Character Tokenization** | Individual letters/symbols | Spell-checking, handwriting recognition, text generation |

### Word Tokenization

Splits text into individual words — this is the most common kind of tokenization.

**Using NLTK:**
```python
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

text = "Natural Language Processing enables computers to understand human language."
tokens = word_tokenize(text)

print(tokens)
```
**Output:**
```
['Natural', 'Language', 'Processing', 'enables', 'computers', 'to', 'understand', 'human', 'language', '.']
```

Notice the period `.` becomes its *own* token — unlike `.split()`, `word_tokenize()` correctly separates punctuation from words.

**Using SpaCy (another popular library):**
```python
import spacy

nlp = spacy.load("en_core_web_sm")
text = "Natural Language Processing enables computers to understand human language."

doc = nlp(text)
tokens = [token.text for token in doc]

print(tokens)
```
This gives the same result — SpaCy is just a different tool that does a similar job (and is often used for more advanced tasks too).

### Sentence Tokenization

Splits text into separate sentences instead of words — useful when you care about sentence-level meaning (like summarizing an article).

```python
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

text = "Natural Language Processing enables computers to understand human language. It is a fascinating field."
sentences = sent_tokenize(text)

print(sentences)
```
**Output:**
```
['Natural Language Processing enables computers to understand human language.', 'It is a fascinating field.']
```

Notice it correctly figured out where one sentence ends and the next begins, using the periods as clues.

### Character Tokenization

Splits text into individual characters — every letter, space, and punctuation mark becomes its own token. This is used less often, but it's important for tasks like spell-checking or handwriting recognition, where individual letters matter.

```python
text = "Natural Language Processing"
characters = list(text)

print(characters)
```
**Output:**
```
['N', 'a', 't', 'u', 'r', 'a', 'l', ' ', 'L', 'a', 'n', 'g', 'u', 'a', 'g', 'e', ' ', 'P', 'r', 'o', 'c', 'e', 's', 's', 'i', 'n', 'g']
```

### Seeing All Three Side-by-Side

```python
import nltk
import spacy
nltk.download('punkt')

nlp = spacy.load("en_core_web_sm")
text = "Natural Language Processing enables computers to understand human language. It is a fascinating field."

# Word tokenization
word_tokens = nltk.word_tokenize(text)
print("Word Tokens:", word_tokens)

# Sentence tokenization
sentence_tokens = nltk.sent_tokenize(text)
print("\nSentence Tokens:", sentence_tokens)

# Character tokenization
char_tokens = list(text)
print("\nCharacter Tokens:", char_tokens)
```

This shows the same piece of text getting sliced up three completely different ways, depending on what level of detail you need — word-by-word, sentence-by-sentence, or letter-by-letter.

---

## Putting the Whole Module Together

Here's the typical order these preprocessing steps happen in a real NLP project:

1. **Look at your raw text** — understand its length, structure, and quirks.
2. **Clean it up** — lowercase everything, strip out punctuation and other noise.
3. **Tokenize it** — split into words or sentences.
4. **Remove stop words** — throw out low-value filler words.
5. **Stem or lemmatize** — standardize different forms of the same word.
6. **(Optional) Use regex** — for structured patterns like emails, dates, or hashtags.

Once your text has been through this pipeline, it's finally clean and structured enough to be turned into numbers — which is exactly what techniques like Bag of Words, TF-IDF, and word embeddings (covered in later material) do next.

**The big takeaway:** messy human language has to be simplified and standardized before a computer can do anything useful with it. Every technique in this module exists to make that translation a little bit better.
