# Module 3: Text Preprocessing in NLP

*Lecture Notes - Module 3*

---

## 3.1 Understanding Text Data

Text data is inherently unstructured and can come in various forms such as articles, social media posts, emails, chat messages, reviews, and more. Unlike numerical data, which is easily analyzable by machines due to its structured nature, text data requires special handling and processing techniques to convert it into a structured format.

This transformation is essential so that algorithms can efficiently process and understand the information contained within the text. The complexity of human language — with its nuances, idioms, and varied syntax — adds an additional layer of challenge to this task.

Sophisticated methods such as natural language processing (NLP), machine learning techniques, and various text mining strategies are employed to make sense of and extract meaningful insights from text data. These methods help in categorizing, summarizing, and even predicting trends based on the textual information available.

### 3.1.1 Nature of Text Data

Text data consists of sequences of characters forming words, sentences, and paragraphs. Each text piece can vary greatly in terms of length, structure, and content. This variability poses challenges for analysis, as the text must be standardized and cleaned before any meaningful processing can occur.

For example, a sentence might contain punctuation, capitalization, and a mixture of different types of words (nouns, verbs, etc.), all of which need to be considered during preprocessing.

Understanding the nature of text data and the necessity of preprocessing is crucial for building effective NLP applications. Proper preprocessing ensures that the text is clean, consistent, and in a format that can be easily analyzed by machine learning models. This includes steps such as tokenization, stop word removal, stemming, lemmatization, and the use of regular expressions to transform raw text into a structured and analyzable format.

**Example text:**
> "Natural Language Processing (NLP) enables computers to understand human language."

This sentence contains punctuation, capitalization, and a mixture of different types of words (nouns, verbs, etc.). Each of these elements must be considered during preprocessing to ensure the text is properly prepared for further analysis.

### 3.1.2 Importance of Text Preprocessing

Preprocessing text data is a crucial step in any NLP pipeline. Key reasons for preprocessing text include:

#### Noise Reduction
Removing irrelevant or redundant information — such as punctuation, stop words, or other non-essential elements — makes the data used for analysis more meaningful and focused, improving model performance.

Key elements of noise reduction:
1. **Punctuation Removal** — Punctuation marks often carry little meaning in text analysis; removing them simplifies text and reduces noise.
2. **Stop Word Removal** — Common words such as "and," "the," "is," and "in" don't contribute much meaning; eliminating them focuses analysis on more meaningful words.
3. **Non-essential Elements** — Removing numbers, special characters, HTML tags, or other elements that don't add value.

When text data is free from unnecessary noise, tokenization, stemming, and lemmatization become more efficient and accurate, ultimately leading to better model performance.

#### Standardization
Converting text to a standardized format — such as lowercasing, stemming, or lemmatization — ensures consistency across the text data, reducing variability and enhancing reliability.

1. **Lowercasing** — Converts all letters to lowercase so words like "Apple" and "apple" aren't treated as different entities.
2. **Stemming** — Reduces words to their base or root form (e.g., "running" → "run"), treating morphological variants as a single term.
3. **Lemmatization** — Similar to stemming but more sophisticated and context-aware; reduces words to their dictionary/canonical form (e.g., "better" → "good"), considering context and part of speech.

#### Feature Extraction
Transforming raw text into features used by machine learning models to learn patterns and make predictions or classifications.

1. **Tokenization** — Breaking text into individual units called tokens (words or phrases), organizing text into manageable, structured pieces.
2. **Vectorization** — Converting tokens into numerical vectors using techniques like Bag of Words (BoW), TF-IDF, and Word2Vec, enabling mathematical operations on text data.
3. **Embedding Representations** — Mapping words/phrases to high-dimensional vectors (Word2Vec, GloVe, BERT) that capture semantic relationships and context.

### 3.1.3 Example: Exploring Raw Text Data

```python
# Sample text
text = "Natural Language Processing (NLP) enables computers to understand human language."

# Display the text
print("Original Text:")
print(text)

# Length of the text
print("\nLength of the text:", len(text))

# Unique characters in the text
unique_characters = set(text)
print("\nUnique characters:", unique_characters)

# Number of words in the text
words = text.split()
print("\nNumber of words:", len(words))

# Display the words
print("\nWords in the text:")
print(words)
```

**Explanation:**
1. **Defining the Sample Text** — a string variable `text` holds the sample sentence.
2. **Displaying the Original Text** — prints the label and the text.
3. **Calculating the Length of the Text** — `len(text)` counts characters, including spaces and punctuation.
4. **Identifying Unique Characters** — `set(text)` removes duplicate characters.
5. **Counting the Number of Words** — `text.split()` breaks the text into words based on spaces.
6. **Displaying the List of Words** — prints each word as a separate list element.

**Output:**
```
Original Text:
Natural Language Processing (NLP) enables computers to understand human language.

Length of the text: 77

Unique characters: {'r', ' ', 'm', 'P', 'N', 'a', 'o', 'u', 'L', 't', 'h', 'c', 'n', '.', 's', 'e', 'l', 'd', 'g', 'p', ')', 'b', '(', 'i'}

Number of words: 10

Words in the text:
['Natural', 'Language', 'Processing', '(NLP)', 'enables', 'computers', 'to', 'understand', 'human', 'language.']
```

This basic exploration helps in understanding the structure and content of the text — an essential step before more advanced processing steps such as tokenization, stemming, lemmatization, and feature extraction.

### 3.1.4 Challenges with Text Data

**Ambiguity** — Words have multiple meanings depending on context (e.g., "bank" as a riverbank vs. a financial institution). Techniques such as word sense disambiguation, context-aware embeddings, and advanced language models like BERT and GPT-4 help tackle this.

**Variability** — Differences in format, style, and structure across sources (social media slang vs. formal academic writing). Tweets are short and concise; blog posts are lengthy and elaborate. Domain-specific jargon and multilingual content add complexity.

**Noisy Data** — Irrelevant or redundant information such as punctuation, numbers, HTML tags, and stop words that obscure meaningful content. Proper preprocessing (removing punctuation, filtering numbers, stripping HTML tags, eliminating stop words) is crucial.

**High Dimensionality** — Each unique word can be a dimension, leading to a very high-dimensional feature space. Challenges include:
1. **Computational Complexity** — more memory and processing power required.
2. **Overfitting** — models may fit noise rather than underlying patterns; mitigated via dimensionality reduction, regularization, cross-validation.
3. **Curse of Dimensionality** — data points become sparse, making pattern-finding harder.
4. **Feature Selection and Engineering** — techniques like TF-IDF, PCA, Word2Vec, and BERT help reduce dimensionality.
5. **Storage and Scalability** — efficient storage and scalable processing frameworks are needed.

Addressed via dimensionality reduction (PCA, SVD, t-SNE), regularization (L1/L2), and advanced embeddings.

**Sentiment and Subjectivity** — Text often contains opinions, emotions, and biases that are hard to quantify. Phrases like "not bad" carry positive sentiment despite containing "bad." Sarcasm and irony (e.g., "Oh great, another meeting") further complicate sentiment analysis. Advanced models like BERT and GPT-3 help improve accuracy.

**Context and Dependency** — Meaning often depends on surrounding words (e.g., "bank" as riverbank vs. financial institution; "not bad" flips sentiment). Models like BERT and GPT-4 use deep learning to better capture context and dependencies.

**Language Diversity** — Multitudes of languages/dialects with unique grammar, vocabulary, and writing systems (alphabetic, logographic, abugida). Resource-poor languages often require transfer learning. Ethical considerations include ensuring fair, unbiased support across linguistic communities.

**Sarcasm and Irony** — Rely on tone, context, and cultural knowledge, which are difficult for algorithms to interpret. Even advanced models like BERT and GPT-4 still struggle here; addressing this requires context-aware models and diverse training datasets.

### 3.1.5 Practical Example: Basic Text Preprocessing Steps

```python
import string

# Sample text
text = "Natural Language Processing (NLP) enables computers to understand human language."

# Convert to lowercase
text = text.lower()
print("Lowercased Text:")
print(text)

# Remove punctuation
text = text.translate(str.maketrans('', '', string.punctuation))
print("\nText without Punctuation:")
print(text)

# Tokenize the text
tokens = text.split()
print("\nTokens:")
print(tokens)
```

**Explanation:**
1. **Import `string`** — provides punctuation characters for removal.
2. **Sample Text** — defined for demonstration.
3. **Lowercase** — `text.lower()` standardizes text so "Language" and "language" are treated the same.
4. **Remove Punctuation** — `str.maketrans` + `translate` maps each punctuation mark to `None`.
5. **Tokenize** — `split()` divides text on whitespace into a list of tokens.

**Output:**
```
Lowercased Text:
natural language processing (nlp) enables computers to understand human language.

Text without Punctuation:
natural language processing nlp enables computers to understand human language

Tokens:
['natural', 'language', 'processing', 'nlp', 'enables', 'computers', 'to', 'understand', 'human', 'language']
```

**Summary:** This example covers fundamental preprocessing steps — lowercasing, removing punctuation, and tokenization — that form the foundation for more advanced text processing and analysis tasks, including stop word removal, stemming, and lemmatization.

---

## 3.2 Text Cleaning: Stop Word Removal, Stemming, Lemmatization

Text cleaning transforms raw, messy, unstructured text into a clean, standardized format suitable for analysis and modeling. This section covers three essential techniques: **stop word removal**, **stemming**, and **lemmatization**.

- **Stop word removal** eliminates common words with little semantic value ("and," "the," "in").
- **Stemming** reduces words to a base/root form by removing suffixes/prefixes (e.g., "running," "runner" → "run").
- **Lemmatization** reduces words to a dictionary form (lemma), considering context (e.g., "better" → "good").

### 3.2.1 Stop Word Removal

Stop words are common words that carry minimal meaningful information (e.g., "the," "is," "in," "and"). Removing them:

1. **Reduces Dimensionality** — makes data easier to manage and analyze.
2. **Increases Processing Speed** — algorithms focus on more informative terms.
3. **Improves Accuracy** — reduces noise/confusion in tasks like text classification and sentiment analysis.

**Example (Python / NLTK):**
```python
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Sample text
text = "Natural Language Processing enables computers to understand human language."

# Tokenize the text
tokens = text.split()

# Remove stop words
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

print("Original Tokens:")
print(tokens)
print("\nFiltered Tokens:")
print(filtered_tokens)
```

**Explanation:** Import `stopwords` from NLTK → download the English stop word list → tokenize sample text via `split()` → build a `stop_words` set → filter tokens with a list comprehension → print results.

**Output:**
```
Original Tokens:
['Natural', 'Language', 'Processing', 'enables', 'computers', 'to', 'understand', 'human', 'language.']

Filtered Tokens:
['Natural', 'Language', 'Processing', 'enables', 'computers', 'understand', 'human', 'language.']
```

### 3.2.2 Stemming

Stemming reduces words to their base/root form, normalizing text so different forms of a word are treated identically.

**Why it matters:**
1. **Dimensionality Reduction** — fewer unique words, less computational complexity.
2. **Improved Accuracy** — e.g., "running," "runner," "runs" → "run."
3. **Resource Efficiency** — smaller vocabulary, faster processing.

**How it works:** Stemming removes suffixes, prefixes, or other affixes. The most common algorithm is the **Porter Stemmer** (Martin Porter, 1980).

**Example (Python / NLTK):**
```python
from nltk.stem import PorterStemmer

# Sample text
text = "Natural Language Processing enables computers to understand human language."

# Tokenize the text
tokens = text.split()

# Initialize the stemmer
stemmer = PorterStemmer()

# Stem the tokens
stemmed_tokens = [stemmer.stem(word) for word in tokens]

print("Original Tokens:")
print(tokens)
print("\nStemmed Tokens:")
print(stemmed_tokens)
```

**Output:**
```
Original Tokens:
['Natural', 'Language', 'Processing', 'enables', 'computers', 'to', 'understand', 'human', 'language.']

Stemmed Tokens:
['natur', 'languag', 'process', 'enabl', 'comput', 'to', 'understand', 'human', 'languag.']
```

**Applications:** search engines, text classification, sentiment analysis.

**Limitations:**
1. **Overstemming** — e.g., "university" → "univers" (loses meaning).
2. **Understemming** — e.g., "organization" and "organizing" may not share the same stem.
3. **Context Ignorance** — e.g., "bank" stems the same regardless of meaning (riverbank vs. financial institution).

### 3.2.3 Lemmatization

Lemmatization transforms words into their dictionary base form (lemma), considering context and part of speech — more sophisticated and accurate than stemming.

**Why it matters:**
1. **Contextual Accuracy** — e.g., "better" → "good."
2. **Improved Text Analysis** — normalizes text for classification, retrieval, sentiment analysis.
3. **Enhanced Search Results** — e.g., a search for "running" also returns "run" and "runs."

**How it works:** Uses a dictionary and morphological analysis; typically requires knowledge of a word's part of speech (e.g., "saw" as noun vs. verb).

**Example (Python / NLTK):**
```python
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

# Sample text
text = "Natural Language Processing enables computers to understand human language."

# Tokenize the text
tokens = text.split()

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Lemmatize the tokens
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]

print("Original Tokens:")
print(tokens)
print("\nLemmatized Tokens:")
print(lemmatized_tokens)
```

**Output:**
```
Original Tokens:
['Natural', 'Language', 'Processing', 'enables', 'computers', 'to', 'understand', 'human', 'language.']

Lemmatized Tokens:
['Natural', 'Language', 'Processing', 'enables', 'computer', 'to', 'understand', 'human', 'language.']
```

**Applications:** search engines, text classification, sentiment analysis.

### 3.2.4 Practical Example: Combining Text Cleaning Techniques

```python
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# Sample text
text = "Natural Language Processing enables computers to understand human language."

# Convert to lowercase
text = text.lower()

# Remove punctuation
import string
text = text.translate(str.maketrans('', '', string.punctuation))

# Tokenize the text
tokens = text.split()

# Remove stop words
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word not in stop_words]

# Initialize the stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Stem and lemmatize the filtered tokens
processed_tokens = [lemmatizer.lemmatize(stemmer.stem(word)) for word in filtered_tokens]

print("Original Text:")
print(text)
print("\nFiltered Tokens (Stop Words Removed):")
print(filtered_tokens)
print("\nProcessed Tokens (Stemmed and Lemmatized):")
print(processed_tokens)
```

**Output:**
```
Original Text:
natural language processing enables computers to understand human language

Filtered Tokens (Stop Words Removed):
['natural', 'language', 'processing', 'enables', 'computers', 'understand', 'human', 'language']

Processed Tokens (Stemmed and Lemmatized):
['natur', 'languag', 'process', 'enabl', 'comput', 'understand', 'human', 'languag']
```

**Recap:**
- **Stop Words** — common, low-information words removed to reduce noise.
- **Stemming** — reduces words to root form (e.g., "running," "runner," "runs" → "run").
- **Lemmatization** — reduces words to dictionary form considering context (e.g., "better" → "good").

---

## 3.3 Regular Expressions

Regular expressions (regex) are powerful tools for searching, matching, and manipulating text based on specific patterns — from simple search-and-replace to complex text extraction and validation (e.g., extracting phone numbers, validating emails, parsing large text files).

### 3.3.1 Basics of Regular Expressions

A regex is a sequence of characters defining a search pattern used to match sequences of characters within text. In Python, regex is implemented via the `re` module (`re.search`, `re.match`, `re.sub`, etc.).

**Example:**
```python
import re

# Sample text
text = "The quick brown fox jumps over the lazy dog."

# Define a pattern to search for the word "fox"
pattern = r"fox"

# Use re.search() to find the pattern in the text
match = re.search(pattern, text)

# Display the match
if match:
    print("Match found:", match.group())
else:
    print("No match found.")
```

**Explanation:**
1. Import the `re` module.
2. Define sample text.
3. Define pattern `r"fox"` (raw string — backslashes treated literally).
4. `re.search()` scans for the pattern, returning a match object or `None`.
5. Print result via `match.group()`.

**Output:**
```
Match found: fox
```

**Practical Applications:**
1. **Text Search** — finding words/phrases or date patterns in documents.
2. **Data Validation** — checking formats like emails or phone numbers.
3. **Text Processing** — extracting/replacing content, e.g., removing HTML tags or extracting hashtags.

### 3.3.2 Common Regex Patterns and Syntax

| Metacharacter | Meaning |
|---|---|
| `.` | Matches any single character except a newline |
| `^` | Matches the start of the string |
| `$` | Matches the end of the string |
| `*` | Matches zero or more repetitions of the preceding character |
| `+` | Matches one or more repetitions of the preceding character |
| `?` | Matches zero or one repetition (makes the character optional) |
| `[]` | Defines a set of characters; matches any one inside the brackets |
| `\d` | Matches any digit, equivalent to `[0-9]` |
| `\w` | Matches any alphanumeric character (and underscore), equivalent to `[a-zA-Z0-9_]` |
| `\s` | Matches any whitespace character (space, tab, newline) |
| `\|` | OR operator — matches one pattern or another |
| `()` | Groups patterns together; can capture for extraction |

### 3.3.3 Practical Examples of Regex in Python

**Example 1 — Extracting Email Addresses**
```python
import re

text = "Please contact us at support@example.com or sales@example.com for further information."
pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
emails = re.findall(pattern, text)

print("Extracted Email Addresses:")
print(emails)
```
Pattern breakdown: `\b` (word boundary) → `[A-Za-z0-9._%+-]+` (local part) → `@` → `[A-Za-z0-9.-]+` (domain) → `\.` (literal dot) → `[A-Z|a-z]{2,}` (TLD, 2+ letters) → `\b`.

**Output:** `['support@example.com', 'sales@example.com']`

Applications: email extraction, data validation, general text processing.

**Example 2 — Validating Phone Numbers**
```python
import re

text = "Contact us at (123) 456-7890 or (987) 654-3210."
pattern = r"\(\d{3}\) \d{3}-\d{4}"
phone_numbers = re.findall(pattern, text)

print("Extracted Phone Numbers:")
print(phone_numbers)
```
Pattern breakdown: `\(` → `\d{3}` → `\)` → space → `\d{3}` → `-` → `\d{4}`.

**Output:** `['(123) 456-7890', '(987) 654-3210']`

Applications: data extraction, data validation, text processing.

**Example 3 — Replacing Substrings**
```python
import re

text = "The quick brown fox jumps over the lazy dog. The fox is clever."
pattern = r"fox"
new_text = re.sub(pattern, "cat", text)

print("Modified Text:")
print(new_text)
```
**Output:**
```
Modified Text:
The quick brown cat jumps over the lazy dog. The cat is clever.
```

Applications: text replacement, data cleaning, data transformation (reformatting dates, standardizing phone numbers, case conversion).

### 3.3.4 Advanced Regex Techniques

**Example 4 — Extracting Dates**
```python
import re

text = "The event is scheduled for 2022-08-15. Another event is on 15/08/2022."
pattern = r"\b(?:\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})\b"
dates = re.findall(pattern, text)

print("Extracted Dates:")
print(dates)
```
Pattern breakdown: `\b` → `(?:...)` (non-capturing group) → `\d{4}-\d{2}-\d{2}` (YYYY-MM-DD) OR `\d{2}/\d{2}/\d{4}` (DD/MM/YYYY) → `\b`.

**Output:** `['2022-08-15', '15/08/2022']`

**Example 5 — Extracting Hashtags from Social Media Text**
```python
import re

text = "Loving the new features of this product! #excited #newrelease #tech"
pattern = r"#\w+"
hashtags = re.findall(pattern, text)

print("Extracted Hashtags:")
print(hashtags)
```
Pattern breakdown: `#` (literal hash) → `\w+` (one or more word characters).

**Output:** `['#excited', '#newrelease', '#tech']`

**Applications:**
- **Social Media Analysis** — trending topics, engagement analysis.
- **Data Cleaning** — extracting hashtags, mentions, URLs from datasets.
- **Content Categorization** — auto-tagging content.
- **Text Processing** — general search/match/manipulate tasks.

---

## 3.4 Tokenization

Tokenization breaks text into smaller units called **tokens** — words, sentences, or characters — converting unstructured text into a structured, analyzable format.

### 2.4.1 Importance of Tokenization

1. **Simplification** — breaks complex text into manageable units for efficient analysis.
2. **Standardization** — creates a consistent, predictable text representation.
3. **Feature Extraction** — enables extraction of meaningful features (words/phrases) as inputs to ML models for prediction, classification, sentiment analysis, etc.

### 3.4.2 Types of Tokenization

1. **Word Tokenization** — splits text into individual words; most common form; useful for text classification, POS tagging, named entity recognition.
2. **Sentence Tokenization** — splits text into sentences; useful for sentiment analysis, summarization, machine translation, topic modeling.
3. **Character Tokenization** — splits text into individual characters; useful for language modeling, character recognition, spell-checking, text generation.

### 3.4.3 Word Tokenization

**Example — NLTK:**
```python
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

text = "Natural Language Processing enables computers to understand human language."
tokens = word_tokenize(text)

print("Word Tokens:")
print(tokens)
```
**Output:**
```
Word Tokens:
['Natural', 'Language', 'Processing', 'enables', 'computers', 'to', 'understand', 'human', 'language', '.']
```
Punctuation is treated as a separate token.

**Example — SpaCy:**
```python
import spacy

nlp = spacy.load("en_core_web_sm")
text = "Natural Language Processing enables computers to understand human language."

doc = nlp(text)
tokens = [token.text for token in doc]

print("Word Tokens:")
print(tokens)
```
**Output:** same as the NLTK result above.

**Benefits:** simplification, standardization, feature extraction.

**Applications:** text classification, sentiment analysis, named entity recognition (NER), machine translation, information retrieval.

### 3.4.4 Sentence Tokenization

**Example — NLTK:**
```python
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

text = "Natural Language Processing enables computers to understand human language. It is a fascinating field."
sentences = sent_tokenize(text)

print("Sentences:")
print(sentences)
```
**Output:**
```
Sentences:
['Natural Language Processing enables computers to understand human language.', 'It is a fascinating field.']
```

**Example — SpaCy:**
```python
import spacy

nlp = spacy.load("en_core_web_sm")
text = "Natural Language Processing enables computers to understand human language. It is a fascinating field."

doc = nlp(text)
sentences = [sent.text for sent in doc.sents]

print("Sentences:")
print(sentences)
```
**Output:** same as the NLTK result above.

**Applications:** summarization, sentiment analysis, machine translation, structural/topic-modeling text analysis.

### 3.4.5 Character Tokenization

```python
text = "Natural Language Processing"
characters = list(text)

print("Characters:")
print(characters)
```
**Output:**
```
Characters:
['N', 'a', 't', 'u', 'r', 'a', 'l', ' ', 'L', 'a', 'n', 'g', 'u', 'a', 'g', 'e', ' ', 'P', 'r', 'o', 'c', 'e', 's', 's', 'i', 'n', 'g']
```

**Applications:** text generation, handwriting recognition, spell checking, text encryption/decryption.

### 3.4.6 Practical Example: Tokenization Pipeline

```python
import nltk
import spacy
nltk.download('punkt')

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

text = "Natural Language Processing enables computers to understand human language. It is a fascinating field."

# Word tokenization (NLTK)
word_tokens = nltk.word_tokenize(text)
print("Word Tokens:")
print(word_tokens)

# Sentence tokenization (NLTK)
sentence_tokens = nltk.sent_tokenize(text)
print("\nSentence Tokens:")
print(sentence_tokens)

# Sentence tokenization (SpaCy)
doc = nlp(text)
spacy_sentence_tokens = [sent.text for sent in doc.sents]
print("\nSentence Tokens (SpaCy):")
print(spacy_sentence_tokens)

# Word tokenization (SpaCy)
spacy_word_tokens = [token.text for token in doc]
print("\nWord Tokens (SpaCy):")
print(spacy_word_tokens)

# Character tokenization
char_tokens = list(text)
print("\nCharacter Tokens:")
print(char_tokens)
```

**Output:**
```
Word Tokens:
['Natural', 'Language', 'Processing', 'enables', 'computers', 'to', 'understand', 'human', 'language', '.', 'It', 'is', 'a', 'fascinating', 'field', '.']

Sentence Tokens:
['Natural Language Processing enables computers to understand human language.', 'It is a fascinating field.']

Sentence Tokens (SpaCy):
['Natural Language Processing enables computers to understand human language.', 'It is a fascinating field.']

Word Tokens (SpaCy):
['Natural', 'Language', 'Processing', 'enables', 'computers', 'to', 'understand', 'human', 'language', '.', 'It', 'is', 'a', 'fascinating', 'field', '.']

Character Tokens:
['N', 'a', 't', 'u', 'r', 'a', 'l', ' ', 'L', 'a', 'n', 'g', 'u', 'a', 'g', 'e', ' ', 'P', 'r', 'o', 'c', 'e', 's', 's', 'i', 'n', 'g', ' ', 'e', 'n', 'a', 'b', 'l', 'e', 's', ' ', 'c', 'o', 'm', 'p', 'u', 't', 'e', 'r', 's', ' ', 't', 'o', ' ', 'u', 'n', 'd', 'e', 'r', 's', 't', 'a', 'n', 'd', ' ', 'h', 'u', 'm', 'a', 'n', ' ', 'l', 'a', 'n', 'g', 'u', 'a', 'g', 'e', '.', ' ', 'I', 't', ' ', 'i', 's', ' ', 'a', ' ', 'f', 'a', 's', 'c', 'i', 'n', 'a', 't', 'i', 'n', 'g', ' ', 'f', 'i', 'e', 'l', 'd', '.']
```

This pipeline demonstrates word, sentence, and character tokenization using both NLTK and SpaCy, showing how different tokenization techniques apply to the same text at varying levels of granularity.
