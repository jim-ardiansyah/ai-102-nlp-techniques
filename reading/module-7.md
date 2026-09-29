# Chapter 6: Sentiment Analysis (Simple Study Guide)

**Sentiment analysis** means teaching a computer to figure out the *feeling* behind a piece of text. Is a review happy, angry, or neutral?

This chapter shows three ways to do it, from simplest to most powerful:

| Approach | How it works | Think of it like... |
|---|---|---|
| **Rule-based** (6.1) | Humans write rules and word lists | Following a recipe card |
| **Machine learning** (6.2) | The computer learns patterns from labeled examples | Learning by looking at graded homework |
| **Deep learning** (6.3) | Big neural networks learn everything on their own from lots of data | A student who reads millions of books |

---

## 6.1 Rule-Based Approaches

### What is it?

In a rule-based system, people write the rules by hand. The main tool is a **sentiment lexicon**: a dictionary where every word has a feeling score. For example, "love" might be +3 and "hate" might be −3. The computer adds up the scores to decide how positive or negative a sentence is.

The big plus: it's **easy to understand**. You can see exactly why the computer made its decision, and you can change the rules whenever you want.

### The 4 steps

1. **Tokenization**: split the text into words.
   "I love sunny days" → ["I", "love", "sunny", "days"]
2. **Normalization**: make words consistent, like changing everything to lowercase and removing punctuation, so "Sunny" and "sunny" count as the same word.
3. **Lexicon lookup**: find each word's feeling score in the dictionary.
4. **Rule application**: combine the scores to get the final answer. For example, "more positive words than negative words → Positive."

### Popular sentiment lexicons

- **AFINN**: gives words a score from −5 (very negative) to +5 (very positive). Simple and popular.
- **SentiWordNet**: gives scores to groups of words with the same meaning (synonyms). More detailed.
- **NRC Emotion Lexicon**: goes beyond positive/negative and tags words with emotions like joy, sadness, anger, and surprise.

### Example 1: TextBlob

```python
from textblob import TextBlob

text = "I love this product! It works wonderfully and the quality is excellent."
blob = TextBlob(text)
print(blob.sentiment)
```

TextBlob gives you two numbers:

- **Polarity**: from −1 (very negative) to +1 (very positive). Here it's **0.625**, so positive.
- **Subjectivity**: from 0 (pure fact) to 1 (pure opinion). Here it's **0.6**, so it's fairly opinion-based.

### Example 2: AFINN with your own rule

```python
from afinn import Afinn

afinn = Afinn()
text = "I hate the traffic in this city. It makes commuting a nightmare."
score = afinn.score(text)

if score > 0:
    sentiment = "Positive"
elif score < 0:
    sentiment = "Negative"
else:
    sentiment = "Neutral"

print(score, sentiment)   # -6.0 Negative
```

AFINN adds up the scores of the words ("hate" and "nightmare" are both negative). Then our simple rule turns the number into a label: above 0 is positive, below 0 is negative, exactly 0 is neutral.

### Pros and cons

**Advantages**
- **Easy to explain**: you can show anyone exactly how the answer was reached. Useful when rules and regulations require explanations.
- **Simple**: no training data and no powerful computer needed.
- **Customizable**: you can add special words for your field (medicine, law, etc.).

**Limitations**
- **Misses things**: slang, idioms, and new words may not be in the dictionary.
- **No sense of context**: it can't catch sarcasm. "I just love waiting in long lines" looks positive because of "love," even though the person is annoyed.
- **Lots of upkeep**: language keeps changing, so someone has to keep updating the word lists.

### Where sentiment analysis is used

- **Customer feedback**: find out what customers like or dislike in reviews and surveys.
- **Social media monitoring**: see how people react to a brand, event, or news in real time.
- **Market research**: spot trends in what people want.
- **Brand management**: track a company's reputation and react quickly to bad press.
- **Finance**: measure the "mood" of the market from news and posts.
- **Healthcare**: track public health concerns and patient experiences.

---

## 6.2 Machine Learning Approaches

### What is it?

Instead of writing rules by hand, we give the computer **lots of examples that are already labeled** ("this review is positive," "this one is negative"). The computer learns the patterns by itself and can then label new text it has never seen.

Machine learning handles tricky language better than rules, because it learns from real examples instead of a fixed word list.

### The 6 steps

1. **Collect data**: gather many texts with sentiment labels (reviews, tweets, surveys).
2. **Clean the data**: split into words, lowercase, remove punctuation.
3. **Extract features**: turn words into numbers (computers only understand numbers).
4. **Train the model**: let the algorithm learn from the labeled examples.
5. **Evaluate**: test the model on examples it didn't see during training.
6. **Predict**: use the model on brand-new text.

### Turning text into numbers (feature extraction)

- **Bag of Words (BoW)**: count how many times each word appears. Word order is ignored, like dumping all the words into a bag.
- **TF-IDF**: like Bag of Words, but smarter. Words that appear everywhere ("the," "is") get a **low** score. Words that are special to one document get a **high** score.
- **Word embeddings** (Word2Vec, GloVe, FastText): turn each word into a list of numbers so that words with similar meanings end up close together. For example, "happy" and "glad" would be near each other.

### TF-IDF explained simply

- **TF (Term Frequency)**: how often a word shows up in *this* document. In an article about cats, "cat" appears a lot, so it has high TF.
- **IDF (Inverse Document Frequency)**: how *rare* the word is across *all* documents. "The" is everywhere, so its IDF is low. A rare word gets a high IDF.
- **TF-IDF = TF × IDF**. A word scores high when it appears a lot in one document but not in many others. That means it's probably important.

### Example: TF-IDF in Python

```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "I love this product! It's amazing.",
    "This is the worst service I have ever experienced.",
    "I am very happy with my purchase.",
    "I am disappointed with the quality of this item."
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(X.toarray())
```

- `fit` learns the vocabulary (all the words) and how rare each one is.
- `transform` turns each sentence into a row of numbers.
- The result is a table: **each row is a sentence, each column is a word**, and each cell is that word's TF-IDF score.

**Where TF-IDF is used:** sorting text into categories (like spam vs. not spam), ranking search results, and measuring how similar two documents are (useful for plagiarism detection and recommendations).

### Common machine learning models

- **Logistic Regression**: simple and fast. Calculates the probability that a text is positive. A great starting point.
- **Support Vector Machine (SVM)**: draws the best possible "dividing line" between positive and negative examples. Works well with text.
- **Naive Bayes**: uses probability and assumes each word acts independently. Simple, but surprisingly good for text.
- **Random Forest**: builds many decision trees and lets them "vote." Harder to fool than a single tree.

### Example: Training a Logistic Regression model

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

corpus = [
    "I love this product! It's amazing.",
    "This is the worst service I have ever experienced.",
    "I am very happy with my purchase.",
    "I am disappointed with the quality of this item."
]
labels = [1, 0, 1, 0]   # 1 = positive, 0 = negative

X = TfidfVectorizer().fit_transform(corpus)

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.25, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)          # learn
y_pred = model.predict(X_test)       # guess

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

What happens:
1. Turn the sentences into TF-IDF numbers.
2. Split the data: 75% for **training** (studying) and 25% for **testing** (the exam).
3. `fit` trains the model.
4. `predict` makes guesses on the test data.
5. Check how many guesses were right.

The book gets 100% accuracy, but that's only because the dataset is tiny. Real projects use thousands of examples.

### How to grade a model

- **Accuracy**: what percent of all guesses were correct. Can be misleading if one class is much bigger than the other.
- **Precision**: of everything the model *called* positive, how much really was positive? Important when false alarms are costly (e.g., spam filters shouldn't block real emails).
- **Recall**: of everything that *really* was positive, how much did the model catch? Important when missing something is costly (e.g., disease screening, fraud detection).
- **F1 score**: one number that balances precision and recall.

### Pros and cons

**Advantages**
- **More accurate** than rules, because it learns complex patterns.
- **Scales up** to huge datasets.
- **Flexible**: can be retrained for different topics or languages.

**Limitations**
- **Needs lots of labeled data**, which takes time and money to create.
- **More complex**: needs tuning and testing.
- **Harder to explain** than rules: it's not always clear why the model made its choice.

---

## 6.3 Deep Learning Approaches

### What is it?

Deep learning uses **neural networks**: large models with many layers, loosely inspired by the brain. They learn everything themselves, including which features matter, straight from the raw text. This is called **end-to-end learning**, and it means no hand-made features are needed.

Deep learning usually gives the best results, especially with large and complicated datasets.

### The four main model types

| Model | Main idea | Good at |
|---|---|---|
| **CNN** | Slides small "windows" over the text to spot short phrases | Catching key phrases like "very happy" |
| **RNN** | Reads words one at a time, in order, remembering what came before | Text where order matters |
| **LSTM** | An upgraded RNN with better long-term memory | Long sentences and paragraphs |
| **Transformer (BERT)** | Looks at all words at once and figures out which ones relate to each other ("attention") | Best results on most tasks today |

### CNNs (Convolutional Neural Networks)

CNNs were first made for images, where they spot edges and shapes. For text, they spot **short word patterns** (called **n-grams**), like "extremely disappointed."

How it works:
1. A **filter** (a small window) slides across the sentence, looking at a few words at a time.
2. It creates a **feature map** showing where patterns were found.
3. **Pooling** keeps only the strongest signals and throws the rest away.
4. The final layers use those signals to decide: positive or negative.

### The shared recipe for the Keras examples

The CNN and LSTM examples use the same steps:

1. **Tokenize**: give each word a number ID. "I love this" → [1, 5, 3].
2. **Pad**: make every sentence the same length (10 here) by adding zeros. Neural networks need equal-sized inputs.
3. **Split** into training and test data.
4. **Build** the model layer by layer.
5. **Compile**: choose how it learns (the `adam` optimizer) and how mistakes are measured (`binary_crossentropy` for two classes).
6. **Train** with `fit` for a few rounds (**epochs**).
7. **Evaluate** on the test data.
8. **Predict** new sentences.

### Example: CNN model (the important part)

```python
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=50))       # word IDs → meaning vectors
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))  # find 5-word patterns
model.add(GlobalMaxPooling1D())                           # keep the strongest patterns
model.add(Dense(10, activation='relu'))                   # combine them
model.add(Dense(1, activation='sigmoid'))                 # output 0 to 1

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

Layer by layer:
- **Embedding**: turns each word ID into a list of 50 numbers that represent its meaning.
- **Conv1D**: 128 filters, each looking at 5 words at a time.
- **GlobalMaxPooling1D**: keeps only the strongest signal from each filter.
- **Dense (sigmoid)**: gives a number between 0 and 1. Above 0.5 means positive.

### RNNs and LSTMs

**RNNs** read text word by word and keep a "memory" (called the **hidden state**) of what they've read so far. That's great for text, because word order matters.

The problem: RNNs **forget** things from early in a long sentence. This is called the **vanishing gradient problem**.

**LSTMs** fix this with a **memory cell** and three **gates** that act like valves:
- **Input gate**: what new information should be stored?
- **Forget gate**: what old information should be thrown away?
- **Output gate**: what information should be passed along?

Example of why this matters: in "The movie started slow and boring, but by the end I absolutely loved it," the final feeling depends on words spread across the whole sentence.

### Example: LSTM model

The code is the same as the CNN example, but the middle layers are replaced by one LSTM layer:

```python
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=50))
model.add(LSTM(100))                        # 100 memory units
model.add(Dense(1, activation='sigmoid'))
```

### Transformers and BERT

**BERT** (Bidirectional Encoder Representations from Transformers) is a very powerful, **pre-trained** model. It has already read a huge amount of text, so it understands language well before you even start.

Its secret is **self-attention**: for every word, it looks at *all* the other words in the sentence (both before and after) and decides which ones matter most. That's what "bidirectional" means.

You don't train BERT from scratch. You **fine-tune** it: take the pre-trained model and train it a little more on your own labeled data.

### Example: BERT (key steps)

```python
from transformers import BertTokenizer, TFBertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
X = tokenizer(corpus, padding=True, truncation=True, max_length=10, return_tensors='tf')

model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```

- BERT uses **its own tokenizer**, because it was trained with a specific vocabulary.
- `padding` makes sentences the same length; `truncation` cuts off sentences that are too long.
- `num_labels=2` means two classes: positive and negative.
- The learning rate is **very small (0.00002)**, so the training only gently adjusts what BERT already knows.
- For prediction, the model outputs two scores and picks the bigger one (`argmax`).

### Pros and cons

**Advantages**
- **Best performance**: top results on sentiment analysis, translation, and summarization.
- **Learns features automatically**: no need to hand-design features.
- **Understands context**: can connect words that are far apart in a text.
- **Can combine data types**: some models use text, images, and audio together.

**Limitations**
- **Needs powerful hardware** (GPUs/TPUs) and uses a lot of energy.
- **Needs a lot of good data**. Bad or biased data leads to bad or biased results.
- **"Black box"**: it's very hard to explain why the model made a decision. That's a problem in fields like healthcare, finance, and law.

---

## Big Picture: Which approach should you use?

| | Rule-based | Machine learning | Deep learning |
|---|---|---|---|
| Needs training data? | No | Yes (medium) | Yes (a lot) |
| Accuracy | Lower | Good | Best |
| Easy to explain? | Very | Somewhat | Hard |
| Computer power needed | Very little | Moderate | A lot |
| Handles sarcasm/context? | Poorly | Somewhat | Best |
| Best when... | You need transparency or have no data | You have a decent labeled dataset | You have lots of data and strong hardware |

---

## Heads-up: problems in the book's examples

Keep these in mind if you run the code:

- **The datasets are way too small.** Every example uses only 4 sentences, and `test_size=0.25` leaves just **1 sentence** for testing. Scores like "100% accuracy" or "50% accuracy" mean almost nothing. (With 1 test sentence, accuracy can only be 0% or 100%, so the "0.5" in the book's deep learning outputs can't actually happen.)
- **Missing import in 6.2**: the logistic regression example uses `TfidfVectorizer` without importing it in that block. Add `from sklearn.feature_extraction.text import TfidfVectorizer`.
- **Labels for Keras**: convert labels with `np.array(labels)` before training. Plain Python lists can cause errors.
- **Newer Keras (TensorFlow 2.16+)**: `tensorflow.keras.preprocessing.text.Tokenizer` has been removed, and the `input_length` argument in `Embedding` is no longer used. Newer code uses the `TextVectorization` layer instead.
- **Model name typos**: the book shows `'bertbase-uncased'`. The correct name is `'bert-base-uncased'`. The learning rate should be `2e-5` (0.00002), not `2e5`.
- **BERT and TensorFlow**: recent versions of the Hugging Face `transformers` library have moved away from TensorFlow, so most current BERT tutorials use PyTorch. Also, the book only passes `input_ids` to BERT; normally you pass the `attention_mask` too.
- **`max_length=10` is very short for BERT**: some of the sample sentences get cut off. Values like 64 or 128 are more typical.
