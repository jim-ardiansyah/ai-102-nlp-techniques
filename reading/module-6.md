# Chapter 5: Understanding Sentence Structure (Simple Study Guide)

This chapter covers three tools that help computers understand how sentences are built: **POS tagging** (what kind of word is this?), **Named Entity Recognition** (is this word a name, place, or amount?), and **Dependency Parsing** (how do the words connect to each other?).

---

## 5.1 Parts of Speech (POS) Tagging

### What is it?

POS tagging means labeling each word in a sentence with its word type, like noun, verb, or adjective. You did this in grammar class; POS tagging just has a computer do it.

It's not always easy, because the same word can play different roles. "Run" is a verb in "I run every morning" but a noun in "I went for a run." The computer has to look at the surrounding words to decide.

POS tagging matters because many bigger NLP tasks (like translation or finding names in text) are built on top of it. If the tags are wrong, everything built on them gets worse.

### Common tags

| Tag | Meaning | Examples |
|---|---|---|
| NN | Noun (person, place, thing, idea) | dog, city, happiness |
| VB | Verb (action or state) | run, is, think |
| JJ | Adjective (describes a noun) | big, blue, interesting |
| RB | Adverb (describes a verb, adjective, or adverb) | quickly, very, seldom |
| PRP | Pronoun (replaces a noun) | he, they, it, we |
| IN | Preposition (shows position, direction, or time) | on, in, under, before |

You'll also see variations, like **NNP** (proper noun, such as "Python"), **VBZ** (verb like "is"), and **VBG** (verb ending in -ing).

### Doing it in Python (NLTK)

```python
import nltk
from nltk import word_tokenize, pos_tag

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

text = "Natural Language Processing with Python is fascinating."
tokens = word_tokenize(text)   # split sentence into words
tags = pos_tag(tokens)         # label each word
print(tags)
```

What happens, step by step:

1. `word_tokenize` splits the sentence into pieces called **tokens**: "Natural", "Language", "Processing", and so on.
2. `pos_tag` gives each token a label using a model that was already trained.
3. The result is a list of (word, tag) pairs, for example `('Natural', 'JJ')` and `('Python', 'NNP')`.

### How do we know if a tagger is good?

We test it on sentences where humans already wrote the correct tags (called the **gold standard**) and compare. Useful measures:

- **Accuracy**: what percent of words got the right tag overall.
- **Precision**: when the tagger says "this is a noun," how often is it actually a noun?
- **Recall**: of all the real nouns, how many did the tagger catch?
- **F1 score**: one number that balances precision and recall.

The book's example uses the **Treebank** corpus (a collection of already-tagged sentences) as the test set, then uses scikit-learn to compute these numbers.

Taggers can perform worse when:

- the text style is different from what it trained on (trained on news, tested on tweets),
- the language is different,
- words are ambiguous ("run"),
- the training data was low quality.

### Training your own tagger

Sometimes you need a tagger for a special field, like medicine or law, with unusual vocabulary. NLTK lets you train simple taggers:

- **Unigram tagger**: gives each word the tag it had most often in training. Simple but ignores context.
- **Bigram tagger**: also looks at the tag of the previous word, so it's a bit smarter. If it can't decide, it "backs off" to the unigram tagger.

In the book's test, the unigram tagger scored about 86.5% and the bigram tagger about 89%. Looking at context helps.

Things to keep in mind: you need good labeled data, training can take computing power, and you should always test on data the model hasn't seen.

### Where POS tagging is used

Parsing sentence structure, finding names (proper nouns often signal names), sentiment analysis (adjectives like "happy" or "sad" carry feeling), pulling facts out of text, machine translation, text-to-speech (knowing word roles helps with natural-sounding speech), and grammar checkers.

---

## 5.2 Named Entity Recognition (NER)

### What is it?

NER finds the "important names and things" in text and sorts them into groups. For example, in "Apple is buying a U.K. startup for $1 billion," NER finds:

- **Apple**, an organization
- **U.K.**, a place
- **$1 billion**, an amount of money

This turns messy text into organized information a computer can use.

### Common categories

- **Person (PER)**: names of people, like "Albert Einstein."
- **Organization (ORG)**: companies, schools, governments, like "Google."
- **Location (LOC)**: cities, countries, rivers, mountains, like "Paris."
- **Miscellaneous (MISC)**: the book puts dates, percentages, and money here.

spaCy uses more specific labels than this list, such as **GPE** (countries, cities, states), **MONEY**, **DATE**, and **PERCENT**.

### Doing it in Python (spaCy)

First install spaCy and its small English model:

```
pip install spacy
python -m spacy download en_core_web_sm
```

```python
import spacy

nlp = spacy.load('en_core_web_sm')
doc = nlp("Apple is looking at buying U.K. startup for 1 billion.")

for ent in doc.ents:
    print(ent.text, ent.label_)
```

Output:

```
Apple ORG
U.K. GPE
1 billion MONEY
```

The model reads the sentence, finds the entities, and stores them in `doc.ents`. Each one has its text and its label.

### How do we know if NER is good?

We compare the model's answers to a human-labeled version and count:

- **True positives**: entities it found correctly.
- **False positives**: things it called entities that weren't.
- **False negatives**: real entities it missed.

From those counts:

- **Precision** = true positives ÷ (true positives + false positives). "Of what it found, how much was right?"
- **Recall** = true positives ÷ (true positives + false negatives). "Of what existed, how much did it find?"
- **F1** = 2 × (precision × recall) ÷ (precision + recall). A balance of the two.

A model trained on news may do poorly on medical or legal text, because the vocabulary is different.

### Training your own NER model

If you need to find things a normal model doesn't know about (like gadget names), you can train a custom model in spaCy. The basic steps:

1. Start with a blank English model.
2. Add an NER component to it.
3. Add your new label, for example `GADGET`.
4. Give it example sentences with the exact character positions of each entity.
5. Train it for several rounds (called **epochs**). The "loss" number should go down as it learns.
6. Test it on a new sentence, like "I just bought a new iPhone," and check that it finds "iPhone" as a GADGET.

In real life you'd need hundreds or thousands of examples, not just two.

### Where NER is used

Search engines (finding documents about a specific person or company), question answering ("Who is the CEO of Google?"), organizing news articles by people and places, and customer support (spotting which product a customer is complaining about).

---

## 5.3 Dependency Parsing

### What is it?

Dependency parsing figures out **which words depend on which other words** in a sentence. Every connection has:

- a **head**: the main word, and
- a **dependent**: the word that describes or attaches to it.

Think of it as a family tree for a sentence. The main verb is usually at the top (the **root**), and other words hang off it.

### Example: "The cat sat on the mat."

| Word | Role | Attached to |
|---|---|---|
| The | det (determiner) | cat |
| cat | nsubj (subject) | sat |
| sat | ROOT (main verb) | — |
| on | prep (preposition) | sat |
| the | det (determiner) | mat |
| mat | pobj (object of preposition) | on |
| . | punct (punctuation) | sat |

So "sat" is the center, "cat" is who did the sitting, and "on the mat" tells where.

### Doing it in Python (spaCy)

```python
import spacy
from spacy import displacy

nlp = spacy.load('en_core_web_sm')
doc = nlp("The cat sat on the mat.")

for token in doc:
    print(f"{token.text} ({token.dep_}): {token.head.text}")

displacy.render(doc, style="dep", jupyter=True)  # draws the tree
```

- `token.dep_` gives the relationship label (like nsubj).
- `token.head.text` gives the word it's attached to.
- `displacy.render` draws a picture of the tree with arrows (works in Jupyter Notebook).

### How do we know if a parser is good?

Two scores:

- **UAS (Unlabeled Attachment Score)**: percent of words attached to the correct head. It only checks the arrow, not the label.
- **LAS (Labeled Attachment Score)**: percent of words with the correct head **and** the correct label. Stricter.

Example: if the parser knows "cat" connects to "sat," that counts for UAS. For LAS, it must also know the connection is "nsubj."

Like the other tools, parsers trained on news or academic text may struggle with slang, social media, or technical jargon. Extra training on your own kind of text (called **fine-tuning** or **domain adaptation**) can help.

### Training your own parser

The steps are almost the same as custom NER: start with a blank model, add a "parser" component, add labels (like nsubj, dobj, prep), give it sentences with the correct head and label for every word, train for several rounds, then test.

### Where dependency parsing is used

- **Information extraction**: from "Barack Obama was born in Hawaii," link the person to the place.
- **Machine translation**: knowing subject, verb, and object helps put words in the right order in another language.
- **Sentiment analysis**: in "I don't like the new design," the parser shows "don't" is attached to "like," so the sentence is negative.
- **Question answering**: breaking down "Who is the CEO of Google?" into its parts.
- **Summarization**: finding the main ideas in a text.
- **Coreference resolution**: working out that "He" means "John" and "it" means "car."
- **Text generation**: helping produce grammatically correct sentences.

---

## Quick Comparison

| Tool | Question it answers | Example output |
|---|---|---|
| POS tagging | What type of word is this? | "cat" → noun |
| NER | Is this a name, place, date, or amount? | "Apple" → organization |
| Dependency parsing | How are the words connected? | "cat" → subject of "sat" |

---

## Heads-up: a few problems in the book's code

If you try running the examples, watch out for these:

- **NLTK downloads**: newer NLTK versions need `nltk.download('punkt_tab')` and `nltk.download('averaged_perceptron_tagger_eng')` instead of the older names.
- **Old spaCy commands**: `nlp.begin_training()` is from spaCy version 2. In version 3 (the current one), use `nlp.initialize()`.
- **iPhone position mistake**: in the custom NER training data, "iPhone" in "Apple is releasing a new iPhone." is at characters 25 to 31, not 26 to 32. With the wrong numbers, spaCy will complain or learn the wrong thing.
- **NER evaluation example**: the book passes lists of entity strings to scikit-learn's precision and recall functions. That treats each position as a class label rather than truly matching entities, so it isn't how NER is really evaluated. Real NER evaluation compares entity spans and labels (tools like `seqeval` or spaCy's built-in scorer do this).
- **Custom parser training data**: the example labels are inconsistent. For instance, "playing" is labeled "aux" and "tennis" is labeled "prep," which don't match the sentence. The sample output "books (pobj): reading" is also wrong, since "books" should be the direct object (dobj). Treat that example as showing the *steps*, not correct grammar.
