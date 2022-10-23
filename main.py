"""
NLP-Pipeline - spaCy pipelines for NLP

"""
import pprint
import spacy

# spaCy version
# currently 3.4.2
print(f"spaCy version {spacy.__version__}")

# Load a model and construct the basic pipeline
nlp = spacy.load("en_core_web_sm")

# Basic Pipeline is...
# currently: tok2vec, tagger, parser, attribute_ruler, lemmatizer, ner
print(nlp.pipe_names)

text = """Tesla’s third-quarter revenue fell short of Wall Street expectations on Wednesday, prompting its stock price to drop more than 4% after markets closed."""

# Tokenization
# note: doc is a generator for tokens
doc = nlp(text)
print([t.text for t in doc])

# Lemmatization
# note: most text properties end with "_" e.g. lemma_, tag_, dep_
for t in doc:
    print(f"{t.text} -> {t.lemma_}")

# Part of Speech Tagging
# NN   Noun
# NNS  Plural noun
# PRP  Pronoun
# PRP$ Possesive pronoun
# VB   Verb
# VBD  Past tense verb
# VBG  Participle
# JJ   Adjective
for t in doc:
    print(f"{t.text} -> {t.tag_}")

# Syntactic Dependency
for t in doc:
    print(f"{t.text},{t.tag_},{t.dep_}")

# Named Entity Recognition
# ORG
# DATE
# PERCENT
for t in doc:
    print(f"{t.text},{t.ent_type_}")
    
# Format as an array of objects
def token_to_dict(t):
    return {
        "text": t.text,
        "pos": t.tag_,
        "dep": t.dep_,
        "ent": t.ent_type_
    }

data = [token_to_dict(t) for t in doc]
pprint.pprint(data)

# Tabulate data
# note: tabulate clashes with spacy 
#header = data[0].keys()
#rows =  [x.values() for x in data]
#print(tabulate.tabulate(rows, header))
