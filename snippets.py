#! /usr/bin/env python
"""
Entity snippets
"""
import spacy

# Load a model and construct the basic pipeline
nlp = spacy.load("en_core_web_sm")


text = """Tesla’s third-quarter revenue fell short of Wall Street expectations on Wednesday, prompting its stock price to drop more than 4% after markets closed. Some experts don’t think the market slowdown, or the Twitter deal, will hurt Tesla’s position as a leader in the electric vehicle industry."""
doc = nlp(text)
print([t.text for t in doc])

# Find sentence containing an entity
def find_sentence(doc, ent):
    for sent in doc.sents:
        if (ent.start >= sent.start) and (ent.end <= sent.end):
            return sent
    return None

def get_entity_in_context(doc, ent):
    sent = find_sentence(doc, ent)
    n_tokens = 4
    start = ent.start
    end = ent.end
    while (n_tokens > 2) and (start >= sent.start):
        start -= 1
        n_tokens -= 1
    while (n_tokens > 0) and (end <= sent.end):
        end += 1
        n_tokens -= 1
    context = sent[start:end+1]
    return context
        
for ent in doc.ents:
    if ent.label_ == "ORG":
        sent = find_sentence(doc, ent)
        print(f"{ent} {ent.label_} {sent}")
        for t in ent:
            print(f"{t.text} {t.dep_} {t.head.text}")

for ent in doc.ents:
    if ent.label_ == "ORG":
        context = get_entity_in_context(doc, ent)
        print(ent, context)
