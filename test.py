import spacy
from spacy import displacy


import concise_concepts

data = {
    "fruit": ["apple", "pear", "orange"],
    "vegetable": ["broccoli", "spinach", "tomato", "Onion"],
    "meat": ["beef", "pork", "fish", "lamb"],
}

text = """
    Heat the oil in a large pan and add the Onion, celery and carrots.
    Then, cook over a medium–low heat for 10 minutes, or until softened.
    Add the courgette, garlic, red peppers and oregano and cook for 2–3 minutes.
    Later, add some oranges and chickens. """


nlp = spacy.load("en_core_web_lg", disable=["ner"])

nlp.add_pipe(
    "concise_concepts",
    config={
        "data": data,
        "ent_score": True,  # Entity Scoring section
        "verbose": True,
        "exclude_pos": ["VERB", "AUX"],
        "exclude_dep": ["DOBJ", "PCOMP"],
        "include_compound_words": False,
        "json_path": "./fruitful_patterns.json",
    },
)
doc = nlp(text)

options = {
    "colors": {"fruit": "darkorange", "vegetable": "limegreen", "meat": "salmon"},
    "ents": ["fruit", "vegetable", "meat"],
}

ents = doc.ents

for ent in ents:
    new_label = f"{ent.label_} ({ent._.ent_score:.0%})"
    options["colors"][new_label] = options["colors"].get(ent.label_.lower(), None)
    options["ents"].append(new_label)
    ent.label_ = new_label
doc.ents = ents

# Construction 2
from spacy.tokens import Doc

words = ["hello", "world", "!"]
spaces = [True, False, False]
doc = Doc(nlp.vocab, words=words, spaces=spaces)

print(doc)
