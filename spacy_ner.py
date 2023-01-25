import re
import spacy
from spacy.util import minibatch, compounding
import random
import pandas as pd
from load_data import load_data
from spacy.training.example import Example

#Step 0: Let's load the data
df_train = load_data('data/train.few.rel.json')


def process_review(review):
    processed_token = []
    for token in review.split():
        token = ''.join(e.lower() for e in token if e.isalnum())
        processed_token.append(token)
    return ' '.join(processed_token)

#Step 1: Let's create the training data
count = 0
TRAIN_DATA = []

for _, item in df_train.iterrows() :
    entity_mentions = item['entity_mentions']
    review = process_review(item['sentence'])

    ent_dict = {}
    entities = []

    count = 0
    for entity in entity_mentions :
        try :
            for i in re.finditer(entity['text'], review):
                entity = (i.span()[0], i.span()[1], entity['entity_type'])
                entities.append(entity)
        except :
            pass

    ent_dict['entities'] = entities
    train_item = (review, ent_dict)
    TRAIN_DATA.append(train_item)

#Step 2: Let's train the model
n_iter = 1000
def train():
    nlp = spacy.blank("en")  # create blank Language class
    print("Created blank 'en' model")
    
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe('ner', last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")
        
    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            
            ner.add_label(ent[2])
            
    nlp.begin_training()
    for itn in range(n_iter):
        random.shuffle(TRAIN_DATA)
        losses = {}
        # batch up the examples using spaCy's minibatch
        batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            texts, annotations = zip(*batch)

            example = []
            # Update the model with iterating each text
            try :
                for i in range(len(texts)):
                    doc = nlp.make_doc(texts[i])
                    example.append(Example.from_dict(doc, annotations[i]))

                nlp.update(example,  # batch examples
                    drop=0.5,  # dropout - make it harder to memorise data
                    losses=losses,
                )
            except :
                #print(example)
                #print(annotations)
                #print(texts)
                #print(batch)
                print('error')
                #break
           
        print("Losses", losses)
    return nlp

nlp = train()

#Step 3: Let's test the model

doc = "that it is very difficult to discern whether or not that is  in point of fact  the missiles that we have become accustomed to striking baghdad and its environs  or whether it is the large 2  000 pound jdam bombs that have been striking targets sclected by the coalition in baghdad  or whether these flashes now represent artillery barrages as fighting continues to take control of the  to take control of saddam international airport"

nlp = spacy.load("en_core_web_lg")

# Test the model
for text, _ in TRAIN_DATA[:100]:
    doc = nlp(text)
    print(doc)
    print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
