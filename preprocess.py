import re
import spacy
from spacy.tokens import DocBin
from load_data import load_data
from spacy.training.example import Example
from spacy.tokens import Doc, Span

nlp = spacy.blank("en")

def data_to_spacy(data_df):
    data =  []
    TYPES = []
    for _, item in data_df.iterrows() :
        entity_mentions = item['entity_mentions']
        sentence = item['tokens']
        labels = ['O']*len(sentence)

        ent_dict = {}
        entities = []
        visited_items = []

        doc = Doc(nlp.vocab, sentence)
        
        for entity in entity_mentions :
            try :
                entities.append(Span(doc, entity['start'], entity['end'], entity['entity_type'])) 
            except :
                pass
        
        doc.set_ents(entities)

        ent_dict['entities'] = entities
      

        data.append(doc)

    return data

def preprocess(data_df):
    data =  []
    for _, item in data_df.iterrows() :
        entity_mentions = item['entity_mentions']
        sentence = item['tokens']
        labels = ['O']*len(sentence)
        spaces = [True]*len(sentence)
        spaces[-1] = False
        
        for entity in entity_mentions :
            try :
                for i in range(entity['start'], entity['end']):
                    if i == entity['start']:
                        labels[i] = 'B-'+entity['entity_type']
                    else:
                        labels[i] = 'I-'+entity['entity_type']
            except :
                pass
        
        doc = Doc(nlp.vocab, words=sentence, spaces=spaces, ents=labels)
        #doc.ents = labels

        data.append(doc)

    return data

if __name__ == "__main__":

    df_train = load_data('data/train.few.rel.json')
    df_test = load_data('data/test.few.rel.json')
    df_dev = load_data('data/dev.few.rel.json')

    training_data = preprocess(df_train)
    test_data = preprocess(df_test)
    dev_data = preprocess(df_dev)
    
    #training_data = data_to_spacy(df_train)
    #test_data = data_to_spacy(df_test)
    #dev_data = data_to_spacy(df_dev)

    for doc in training_data[:10]:
        for ent in doc.ents:
            print(ent.text, ent.start_char, ent.end_char, ent.label_)

    db = DocBin()

    for doc in training_data:
        db.add(doc)

    db.to_disk("./train.spacy")

    db = DocBin()
    for doc in test_data:
        db.add(doc)

    db.to_disk("./test.spacy")

    db = DocBin()
    for doc in dev_data:
        db.add(doc)
    
    db.to_disk("./dev.spacy")
    nlp.to_disk("checkpoint/spacy_ner")
