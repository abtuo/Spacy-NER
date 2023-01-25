import random
from spacy.util import minibatch, compounding
from pathlib import Path
import spacy
from spacy.tokens import Doc, Span, DocBin

from spacy.training.example import Example

from load_data import load_data

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
        
    TRAIN_DATA = []
    TEST_DATA = []
    nlp = spacy.blank("en")
    #nlp=spacy.load('en_core_web_lg')

    df_train = load_data('data/train.few.rel.json')
    df_test = load_data('data/test.few.rel.json')
    df_dev = load_data('data/dev.few.rel.json')
    
    train_data = preprocess(df_train)
    test_data = preprocess(df_test)


    for doc in train_data:
        entities = []
        for ent in doc.ents:
            entities.append((ent.start_char, ent.end_char, ent.label_))
        TRAIN_DATA.append((doc.text, {"entities": entities}))

    for text, entities in TRAIN_DATA:
        ent = entities['entities']
        doc = nlp(text)

        if '-' in spacy.training.offsets_to_biluo_tags(nlp.make_doc(text), ent):
            print(spacy.training.offsets_to_biluo_tags(nlp.make_doc(text), ent))

    # Getting the pipeline component
    nlp.add_pipe('ner')

    nlp.begin_training()

    ner=nlp.get_pipe("ner")
    
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    # Add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    with nlp.disable_pipes(*unaffected_pipes):

        # Training for 30 iterations
        for iteration in range(30):

            # shuufling examples  before every iteration
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                example = []
                texts, annotations = zip(*batch)
                for i in range(len(texts)):
                    
                    example.append(Example.from_dict(doc, annotations[i]))

                nlp.update(
                            example,
                            drop=0.5,  # dropout - make it harder to memorise data
                            losses=losses,
                        )

            print("Losses", losses)

    output_dir = './checkpoint/'

    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)