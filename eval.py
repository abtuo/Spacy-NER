import spacy
from spacy.scorer import Scorer
from spacy.tokens import Doc
from spacy.training.example import Example
from load_data import load_data

df_test = load_data('data/test.few.rel.json')
nlp = spacy.load("checkpoint/spacy_ner")

TEST_DATA = []
for _, item in df_test.iterrows() :
    entity_mentions = item['entity_mentions']
    sentence = item['sentence']
    
    labels = ['O']*len(sentence.split())
    ent_dict = {}
    entities = []
    for entity in entity_mentions :
      
        try :
            if len(entity['text'].split()) == 1:
                labels[entity['start']] = 'U-'+entity['entity_type']

            else:
                labels[entity['start']] = 'B-'+entity['entity_type']
                for i in range(entity['start']+1, entity['end']):
                    labels[i] = 'L-'+entity['entity_type']
            ent_dict['entities'] = labels
            train_item = (sentence, ent_dict)
            TEST_DATA.append(train_item)
        except :
            pass

print(TEST_DATA[:10])


def evaluate(ner_model, examples):
    scorer = Scorer()
    example = []
    for input_, annot in examples:
        pred = ner_model(input_)
        print(pred,annot)
        temp = Example.from_dict(pred, dict.fromkeys(annot))
        example.append(temp)
    scores = scorer.score(example)
    return scores

ner_model = spacy.load("checkpoint/spacy_ner")
results = evaluate(ner_model, TEST_DATA)
print(results)