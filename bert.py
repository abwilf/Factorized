from transformers import pipeline, AutoTokenizer
import sys; sys.path.append('/work/awilf/utils/'); from alex_utils import *

def get_bert_features(words, model_name='bert-base-uncased', remove_pooled=True):
    ''' words is a string (e.g. 'hi alex').  not a list of lists'''
    if not hasattr(get_bert_features, 'feature_extraction'): # only load it once for however many function calls
        get_bert_features.feature_extraction = pipeline('feature-extraction', model=model_name, tokenizer=model_name, device=0)
        print('loading model')
    
    features = get_bert_features.feature_extraction(words)
    features = ar(features)
    if remove_pooled:
        features = features[0,1:-1,:]
    return features

if __name__ == '__main__':
    words = 'how does the crowd of geese feel?'
    b = get_bert_features(words)
    a = 1