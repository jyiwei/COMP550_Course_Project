import nltk
import spacy
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import KeyedVectors
import itertools
from utils import *

nlp = spacy.load("en_core_web_sm")

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return "_ADJ"
    elif treebank_tag.startswith('V'):
        return "_VERB"
    elif treebank_tag.startswith('N'):
        return "_NOUN"
    elif treebank_tag.startswith('R'):
        return "_ADV"
    else:
        return "_X"  

def lemmatize_with_pos(word, pos):
    lemmatizer = WordNetLemmatizer()
    lemma = lemmatizer.lemmatize(word)
    return lemma + get_wordnet_pos(pos)

def word_similarity(word1, word2, model):
    if word1 in model.key_to_index and word2 in model.key_to_index:
        return model.similarity(word1, word2)
    else:
        return 0

def get_synset_signature(synset):
    definition = word_tokenize(synset.definition())
    examples = list(itertools.chain(*[word_tokenize(example) for example in synset.examples()]))
    hypernyms = list(itertools.chain(*[word_tokenize(hypernym.definition()) for hypernym in synset.hypernyms()]))
    hyponyms = list(itertools.chain(*[word_tokenize(hyponym.definition()) for hyponym in synset.hyponyms()]))
    return definition + examples + hypernyms + hyponyms

def enhanced_lesk(word, context, pos=None, model=None):
    best_sense = None
    max_similarity = 0
    word = word.lower()

    if isinstance(context, tuple):
        context_list, target_word = context
    else:
        context_list = context.split()

    pos_tags = nltk.pos_tag(context_list)
    context_list = [lemmatize_with_pos(word, pos) for word, pos in pos_tags]

    for sense in wn.synsets(word, pos=pos):
        signature = get_synset_signature(sense)
        signature_tags = nltk.pos_tag(signature)
        signature = [lemmatize_with_pos(word, pos) for word, pos in signature_tags]

        similarity = sum(word_similarity(context_word, sig_word, model) for context_word in context_list for sig_word in signature)
        if similarity > max_similarity:
            max_similarity = similarity
            best_sense = sense

    return best_sense

def lesk_word2vec(instances, keys):
    sum_correct = 0
    count = len(instances)
    model = KeyedVectors.load_word2vec_format('C:\\Users\\Zihan\\Desktop\\Workspace\\COMP550\\model.bin', binary=True)

    for entry in instances:
        context = instances[entry].context, instances[entry].lemma
        word = instances[entry].lemma
        pos = tagConvert(instances[entry].pos)

        predicted_sense = enhanced_lesk(word, context, model=model)
        
        if predicted_sense:
            predicted_keys = [lemma.key() for lemma in predicted_sense.lemmas()]

            for key in keys[entry]:
                if key in predicted_keys:
                    sum_correct += 1
                    break

    accuracy = float(sum_correct) / len(instances)
    print(f"Word2vec lesk accuracy: {accuracy:.2f}%")
    return accuracy

def check_vocabulary_coverage(model, words):
    covered = 0
    total = len(words)
    for word in words:
        if word in model.key_to_index: 
            covered += 1

    coverage_percent = (covered / total) * 100
    return covered, total, coverage_percent

def check_c(instances, keys):
    model = KeyedVectors.load_word2vec_format('C:\\Users\\Zihan\\Desktop\\Workspace\\COMP550\\model.bin', binary=True)
    vo = []
    for entry in instances:
        word = instances[entry].context
        pos_tags = nltk.pos_tag(word)
        for j in pos_tags:
            vo.append(lemmatize_with_pos(j[0], j[1]))
    vo = list(set(vo))        
    covered, total, coverage_percent = check_vocabulary_coverage(model, vo)

    print(f"Vocabulary Coverage: {covered}/{total} words ({coverage_percent:.2f}%)")
    return coverage_percent


