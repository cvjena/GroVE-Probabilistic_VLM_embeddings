""" a script for making vocabulary pickle.

Original code:
https://github.com/yalesong/pvse/blob/master/vocab.py
"""

import nltk
nltk.download('punkt')
import pickle
from collections import Counter
import fire
import os
from tqdm import tqdm


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def from_txt(txt):
    captions = []
    with open(txt, 'rb') as f:
        for line in f:
            captions.append(line.strip())
    return captions


def build_vocab(data_path, threshold):
    """Build a simple vocabulary wrapper."""
    counter = Counter()
    captions = []
    for cname in os.listdir(data_path):
        for fname in os.listdir(os.path.join(data_path, cname)):
            full_path = os.path.join(data_path, cname, fname)
            captions.extend(from_txt(full_path))
    
    caption_len = len(captions)
    captions.clear()
    '''
    for i, caption in tqdm(enumerate(captions), total=len(captions)):
        tokens = nltk.tokenize.word_tokenize(
            caption.lower().decode('utf-8', errors='ignore'))
        counter.update(tokens)
    '''

    ###

    # Process captions in batches
    batch_size = 10000
    for i in tqdm(range(0, len(captions), batch_size)):
        batch_captions = captions[i:i + batch_size]
        for caption in batch_captions:
            tokens = nltk.tokenize.word_tokenize(caption.lower().decode('utf-8', errors='ignore'))
            counter.update(tokens)

        # Save the counter to disk periodically
        with open(f'counter_{i//batch_size}.pkl', 'wb') as f:
            pickle.dump(counter, f)
                                                            
        # Clear the counter to free memory
        counter.clear()
    
    '''
    # Load all counters and merge them
    final_counter = Counter()
    for i in range(0, len(captions), batch_size):
        with open(f'counter_{i//batch_size}.pkl', 'rb') as f:
            batch_counter = pickle.load(f)
            final_counter.update(batch_counter)
    '''
    '''
    batch_size = 10000
    def read_pickle_files():
        for i in range(0, len(captions), batch_size):
            with open(f'counter_{i//batch_size}.pkl', 'rb') as f:
                yield from pickle.load(f)

    class Vocabulary:
        def __init__(self):
            self.word2idx = {}
            self.idx2word = {}
            self.idx = 0

        def add_word(self, word):
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1 

    final_counter = Counter()

    # Read and update the final counter from individual pickle files
    for counter in read_pickle_files():
        final_counter.update(counter)

    # Discard if the occurrence of the word is less than the threshold
    words = [word for word, cnt in final_counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add words to the vocabulary
    for word in words:
        vocab.add_word(word)

    dict_vocab = {
        'idx': vocab.idx,
        'idx2word': vocab.idx2word,
        'word2idx': vocab.word2idx,
    }

    return dict_vocab
    '''
    ###

    '''
    # Discard if the occurrence of the word is less than min_word_cnt.
    words = [word for word, cnt in final_counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    dict_vocab = {
        'idx': vocab.idx,
        'idx2word': vocab.idx2word,
        'word2idx': vocab.word2idx,
    }
    return dict_vocab
    '''

def main(data_path):
    vocab = build_vocab(data_path, threshold=4)
    with open('./vocab_local.pkl', 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    fire.Fire(main)
