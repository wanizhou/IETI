import pycorrector
import unicodedata
import sys
import logging
from gensim.models.phrases import Phrases
import numpy as np
from nltk.corpus import stopwords
import itertools
from gensim import corpora
from scipy import sparse
import re

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)

from nltk.stem.wordnet import WordNetLemmatizer
import importlib

importlib.reload(sys)

unicode_punc_tbl = dict.fromkeys(i for i in range(128, sys.maxunicode)
                                 if unicodedata.category(chr(i)).startswith('P'))


# Data cleaning
def extractSentenceWords(doc, remove_url=True, remove_punc="utf-8", min_length=1, lemma=False, sent=True,
                         replace_digit=False, repeat=False):
    if remove_punc:
        # Unified coding
        if not isinstance(doc, str):
            encoding = remove_punc
            doc_u = doc.decode(encoding)
        else:
            doc_u = doc
        # remove unicode punctuation marks, keep ascii punctuation marks
        doc_u = doc_u.translate(unicode_punc_tbl)
        if not isinstance(doc, str):
            doc = doc_u.encode(encoding)
        else:
            doc = doc_u

    if remove_url:
        re_url = r"(https?:\/\/)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)"
        doc = re.sub(re_url, "", doc)

    sentences = re.split(r"\s*[;:`\"()?!{}]\s*|--+|\s*-\s+|''|\.\s|\.$|\.\.+|锟斤拷|锟斤拷", doc)  

    wc = 0
    wordsInSentences = []
    wnl = WordNetLemmatizer()

    for sentence in sentences:
        if sentence == "":
            continue

        if not re.search("[A-Za-z0-9]", sentence):
            continue

        words = re.split(r"[\s+,\-*\/&%=_<>\[\]~\|\@\$\\]", sentence)
        words = filter(lambda w: w, words)
        words = list(map(lambda w: w.lower(), words))
        if replace_digit:
            words = list(map(lambda w: re.sub(r'\d+', '<digit>', w), words))
        if lemma:
            words = list(map(lambda w: wnl.lemmatize(w, 'v'), words))
        if repeat:
            _words = list(set(words))
            _words.sort(key=words.index)
            words = _words
        if len(words) >= min_length:
            wordsInSentences.append(words)
            wc += len(words)

    if not sent:
        return list(itertools.chain.from_iterable(wordsInSentences)), wc
    return wordsInSentences, wc


def build_input(app_files):
    doc_sent_word = []
    num_words = 0
    num_docs = 0
    l_id = 0
    with open(app_files) as fin:
        for line in fin.readlines():
            line = line.strip()
            line = line.split("******")
            words_sents, wc = extractSentenceWords(line[1], lemma=True)
            doc_sent_word.append(words_sents)
            num_docs += 1
            num_words += wc
            if l_id % 1000 == 0:
                logging.info("processed %d docs of %s" % (l_id, app))
            l_id += 1
    logging.info("Read %d docs, %d words!" % (num_docs, num_words))
    return doc_sent_word


def extract_phrases(app_files, bigram_min, trigram_min):
    rst = build_input(app_files)
    gen = list(itertools.chain.from_iterable(rst))  # 列表平滑处理

    bigram = Phrases(gen, threshold=6, min_count=bigram_min)
    trigram = Phrases(bigram[gen], threshold=4, min_count=trigram_min)

    bigram.save('model/%s_bigram_model.pkl' % (app))
    trigram.save('model/%s_trigram_model.pkl' % (app))


def load_phrase():
    global bigram
    global trigram
    bigram = Phrases.load('model/%s_bigram_model.pkl' % (app))
    trigram = Phrases.load('model/%s_trigram_model.pkl' % (app))


def build_phrase(doc):
    return trigram[bigram[doc]]


def replace_digit(sent):
    for w in sent:
        if w.isdigit():
            yield '<digit>'
        else:
            yield w


def extract_review():
    timed_reviews = {}
    num_docs = 0
    num_words = 0
    timed_reviews[app] = []

    with open(app_files) as fin:
        lines = fin.readlines()
    for l_id, line in enumerate(lines):
        line = line.strip()
        terms = line.split("******")
        if len(terms) != 6:
            logging.error("review format error at %s in %s" % (app, line))
            continue
        if not StoreNum:  # for ios
            date = terms[3]
            version = terms[4]
        else:  # for android
            date = terms[2]
            version = terms[3]

        review_o = terms[1]
        review_p, wc = extractSentenceWords(review_o, repeat=True)

        for list_text in review_p:
            for index, value in enumerate(list_text):
                list_text[index] = pycorrector.en_correct(value)
        review = list(build_phrase(review_p))
        review = [list(replace_digit(s)) for s in review]
        rate = float(terms[0]) if re.match(r'\d*\.?\d+', terms[0]) else 2.0  
        timed_reviews[app].append({"review": review, "date": date, "rate": rate, "version": version})
        num_docs += 1
        num_words += wc
        if l_id % 1000 == 0:
            logging.info("processed %d docs of %s" % (l_id, app))
    logging.info("total read %d reviews, %d words." % (num_docs, num_words))
    return timed_reviews


def obtm_input():
    for apk, reviews in timed_reviews.items():
        # build a dictionary to store the version and review
        version_dict = {}
        input = []
        rate = []
        tag = []

        for review in reviews:
            review_version = review['version']
            if review_version == "Unknown":
                continue
            if review_version not in version_dict:
                version_dict[review_version] = ([], [])
            version_dict[review_version][0].append(review['review'])
            version_dict[review_version][1].append(review['rate'])

        for ver in sorted(version_dict.keys(), key=lambda s: list(map(int, s.split('.')))):
            if len(version_dict[ver][0]) > 50:  # skip versions with not enough reviews
                tag.append(ver)
                input.append(version_dict[ver][0])
                rate.append(version_dict[ver][1])
        dict_input = list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(input))))
        dictionary = corpora.Dictionary(dict_input)  
        dictionary.filter_tokens(map(dictionary.token2id.get, stoplist))  
        dictionary.compactify()  
        dictionary.filter_extremes(no_below=2, keep_n=None)  
        dictionary.compactify()  

        input_X = []
        for t_i, text_period in enumerate(input):
            # construct sparse matrix  
            text_period = list(itertools.chain.from_iterable(text_period))
            row = []
            col = []
            value = []
            r_id = 0

            for k, text in enumerate(text_period):
                empty = True
                for i, j in dictionary.doc2bow(text):
                    row.append(r_id)
                    col.append(i)
                    value.append(j)
                    empty = False
                if not empty:
                    r_id = r_id + 1
            input_X.append(sparse.coo_matrix((value, (row, col)), shape=(r_id, len(dictionary))))
        OLDA_input[apk] = (dictionary, input_X, input, rate, tag)

    voca = dictionary.token2id
    file = open('OBTM/input/voca.txt', 'a', encoding='utf-8')
    for index, value in enumerate(voca):
        s = '%d\t%s\r' % (index, value)
        file.write(s)
    file.close()
    print('len of dict:', len(voca))

    for ver, text in enumerate(input):
        file1 = open('OBTM/input/texts/%s.txt' % (ver), 'a', encoding='utf-8')
        file2 = open('OBTM/input/doc_wids/%s.txt' % (ver), 'a', encoding='utf-8')
        for j in text:
            for i in range(len(j)):
                review_half = j[i]
                s_2id = dictionary.doc2idx(j[i])

                index = [i for i, x in enumerate(s_2id) if x != -1]

                new_review_half = list(np.array(review_half)[index])
                new_s_2id = list(np.array(s_2id)[index])

                s1 = str(new_review_half).replace('[', '').replace(']', '')
                s1 = s1.replace("'", '').replace(',', '') + '\r'

                s2 = str(new_s_2id).replace('[', '').replace(']', '')
                s2 = s2.replace("'", '').replace(',', '') + '\r'

                file1.write(s1)
                file2.write(s2)

        file1.close()
        file2.close()
    return OLDA_input


if __name__ == '__main__':
    OLDA_input = {}

    # parameter
    app_files = 'data/ios/youtube/total_info.txt'
    app = 'youtube'
    bigram_min = 6
    trigram_min = 3
    StoreNum = 0  # 0 for ios, 1 for android
    my_stoplst = ["app", "good", "excellent", "awesome", "please", "they", "very", "too", "like", "love", "nice",
                  "yeah", "amazing", "lovely", "perfect", "much", "bad", "best", "yup", "suck", "super", "thank",
                  "great",
                  "really", "omg", "gud", "yes", "cool", "fine", "hello", "alright", "poor", "plz", "pls", "google",
                  "facebook",
                  "three", "ones", "one", "two", "five", "four", "old", "new", "asap", "version", "times", "update",
                  "star",
                  "first",
                  "rid", "bit", "annoying", "beautiful", "dear", "master", "evernote", "per", "line", "oh", "ah",
                  "cannot", "doesnt", "won't", "dont", "unless", "you're", "aren't", "i'd", "can't", "wouldn't",
                  "around",
                  "i've", "i'll", "gonna", "ago", "you'll", "you'd", "28th", "gen", "it'll", "vice", "would've",
                  "wasn't",
                  "year", "boy", "they'd",
                  "isnt", "1st", "i'm", "nobody", "youtube", "isn't", "don't", "2016", "2017", "since", "near", "god"]

    stoplist = stopwords.words('english') + my_stoplst

    extract_phrases(app_files, bigram_min, trigram_min)
    load_phrase()
    timed_reviews = extract_review()
    np.save('data/process_data/process.npy', timed_reviews)
    timed_reviews = np.load('data/process_data/process.npy', allow_pickle=True).item()
    OBTM_input = obtm_input()
    np.save('data/process_data/obtm_input.npy', OBTM_input)
