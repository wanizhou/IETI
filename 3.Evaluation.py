import numpy as np
from collections import defaultdict
import nltk
import re
from nltk.corpus import stopwords
import itertools
from scipy.stats import entropy
import os
from gensim.models import Word2Vec

candidate_num = 3
n_topics_num = 10  # K
window_size_num = 3
app = 'youtube'
mu, lam, theta = 0.75, 0.5, 0.1
Evaluate = 0  # Evaluate or not


# OBTM out
docs_topic = []
for file_name in range(16):
    with open('OBTM/output/k10.day%s.pw_z' % (file_name)) as fin:
        lines = fin.readlines()
        doc_topic = []
        for line in lines:
            s = line.split()
            s = np.array(s).astype(np.float64)
            doc_topic.append(s)
        doc_topic = np.array(doc_topic)
    docs_topic.append(doc_topic)

# dict
OBTM_input = np.load('data/process_data/obtm_input.npy', allow_pickle=True).item()

my_stoplst = ["app", "good", "excellent", "awesome", "please", "they", "very", "too", "like", "love", "nice",
              "yeah", "amazing", "lovely", "perfect", "much", "bad", "best", "yup", "suck", "super", "thank", "great",
              "really", "omg", "gud", "yes", "cool", "fine", "hello", "alright", "poor", "plz", "pls", "google",
              "facebook",
              "three", "ones", "one", "two", "five", "four", "old", "new", "asap", "version", "times", "update", "star",
              "first",
              "rid", "bit", "annoying", "beautiful", "dear", "master", "evernote", "per", "line", "oh", "ah",
              "cannot", "doesnt", "won't", "dont", "unless", "you're", "aren't", "i'd", "can't", "wouldn't", "around",
              "i've", "i'll", "gonna", "ago", "you'll", "you'd", "28th", "gen", "it'll", "vice", "would've", "wasn't",
              "year", "boy", "they'd",
              "isnt", "1st", "i'm", "nobody", "youtube", "isn't", "don't", "2016", "2017", "since", "near", "god"]

stoplist = stopwords.words('english') + my_stoplst

# Filter tag phrases
phrases = {}
for apk, item in OBTM_input.items():
    dic, _, _1, _2, _3 = item
    phrases[apk] = defaultdict(int)
    for i in dic.values():
        if '_' in i:
            phrase = i
            words, tags = zip(*nltk.pos_tag(phrase.split('_')))
            match = False
            for tag in tags:
                if re.match(r"^NN", tag):
                    match = True
                    continue
                if re.match(r"DT", tag):
                    match = False
                    break
                if re.match(r"RB", tag):
                    match = False
                    break

            for word in words:
                if word in stopwords.words('english') + my_stoplst:
                    match = False
                    break
                if len(word) < 3:
                    match = False
                    break
                if "\\'" in word:
                    match = False
                    break
            if match:
                phrases[apk][phrase] = 1

# Emerging topic detection

"""
Returns IDs of all relevant phrases that meet the conditions under different time slices (corresponding to APP versions)
"""


def get_candidate_label_ids(dic, labels, rawinput):
    all_label_ids = list(map(dic.token2id.get, labels))  
    label_ids = []

    for rawinput_i in rawinput:
        count = defaultdict(int)
        for input in list(itertools.chain.from_iterable(rawinput_i)):
            bow = dic.doc2bow(input)
            for id, value in bow:
                if id in all_label_ids:
                    count[id] += value
        label_ids.append(count.keys())
    return label_ids


"""
A time slice (app version)
If a word (or phrase) in the review text is the same as a candidate phrase in the candidate label (LABEL_IDS [time slice]) under the current time slice
Then print all the words (or phrases) in the comment once (usually 1).
The output format is (the candidate phrase ID belongs to, and the other word ID under this comment).
"""


def count_occurence(dic, rawinput, label_ids):
    count = []
    for d_i, rawinput_i in enumerate(rawinput):
        count_i = defaultdict(int)
        for input in list(itertools.chain.from_iterable(rawinput_i)):
            bow = dic.doc2bow(input)
            for id, value in bow:
                count_i[id] += value
                if id in label_ids[d_i]:
                    for idx, valuex in bow:
                        count_i[id, idx] += min(value, valuex)
        count.append(count_i)
    return count


"""
Output the total number of times the word (or phrase) appears in the comment under all time slices
"""


def total_count_(dic, rawinput):
    total_count = []
    for rawinput_i in rawinput:
        total_count_i = 0
        for input in list(itertools.chain.from_iterable(rawinput_i)):
            bow = dic.doc2bow(input)
            for id, value in bow:
                total_count_i += value
        total_count.append(total_count_i)
    return total_count


"""
The average comment length and average comment star value of the comments contained in each candidate phrase for 
different time segments, as calculated by a formula 
The calculation formula is: e ^ (-mean_rate / log(1 + mean_length) ) 
"""


def get_sensitivities(dic, rawinput, rates, label_ids):
    sensi = []
    for t_i, rawinput_i in enumerate(rawinput):
        sensi_t = []
        label_sensi = [[] for _ in label_ids[t_i]]
        for d_i, input in enumerate(rawinput_i):
            doc_input = list(itertools.chain.from_iterable(input))
            bow = dic.doc2bow(doc_input)

            for id, value in bow:
                if id in label_ids[t_i]:
                    label_sensi[list(label_ids[t_i]).index(id)].append([rates[t_i][d_i], len(doc_input)])

        for rl in label_sensi:
            rl = np.array(rl)
            m_rl = np.mean(rl, 0)
            sensi_t.append(np.exp(- m_rl[0] / np.log(1 + m_rl[1])))  #
        sensi.append(np.array(sensi_t))
    return sensi


"""
Different time slice
Do not distinguish a single complete comment statement, only half a sentence 
Returns the serial number of the comment text with more than 5 words in half a sentence (under different time slices, 
no single complete comment statement is distinguished, but the comment set with only half a sentence) and the star 
rating corresponding to the comment
"""


def get_candidate_sentences_ids(rawinput, rates):
    sent_ids = []
    sent_rates = []
    index = 0
    for t_i, rawinput_i in enumerate(rawinput):
        sent_id = []
        sent_rate = []
        for i_d, input_d in enumerate(rawinput_i):
            for i_s, input_s in enumerate(input_d):
                if len(input_s) < 4:
                    continue
                sent_id.append(index + i_s)
                sent_rate.append(rates[t_i][i_d])
            index += len(input_d)
        sent_ids.append(sent_id)
        sent_rates.append(sent_rate)
    return sent_ids, sent_rates


"""
Calculates  value of the candidate sentence
The calculation formula is: e ^ (-rate / log(lenth))
"""


def get_sensitivities_sent(rawinput_sent, sent_rates, sent_ids):
    sensi = []
    for t_i, sent_id in enumerate(sent_ids):
        sensi_i = []
        for id, s_id in enumerate(sent_id):
            r = sent_rates[t_i][id]
            l = len(rawinput_sent[s_id])
            sensi_i.append(np.exp(- r / float(np.log(l))))
        sensi.append(np.array(sensi_i))
    return sensi


"""
Calculate the score for the phrase related to the topic
"""


def rank_topic_label(count, total_count, phi, label_ids, mu=0.2):
    # matrix implementation for speed-up
    # construct topic matrix
    mu_div = mu / (len(phi) - 1)
    c_phi = phi * (1 + mu_div) - np.sum(phi, 0) * mu_div
    # construct label count matrix
    c_label_m = np.empty((len(label_ids), len(phi[0])), dtype=float)
    for ind, label_id in enumerate(label_ids):
        for w_id in range(len(phi[0])):
            c_label_m[ind, w_id] = count.get((label_id, w_id)) * total_count / float(
                (count.get(w_id) + 1) * (count.get(label_id) + 1)) if (label_id, w_id) in count else 1.0
    c_label_m = np.log(c_label_m)
    # compute score matrix
    topic_label_scores = np.dot(c_phi, np.transpose(c_label_m))
    return topic_label_scores


def topic_labeling_(count, total_count, label_ids, sensi, phi, mu, lam):
    topic_label_scores = rank_topic_label(count, total_count, phi, label_ids, mu)
    topic_label_scores += lam * sensi
    return topic_label_scores


"""
Calculate scores for sentences related to the topic
"""


def topic_label_sent(dic, phi, rawinput_sents, sent_ids, sensi, mu, lam):
    # construct topic matrix
    mu_div = mu / (len(phi) - 1)
    c_phi = phi * (1 + mu_div) - np.sum(phi, 0) * mu_div
    # construct residual
    phi_logphi = phi * np.log(phi)
    residual_1 = mu_div * np.sum(phi_logphi)  # residual_1 is a value
    residual_2 = (1 + mu_div) * np.sum(phi_logphi, 1, keepdims=True)  # residual_2 is a n_topic*1
    # construct sentence count matrix
    sent_count = np.empty((len(sent_ids), len(phi[0])), dtype=float)
    for ind, s_id in enumerate(sent_ids):
        bow = dic.doc2bow(rawinput_sents[s_id])
        len_s = len(rawinput_sents[s_id])
        for w_id in range(len(phi[0])):
            sent_count[ind, w_id] = 0.00001
        for k, v in bow:
            sent_count[ind, k] = v / float(len_s)

    phi_sent = np.dot(c_phi, np.transpose(np.log(sent_count))) + residual_1 - residual_2 + lam * sensi
    return phi_sent


def count_width(dictionary, label_phrases_ver, counts, sensi_labels, label_ids):
    count_width_rst = []
    for phrases in label_phrases_ver:
        t_count = 0
        for phrase in phrases:
            pid = dictionary.token2id.get(phrase)
            t_count += np.log(counts.get(pid) + 1) * sensi_labels[list(label_ids).index(pid)]
        count_width_rst.append(t_count)
    return np.array(count_width_rst)


"""
scipy.stats.entropy(pk, qk=None, base=None, axis=0)
Calculates the entropy of the distribution at a given probability value.
If qk is not None, then the divergence of KL is calculated
"""


def JSD(P, Q):
    """
    Jensen-Shannon divergence
    :param P:
    :param Q:
    :return:
    """
    _M = 0.5 * (P + Q)
    return 0.5 * (entropy(P, _M) + entropy(Q, _M))


def topic_detect(rawinput_sents, dic, phi, last_phi, count, last_count, total_count, last_total_count, label_ids,
                 sent_ids, sensi_label, sensi_sent, jsds, theta, mu, lam):
    # matrix implementation for speed-up

    # construct count label matrix
    c_label_m = np.empty((len(label_ids), len(phi[0])), dtype=float)
    c_last_label_m = np.empty((len(label_ids), len(phi[0])), dtype=float)

    for ind, label_id in enumerate(label_ids):
        for w_id in range(len(phi[0])):
            c_label_m[ind, w_id] = count.get((label_id, w_id)) * total_count / float(
                (count.get(w_id) + 1) * (count.get(w_id) + 1)) if (label_id, w_id) in count else 1.0
            c_last_label_m[ind, w_id] = last_count.get((label_id, w_id)) * last_total_count / float(
                (last_count.get(w_id) + 1) * (last_count.get(label_id) + 1)) if (label_id, w_id) in last_count else 1.0

    c_label_m = np.log(c_label_m)
    c_last_label_m = np.log(c_last_label_m)

    # construct sentence count matrix
    sent_count = np.empty((len(sent_ids), len(phi[0])), dtype=float)
    for ind, s_id in enumerate(sent_ids):
        bow = dic.doc2bow(rawinput_sents[s_id])
        len_s = len(rawinput_sents[s_id])
        for w_id in range(len(phi[0])):
            sent_count[ind, w_id] = 0.00001
        for k, v in bow:
            sent_count[ind, k] = v / float(len_s)

    # read topic distribution \phi
    emerging_label_scores_rst = np.zeros((len(phi), len(label_ids)))
    emerging_sent_scores_rst = np.zeros((len(phi), len(sent_ids)))

    js_d = []
    for t_i, phi_i in enumerate(phi):
        # labeling
        js_divergence = JSD(phi_i, last_phi[t_i])
        js_d.append(js_divergence)
        jsds.append(js_divergence)

    js_mean = np.mean(jsds[:-3 * len(phi) - 1:-1])
    js_std = np.std(jsds[:-3 * len(phi) - 1:-1])

    """
    Classical outlier detection method
    It is assumed that the divergence obeys the Gaussian distribution, and the mean and variance are js_mean and js_std respectively
    The calculation method is as follows:
    1.Firstly, the JS divergence of all the topic probability vectors under a certain time slice is calculated to obtain the JS divergence matrix DJS
    2.Calculate the mean and variance of the DJS
    3.Set the threshold = js_mean + 1.25js_std, where 1.25 represents 10% acceptance of the exception topic
    """
    emerging_index = np.array(js_d) > js_mean + 1.25 * js_std

    # Marking exception topics
    phi_e = phi[emerging_index]
    phi_last_e = last_phi[emerging_index]
    E = float(np.sum(emerging_index))

    if E == 0:
        return emerging_label_scores_rst, emerging_sent_scores_rst

    # TOPIC DETECT: construct phi - last_phi
    phi_m = (1 + mu / E) * phi_e - theta * last_phi[emerging_index] - mu / E * np.sum(phi_e, 0)
    # TOPIC DETECT: construct residuals
    residuals_m = (1 + mu / E) * np.log(phi_e) * phi_e - theta * np.log(phi_last_e) * phi_last_e - mu / E * np.sum(
        np.log(phi_e) * phi_e, 0)
    # TOPIC DETECT: compute labels
    emerging_label_scores = np.dot((1 + mu / E) * phi_e - mu / E * np.sum(phi_e, 0),
                                   np.transpose(c_label_m)) - theta * np.dot(last_phi[emerging_index], np.transpose(
        c_last_label_m)) + lam * sensi_label
    emerging_sent_scores = np.dot(phi_m, np.transpose(np.log(sent_count))) - np.sum(residuals_m, 1,
                                                                                    keepdims=True) + lam * sensi_sent

    emerging_label_scores_rst[emerging_index] = emerging_label_scores
    emerging_sent_scores_rst[emerging_index] = emerging_sent_scores
    return emerging_label_scores_rst, emerging_sent_scores_rst


apk_jsds = {}

for apk, item in OBTM_input.items():
    dictionary, _, rawinput, rates, tag = item
    phis = docs_topic
    labels = phrases[apk].keys()

    # The candidate phrases
    label_ids = get_candidate_label_ids(dictionary, labels, rawinput)
    count = count_occurence(dictionary, rawinput, label_ids)
    total_count = total_count_(dictionary, rawinput)
    sensi_label = get_sensitivities(dictionary, rawinput, rates, label_ids)

    rawinput_sent = list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(rawinput))))

    # The candidate sentences
    sent_ids, sent_rates = get_candidate_sentences_ids(rawinput, rates)
    sensi_sent = get_sensitivities_sent(rawinput_sent, sent_rates, sent_ids)

    jsds = []
    label_phrases = [];
    label_sents = [];
    emerge_phrases = [];
    emerge_sents = []

    result_path = "result/%s" % apk
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    fout_labels = open(os.path.join(result_path, "topic_labels.txt"), 'w')
    fout_sents = open(os.path.join(result_path, "topic_sents.txt"), "w")

    fout_emerging = open(os.path.join(result_path, "emerging_topic_labels.txt"), 'w')
    fout_emerging_sent = open(os.path.join(result_path, "emerging_topic_sents.txt"), 'w')
    fout_topic_width = open(os.path.join(result_path, "topic_width.txt"), 'w')

    for t_i, phi in enumerate(phis):
        print("labeling topic at %s slice of %s" % (t_i, apk))

        topic_label_scores = topic_labeling_(count[t_i], total_count[t_i], label_ids[t_i], sensi_label[t_i], phi, mu,
                                             lam)
        topic_label_sent_score = topic_label_sent(dictionary, phi, rawinput_sent, sent_ids[t_i], sensi_sent[t_i], mu,
                                                  lam)

        fout_labels.write('\n')
        fout_labels.write("time slice %s, tag: %s\n" % (t_i, tag[t_i]))
        for tp_i, label_scores in enumerate(topic_label_scores):
            fout_labels.write("Topic %d:" % tp_i)
            for w_id in np.argsort(label_scores)[:-candidate_num - 1:-1]:
                fout_labels.write("%s\t%f\t" % (dictionary[list(label_ids[t_i])[w_id]], label_scores[w_id]))
            fout_labels.write('\n')

        fout_sents.write('\n')
        fout_sents.write("time slice %s, tag: %s\n" % (t_i, tag[t_i]))
        for tp_i, sent_scores in enumerate(topic_label_sent_score):
            fout_sents.write("Topic %d:" % tp_i)
            for s_id in np.argsort(sent_scores)[:-candidate_num - 1:-1]:
                fout_sents.write("%s\t%f\t" % (" ".join(rawinput_sent[sent_ids[t_i][s_id]]), sent_scores[s_id]))
            fout_sents.write('\n')

        # store for verification
        label_phrases_ver = []
        label_sents_ver = []
        for tp_i, label_scores in enumerate(topic_label_scores):
            label_phrases_ver.append(
                [dictionary[list(label_ids[t_i])[w_id]] for w_id in np.argsort(label_scores)[:-candidate_num - 1:-1]])
        label_phrases.append(list(itertools.chain.from_iterable(label_phrases_ver)))

        for tp_i, sent_scores in enumerate(topic_label_sent_score):
            label_sents_ver.append(
                [rawinput_sent[sent_ids[t_i][s_id]] for s_id in np.argsort(sent_scores)[:-candidate_num - 1:-1]])
        label_sents.append(list(itertools.chain.from_iterable(label_sents_ver)))

        if t_i == 0:
            topic_width = count_width(dictionary, label_phrases_ver, count[t_i], sensi_label[t_i], label_ids[t_i])
            for theta in topic_width:
                fout_topic_width.write("%f\t" % theta)
            fout_topic_width.write("\n")
            continue

        # Detects the exception topic and returns the related phrases and sentences marked by the exception topic tag
        emerging_label_scores, emerging_sent_scores = topic_detect(rawinput_sent,  # All reviews, regardless of time slice, whole sentence, divided into half sentence
                                                                   dictionary,  # review on the dictionary
                                                                   phi,  # Probability vectors of all topics under a time slice
                                                                   phis[t_i - 1],  # Probability vectors of all topics under the last time slice
                                                                   count[t_i],  # The frequency of all reviews at a given time
                                                                   count[t_i - 1],
                                                                   total_count[t_i],  # The total frequency of all reviews at a given time
                                                                   total_count[t_i - 1],
                                                                   label_ids[t_i],  # The id of the relevant phrase that matches the condition at a given time
                                                                   sent_ids[t_i],  # The id of the sentence that matches the condition at a given time
                                                                   sensi_label[t_i],  # The id of the sentence that matches the condition at a given time
                                                                   sensi_sent[t_i],  # The score of the relevant sentences at a given time
                                                                   jsds, theta, mu, lam)

        fout_emerging.write('\n')
        fout_emerging.write("time slice %s, tag: %s\n" % (t_i, tag[t_i]))
        for tp_i, label_scores in enumerate(emerging_label_scores):
            fout_emerging.write("Topic %d: " % tp_i)
            if np.sum(label_scores) == 0:
                fout_emerging.write('None\n')
            else:
                for w_id in np.argsort(label_scores)[:-4:-1]:
                    fout_emerging.write("%s\t%f\t" % (dictionary[list(label_ids[t_i])[w_id]], label_scores[w_id]))
                fout_emerging.write('\n')

        fout_emerging_sent.write('\n')
        fout_emerging_sent.write("time slice %s, tag: %s\n" % (t_i, tag[t_i]))
        for tp_i, sent_scores in enumerate(emerging_sent_scores):
            fout_emerging_sent.write("Topic %d: " % tp_i)
            if np.sum(sent_scores) == 0:
                fout_emerging_sent.write('None\n')
            else:
                for s_id in np.argsort(sent_scores)[:-4:-1]:
                    fout_emerging_sent.write(
                        "%s\t%f\t" % (" ".join(rawinput_sent[sent_ids[t_i][s_id]]), sent_scores[s_id]))
                fout_emerging_sent.write('\n')

        # store for verification
        emerge_phrases_ver = []
        emerge_sents_ver = []
        emerge_phrases_width_ver = []

        for tp_i, label_scores in enumerate(emerging_label_scores):
            if np.sum(label_scores) == 0:
                emerge_phrases_width_ver.append([])
                continue
            emerge_phrases_ver.append(
                [dictionary[list(label_ids[t_i])[w_id]] for w_id in np.argsort(label_scores)[:-4:-1]])
            emerge_phrases_width_ver.append(
                [dictionary[list(label_ids[t_i])[w_id]] for w_id in np.argsort(label_scores)[:-4:-1]])
        emerge_phrases.append(emerge_phrases_ver)

        # merge emerge to label
        label_emerge_ver = [set(l) | set(e) for l, e in zip(label_phrases_ver, emerge_phrases_width_ver)]
        #         topic_width = count_width(dictionary, label_emerge_ver, count[t_i], sensi_label[t_i], label_ids[t_i])
        for tp_i, sent_scores in enumerate(emerging_sent_scores):
            if np.sum(sent_scores) == 0:
                continue
            emerge_sents_ver.append(
                [rawinput_sent[sent_ids[t_i][s_id]] for s_id in np.argsort(sent_scores)[:-4:-1]])
        emerge_sents.append(emerge_sents_ver)

    fout_labels.close()
    fout_sents.close()
    fout_emerging.close()
    fout_emerging_sent.close()
    fout_topic_width.close()

print('**********************************************************')
print('topic_nums:{},mu:{},lam:{},theta:{}'.format(n_topics_num, mu, lam, theta))

wv_model = Word2Vec.load('model/wv/word2vec_app.model')

def sim_w(w1, w2, wv_model):
    if w1 not in wv_model or w2 not in wv_model:
        return 0.0
    return wv_model.similarity(w1, w2)

sim = 0.5
label_phrase_precisions = [];
label_phrase_recalls = []
label_sent_precisions = [];
label_sent_recalls = []

em_phrase_precisions = [];
em_phrase_recalls = []
em_sent_precisions = [];
em_sent_recalls = []

if Evaluate:
    clog = []
    with open('data/ios/youtube/changelog_radar.txt') as fin:
        for line in fin.readlines():
            line = line.strip()
            issue_kw = list(map(lambda s: s.strip().split(), line.split(",")))
            clog.append(issue_kw)

    for id, ver in enumerate(clog):
        if ver == [[]]:  # skip the empty version changelog
            continue

        label_phrase_match_set = set();
        label_phrase_issue_match_set = set()
        label_sent_match_set = set();
        label_sent_issue_match_set = set()

        em_phrase_match_set = set();
        em_phrase_issue_match_set = set()
        em_sent_match_set = set();
        em_sent_issue_match_set = set()

        # merge changelog with next version
        if id != len(clog) - 1 and clog[id + 1] != [[]]:
            m_ver = ver + clog[id + 1]
        else:
            m_ver = ver

        # phrase
        for issue in m_ver:
            for kw in issue:
                kw_match = False
                for w in label_phrases[id]:
                    label_match = False
                    for w_s in w.split("_"):
                        if sim_w(kw, w_s, wv_model) > sim:
                            label_match = True
                            kw_match = True
                            break
                    if label_match:  # if label match found, add label to match set
                        label_phrase_match_set.add(w)
                if kw_match:  # if kw match found, add issue to match set
                    label_phrase_issue_match_set.add("_".join(issue))

        # sentence
        for issue in m_ver:
            for kw in issue:
                kw_match = False
                for sent in label_sents[id]:
                    for w in sent:
                        label_match = False
                        for w_s in w.split("_"):
                            if sim_w(kw, w_s, wv_model) > sim:

                                label_match = True
                                kw_match = True
                                break
                        if label_match:
                            label_sent_match_set.add("_".join(sent))  # if label match found, skip to next sentence
                            break
                if kw_match:
                    label_sent_issue_match_set.add("_".join(issue))

        if id != 0:  # skip the first epoch
            for issue in m_ver:
                for kw in issue:
                    kw_match = False
                    for tws in emerge_phrases[id - 1]:
                        for w in tws:
                            label_match = False
                            for w_s in w.split("_"):
                                if sim_w(kw, w_s, wv_model) > sim:

                                    label_match = True
                                    kw_match = True
                                    break
                            if label_match:
                                em_phrase_match_set.add("_".join(tws))
                                break
                    if kw_match:
                        em_phrase_issue_match_set.add("_".join(issue))

                        # sentence
            for issue in m_ver:
                for kw in issue:
                    kw_match = False
                    for tsents in emerge_sents[id - 1]:
                        sent = list(itertools.chain.from_iterable(tsents))
                        label_match = False
                        for w in sent:
                            for w_s in w.split("_"):
                                if sim_w(kw, w_s, wv_model) > sim:
                                    # hit
                                    # logging.info("hit: %s -> %s" % (w, kw))
                                    label_match = True
                                    kw_match = True
                                    break
                            if label_match:
                                em_sent_match_set.add("_".join(sent))  # if label match found, skip to next sentence
                                break
                    if kw_match:
                        em_sent_issue_match_set.add("_".join(issue))

                        # compute
        label_phrase_precision = len(label_phrase_match_set) / float(len(label_phrases[id]))
        label_phrase_recall = len(label_phrase_issue_match_set) / float(len(m_ver))

        label_sent_precision = len(label_sent_match_set) / float(len(label_sents[id]))
        label_sent_recall = len(label_sent_issue_match_set) / float(len(m_ver))

        label_phrase_precisions.append(label_phrase_precision)
        label_phrase_recalls.append(label_phrase_recall)

        label_sent_precisions.append(label_sent_precision)
        label_sent_recalls.append(label_sent_recall)

        if id != 0:
            if len(emerge_phrases[id - 1]) != 0:
                em_phrase_precision = len(em_phrase_match_set) / float(len(emerge_phrases[id - 1]))
                em_phrase_precisions.append(em_phrase_precision)
            em_phrase_recall = len(em_phrase_issue_match_set) / float(len(ver))

            if len(emerge_sents[id - 1]) != 0:
                em_sent_precision = len(em_sent_match_set) / float(len(emerge_sents[id - 1]))
                em_sent_precisions.append(em_sent_precision)
            em_sent_recall = len(em_sent_issue_match_set) / float(len(ver))

            em_phrase_recalls.append(em_phrase_recall)
            em_sent_recalls.append(em_sent_recall)

    label_phrase_fscore = 2 * np.mean(label_phrase_recalls) * np.mean(em_phrase_precisions) / (
            np.mean(label_phrase_recalls) + np.mean(em_phrase_precisions))
    label_sent_fscore = 2 * np.mean(label_sent_recalls) * np.mean(em_sent_precisions) / (
            np.mean(label_sent_recalls) + np.mean(em_sent_precisions))

# print("Phrase label precision: %s\t recall: %f" % (np.mean(label_phrase_precisions), np.mean(label_phrase_recalls)))
# print("Sentence label precision: %s\t recall: %f" % (np.mean(label_sent_precisions), np.mean(label_sent_recalls)))
print("phrase precision: %s\t recall: %f" % (np.mean(em_phrase_precisions), np.mean(em_phrase_recalls)))
print("sentence precision: %s\t recall: %f" % (np.mean(em_sent_precisions), np.mean(em_sent_recalls)))
print("phrase F1 score: %f" % label_phrase_fscore)
print("sentence F1 score: %f" % label_sent_fscore)
