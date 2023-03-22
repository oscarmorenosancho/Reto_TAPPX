import json
import re
import math
import statistics
import nltk
import spacy
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stemmer = SnowballStemmer('spanish')
nlp = spacy.load("es_dep_news_trf")


def remove_punt(s):
    if (s and len(s)>0):
        ret = s.replace('.', ' ').replace(':', ' ').replace(',', ' ')
        ret = ret.replace('"', ' ').replace("'", ' ').replace('-', ' ').replace('|', ' ')
        ret = ret.replace(')', ' ').replace('(', ' ').replace('}', ' ').replace('{', ' ')
        ret = ret.replace(']', ' ').replace('[', ' ').replace('\\', ' ').replace('/', ' ')
        ret = re.sub('\d+', '', ret)
        ret = ret.lower()
    return ret

to_lower = lambda s: s.lower()

def keyword_to_root(s):
    return stemmer.stem(s)

not_discard = lambda s: len(s) > 3

# def tokenize_text(text, stop_words):
#     words = word_tokenize(text.lower())
#     tokens = []
#     for word in words:
#         if word not in stop_words:
#             tokens.append(word)
#     return tokens
def tokenize_text(text):
    text = remove_punt(text)
    doc = nlp(text)
    lst = []
    for token in doc:
        lst.append([token.text, token.pos_])
    # avoid = ['DET', 'ADP', 'NUM', 'PUNCT', 'SYM', 'CCONJ', 'SCONJ', 'AUX', 'PRON', 'SPACE', 'ADV', 'VERB']
    avoid = ['DET', 'ADP', 'NUM', 'PUNCT', 'SYM', 'CCONJ', 'SCONJ', 'AUX', 'PRON', 'SPACE', 'ADV']
    lst = list(filter(lambda x: not x[1] in avoid , lst))
    lst = list(map(lambda x: x[0], lst))
    lst = list(filter(lambda x: len(x)>3, lst))

    return lst

# Terms extraction from document, from text title and keywords
def doc_terms_extract(item):
    # stop_words = set(stopwords.words('spanish'))
    # terms = tokenize_text(text, stop_words)
    # Tokenize the text
    text = item['text']
    terms = tokenize_text(text)
    # roots = []
    # terms = list(filter (not_discard, terms))
    # for term in terms:
    #     roots.append(stemmer.stem(term))
    # terms = roots

    keywords = list(filter(not_discard, item['keywords']))
    atom_keywords = [*keywords]
    to_split = [*keywords]
    for keyword in to_split:
        atom_keywords += tokenize_text(keyword)
    terms = [*terms, *atom_keywords]
    if 'title' in item:
        title = tokenize_text(item['title'])
        terms = [*terms, *title, *title]
    return terms
  
# Coptutes the occurrences of terms in a document
def compute_str_ocur(item):
    terms = doc_terms_extract (item)
    terms_set = list(set(terms))
    terms_ocur = {}
    total = 0
    for term in terms_set:
        ocurs = len (list(filter(lambda x: x == term, terms)))
        terms_ocur[term] = ocurs
        total += ocurs
    return ({'terms_ocur': terms_ocur, 'total': total})

# Update ocurrences in Totals of Corpus
def update_total_str_ocur(totals, item):
    total_keys = totals.keys()
    if not 'terms_ocur' in total_keys:
        totals['terms_ocur'] = {}
    if not 'docs_rep' in total_keys:
        totals['docs_rep'] = {}
    if not 'total' in total_keys:
        totals['total'] = 0
    if not 'docs_cnt' in total_keys:
        totals['docs_cnt'] = 1
    else:
        totals['docs_cnt'] += 1
    total_terms_ocur_keys = list(totals['terms_ocur'].keys())
    total_docs_rep_keys = list(totals['docs_rep'].keys())
    item_keys = list(item['tf-idf']['terms_ocur'].keys())
    for key in item_keys:
        if not key in total_terms_ocur_keys:
            totals['terms_ocur'][key] = 0
        if not key in total_docs_rep_keys:
            totals['docs_rep'][key] = 0
        amount_in_key = item['tf-idf']['terms_ocur'][key]
        totals['terms_ocur'][key] += amount_in_key
        totals['docs_rep'][key] += 1
        totals['total'] += amount_in_key
    return totals

# Computes IDF of term appeard in documents corpus
def compute_idf(totals):
    total_keys = totals.keys()
    check_c = 'terms_ocur' in totals
    check_c = check_c and 'docs_rep' in totals
    check_c = check_c and 'total' in totals
    check_c = check_c and 'docs_cnt' in totals
    if check_c:
        if not 'terms_idf' in total_keys:
            totals['terms_idf'] = {}
        for key in totals['docs_rep'].keys():
            totals['terms_idf'][key] = math.log10(totals['docs_cnt'] / totals['docs_rep'][key])
    return totals

# Computes the measure of TF-IDF
def compute_tfidf(totals,item):
    total_keys = totals.keys()
    item_keys = item.keys()
    ckeck_c = 'terms_idf' in total_keys
    ckeck_c = ckeck_c and 'tf-idf' in item_keys
    if ckeck_c:
        if not 'terms_tfidf' in item['tf-idf']:
            item['tf-idf']['terms_tfidf'] = {}
        tot_terms = item['tf-idf']['total']
        it_w_tfidf = item['tf-idf']['terms_tfidf']
        for key in item['tf-idf']['terms_ocur']:
            key_ocur = item['tf-idf']['terms_ocur'][key]
            key_tf = key_ocur / tot_terms
            it_w_tfidf[key] = key_tf * totals['terms_idf'][key]
        sorted_dict = sorted(it_w_tfidf.items(), key=lambda x:x[1],reverse=True)
        # sorted_dict = sorted_dict[:70]
        item['tf-idf']['terms_tfidf'] = dict(sorted_dict)
        del item['tf-idf']['terms_ocur']
    return

# Computes the similarity beteen documents
def compute_projection(doc1, doc2):
    doc1_terms = doc1['tf-idf']['terms_tfidf']
    doc2_terms = doc2['tf-idf']['terms_tfidf']
    cross_prod = {}
    acum = 0
    for doc_term in doc1_terms:
        if doc_term in doc2_terms:
            cross_prod[doc_term] = doc1_terms[doc_term] * doc2_terms[doc_term]
            acum += cross_prod[doc_term]
    return {"cross_prod": cross_prod, "acum": acum}

# This converst list of result to final result
def transform_in_dict(list_cross):
    res = dict()
    for pairing in list_cross:
        if not pairing[0] in res:
            res[pairing[0]] = dict()
            res[pairing[0]]['len'] = 0 
        res[pairing[0]][pairing[1]] = {'matchs': pairing[2], 'score': pairing[3]}
        res[pairing[0]]['len'] += 1
    return res

# Main
with open('articles.json') as user_file:
    articles = json.load(user_file)
with open('videos.json') as user_file:
    videos = json.load(user_file)

for article_id in articles:
    article = articles[article_id]
    print (article_id)
    article['tf-idf'] = compute_str_ocur(article)

for video_id in videos:
    video = videos[video_id]
    print (video_id)
    video['tf-idf'] = compute_str_ocur(video)


totals = { }
for item in articles:
    totals = update_total_str_ocur(totals, articles[item])

for item in videos:
    totals = update_total_str_ocur(totals, videos[item])

compute_idf (totals)
# articles.update(videos)
for article_id in articles:
    article = articles[article_id]
    compute_tfidf(totals,article)

for video_id in videos:
    video = videos[video_id]
    compute_tfidf(totals,video)

cross_project = []
for article_id in articles:
    for video_id in videos:
        proj = compute_projection(articles[article_id], videos[video_id])
        if (proj["acum"] > 0):
            cross_project.append([article_id, video_id, proj['cross_prod'], proj['acum']])


# Function to decide popsition to filter
def score_less_than_mean(x):
    return  x[3] >= 0.2 * score_mean

# Function to decide popsition to filter in article
def score_less_than_mean_art(x):
    return  x[3] >= score_mean + score_stdev


data = list(map(lambda x:x[3], cross_project))
score_mean = (statistics.mean(data))
score_stdev =  (statistics.stdev(data))
print (f"Totals stats: mean {score_mean}, stdev {score_stdev}")
cross_project = filter(score_less_than_mean, cross_project)

cross_project = sorted(cross_project, key=lambda x:x[3],reverse=True)

result = []
for art_id in articles:
    pairs = list(filter(lambda x:x[0] == art_id, cross_project))
    data = list(map(lambda x:x[3], pairs))
    score_mean = (statistics.mean(data))
    score_stdev =  (statistics.stdev(data))
    print (f"Article: {art_id} stats: mean {score_mean}, stdev {score_stdev}")
    pairs_filt = list(filter(score_less_than_mean_art, pairs))
    if len(pairs_filt) < 2:
        pairs_filt = pairs[:2]
    result += pairs_filt
    
result = transform_in_dict(result)

json_object = json.dumps(articles)
with open("result_art.json", "w") as outfile:
    outfile.write(json_object)

json_object = json.dumps(videos)
with open("result_vid.json", "w") as outfile:
    outfile.write(json_object)

json_object = json.dumps(totals)
with open("totals.json", "w") as outfile:
    outfile.write(json_object)

json_object = json.dumps(result)
with open("cross_project.json", "w") as outfile:
    outfile.write(json_object)

print (len(result))
