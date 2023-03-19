import json
import re
import math

with open('articles.json') as user_file:
    articles = json.load(user_file)
with open('videos.json') as user_file:
    videos = json.load(user_file)

def remove_punt(s):
    if (s and len(s)>0):
        ret = s.replace('.', ' ').replace('.', ' ').replace(':', ' ').replace(',', ' ')
        ret = ret.replace('"', '').replace("'", '').replace('-', ' ').replace('|', '')
        ret = ret.replace(')', ' ').replace('(', ' ').replace('}', ' ').replace('{', ' ')
        ret = ret.replace(']', ' ').replace('[', ' ').replace('\\', ' ').replace('/', ' ')
        ret = re.sub('\d+', '', ret)
    return ret

to_uppercase = lambda s: s.upper()

def compute_str_ocur(item):
    text = remove_punt(item['text']).upper()
    words = text.split()
    words_set = list(set(words))
    words_ocur = {}
    total = 0
    for word in words_set:
        ocurs = text.count(word)
        words_ocur[word] = ocurs
        total += ocurs
    return ({'words_ocur': words_ocur, 'total': total})

def update_total_str_ocur(totals, item):
    total_keys = totals.keys()
    if not 'words_ocur' in total_keys:
        totals['words_ocur'] = {}
    if not 'docs_rep' in total_keys:
        totals['docs_rep'] = {}
    if not 'total' in total_keys:
        totals['total'] = 0
    if not 'docs_cnt' in total_keys:
        totals['docs_cnt'] = 1
    else:
        totals['docs_cnt'] += 1
    total_words_ocur_keys = list(totals['words_ocur'].keys())
    total_docs_rep_keys = list(totals['docs_rep'].keys())
    item_keys = list(item['tf-idf']['words_ocur'].keys())
    for key in item_keys:
        if not key in total_words_ocur_keys:
            totals['words_ocur'][key] = 0
        if not key in total_docs_rep_keys:
            totals['docs_rep'][key] = 0
        amount_in_key = item['tf-idf']['words_ocur'][key]
        totals['words_ocur'][key] += amount_in_key
        totals['docs_rep'][key] += 1
        totals['total'] += amount_in_key
    return totals

def compute_idf(totals):
    total_keys = totals.keys()
    check_c = 'words_ocur' in totals
    check_c = check_c and 'docs_rep' in totals
    check_c = check_c and 'total' in totals
    check_c = check_c and 'docs_cnt' in totals
    if check_c:
        if not 'words_idf' in total_keys:
            totals['words_idf'] = {}
        for key in totals['docs_rep'].keys():
            totals['words_idf'][key] = math.log10(totals['docs_cnt'] / totals['docs_rep'][key])
    return totals

def compute_tfidf(totals,item):
    total_keys = totals.keys()
    item_keys = item.keys()
    ckeck_c = 'words_idf' in total_keys
    ckeck_c = ckeck_c and 'tf-idf' in item_keys
    if ckeck_c:
        if not 'words_tfidf' in item['tf-idf']:
            item['tf-idf']['words_tfidf'] = {}
        tot_words = item['tf-idf']['total']
        it_w_tfidf = item['tf-idf']['words_tfidf']
        for key in item['tf-idf']['words_ocur']:
            key_ocur = item['tf-idf']['words_ocur'][key]
            key_tf = key_ocur / tot_words
            it_w_tfidf[key] = key_tf * totals['words_idf'][key]
        sorted_dict = sorted(it_w_tfidf.items(), key=lambda x:x[1],reverse=True)
        item['tf-idf']['words_tfidf'] = dict(sorted_dict)
    return

for article_id in articles:
    article = articles[article_id]
    article['tf-idf'] = compute_str_ocur(article)

for video_id in videos:
    video = videos[video_id]
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

json_object = json.dumps(articles)
with open("result_art.json", "w") as outfile:
    outfile.write(json_object)

json_object = json.dumps(videos)
with open("result_vid.json", "w") as outfile:
    outfile.write(json_object)

json_object = json.dumps(totals)
with open("totals.json", "w") as outfile:
    outfile.write(json_object)
