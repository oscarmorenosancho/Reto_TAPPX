import json

with open('articles.json') as user_file:
    articles = json.load(user_file)
with open('videos.json') as user_file:
    videos = json.load(user_file)

def remove_punt(s):
    ret = s.replace('. ', ' ').replace('.', ' ').replace(': ', ' ').replace(', ', ' ')
    ret = ret.replace('"', '').replace('-', ' ').replace('|', '')
    ret = ret.replace(')', '').replace('(', ' ').replace('}', '').replace('{', ' ')
    ret = ret.replace(']', '').replace('[', ' ')
    return ret

to_uppercase = lambda s: s.upper()

def compute_str_ocur(text):
    text = remove_punt(article['text']).upper()
    words = text.split()
    words_set = list(set(words))
    words_ocur = dict()
    total = 0
    for word in words_set:
        ocurs = text.count(word)
        words_ocur[word] = ocurs
        total += ocurs
    return ({'words_ocur': words_ocur, 'total': total})

def update_total_str_ocur(totals, item):
    total_keys = list(totals['words_ocur'].keys())
    item_keys = list(item['str_ocur']['words_ocur'].keys())
    for key in item_keys:
        if not key in total_keys:
            totals['words_ocur'][key] = 0
            amount_in_key = item['str_ocur']['words_ocur'][key]
            totals['words_ocur'][key] = amount_in_key
            totals['total'] += amount_in_key
    return totals

for article_id in articles:
    article = articles[article_id]
    article['str_ocur'] = compute_str_ocur(article)

for video_id in videos:
    video = videos[video_id]
    video['str_ocur'] = compute_str_ocur(video)

articles.update(videos)

totals = { 'words_ocur': {}, 'total': 0}
for item in articles:
    totals = update_total_str_ocur(totals, articles[item])

json_object = json.dumps(articles)
with open("result.json", "w") as outfile:
    outfile.write(json_object)

json_object = json.dumps(totals)
with open("totals.json", "w") as outfile:
    outfile.write(json_object)
