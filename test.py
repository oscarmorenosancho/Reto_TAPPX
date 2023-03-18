# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    test.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: omoreno- <omoreno-@student.42barcelona.    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/03/18 11:27:13 by omoreno-          #+#    #+#              #
#    Updated: 2023/03/18 14:02:49 by omoreno-         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

# import pandas as pd

# articles = pd.read_json('articles.json')

# dict_art = articles.__dict__
# print (dict_art)

import json

# with open('articles.json') as user_file:
#   articles = user_file.read()
  
# print(type(articles))

with open('articles.json') as user_file:
    articles = json.load(user_file)
with open('videos.json') as user_file:
    videos = json.load(user_file)

articles_keys = []
videos_keys = []
key_intersect_res = []

to_uppercase = lambda s: s.upper()

for article_id in articles:
    article = articles[article_id]
    keys_set = set(map(to_uppercase, article['keywords'])).difference(set(['']))
    for key in article['keywords']:
         splitted_set = map(to_uppercase, key.split())
         keys_set.union(splitted_set)
    articles_keys.append([article_id, keys_set])

for video_id in videos:
    video = videos[video_id]
    keys_set = set(map(to_uppercase, video['keywords'])).difference(set(['']))
    for key in video['keywords']:
         splitted_set = map(to_uppercase, key.split())
         keys_set.union(splitted_set)
    videos_keys.append([video_id, keys_set])

for article in articles_keys:
    for video in videos_keys:
        key_intersect_set = list(video[1].intersection(article[1]))
        if (len(key_intersect_set)):
            key_intersect_res.append([article[0], video[0], len(key_intersect_set), key_intersect_set])

# print (articles_keys)
# print()
# print (videos_keys)
# print()

for rel in key_intersect_res:
    print(rel)