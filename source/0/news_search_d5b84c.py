# https://github.com/sebseager/qiskit-fall-fest-2021/blob/1c403c7e5ef1e555fd445e31f234cb48ba6cef43/news_search.py
# Copyright 2021 Seb Seager

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import requests
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import qiskit
from qiskit.quantum_info import Statevector
import streamlit as st


def keywords_satisfied(text, keywords):
    ans = [0] * len(keywords)
    for j in range(len(text)):
        for i in range(len(keywords)):
            if keywords[i] == text[j]:
                ans[i] = 1
    return sum(ans)


def oracle(article, keywords):
    n = keywords_satisfied(article, keywords)
    return 1 / (2 ** (len(keywords) - n))


# change article.text to whatever the API defines
def encode_articles(articles, keywords):
    sum = 0
    results = []
    for i in range(len(articles)):
        results.append(oracle(articles[i], keywords))
        sum += results[i] ** 2
    for i in range(len(articles)):
        results[i] = (1 / (sum ** (1 / 2))) * results[i]
    return results


# encoding step
def find_article(articles, keywords):
    encoded_vector = list(encode_articles([x[1] for x in articles], keywords))
    encoded_vector = np.array(encoded_vector)
    state_vector = Statevector(data=encoded_vector)
    # print(state_vector.measure())
    # idx = int(state_vector.measure()[0], 2)
    idx = int(state_vector.measure()[0])
    chosen_article = articles[idx]
    return chosen_article[0], chosen_article[2]


@st.cache()
def pull_articles(news_country, api_key):
    # api call
    base_url = "https://newsapi.org/v2/top-headlines"
    country = f"country={news_country}&" if len(news_country) else ""
    x = requests.get(f"{base_url}?{country}apiKey={api_key}&pageSize=100")
    try:
        articles = dict(x.json())["articles"]
    except KeyError:
        st.error("Invalid API call")
        return

    # process articles from api
    processed_articles = []
    for a in articles:
        if a["content"]:
            new_content = a["content"].lower().replace(".,![]{}-—_;':?", "").split()
            new_title = a["title"]
            new_url = a["url"]
            processed_articles.append((new_title, new_content, new_url))

    return processed_articles


def main():
    st.title("Quantum Inspired News Suggestions")
    st.header("1. Pull articles from NewsAPI")

    api_key = st.text_input("NewsAPI key", value="b72aad9271604a1b888e9479c9de4fb7")
    news_country = st.text_input("News country/region to search", value="us")

    # keywords to check for
    st.header("2. Perform search")
    search_input = st.text_input("Keywords to search for (separated by spaces)")

    if st.button("Run"):
        all_articles = pull_articles(news_country, api_key)
        article_summary = st.text(f"Articles found: {len(all_articles)}")
        search_keywords = search_input.split(" ")
        received_article = find_article(all_articles, search_keywords)
        try:
            st.text(f"Article: {received_article[0]}")
            st.text(f"URL: {received_article[1]}")
        except IndexError:
            st.error("No articles found")

    st.header("3. Display article frequencies")
    nrep = st.slider("Sample size", 10, 5000, 1000)

    if st.button("Build article frequency graph"):
        all_articles = pull_articles(news_country, api_key)
        article_summary = st.text(f"Articles found: {len(all_articles)}")
        search_keywords = search_input.split(" ")
        occurrence_dict = {}
        for i in range(nrep):
            received_article = find_article(all_articles, search_keywords)
            try:
                occurrence_dict[received_article[0]] += 1
            except KeyError:
                occurrence_dict[received_article[0]] = 1

        st.bar_chart(
            pd.DataFrame.from_dict(occurrence_dict, orient="index"),
            height=500,
        )


if __name__ == "__main__":

    main()
