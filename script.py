import chardet
import gensim
import json
import nltk
import os
import string
import numpy as np
import pandas as pd
import requests
import sqlite3
import xlrd
import re
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim import corpora, models


# Number used to split essays into clusters.
NUM_OF_CLUSTERS = 7


def upload_essays(essay_path):
    """Uploads essays from given path and stores them in a dictionary.

    Args:
        essay_path: A string representing the path to the essays directory.

    Returns:
        A dictionary of filename (string) -> essay corpus (string).

    Raises:
        TypeError if non-text file found in the essay directory.
    """
    # Should raise error if path not valid: maybe use try/except?
    files = os.listdir(essay_path)

    essays = {}
    for file in files:

        # Verify all files are text files.
        if ".txt" not in file:
            raise TypeError("All essays must be .txt files, error with: "+file)

        # Attempt to confidently guess encoding;
        # Otherwise, default to ISO-8859-1.
        encoding = "ISO-8859-1"
        guess = chardet.detect(open(essay_path + file, "rb").read())

        if (guess["confidence"] >= 0.95):
            encoding = guess["encoding"]

        with open(essay_path + file, "r", encoding=encoding) as f:
            essays[file] = f.read()

    return essays


def build_dict_of_topics_and_process_compound_terms(essays, sheet_path):
    """
        - Reads a spreadsheet of topics/defining terms, and builds a Dictionary of
    topic -> (defining_term -> score).
        - Checks for compound defining terms and processes them in essays.

    Args:
        essays: Dictionary of filename (string) -> essay (string).

    Returns:
        A tuple: Dictionary of topics, Dictionary of essays w/ procecessed
        compound terms.
    """

    workbook = xlrd.open_workbook(sheet_path)
    sheet = workbook.sheet_by_index(0)

    topic_term_dict = {}

    # Read the first column (TOPIC) and add topics as keys to the dictionary.
    current_topic = ""
    for i in range(1, sheet.nrows):
        topic = sheet.cell_value(i, 0)
        if topic:
            topic_term_dict[topic] = {}
            current_topic = topic

        term = sheet.cell_value(i, 1)
        if term:
            # Compound ? If yes, remove spacing.
            if ' ' in term:
                spacefree_term = ''.join(term.split(' '))

                # Replace all occurences of compound terms by removing spaces.
                for (label, corpus) in essays.items():
                    essays[label] = re.sub(term, spacefree_term, corpus)

                term = spacefree_term

            # Lemmatize.
            lemmatized_term = lemmatize_word(term)

            # Append lemmatized terms.
            adjusted_score = 10 - int(sheet.cell_value(i, 2)) + 1

            # Store term + score in dictionary (no duplicates).
            if lemmatized_term not in topic_term_dict[current_topic]:
                topic_term_dict[current_topic][lemmatized_term] = adjusted_score

    return topic_term_dict, essays


def tokenize_essays(essays):
    """
    Converts each essay from a string to a list of strings (tokens), while
    disregarding words that are too short/long.

    Args:
        essays: A dictionary of filename (string) -> essay corpus (string).

    Returns:
        A dicionary of filename (string) -> tokenized corpus (list of strings).
    """

    tokenized_essays = {}
    for (filename, corpus) in essays.items():
        tokenized_essays[filename] = gensim.utils.simple_preprocess(
            corpus, deacc=True, min_len=2, max_len=20)

    return tokenized_essays


def lemmatize_word(word):
    """
    Converts a given word (string) to its lemmatized version.

    Args:
        word (string)

    Returns:
        The lemmatized version of the word (string).
    """

    # determine part of speech of word
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": nltk.corpus.wordnet.ADJ,
                "N": nltk.corpus.wordnet.NOUN,
                "V": nltk.corpus.wordnet.VERB,
                "R": nltk.corpus.wordnet.ADV}
    part_of_speech = tag_dict.get(tag, nltk.corpus.wordnet.NOUN)

    # lemmatize word according to part of speech
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return lemmatizer.lemmatize(word, part_of_speech)


def lemmatize_essays(tokenized_essays):
    """
    Converts the tokens (words) of each essay into lemmatized tokens.

    Args:
        A dicionary of filename (string) -> tokenized corpus (list of strings).

    Returns:
        A dicionary of filename (string) -> tokenized+lemmatized corpus.
    """

    lemmatized_essays = {}
    for (label, word_lst) in tokenized_essays.items():
        lemmatized_essays[label] = []
        for word in word_lst:
            lemmatized_essays[label].append(lemmatize_word(word))

    return lemmatized_essays


def remove_stopwords(lemmatized_essays):
    """
    Removes any tokens charactrized as stop words from the essay tokens.

    Args:
        A dicionary of filename (string) -> tokenized+lemmatized corpus.

    Returns:
        A dicionary of filename (string) -> essay corpus w/o stop words.
    """

    english_stopwords = nltk.corpus.stopwords.words('english')
    custom_stopwords = open("custom_stopwords.txt", "r").read().splitlines()

    stopwords_free_essays = {}
    for (label, word_lst) in lemmatized_essays.items():
        stopwords_free_essays[label] = []
        for word in word_lst:
            if word not in english_stopwords + custom_stopwords:
                stopwords_free_essays[label].append(word)

    return stopwords_free_essays


def vectorize_essays(preprocessed_essays):
    """
    Converts each essay into a vector representation using Doc2Vec.

    Args:
        A dictionary of tokenized + lemmatized + stopwords_free essays.

    Returns:
        A Dataframe of essays (rows) and vector representation (cols) in
        100 dimensions.
    """

    # Vectorize w/ doc2vec.
    documents = []
    for i, doc in enumerate(preprocessed_essays.values()):
        documents.append(TaggedDocument(doc, [i]))

    d2v_model = Doc2Vec(documents, vector_size=100)
    vectorized_df = pd.DataFrame(d2v_model.docvecs.vectors_docs)

    # Feature scaling through standardization.
    stdsclr = StandardScaler()
    return pd.DataFrame(stdsclr.fit_transform(vectorized_df.astype(float)))


def cluster_with_kmeans(standardized_df, preprocessed_essays):
    """
    Partitions essays into clusters using k-means.

    Args:
        standardized_df: Vector representation of essays (DataFrame).
        preprocessed_essays: Dictionary of essays (filename -> corpus tokens).

    Returns:
        A DataFrame of essays w/ corresponding cluster number
        (row: essays, cols: cluster id, essay corpus, filename).
    """

    kmeans = KMeans(n_clusters=NUM_OF_CLUSTERS, init="k-means++", max_iter=100)
    kmeans.fit(standardized_df.values)

    cluster_df = standardized_df
    cluster_df['cluster'] = kmeans.labels_
    cluster_df['essay'] = preprocessed_essays.values()
    cluster_df['filename'] = preprocessed_essays.keys()

    return cluster_df


def update_df_with_topic_scores(cluster_df, topic_term_dict):
    """
    Updates the Dataframe containing essays w/ clusters, w/ scores corresponding
    to the topics.

    Args:
        cluster_df : A DataFrame of essays w/ corresponding cluster number.
        topic_term_dict : Dictionary of topics w/ defining terms.

    Returns:
        An updated version of the given dataframe (cluster_df).
    """

    # Add cloumn for each topic, and initialize all essay scores to 0.
    for topic in topic_term_dict:
        cluster_df[topic] = 0

    # Get filenames per cluster.
    filenames_per_cluster = {}
    for i in range(NUM_OF_CLUSTERS):
        filenames_per_cluster[i] = list(
                                cluster_df[cluster_df.cluster == i].filename)

    for i in range(NUM_OF_CLUSTERS):
        for filename in filenames_per_cluster[i]:
            essay = list(cluster_df[cluster_df.filename == filename].essay)
            dictionary = corpora.Dictionary(essay)
            essay_corpus = [dictionary.doc2bow(token) for token in essay]
            lda = models.ldamodel.LdaModel(corpus=essay_corpus,
                                           id2word=dictionary,
                                           num_topics=1, passes=10)

            # Get"topic terms" for each essay using LDA.
            essay_term_score = {}
            for idx, terms in lda.print_topics(0, 100):

                # LDA generates topic terms in the format: "term1*score1 + term2*score2 + ..."".
                for term_with_score in terms.split('+'):

                    # Separate terms/scores from LDA generated string.
                    term = term_with_score.split('*')[1][1:-2]
                    score = term_with_score.split('*')[0]

                    # Build a dictionary of all the topic terms of the essay w/ corresponding scores.
                    essay_term_score[term] = float(score)

            # For each topic term extracted for an essay, check if it's in the topic dictionary.
            essay_topic_term_score = {}
            for term in essay_term_score.keys():
                for topic in topic_term_dict.keys():
                    if term in topic_term_dict[topic].keys():
                        # If a term is found in the the topic dictionary, compute it's score and update its value.
                        score = essay_term_score[term] * topic_term_dict[topic][term]
                        if topic in essay_term_score:
                            essay_topic_term_score[topic] += score
                        else:
                            essay_topic_term_score[topic] = score

            # For each essay, add a score corresponding to a topic that corresponds to it.
            for topic, score in essay_topic_term_score.items():
                cluster_df.loc[cluster_df[cluster_df['filename'] == filename].index, topic] = score

    return cluster_df


def update_essay_rank(cluster_df, topic_term_dict):
    """
    Update essay rank by multiplying topic strength by distance to local and
    global topic centroids.

    Args:
        cluster_df : A DataFrame of essays w/ corresponding cluster number.
        topic_term_dict : Dictionary of topics w/ defining terms.

    Returns:
        An updated version of the given dataframe (cluster_df), updating rank
        and removing 100-dim vector for each essay.
    """

    def calculate_distance(v1, v2):
        """ calculate L2 norm (Euclidean distance) between two vectors """
        return np.linalg.norm(v1-v2)

    def calculate_centroid(df, topic):
        """ calculate centroid of vectors in df that have passed topic """
        # Get all essays that match topic.
        topic_df = df[df[topic] != 0].iloc[:, :100]

        # Get number of vectors with that topic.
        n = len(topic_df)

        # Calculate mean vector.
        return sum([topic_df.iloc[i] for i in range(n)])/n

    # Calculate global centroids for each topic.
    global_centroid_by_topic = {}
    for topic in topic_term_dict.keys():
        global_centroid_by_topic[topic] = calculate_centroid(cluster_df, topic)

    # Update essay rankings.
    for cluster in range(NUM_OF_CLUSTERS):
        # get essays within cluster
        sub_df = cluster_df[cluster_df['cluster'] == cluster]

        for topic in topic_term_dict.keys():
            # Calculate local centroid within cluster by topic.
            local_centroid = calculate_centroid(sub_df, topic)

            # Calculate dist between topic local_centroid and global_centroid.
            d1 = calculate_distance(global_centroid_by_topic[topic],
                                    local_centroid)

            # Iterate across all essays that contain topic within cluster.
            for essay in sub_df[sub_df[topic] != 0].index:
                essay_vector = cluster_df.iloc[essay, :100]

                # Calculate distance from essay to local_centroid.
                d2 = calculate_distance(essay_vector, local_centroid)

                # Update ranking, multiply by distances.
                cluster_df.at[essay, topic] = cluster_df.at[essay, topic]*d1*d2

    # Drop 100-dim vectors from dataframe, keep only essay filenames and scores.
    return cluster_df.iloc[:, 102:]


def create_db_from_df(essay_topic_df):
    """
    Creates a sqlite3 database from given essay dataframe.

    Args:
        essay_topic_df: A dataframe with essay scores w.r.t topics.

    Returns:
        A sqlite3 database with the essays as rows and topic as columns. Each
        essay has an associated score corresponding to each of the topics.
    """

    # Remove database if it already exists in the directory.
    exists = os.path.isfile("db.sqlite3")
    if exists:
        os.remove("db.sqlite3")

    # Connect to the database.
    conn = sqlite3.connect("db.sqlite3")
    c = conn.cursor()

    # Create essay table.
    table = "Essays"
    pd.set_option("display.max_colwidth", 10000)
    essay_topic_df['link'] = essay_topic_df['filename'].map(
        lambda x: "https://apw.dhinitiative.org/islandora/object/apw%3A" + x[
                                                        4:x.find('.')] + "?")
    essay_topic_df['title'] = essay_topic_df['link'].map(
        lambda x: BeautifulSoup(requests.get(x).text).find('h1'))

    index_names = []
    for index, row in essay_topic_df.iterrows():
    	try:
    		row['title'].text
    	except:
    		index_names.append(index)

    essay_topic_df.drop(index_names, inplace=True)
    essay_topic_df['title'] = essay_topic_df['title'].map(lambda x: x.text)

    essay_topic_df.to_sql(table, conn)

    # Commit changes.
    conn.commit()

    # Close the connection to the database file.
    conn.close()


def main():
    # Upload (from some directory) and store the essays.
    root = os.path.dirname(os.path.realpath('__file__'))
    essay_path = root + '/../essays/'
    essays = upload_essays(essay_path)

    # Read in spreadsheet of topics/defining terms in order to:
    #   1. Build a dictionary of topics w/ defining terms + scores.
    #   2. Preprocess compound defining terms in the essays.
    spreadsheet_path = "topic_term_sheet.xlsx"
    topic_dict, essays = build_dict_of_topics_and_process_compound_terms(
                                essays, spreadsheet_path)

    # Tokenize essays.
    essays = tokenize_essays(essays)

    # Lemmatize all essay tokens.
    essays = lemmatize_essays(essays)

    # Remove any tokens identified as stop words.
    essays = remove_stopwords(essays)

    # Get the vector representation of essays.
    vectorized_essays_df = vectorize_essays(essays)

    # Cluster essays using k-means.
    essays_with_assigned_cluster_df = cluster_with_kmeans(vectorized_essays_df,
                                                          essays)

    # For each essay, assign an initial score to each of its relevent topics.
    essay_with_topic_scores_df = update_df_with_topic_scores(
                                    essays_with_assigned_cluster_df,
                                    topic_dict)

    # Adjust the initially assigned scores.
    essay_topic_df = update_essay_rank(essay_with_topic_scores_df, topic_dict)

    # Create essay database in current directory.
    create_db_from_df(essay_topic_df)


if __name__ == "__main__":
    main()
