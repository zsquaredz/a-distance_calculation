import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


# extract the reviews text from an XML file (for Blitzer's dataset)
def xml_extractor(path, max_len=1000):
    reviews = []
    tree = ET.parse(path)
    root = tree.getroot()
    for rev in root.iter('review'):
        reviews.append(rev.text)
    return reviews[:max_len]


# transform the raw text to vectors
def raw_to_ngram(reviews, ngram, min_freq):
    labels = [0] * (int(len(reviews)/2)) + [1] * (int(len(reviews)/2))
    X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.2, random_state=42)
    ngram_vectorizer = CountVectorizer(ngram_range=(1, ngram), token_pattern=r'\b\w+\b',
                                                  min_df=min_freq, binary=True)
    train_feature_matrix = ngram_vectorizer.fit_transform(X_train).toarray()
    test_feature_matrix = ngram_vectorizer.transform(X_test).toarray()
    return train_feature_matrix, y_train, test_feature_matrix, y_test


def get_features(src_path, trg_path, ngram, min_freq):
    src_reviews = xml_extractor(src_path)
    trg_reviews = xml_extractor(trg_path)
    all_reviews = src_reviews + trg_reviews

    train_feature_matrix, y_train, test_feature_matrix, y_test = raw_to_ngram(all_reviews, ngram, min_freq)
    return train_feature_matrix, y_train, test_feature_matrix, y_test

