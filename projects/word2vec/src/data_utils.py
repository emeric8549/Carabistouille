from gensim.corpora import WikiCorpus

def load_wiki_texts(wiki_dump_path, limit=None):
    wiki = WikiCorpus(wiki_dump_path, lemmatize=False, dictionary=None)
    for i, tokens in enumerate(wiki.get_texts()):
        yield [word.decode('utf-8') for word in tokens]
        if limit and i + 1 >= limit:
            break