from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

def all_punct(word):
    return all([not c.isalnum() for c in word])

with open("./allRecsWithAbstracts.xml") as f:
    b = BeautifulSoup(f, features='xml')

all_subjects = set()

en_stopwords = set(stopwords.words('english'))

rec_count = 0
for r in b.records.find_all(text=False, recursive=False):
    if r.name != "record":
        raise Exception(u"Unexpected child node of records: {} != record".format(r.name))

    title = r.title.string
    abstract = r.abstract.string
    subjects = [x.string for x in r.find_all("subject", recursive=False)]

    all_subjects |= set(subjects)

    if rec_count < 100:
        print(u"=============")
        print(u"Title: {}".format(title))
        print(u"Abstract: {}".format(abstract))
        print(u"Subjects ({}): {}".format(len(subjects), u"; ".join(subjects)))

        # tokens = word_tokenize(abstract)

        # tokens = [w for w in word_tokenize(abstract) if not all_punct(w)]

        # tokens = [w.lower() for w in word_tokenize(abstract) if not all_punct(w) and w.lower() not in en_stopwords]

        tokens = [w.lower() for w in word_tokenize(abstract) if not all_punct(w) and w.lower() not in en_stopwords]

        # tokens = []
        # for sentence in sent_tokenize(abstract):
        #     tokens.append(word_tokenize(sentence))

        # tokens = []
        # for sentence in sent_tokenize(abstract):
        #     words = [w for w in word_tokenize(sentence) if not all_punct(w)]
        #     tokens.append(words)

        print(tokens)

    rec_count += 1

all_subjects = sorted(list(all_subjects))
print(u"=============")
print(u"Total number of records: {}".format(rec_count))
print(u"Total number of subjects: {}".format(len(all_subjects)))


# TODO: class Tokenizer(object):  with plenty of options regarding filtering, stemming, stopwords...