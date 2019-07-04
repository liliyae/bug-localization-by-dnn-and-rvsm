""" 
Helper functions mostly for feature extraction.

Used many modified and intact code blocks from 
'https://github.com/jeffrey-palmerino/BugLocalizationDNN'
"""

import csv
import re
import os
import random
import timeit
from datetime import datetime
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer


def git_clone(repo_url, clone_folder):
    """ Clones the git repo from 'repo_ur' into 'clone_folder'

    Arguments:
        repo_url {string} -- Url of git repository
        clone_folder {string} -- path of a local folder to clone the repository 
    """
    repo_name = repo_url[repo_url.rfind('/')+1:-4]
    if os.path.isdir(clone_folder + repo_name):
        print("Already cloned")
        return
    cwd = os.getcwd()
    if not os.path.isdir(clone_folder):
        os.mkdir(clone_folder)
    os.chdir(clone_folder)
    os.system("git clone {}".format(repo_url))
    os.chdir(cwd)


def tsv2dict(tsv_path):
    """ Converts a tab separated values (tsv) file into a dictionary

    Arguments:
        tsv_path {string} -- path of the tsv file
    """
    reader = csv.DictReader(open(tsv_path, 'r'), delimiter='\t')
    dict_list = []
    for line in reader:
        line["files"] = [os.path.normpath(f[8:]) for f in line["files"].strip(
        ).split() if f.startswith("bundles/") and f.endswith(".java")]
        line["raw_text"] = line["summary"] + line["description"]
        # line["summary"] = clean_and_split(line["summary"][11:])
        # line["description"] = clean_and_split(line["description"])
        line["report_time"] = datetime.strptime(
            line["report_time"], "%Y-%m-%d %H:%M:%S")

        dict_list.append(line)
    return dict_list


def csv2dict(csv_path):
    """ Converts a comma separated values (csv) file into a dictionary

    Arguments:
        csv_path {string} -- path to csv file
    """
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')
        csv_dict = list()
        for line in reader:
            csv_dict.append(line)

    return csv_dict


def clean_and_split(text):
    """ Remove all punctuation and split text strings into lists of words

    Arguments:
        text {string} -- input text
    """
    table = str.maketrans(dict.fromkeys(string.punctuation))
    clean_text = text.translate(table)
    word_list = [s.strip() for s in clean_text.strip().split()]
    return word_list


def top_k_wrong_files(right_files, br_raw_text, java_files, k=50):
    """ Randomly samples 2*k from all wrong files and returns metrics
        for top k files according to rvsm similarity.

    Arguments:
        right_files {list} -- list of right files
        br_raw_text {string} -- raw text of the bug report
        java_files {dictionary} -- dictionary of source code files

    Keyword Arguments:
        k {integer} -- the number of files to return metrics (default: {50})
    """

    # Randomly sample 2*k files
    randomly_sampled = random.sample(
        set(java_files.keys()) - set(right_files), 2*k)

    all_files = []
    for filename in randomly_sampled:
        try:
            src = java_files[filename]

            rvsm = cosine_sim(br_raw_text, src)
            cns = class_name_similarity(br_raw_text, src)

            all_files.append((filename, rvsm, cns))
        except:
            pass

    top_k_files = sorted(all_files, key=lambda x: x[1], reverse=True)[:k]

    return top_k_files


def stem_tokens(tokens):
    """ Remove stopword and stem

    Arguments:
        tokens {list} -- tokens to stem 
    """
    stemmer = PorterStemmer()
    removed_stopwords = [stemmer.stem(
        item) for item in tokens if item not in stopwords.words("english")]

    return removed_stopwords


def normalize(text):
    """ Lowercase, remove punctuation, tokenize and stem

    Arguments:
        text {string} -- A text to normalize
    """
    remove_punc_map = dict((ord(char), None) for char in string.punctuation)
    removed_punc = text.lower().translate(remove_punc_map)
    tokenized = word_tokenize(removed_punc)
    stemmed_tokens = stem_tokens(tokenized)

    return stemmed_tokens


def cosine_sim(text1, text2):
    """ Cosine similarity with tfidf

    Arguments:
        text1 {string} -- first text
        text2 {string} -- second text
    """
    vectorizer = TfidfVectorizer(
        tokenizer=normalize, min_df=1, stop_words='english')
    tfidf = vectorizer.fit_transform([text1, text2])
    sim = ((tfidf * tfidf.T).A)[0, 1]

    return sim


def get_all_source_code(start_dir):
    """ Creates corpus starting from 'start_dir'

    Arguments:
        start_dir {string} -- directory path to start
    """
    files = {}
    start_dir = os.path.normpath(start_dir)
    for dir_, dir_names, file_names in os.walk(start_dir):
        for filename in [f for f in file_names if f.endswith(".java")]:
            src_name = os.path.join(dir_, filename)
            with open(src_name, 'r') as src_file:
                src = src_file.read()

            file_key = src_name.split(start_dir)[1]
            file_key = file_key[len(os.sep):]
            files[file_key] = src

    return files


def get_months_between(d1, d2):
    """ Calculates the number of months between two date strings

    Arguments:
        d1 {datetime} -- date 1
        d2 {datetime} -- date 2
    """

    diff_in_months = abs((d1.year - d2.year) * 12 + d1.month - d2.month)

    return diff_in_months


def most_recent_report(reports):
    """ Returns the most recently submitted previous report that shares a filename with the given bug report

    Arguments:
        filename {string} -- the name of the shared Java file
        current_date {datetime} -- until date
        bug_reports {list of dictionaries} -- list of all bug reports
    """

    if len(reports) > 0:
        return max(reports, key=lambda x: x.get("report_time"))

    return None


def previous_reports(filename, until, bug_reports):
    """ Returns a list of previously filed bug reports that share a file with the current bug report

    Arguments:
        filename {string} -- the name of the shared Java file
        until {datetime} -- until date
        bug_reports {list of dictionaries} -- list of all bug reports
    """
    return [br for br in bug_reports if (filename in br["files"] and br["report_time"] < until)]


def bug_fixing_recency(br, prev_reports):
    """ Calculates the Bug Fixing Recency as defined by Lam et al.

    Arguments:
        report1 {dictionary} -- current bug report
        report2 {dictionary} -- most recent bug report
    """
    mrr = most_recent_report(prev_reports)

    if br and mrr:
        return 1/float(get_months_between(br.get("report_time"), mrr.get("report_time")) + 1)

    return 0


def collaborative_filtering_score(raw_text, prev_reports):
    """[summary]

    Arguments:
        raw_text {string} -- raw text of the bug report 
        prev_reports {list} -- list of previous reports
    """

    prev_reports_merged_raw_text = ""
    for report in prev_reports:
        prev_reports_merged_raw_text += report["raw_text"]

    cfs = cosine_sim(raw_text, prev_reports_merged_raw_text)

    return cfs


def class_name_similarity(raw_text, source_code):
    """[summary]

    Arguments:
        raw_text {string} -- raw text of the bug report 
        source_code {string} -- java source code 
    """
    classes = source_code.split(" class ")[1:]
    class_names = [c[:c.find(" ")] for c in classes]
    class_names_text = ' '.join(class_names)

    class_name_sim = cosine_sim(raw_text, class_names_text)

    return class_name_sim


class CodeTimer:
    def __init__(self, message=""):
        self.message = message

    def __enter__(self):
        print(self.message)
        self.start = timeit.default_timer()

    def __exit__(self, exc_type, exc_value, traceback):
        self.took = timeit.default_timer() - self.start
        print("Finished in {0:0.5f} secs.".format(self.took))
