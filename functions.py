from nltk.stem.porter import PorterStemmer
from itertools import groupby
import numpy as np

def clean_essay(essay):
    """
    Clean the input essay by converting all words to lower case,
    removing punctuation marks and removing words that start with '@'.

    Arguments:
        essay: essay content directly from dataset
    Returns:
        essay: Cleaned essay
        tagged_words_count: Count of words that start with '@' in the dataset.
    """
    # Converting all words in the essay to lower case.
    essay = essay.lower()

    # Removing punctuation marks from essay by replacing them with nothing.
    essay = essay.replace(".", "") 
    essay = essay.replace(",", "")
    essay = essay.replace("'", "")  
    essay = essay.replace("\"", "")
    essay = essay.replace("!", "")
    essay = essay.replace("?", "")
    essay = essay.replace("-", "")
    essay = essay.replace(":", "")
    essay = essay.replace(";", "")
    essay = essay.replace("#", "")
    essay = essay.replace("\\", "")
    essay = essay.replace("[", "")
    essay = essay.replace("]", "")
    essay = essay.replace("{", "")
    essay = essay.replace("}", "")

    # Replacing '(', ')', '/' with space so as to separate words 
    essay = essay.replace("(", " ")
    essay = essay.replace(")", " ")      
    essay = essay.replace("/", " ")
    
    temp_list = []
    tagged_words_count = 0
    
    # Remove words that start with '@'
    for word in essay.split():
        if word.startswith('@'):
            tagged_words_count += 1
        else:
            temp_list.append(word)
    essay = ' '.join(temp_list)

    return essay, tagged_words_count

 
def words(essay_tokens):
    return filter(lambda w: len(w) , [w.strip("0123456789!:,.?(){}[]") for w in essay_tokens.split()])
 
def yule(essay_tokens):
    """
    Returns yule's I measure (the inverse of yule's K measure)
    Higher number depicts higher diversity that is richer vocabulary.

    Arguments:
        essay_tokens: essay content in the form of tokens   
    """
    d = {}
    stemmer = PorterStemmer()
    for w in essay_tokens:
        w = stemmer.stem(w).lower()
        try:
            d[w] += 1
        except KeyError:
            d[w] = 1
 
    M1 = float(len(d))
    M2 = sum([len(list(g)) * (freq**2) for freq,g in groupby(sorted(d.values()))])
 
    try:
        return (M1 * M1) / (M2 - M1)
    except ZeroDivisionError:
        return 0

def matrix_o(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings

    Arguments:
        rater_a: Scores by rater a
        rater_b: Scores by rater b
        min_rating: the minimum possible rating
        max_rating: max_rating is the maximum possible rating
    """

    assert(len(rater_a) == len(rater_b))

    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)

    number_of_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(number_of_ratings)] for j in range(number_of_ratings)]

    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
        
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made 
    
    Arguments: 
        ratings: Scores by a rater
        min_rating: the minimum possible rating
        max_rating: max_rating is the maximum possible rating 
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)

    number_of_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(number_of_ratings)]

    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Calculates the quadratic weighted kappa which is a measure of 
    inter-rater agreement between two raters that provide discrete numeric ratings.  
    
    Potential values range from -1(representing complete disagreement) to 
    1 (representing complete agreement).  

    A kappa value of 0 is expected if all agreement is due to chance.

    Arguments:
        rater_a: Scores by rater a
        rater_b: Scores by rater b
        min_rating: the minimum possible rating
        max_rating: max_rating is the maximum possible rating

    """
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)

    assert(len(rater_a) == len(rater_b))
    
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))

    Oij = matrix_o(rater_a, rater_b, min_rating, max_rating)

    number_of_ratings = len(Oij)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(number_of_ratings):
        for j in range(number_of_ratings):
            Eij = (hist_rater_a[i] * hist_rater_b[j] / num_scored_items)
            Wij = pow(i - j, 2.0) / pow(number_of_ratings - 1, 2.0)

            numerator += Wij * Oij[i][j] / num_scored_items
            denominator += Wij * Eij / num_scored_items

    return 1.0 - numerator / denominator





