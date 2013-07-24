#from __future__ import unicode_literals  # force unadorned strings "" to be unicode without prepending u""
# -*- coding: utf-8 -*-

"""
This is a custom tweet tokenizer using some functions and ideas from http://sentiment.christopherpotts.net/code-data/happyfuntokenizing.py in addition to checking if
    the tweet includes capitalized versions of the brand at hand
"""


from collections import Counter
from sklearn.feature_extraction import DictVectorizer
import re
import htmlentitydefs

emoticon_string = r"""
    (?:
    [<>]?
    [:;=8]                     # eyes
    [\-o\*\']?                 # optional nose
    [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
    |
    [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
    [\-o\*\']?                 # optional nose
    [:;=8]                     # eyes
    [<>]?
    )"""

# The components of the tokenizer:
regex_strings = (
                 # Phone numbers:
                 r"""
                     (?:
                     (?:            # (international)
                     \+?[01]
                     [\-\s.]*
                     )?
                     (?:            # (area code)
                     [\(]?
                     \d{3}
                     [\-\s.\)]*
                     )?
                     \d{3}          # exchange
                     [\-\s.]*
                     \d{4}          # base
                     )"""
                 ,
                 # Emoticons:
                 emoticon_string
                 ,
                 # HTML tags:
                 r"""<[^>]+>"""
                 ,
                 # Twitter username:
                 r"""(?:@[\w_]+)"""
                 ,
                 # Twitter hashtags:
                 r"""(?:\#+[\w_]+[\w\'_\-]*[\w_]+)"""
                 ,
                 # Remaining word types:
                 r"""
                     (?:[a-z][a-z'\-_]+[a-z])       # Words with apostrophes or dashes.
                     |
                     (?:[+\-]?\d+[,/.:-]\d+[+\-]?)  # Numbers, including fractions, decimals.
                     |
                     (?:[\w_]+)                     # Words without apostrophes or dashes.
                     |
                     (?:\.(?:\s*\.){1,})            # Ellipsis dots.
                     |
                     (?:\S)                         # Everything else that isn't whitespace.
                     """
                 )

######################################################################
# This is the core tokenizing regex:

word_re = re.compile(r"""(%s)""" % "|".join(regex_strings), re.VERBOSE | re.I | re.UNICODE)

# The emoticon string gets its own regex so that we can preserve case for them as needed:
emoticon_re = re.compile(regex_strings[1], re.VERBOSE | re.I | re.UNICODE)

# These are for regularizing HTML entities to Unicode:
html_entity_digit_re = re.compile(r"&#\d+;")
html_entity_alpha_re = re.compile(r"&\w+;")
amp = "&amp;"


class TweetTokenizer():
    
    def __html2unicode(self, s):
        """
            Internal metod that seeks to replace all the HTML entities in
            s with their corresponding unicode characters.
            """
        # First the digits:
        ents = set(html_entity_digit_re.findall(s))
        if len(ents) > 0:
            for ent in ents:
                entnum = ent[2:-1]
                try:
                    entnum = int(entnum)
                    s = s.replace(ent, unichr(entnum))
                except:
                    pass
        # Now the alpha versions:
        ents = set(html_entity_alpha_re.findall(s))
        ents = filter((lambda x : x != amp), ents)
        for ent in ents:
            entname = ent[1:-1]
            try:
                s = s.replace(ent, unichr(htmlentitydefs.name2codepoint[entname]))
            except:
                pass
            s = s.replace(amp, " and ")
        return s
    
    def tokenize(self, s):
        """
            Argument: s -- any string or unicode object
            Value: a tokenize list of strings; conatenating this list returns the original string if preserve_case=False
            """
        # Try to ensure unicode:
        try:
            s = unicode(s)
        except UnicodeDecodeError:
            s = str(s).encode('string_escape')
            s = unicode(s)
        # Fix HTML character entitites:
        s = self.__html2unicode(s)
                
        #Remove http links
        p=re.compile(r'\<http.+?\>', re.DOTALL)
        
        #If we want, we can remove all http urls
        #s = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', '', s, flags=re.S)

        # Tokenize:
        words = word_re.findall(s)
        # Possible alter the case, but avoid changing emoticons like :D into :d:
        if not self.preserve_case:
            words = map((lambda x : x if emoticon_re.search(x) else x.lower()), words)

        #ADD A TAG IF CAPITALZED BRAND WORD FOUND
        if unicode(self.brand.capitalize()) in words:
            words.append(unicode("CAPITALIZED_BRAND"))
        return words


    def __init__(self,brand,preserve_case=True):
        self.vectorizer = DictVectorizer()
        self.brand = brand
        self.preserve_case = preserve_case

    
    def split_for_dict_vectorizer(self,dataset):
        if not(isinstance(dataset,Counter)):
            return [Counter(self.tokenize(s)) for s in dataset]
        else:
            return dataset


    def transform(self,dataset):
        mydataset = self.split_for_dict_vectorizer(dataset)
        return self.vectorizer.transform(self.split_for_dict_vectorizer(dataset))


    def fit_transform(self,dataset):
        mydataset = self.split_for_dict_vectorizer(dataset)

        return self.vectorizer.fit_transform(mydataset)


    def get_feature_names(self):
        return self.vectorizer.get_feature_names()

   



