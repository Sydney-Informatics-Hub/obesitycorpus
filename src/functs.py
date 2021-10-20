import datetime
from bs4 import UnicodeDammit
import re
import spacy
from spacy.tokens import Span
import pandas as pd
from unidecode import unidecode
import unicodedata
import dateparser
import zipfile
import os

def apply_to_titlebody(df, function):
    df["body"] = df["body"].apply(function)
    df["title"] = df["title"].apply(function)

def obesitylist(*args):
    mylist = ['obesity', 'obese', "obesogenic", "obesogen"]
    for x in args:
        mylist.append(x)
    return mylist


def readfilesin(file_path, encoding):
    if encoding in ['ascii', 'Windows-1252', 'ISO-8859-1']:
        with open(file_path, encoding='Windows-1252') as file:
            data = file.read()
    elif encoding == 'utf-8':
        with open(file_path, encoding='utf-8') as file:
            data = file.read()
    else:
        try:
            with open(file_path, 'rb') as non_unicode_file:
                content = non_unicode_file.read()
                dammit = UnicodeDammit(content, ['Windows-1252'])
                data = dammit.unicode_markup
        except Exception as e:
            raise ValueError('Can\'t return dictionary from empty or invalid file %s due to %s' % (file_path, e))
    return data.replace("\r", "").replace("\nClassification\n\n\n", "").strip()


def convert_month(month):
    if month in ["Jan", "Feb", "Mar", "Apr", "May",
                 "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]:
        return (datetime.datetime.strptime(month, "%b").strftime("%m"))
    elif month in ["January", "February", "March", "April", "May",
                   "June", "July", "August", "September", "October", "November", "December"]:
        return (datetime.datetime.strptime(month, "%B").strftime("%m"))
    else:
        print("The following month name is not valid: '", month, "'")


def where_is_byline(contentslist):
    tmplist = [x.lower().find('byline') for x in contentslist]
    return [i for i, x in enumerate(tmplist) if x != -1]


def get_byline(contents):
    byline_withnewline = re.search('Byline: (.*)\n', contents, re.IGNORECASE)
    byline_nonewline = re.search('Byline: (.*)', contents, re.IGNORECASE)
    if byline_withnewline:
        # and convert to title case
        return byline_withnewline.group(1).title()
    elif byline_nonewline:
        # it was at the end of the metadata so we already stripped the newline
        return byline_nonewline.group(1).title()
    else:
        return None

def get_wordcount_from_metadata(contents):
    wordcount_specified = re.search('Length: (\d+) words', contents, re.IGNORECASE)
    if wordcount_specified:
        return wordcount_specified.group(1)
    else:
        return None

def standard_outputfilename(row):
    return f"{row.source}_{row.year}_{row.numeric_month}_{row.fourdigitcode}_{make_slug(row.title)}.txt"

def parse_filename(path):
    source, year, month = path.split('/')[2].split("_")
    month = month.replace("txt", "").strip()
    numeric_month = convert_month(month)
    return pd.Series({
        "source": source,
        "year": year,
        "numeric_month": numeric_month
    })


def get_text4digitcode(contents):
    code4digits = re.search(r'(\d+)txt', contents, re.IGNORECASE)
    if code4digits:
        return code4digits.group(1)
    else:
        # some articles from the Age and Brisbane Times have (1) (2) etc numeration instead
        # extract and pad these to four digits to make consistent
        codeXdigits = re.search(r'\((\d+)\)\.txt', contents, re.IGNORECASE).group(1).zfill(4)
        return codeXdigits


def clean_nonascii(body, replacementcsvfile="replacements.csv"):
    '''
    Removes non-unicode characters in bodies stored one per row in column of pandas df
    '''
    # load in replacement dictionary
    replacementdictionary = {"Â\xad": "' ", "~\xad": "-", "\\xE2Ä(tm)": "'", "\\xE2Äú": "\"", \
                             "\\xE2Ä\"": "-", "\xE2Äò": "\"", "\\xE2€(tm)": "'", "\\xE2€": "'"}
    replacementdictionary.update(pd.read_csv(replacementcsvfile, quotechar="'", escapechar="\\", \
                                             keep_default_na=False).set_index('word')['replacement'].to_dict())
    # clean using that
    messybody = body
    bodies_replaced_with_my_dict = mystringreplace(messybody, replacementdictionary)
    # clean up using unidecode
    # this needs to happen after the replacement dictionary replacement as it will
    # automatically incorrectly replace all non-ascii characters
    ascii_replaced = unidecode(bodies_replaced_with_my_dict)
    # clean up using unidecodedata.normalise
    cleaned_bodies = unicodedata.normalize("NFKD",ascii_replaced)
    return cleaned_bodies

def strip_newlines(column):
    column = column.strip("\n")
    return column

def clean_quotes(column):
    '''
    Cleans up quotes in body or title
    '''
    mytext = column
    # manually checked 3 quotes are used instead of two in the corpus
    mytext = mytext.replace('```', '"')
    # two open quotes
    mytext = mytext.replace('``', '"')
    # two close quotes
    mytext = mytext.replace("''", "\"")
    # single quote
    mytext = mytext.replace("`", "'")

    return mytext

def clean_wa(column):
    # replaces some odd tags in the WA
    mytext = column
    # replacing ";  -----QUOTE----" and ";  -----info box----" detected in the title of some of the WA articles
    mytext = re.sub('; {2}-----QUOTE----', '', mytext, flags=re.IGNORECASE)
    mytext = re.sub(';  -----info box----', '', mytext, flags=re.IGNORECASE)
    return mytext



def clean_quot(column):
    '''
    Deals with &quot; strings in body or metadata
    '''
    mytext = column
    mytext = mytext.replace('&quot;&quot;&quot;', '"')
    mytext = mytext.replace('&quot;&quot;', '"')
    mytext = mytext.replace('&quot;', '"')
    mytext = mytext.replace('&Quot;', '"')
    return mytext

def clean_page_splits(bodytext):
    '''
    This is useful not only because it removes From Page/Continued Page
    But also and more so because the byline or it's variant can be also repeated within those two tags
    Leading to possible duplication of key terms that were NOT in the original article
    '''
    # Advertiser +  NT news + Courier Mail
    # note that text byline is duplicated in this example between the two page references!
    bodytext = re.sub(r'\nContinued Page \d+\n\w+.*\nFrom Page \d+\n', ' ', bodytext)
    # canberra times
    bodytext = re.sub(r'\nFrom Page\d+ ', ' ', bodytext)
    bodytext = re.sub(r'\nFrom Page \d+ ', ' ', bodytext)
    # Herald sun (also matches Hobart Mercury)
    bodytext = re.sub(r'\nContinued Page \d+\nFrom Page \d+\n', ' ', bodytext)
    bodytext = re.sub(r'\nContinued Page \d+ From Page \d+\n', ' ', bodytext)
    # NT news
    bodytext = re.sub(r'\nCONTINUED Page \d+\n\w+.*\nFROM Page \d+\n', ' ', bodytext)
    # SydHerald and Age and Telegraph
    # no evidence of splits
    # Australian
    bodytext = re.sub(r'\nContinued on Page \d+\nContinued from Page \d+\n', ' ', bodytext)
    # West Australian
    bodytext = re.sub(r'\nContinued page \d+\nFrom page \d+', ' ', bodytext)
    return bodytext


def clean_redundant_phrases(bodytext):
    '''
    This function cleans some social media references at the end of texts that are unrelated to the content of the body.
    '''
    bodytext = re.sub(r'\nTo read more from Good Weekend magazine, visit our page at The Sydney Morning Herald or            The Age.', '', bodytext)
    bodytext = re.sub(r'           Stay informed. Like the Brisbane Times Facebook page           .', '', bodytext)
    # can times
    bodytext = re.sub(r'\nFollow \w.+ on Twitter and \s+ Facebook\n', ' ', bodytext)
    bodytext = re.sub(r'Follow us on Facebook', ' ', bodytext)
    return bodytext

def replace_six_questionmarks(column):
    '''
    Replaces 6 ?????? with a double quote
    '''
    # [^?](\?){6}[^?] TODO 
    mytext = re.sub(r'[^?](\?){6}[^?]', '"', column)
    return mytext

def count_keywords(series, keyword):
    return series.map(lambda x: x.lower().count(keyword))

def make_slug(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespace with a single dash
    s = re.sub(r"\s+", '-', s)
    # truncate ultra-long titles
    s = s[:150] if len(s) > 150 else s
    return s


# to clean the corpus - cleaning symbols in messy bodies

def mystringreplace(string, replacementobject):
    if string is None:
        return None
    elif isinstance(replacementobject, list):
        for word in replacementobject:
            string = string.replace(word, " ")
        return string
    else:
        for word, replacement in replacementobject.items():
            string = string.replace(word, replacement)
        return string


def find_problems(start, end, corpusdf, colname="cleaned_bodies"):
    # finding problematic sentences
    matches = [re.findall(r'\w+.[^\x00-\x7F].+', x)
               for x in corpusdf[colname].iloc[start:end].tolist()]
    return [item
            for sublist in matches
            for item in sublist]


def find_specific_character_with_preceding(character, start, end, corpusdf, colname="cleaned_bodies"):
    # finding a specific character with the preceding characters
    pattern = r'\w+.' + re.escape(character) + '+.*'
    return [item for sublist in [re.findall(pattern, x) for x in corpusdf[colname].tolist()[start:end]] for item in
            sublist]


def find_specific_character_wout_preceding(character, start, end, corpusdf, colname="cleaned_bodies"):
    # finding a specific character where that character starts a word
    pattern = r'' + re.escape(character) + '+.*'
    return [item for sublist in [re.findall(pattern, x) for x in corpusdf[colname].tolist()[start:end]] for item in
            sublist]


def find_filename_from_string(string, corpusdf):
    return corpusdf[corpusdf['body'].str.contains(string)]['filename'].to_list()


def display_body_from_string(string, corpusdf):
    return corpusdf[corpusdf['body'].str.contains(string)]['body'].to_list()


def get_date(string):
    pattern = re.compile(
        "(Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|"
        "Jul(y)?|Aug(ust)?|Sep(tember)?|Oct(ober)?|Nov(ember)?|"
        "Dec(ember)?)\s+\d{1,2},\s+\d{4}")
    if pattern.search(string) is not None:
        return dateparser.parse(pattern.search(string).group())
    else:
        return None


def write_corpus_titlebody(df, directoryname="corpus-titlebody"):
    '''
    Writes our corpus with title and body, without any tags or metadata
    '''
    archive = zipfile.ZipFile(f"../200_data_clean/{directoryname}.zip", "w", zipfile.ZIP_DEFLATED)
    for index, row in df.iterrows():
        outputfilename = standard_outputfilename(row)
        content = row['title'] + row['body']
        archive.writestr(outputfilename, content)
    archive.close()

def write_corpus_nested(df, directoryname="corpus-nested"):
    '''
    Writes our corpus with title and body, nested by source/year/month
    '''
    for index, row in df.iterrows():
        outputdir = "../200_data_clean/" + directoryname + f"/{row.source}/{row.year}/{row.numeric_month}/"
        outputfilename = outputdir + standard_outputfilename(row)
        os.makedirs(os.path.dirname(outputdir), exist_ok=True)
        content = row['title'] + row['body']
        f = open(outputfilename, 'w', encoding='utf-8')
        f.write(content)
        f.close()


def cqpweb_metadata(df, directoryname="corpus-titlebody"):
    '''
    Writes our corpus with title and body, without any tags or metadata
    '''
    # ../200_data_clean/
    # TODO Fix this to use standard_outputfilename instead of the below code
    outputdf = df.copy()
    outputdf['slug'] = outputdf['title'].apply(lambda x: make_slug(x))
    outputdf['outputputfile'] = outputdf[['source', 'year', 'numeric_month', 'fourdigitcode', 'slug']].agg('_'.join, axis=1)
    # this removed columns that don't need to be in the metadata
    outputdf.drop(['filename', 'encoding','confidence','fullpath','fourdigitcode','year','numeric_month','body'], axis=1, inplace=True)
    outputdf.to_csv(f'../200_data_clean/{directoryname}_metadata.csv', index=False)
    outputdf.to_csv(f'../200_data_clean/{directoryname}_metadata.tsv', sep='\t', index=False)


def write_corpus_sketchengine(df, directoryname="corpus-sketchengine"):
    '''
    Writes our corpus with title and body, with tags in the format accepted by sketch engine
    '''
    archive = zipfile.ZipFile(f"../200_data_clean/{directoryname}.zip", "w", zipfile.ZIP_DEFLATED)
    for index, row in df.iterrows():
        outputfilename = standard_outputfilename(row)
        sketchenginetags = '<doc date="' + row['date'].strftime("%Y-%m-%d") + '" publication="' + row['source'] + '" wordcountTotal="' + str(row['wordcount_total']) + '">'
        content = sketchenginetags + "\n<head>" + row['title'] + "</head>\n<body>\n" + row['body'] + "\n</body>\n</doc>"
        archive.writestr(outputfilename, content)
    archive.close()


# Related to SPACY ------------------

def explore_tokens(sentencenlp_list, obesitynames):
    sentencesummarylist = []
    for sentence in sentencenlp_list:
        # displacy.serve(sentence, style="dep")
        for token in sentence:
            if token.lemma_ in obesitynames:
                mydict = {
                    'sentence': sentence.text,
                    'text': token.text,
                    'tag': token.tag_,
                    'dep': token.dep_,
                    'head': token.head,
                    'left': token.left_edge.orth_,
                    'right': token.right_edge.orth_}
                sentencesummarylist.append(mydict)
    return sentencesummarylist
