import datetime
import os
import re
import spacy
from spacy.tokens import Span
import pandas as pd
import numpy as np
import zipfile
import dateparser
from bs4 import UnicodeDammit
from unidecode import unidecode

def obesitylist(*args):
    mylist = ['obesity', 'obese', "obesogenic", "obesogen"]
    for x in args:
        mylist.append(x)
    return mylist

def abbreviate_source(df, source_column):
    conditions = [
        df[source_column].eq("HeraldSun"),
        df[source_column].eq("SydHerald"),
        df[source_column].eq("Advertiser") ,
        df[source_column].eq("CourierMail") ,
        df[source_column].eq("Age") ,
        df[source_column].eq("CanTimes") ,
        df[source_column].eq("Australian") ,
        df[source_column].eq("WestAus") ,
        df[source_column].eq("HobMercury") ,
        df[source_column].eq("Telegraph"),
        df[source_column].eq("NorthernT"),
        df[source_column].eq("BrisTimes")
         ]
    choices = ["HS", "SM", "AD", "CM", "AG", "CT", "AU", "WA", "HM", "TE", "NT", "BT"]
    return np.select(conditions, choices, default="Missing")


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
    # TODO get this to use pandas.to_datetime is your friend (and dateutil which underlies it).
    # even though this isn't coming from Pandas...
    if month in ["Jan", "Feb", "Mar", "Apr", "May",
                 "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]:
        return (datetime.datetime.strptime(month, "%b").strftime("%m"))
    elif month in ["January", "February", "March", "April", "May",
                   "June", "July", "August", "September", "October", "November", "December"]:
        return (datetime.datetime.strptime(month, "%B").strftime("%m"))
    else:
        print("The following month name is not valid: '", month, "'")

def apply_to_titlebody(df, function):
    df["body"] = df["body"].apply(function)
    df["title"] = df["title"].apply(function)

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
    mytext = re.sub('; {2}-----info box----', '', mytext, flags=re.IGNORECASE)
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
    bodytext = re.sub(r'\nFrom Page ?\d+ ', ' ', bodytext)
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

def make_slug(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespace with a single dash
    s = re.sub(r"\s+", '-', s)
    # truncate ultra-long titles
    s = s[:150] if len(s) > 150 else s
    return s

def mystringreplace(string, replacementobject):
    # to clean the corpus - cleaning symbols in messy bodies
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
    if (match := pattern.search(string)) is not None:
        return dateparser.parse(match.group())
    else:
        return None


def write_corpus_titlebody(df, cleandatapath, directoryname="corpus-titlebody"):
    '''
    Writes our corpus with title and body, without any tags or metadata
    '''
    archive = zipfile.ZipFile(f"{cleandatapath}/{directoryname}.zip", "w", zipfile.ZIP_DEFLATED)
    for index, row in df.iterrows():
        outputfilename = standard_outputfilename(row)
        content = row['title'] + row['body']
        archive.writestr(outputfilename, content)
    archive.close()

def write_corpus_nested(df, cleandatapath, directoryname="corpus-nested"):
    '''
    Writes our corpus with title and body, nested by source/year/month
    '''
    for index, row in df.iterrows():
        outputdir = str(cleandatapath) + "/" + directoryname + f"/{row.source}/{row.year}/{row.numeric_month}/"
        outputfilename = outputdir + standard_outputfilename(row)
        os.makedirs(os.path.dirname(outputdir), exist_ok=True)
        content = row['title'] + "\n" + row['body']
        f = open(outputfilename, 'w', encoding='utf-8')
        f.write(content)
        f.close()

def clean_unsafe(df):
    """
    Cleans markup that could be unsafe in sgml
    """
    # Replace ampersands with safe replacements
    clean_amp = lambda x: (re.sub(r'&', '&amp;', x))
    apply_to_titlebody(df, clean_amp)
    # Replace >
    clean_gt = lambda x: (re.sub(r'>', '&gt;', x))
    apply_to_titlebody(df, clean_gt)
    # Replace <
    clean_lt = lambda x: (re.sub(r'<', '&lt;', x))
    apply_to_titlebody(df, clean_lt)
    return df

def write_corpus_cqpweb(df, cleandatapath, directoryname="corpus-cqpweb"):
    '''
    Writes our corpus and metadata as per CQP web sample format
    '''
    outputpath = str(cleandatapath) + "/"
    archive = zipfile.ZipFile(f"{outputpath}{directoryname}.zip", "w", zipfile.ZIP_DEFLATED)
    # Cleans markup that could be unsafe in sgml
    df = clean_unsafe(df)
    for index, row in df.iterrows():
        outputfilename = standard_outputfilename(row)
        cqpwebtags = '<text id="' + row['text_id'] + '">\n'
        content = cqpwebtags + row['body'] + "\n</text>\n"
        archive.writestr(outputfilename, content)
    archive.close()
    # create an extra column as per CQP web sample file
    df['yearmo'] = df.year + df.numeric_month
    # reorder columns so ones Andrew requires are placed first
    andrewcols = ['text_id', 'shortcode', 'year', 'numeric_month', 'yearmo', 'rownumber']
    df = df[ andrewcols + [ col for col in df.columns if col not in andrewcols]]
    # get rid of the index column & some unnecessary columns
    df = df.loc[:, ~df.columns.str.match('Unnamed')]
    df.drop(['filename', 'encoding','confidence','fullpath','fourdigitcode','body'], axis=1, inplace=True)
    df.to_csv(f'{outputpath}{directoryname}_metadata.csv', index=False)
    # drop the metadata column before going to tsv for cqpweb
    df.drop(['metadata'], axis=1, inplace=True)
    df.to_csv(f'{outputpath}{directoryname}_metadata.tsv', sep='\t', index=False)


def write_corpus_sketchengine(df, cleandatapath, directoryname="corpus-sketchengine"):
    '''
    Writes our corpus with title and body, with tags in the format accepted by sketch engine
    '''
    # Cleans markup that could be unsafe in sgml
    archive = zipfile.ZipFile(f"{cleandatapath}/{directoryname}.zip", "w", zipfile.ZIP_DEFLATED)
    df = clean_unsafe(df)
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
