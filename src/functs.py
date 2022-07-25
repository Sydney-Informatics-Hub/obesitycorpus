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
import unicodedata
import string
from nltk.tokenize import RegexpTokenizer
import html

def obesitylist(*args):
    # nb: obesogen will also pick up obesogenic
    mylist = ["obesity","obesity's", "obese", "obesogen"]
    for x in args:
        mylist.append(x)
    return mylist

def source_mapping():
    source_to_abbr = {
    "HeraldSun": "HS",
    "SydHerald": "SM",
    "Advertiser": "AD",
    "CourierMail": "CM" ,
    "Age": "AG",
    "CanTimes": "CT",
    "Australian": "AU",
    "WestAus": "WA",
    "HobMercury": "HM",
    "Telegraph": "TE",
    "NorthernT": "NT",
    "BrisTimes": "BT"
    }
    return source_to_abbr

def abbreviate_source(source):
    source_to_abbr = source_mapping()
    return source.map(source_to_abbr).fillna("Missing")

def expand_source(shortsource):
    abbr_to_source = {v: k for k, v in source_mapping().items()}
    return shortsource.map(abbr_to_source).fillna("Missing")

def get_record_from_corpus_df(corpusdf, source, year, orinummonth, fourdigitcode):
    '''
    Returns a key/value dict for each column of the pandas dataframe that matches the filtering
    '''
    return corpusdf.query("source == @source and year == @year and original_numeric_month == @orinummonth and fourdigitcode == @fourdigitcode").to_dict()

def get_record_by_article_id(corpusdf, article_id):
    '''
    Returns a key/value dict for each column of the pandas dataframe that matches the filtering
    '''
    return corpusdf[(corpusdf['article_id'] == article_id)].to_dict('records')

def get_body_from_article_id(corpusdf, article_id):
    '''
    Returns a key/value dict for each column of the pandas dataframe that matches the filtering
    '''
    return corpusdf[(corpusdf['article_id'] == article_id)]['body'].values[0]


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

def remove_australian_authordeets(body):
    '''
    The Australian has extra text at the end of the body
    deliniated by 1-2 "____" lines
    ex. \n______________________________\n>> Christen Pears is a personal
    trainer and pilates instructor in Western Australia.
    This keeps only everything before the first of these in the corpus
    '''
    # actually continues with content
    if body.split('\n______________________________\n', 1)[1][0:12] == ">> NEXT WEEK":
        return body
    else:
        return body.split('\n______________________________\n', 1)[0]

def clean_couriermail_talktous(bodytext):
    '''
    The courier mail provides many lines of
    contact details at the end of it's talk to us session. Remove these.
    '''
    bodytext = bodytext.split('TALK TO US', 1)[0]
    # remove additional courier mail requests for feedback
    bodytext = re.sub(r'\nWhat do you think\? Email yournews@thesundaymail.com.au or write to us at GPO Box 130, Brisbane, 4001.', '', bodytext)
    bodytext = re.sub(r'\nWhat do you think\? Email yournews@thesundaymail .com.au or write to us at GPO Box 130, Brisbane, 4001.', '', bodytext)
    bodytext = re.sub(r'\nWhat do you think\? Email yournews@thesundaymail.com.au', '', bodytext)
    return bodytext

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
        return int(wordcount_specified.group(1))
    else:
        return None

def count_words(text):
    '''
    My way of counting words. Gets rid of punctuation and counts numbers.
    Hyphenated/contracted words are counted as one word.
    '''
    # remove punctuation
    # from "It's my life today 2 - wohoo joy-ya obesity's U.S." to 
    # Its my life today 2 wohoo joyya obesitys US
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    # 9 for the above string
    return len(tokens)

def sum_all_keywords(text, keywordslist):
    sum = 0
    for keyword in keywordslist:
        sum += text.lower().count(keyword)
    return sum

def standard_outputfilename(row):
    return f"{row.source}_{row.year}_{row.original_numeric_month}_{row.fourdigitcode}_{make_slug(row.title)}.txt"

def parse_filename(path):
    source, year, month = path.split('/')[2].split("_")
    month = month.replace("txt", "").strip()
    original_numeric_month = convert_month(month)
    return pd.Series({
        "source": source,
        "year": year,
        "original_numeric_month": original_numeric_month
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
                             "\\xE2Ä\"": "-", "\xE2Äò": "\"", "\\xE2€(tm)": "'", "\\xE2€": "'", 
                             "x{2002}":" "}
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

def clean_quotes(mytext):
    '''
    Cleans up quotes in body or title
    '''
    # manually checked 3 quotes are used instead of two in the corpus
    mytext = mytext.replace('```', '"')
    # two open quotes
    mytext = mytext.replace('``', '"')
    # two close quotes
    mytext = mytext.replace("''", "\"")
    # single quote
    mytext = mytext.replace("`", "'")
    return mytext

def clean_wa(mytext):
    # replaces some odd tags in the WA
    # replacing ";  -----QUOTE----" and ";  -----info box----" detected in the title of some of the WA articles
    mytext = re.sub(';  -----QUOTE----', '', mytext, flags=re.IGNORECASE)
    mytext = re.sub(';  -----info box----', '', mytext, flags=re.IGNORECASE)
    return mytext

def clean_quot(mytext):
    '''
    Deals with &quot; strings in body or metadata
    '''
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
    # herald sun
    bodytext = re.sub(r'\nheraldsun.com.au\n', '', bodytext)
    return bodytext

def replace_six_questionmarks(column):
    '''
    Replaces 6 ?????? with a double quote
    '''
    # [^?](\?){6}[^?] TODO 
    mytext = re.sub(r'[^?](\?){6}[^?]', '"', column)
    return mytext

def replace_triple_quote(column):
    '''
    Replaces """ and "" with "
    '''
    mytext = re.sub(r'"""', '"', column)
    mytext = re.sub(r'""', '"', mytext)
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
    archive = zipfile.ZipFile(f"{str(cleandatapath)}/{directoryname}.zip", "w", zipfile.ZIP_DEFLATED)
    for index, row in df.iterrows():
        outputfilename = standard_outputfilename(row)
        content = row['title'] + "\n" + row['body']
        archive.writestr(outputfilename, content)
    archive.close()

def write_corpus_articleid(df, cleandatapath, directoryname="corpus-articleid"):
    '''
    Writes our corpus with title and body, without any tags or metadata, with the filename as article_id
    '''
    archive = zipfile.ZipFile(f"{str(cleandatapath)}/{directoryname}.zip", "w", zipfile.ZIP_DEFLATED)
    for index, row in df.iterrows():
        outputfilename = row['article_id'] + ".txt"
        content = row['title'] + "\n" + row['body']
        archive.writestr(outputfilename, content)
    archive.close()

def write_corpus_nested(df, cleandatapath, directoryname="corpus-nested"):
    '''
    Writes our corpus with title and body, nested by source/year/month
    '''
    for index, row in df.iterrows():
        outputdir = str(cleandatapath) + "/" + directoryname + f"/{row.source}/{row.year}/{row.month_metadata}/"
        outputfilename = outputdir + standard_outputfilename(row)
        os.makedirs(os.path.dirname(outputdir), exist_ok=True)
        content = row['title'] + "\n" + row['body']
        f = open(outputfilename, 'w', encoding='utf-8')
        f.write(content)
        f.close()

def clean_sgml(df):
    """
    Cleans markup that could be unsafe in sgml
    """
    for field in ['title', 'body']:
        df[field] = df[field].apply(lambda x:html.escape(x, quote=False))
    return df

def remove_quote_fill_none(text):
    if text is None:
        return "Unknown"
    else:
        return re.sub("'|\"", "", text)

def write_corpus_cqpweb(inputdf, cleandatapath, directoryname="corpus-cqpweb", write_actual_files=True):
    '''
    Writes our corpus and metadata as per CQP web sample format
    '''
    
    df = inputdf.copy()
    outputpath = str(cleandatapath) + "/"
    if write_actual_files:
        archive = zipfile.ZipFile(f"{outputpath}{directoryname}.zip", "w", zipfile.ZIP_DEFLATED)
    # Cleans markup that could be unsafe in sgml
    df = clean_sgml(df)
    if write_actual_files:
        for index, row in df.iterrows():
            outputfilename = standard_outputfilename(row)
            cqpwebtags = '<text id="' + row['article_id'] + '">\n'
            content = cqpwebtags + '<head>' + row['title'] + '</head>\n<body>\n' + row['body'] + "\n</body>\n</text>\n"
            archive.writestr(outputfilename, content)
        archive.close()
    # create an extra column as per CQP web sample file
    # use the real date here
    df['yearmo'] = df.year + df.month_metadata
    # reorder columns so ones Andrew requires are placed first
    andrewcols = ['article_id', 'shortcode', 'year', 'month_metadata', 'yearmo', 'rownumber']
    df = df[ andrewcols + [ col for col in df.columns if col not in andrewcols]]
    # get rid of the index column & some unnecessary columns
    df = df.loc[:, ~df.columns.str.match('Unnamed')]
    df.drop(['fourdigitcode','body', 'hash', 'matched_list', 'jaccards'], axis=1, inplace=True)
    df.to_csv(f'{outputpath}{directoryname}_metadata.csv', index=False)
    # replace all of the single and double quotes to enable cqpweb import
    df['byline'] = df.apply(lambda x: remove_quote_fill_none(x.byline), axis = 1)
    # drop a large number of unsupported columns before going to tsv for cqpweb
    df.drop(
        ['metadata', 'original_numeric_month', 'title','shortcode', 'rownumber', 'percent_contribution', 'first_sent'],
        axis=1, inplace=True)
    df.drop(list(df.filter(regex = 'count')), axis = 1, inplace = True)
    df.drop(list(df.filter(regex = 'keywords')), axis = 1, inplace = True)
    # cqpweb can't handle dashes so replacing with underscores
    df = df.replace('-','_', regex=True)
    df['date'] = df.apply(lambda x: x.date.strftime("%Y_%m_%d"), axis=1)
    if write_actual_files:
        df.to_csv(f'{outputpath}{directoryname}_metadata.tsv', sep='\t', index=False)


def write_corpus_sketchengine(inputdf, cleandatapath, directoryname="corpus-sketchengine"):
    '''
    Writes our corpus with title and body, with tags in the format accepted by sketch engine
    '''
    df = inputdf.copy()
    # Cleans markup that could be unsafe in sgml
    archive = zipfile.ZipFile(f"{str(cleandatapath)}/{directoryname}.zip", "w", zipfile.ZIP_DEFLATED)
    df = clean_sgml(df)
    for index, row in df.iterrows():
        outputfilename = standard_outputfilename(row)
        sketchenginetags = '<doc date="' + row['date'].strftime("%Y-%m-%d") + '" publication="' + row['source'] + '" wordcountTotal="' + str(row['wordcount_total']) + '">'
        content = sketchenginetags + "\n<head>" + row['title'] + "</head>\n<body>\n" + row['body'] + "\n</body>\n</doc>"
        archive.writestr(outputfilename, content)
    archive.close()

def write_corpus_summary_tables(corpusdf, cleandatapath, articlecounts_name="articlecounts", wordcounts_name="wordcounts"):
    # generate summary of number of articles by source per year
    corpusdf.groupby(['source', 'year']).agg({ 'article_id':'count'}).unstack().fillna(0).to_csv(cleandatapath/f'{articlecounts_name}.csv')
    # generate summaries of word counts by corpus
    corpusdf.groupby(['source', 'year']).agg({ 'wordcount_total':'sum'}).unstack().fillna(0).to_csv(cleandatapath/f'{wordcounts_name}.csv')

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
