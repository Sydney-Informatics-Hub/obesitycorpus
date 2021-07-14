import datetime
from bs4 import UnicodeDammit
import re
import spacy
from spacy.tokens import Span

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
                content = non_unicode_file.read(1024)
                dammit = UnicodeDammit(content, ['Windows-1252'])
                data = dammit.unicode_markup
        except Exception as e:
            raise ValueError('Can\'t return dictionary from empty or invalid file %s due to %s' % (file_path, e))
    return(data.replace("\r", "").replace("\nClassification\n\n\n", "").strip())

def convert_month(month):
    if month in ["Jan", "Feb", "Mar", "Apr", "May", 
    "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]:
        return(datetime.datetime.strptime(month, "%b").strftime("%m"))
    elif month in ["January", "February", "March", "April", "May", 
    "June", "July", "August", "September", "October", "November", "December"]:
        return(datetime.datetime.strptime(month, "%B").strftime("%m"))
    else:
        print("The following month name is not valid: '", month, "'")

def where_is_byline(contentslist):
    tmplist = [x.lower().find('byline') for x in contentslist]
    return([i for i, x in enumerate(tmplist) if x != -1])

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

def cleantitler(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespace with a single dash
    s = re.sub(r"\s+", '-', s)
    # truncate ultra-long titles
    s = s[:200] if len(s) > 200 else s
    return(s)


def explore_tokens(sentencenlp_list):
    sentencesummarylist = []
    for sentence in sentencenlp_list:
        # displacy.serve(sentence, style="dep")
        tokensummarylist = []
        for token in sentence:
            if token.lemma_ in obesitynames:
                mydict = {
                    'text' : token.text,
                    'tag' : token.tag_,
                    'dep' : token.dep_,
                    'head' : token.head,
                    'left': token.left_edge,
                    'right': token.right_edge}
                tokensummarylist.append(mydict)
        sentencesummarylist.append(tokensummarylist)
    return(sentencesummarylist)



byline_replacementlist = [
    "Exclusive ",
 " And ",
  # jobs
 "National",
 "Correspondent",
 "Political",
 "Health " ,
 "Political",
 "Education" ,
 "Commentator",
 "Regional",
 "Agencies",
 "Defence",
 "Fashion",
 "Music",
 "Social Issues",
 "Reporter",
 "Chief",
 "Business",
 "Workplace",
 "Editor",
 "Indulge",
 "Science",
 "Sunday",
 "Saturdays",
 "Writer",
 "Food ",
 "Dr ",
 "Professor ",

 # cities,
 "Las Vegas",
 "Melbourne",
 "Canberra",
 "Brisbane",
 "Sydney",
 "Perth",
 "Adelaide",
 "Chicago",
 "Daily Mail, London" ,
 "London Correspondent",
 "London Daily Mail",
 "Buenos Aires" ,
 "New York",
 "New Delhi",
 "London",
 ", Washington",
 "Beijing",
 "Health",
 # random stuff
 "State ",
 "Council On The Ageing",
 "Words ",
 "Text",
 "By ",
 "With ",
 " In ",
 " - ",
 ":",
 "Proprietor Realise Personal Training Company",
 "B+S Nutritionist",
 "My Week",
 "Medical",
 "Manufacturing",
 "Brought To You",
 "Film ",
 "Reviews ",
 "Comment",
 "Personal",
 "Finance",
 "Natural ",
 "Solutions ",
 "Special ",
 "Report ",
 "Recipe ",
 "Photography ",
 "Photo",
 "-At-Large",
 "Styling ",
 "Preparation ",
 # individual,
 " - Ian Rose Is A Melbourne Writer." ,
 "Cycling Promotion Fund Policy Adviser ",
 ". Dannielle Miller Is The Head Of Enlighten Education. Enlighten Works With Teenage Girls In High Schools On Developing A Positive Self-Esteem And Healthy Body Image." ]