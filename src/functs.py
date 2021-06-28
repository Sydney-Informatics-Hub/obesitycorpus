import datetime
from bs4 import UnicodeDammit

def readfilesin(file_path, encoding):
    if encoding in ['ascii', 'Windows-1252', 'ISO-8859-1']:
        with open(file_path, encoding='Windows-1252') as file:
            data = file.read().replace("\r", "")
    elif encoding == 'utf-8':
        with open(file_path, encoding='utf-8') as file:
            data = file.read().replace("\r", "")
    else:
        try:
            with open(file_path, 'rb') as non_unicode_file:
                content = non_unicode_file.read(1024)
                dammit = UnicodeDammit(content, ['Windows-1252'])
                data = dammit.unicode_markup.replace("\r", "")
        except Exception as e:
            raise ValueError('Can\'t return dictionary from empty or invalid file %s due to %s' % (file_path, e))
    return(data)

def convert_month(month):
    if month in ["Jan", "Feb", "Mar", "Apr", "May", 
    "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]:
        return(month)
    elif month in ["January", "February", "March", "April", "May", 
    "June", "July", "August", "September", "October", "November", "December"]:
        return(datetime.datetime.strptime(month, "%B").strftime("%b"))
    else:
        print("The following month name is not valid: '", month, "'")

def where_is_byline(contentslist):
    tmplist = [x.lower().find('byline') for x in contentslist]
    return([i for i, x in enumerate(tmplist) if x != -1])

def where_is_body(contentslist):
    tmplist = [x.lower().find('body') for x in contentslist]
    return([i for i, x in enumerate(tmplist) if x != -1])


        


# @pytest.mark.parametrize("non_palindrome", [
#     "abc",
#     "abab",
# ])
# def test_is_palindrome_not_palindrome(non_palindrome):
#     assert not is_palindrome(non_palindrome)
