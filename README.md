# Obesity project





## Some notes on analysis

1. The following command revealed that the easiest way to get rid of "empty" files and list search results was to remove rows without the word "body" (case-insensitive), and then to split on this word.

```sh
grep -Lri "body" .
# ./The Age/Age_2013/Age_2013_Julytxt/Age (1).txt
# ./Herald Sun/HeraldSun_2011/HeraldSun_2011_Februarytxt.zip
# ./Hobart Mercury/HobMercury_2008/HobMercury_2008_Octobertxt/0001txt/HobMercury (1).txt
# ./Hobart Mercury/HobMercury_2008/HobMercury_2008_Maytxt/0001txt/HobMercury (1).txt
# ./Hobart Mercury/HobMercury_2008/HobMercury_2008_Septembertxt/0001txt/HobMercury (1).txt
# ./Hobart Mercury/HobMercury_2008/HobMercury_2008_Novembertxt/0001txt/HobMercury (1).txt
# ./Hobart Mercury/HobMercury_2008/HobMercury_2008_Junetxt/0001txt/HobMercury (1).txt
# ./Hobart Mercury/HobMercury_2008/HobMercury_2008_Julytxt/0001txt/HobMercury (1).txt
```

2. Can split by `\nBody\n`

3. Byline is meaningful only in the first chunk of text, in the second it occurs only twice, when mentioned as actual byline-chasin journalists and journalistic byline. So we can extract them by collecting only the first match of byline in the text.

3. 