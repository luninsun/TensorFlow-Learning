# coding=gbk
'''
Created on 2020年1月31日

@author: sunba
'''

from urllib.request import urlopen
from bs4 import BeautifulSoup


print('-----Note---------------')

html = urlopen('http://www.pythonscraping.com/pages/warandpeace.html')
bs = BeautifulSoup(html.read(), 'html.parser')

nameList = bs.find_all('span', {'class' : 'green'})

for name in nameList:
#     print(name.get_text())
    pass

html = urlopen('http://www.pythonscraping.com/pages/page3.html')
bs = BeautifulSoup(html.read(), 'html.parser')

# for child in bs.find('table', {'id' : 'giftList'}).descendants:
#     print(child)

for sibing in bs.find('table', {'id' : 'giftList'}).tr.next_siblings:
    print(sibing)