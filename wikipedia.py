# coding=gbk
'''
Created on 2020年1月31日

@author: sunba
'''

from urllib.request import urlopen
from bs4 import BeautifulSoup

html = urlopen('https://baike.baidu.com/')

bs = BeautifulSoup(html, 'html.parser')

for link in bs.find_all('a'):
    if 'href' in link.attrs:
        print(link.attrs['href'])
