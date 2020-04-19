# coding=gbk
'''
Created on 2020年1月31日

@author: sunba
'''

import urllib
from urllib.request import urlopen
from bs4 import BeautifulSoup
import datetime
import random



def getLinks(url):
    html = urlopen('https://baike.baidu.com/{}'.format(urllib.parse.quote(url)))
    bs = BeautifulSoup(html, 'html.parser')
    
    links = bs.find_all(lambda tag : 'href' in tag.attrs)
    
    return links

if __name__ == '__main__':
    random.seed(datetime.datetime.now())
    links = getLinks('item/2020年新型冠状病毒疫情')
    while len(links) > 0:
        newUrl = links[random.randint(0, len(links)-1)].attrs['href']
        print(newUrl)
        links = getLinks(newUrl)
