# coding=gbk
'''
Created on 2020年1月30日

@author: sunba
'''

import urllib.request
import urllib.parse
import socket
import urllib.error
from urllib import request, parse
from urllib.request import ProxyHandler, build_opener
from pip._vendor.urllib3 import response
from urllib.error import URLError
import http.cookiejar
from Tools.scripts.objgraph import ignore
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

if __name__ == '__main__':
    print('-----Note---------------')
#     response = urllib.request.urlopen('https://www.baidu.com')
#     print(response.read().decode('utf-8'))
#     print(type(response))
#     print(response.status)
#     print(response.getheaders())

#     data = bytes(urllib.parse.urlencode({'word': 'hello'}), encoding='utf8')
#     response = urllib.request.urlopen('http://httpbin.org/post', data = data)
#     print(response.read().decode('utf-8'))
    
#     try:
#         response = urllib.request.urlopen('http://httpbin.org/get', timeout = 0.1)
#     except urllib.error.URLError as e:
#         if isinstance(e.reason, socket.timeout):
#             print('TIMEOUT')

#     url = 'http://httpbin.org/post'
#     headers = {
#         'User-Agent': 'Mozilla/4.0 (compatible; MSIE S. S; Windows NT)',
#         'Host': 'httpbin.org'
#     }
#     dict = {
#         'name': 'Germey'
#     }
#     data = bytes(parse.urlencode(dict), encoding = 'utf8')
#     req = request.Request(url = url, data = data, headers = headers, method = 'POST')
#     response = request.urlopen(req)
#     print(response.read().decode('utf-8'))

#     proxy_handler = ProxyHandler({
#         'http': 'http://127.0.0.1:9743',
#         'https': 'https://127.0.0.1:9743'
#     })
#     opener = build_opener(proxy_handler)
#     
#     try:
#         response = opener.open('https://www.baidu.com')
#         print(response.read().decode('utf-8'))
#     except URLError as e:
#         print(e.reason)

#     filename = 'cookies.txt'
#     cookie = http.cookiejar.MozillaCookieJar(filename)
#     handler = urllib.request.HTTPCookieProcessor(cookie)
#     opener = build_opener(handler)
#     response = opener.open('https://www.baidu.com')
#     cookie.save(ignore_discard=True, ignore_expires=True)
#     
#     for item in cookie:
#         print(item.name + " = " + item.value)

#     url = 'http://www.baidu.com/index.html;user?id=5#comment'
#     result = urlparse(url)
#     print(result)

    rp = RobotFileParser()
    rp.set_url('http://www.jianshu.com/robots.txt')
    rp.read()
    print(rp.can_fetch('*', 'http://www.jianshu.com/p/b67554025d7d'))
    print(rp.can_fetch('*', 'http://www . jianshu.com/search?q=python&page=l&ty pe=collections'))
            
    
    print('------------------------')