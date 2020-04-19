# coding=gbk
'''
Created on 2020年1月30日

@author: sunba
'''

import requests
import re

print('-----Note---------------')

# r = requests.get('https://www.baidu.com/')
# print(type(r))
# print(r.status_code)
# print(type(r.text))
# print(r.cookies)
# print(r.text)

# data = {
#     'name': 'germey',
#     'age': 22
# }
# r = requests.get('http://httpbin.org/get', params=data)
# print(r.text)


# headers = {
#     'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac 0S X 10_11_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36'
# }
# r = requests.get('https://www.zhihu.com/explore', headers = headers)
# pattern = re.compile('explore-feed.*?question_link.*?>(.*?)＜ /a>', re.S)
# titles = re.findall(pattern, r.text)
# print(titles)

# r = requests.get('https://github.com/favicon.ico')
# with open('favicon.ico', 'wb') as f:
#     f.write(r.content)

# data = {'name' : 'germey', 'age' : 22}
# r = requests.post('http://httpbin.org/post', data = data)
# print(r.text)

response = requests.get('https://www.12306.cn')
print(response.status_code)

print('------------------------')