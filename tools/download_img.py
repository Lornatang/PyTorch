import os
import random
import re

import requests

if not os.path.isdir('data'):
  os.makedirs('data')


def spider_pic(html, keyword):
  print('Looking for ' + keyword + ' Corresponding picture, download, '
                                   'please later. ')
  # find url address
  for address in re.findall('"objURL":"(.*?)"', html, re.S):
    # crawl addresses longer than 30 times, using'.' Replace what follows
    print('Crawling over URL addresses：' + str(address)[0:30] + '...')

    try:
      pics = requests.get(address, timeout=10)  # 请求URL时间（最大10秒）
    except requests.exceptions.ConnectionError:
      print('Error occurred on your current request URL address')
      continue

    fq = open('./data/' + (str(random.randrange(0, 10000, 4)) + '.jpg'), 'wb')  # 下载图片，并保存和命名
    fq.write(pics.content)
    fq.close()


if __name__ == '__main__':
  word = input('Please enter your search keywords:')
  result = requests.get(
    'http://image.baidu.com/search/index?tn=baiduimage&ps=1&ct=201326592&lm=-1&cl=2&nc=1&ie=utf-8&word=' + word)

  # call function
  spider_pic(result.text, word)
