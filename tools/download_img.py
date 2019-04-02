import re  # 导入正则表达式模块
import requests  # python HTTP客户端 编写爬虫和测试服务器经常用到的模块
import random  # 随机生成一个数，范围[0,1]
import os


if not os.path.isdir('data'):
    os.makedirs('data')


# 定义函数方法
def spider_pic(html, keyword):
    print('正在查找 ' + keyword + ' 对应的图片,下载中，请稍后......')
    for addr in re.findall('"objURL":"(.*?)"', html, re.S):  # 查找URL
        print('正在爬取URL地址：' + str(addr)[0:30] + '...')  # 爬取的地址长度超过30时，用'...'代替后面的内容

        try:
            pics = requests.get(addr, timeout=10)  # 请求URL时间（最大10秒）
        except requests.exceptions.ConnectionError:
            print('您当前请求的URL地址出现错误')
            continue

        fq = open('./data/' + (str(random.randrange(0, 10000, 4)) + '.jpg'), 'wb')  # 下载图片，并保存和命名
        fq.write(pics.content)
        fq.close()


# python的主方法
if __name__ == '__main__':
    word = input('请输入你要搜索的图片关键字：')
    result = requests.get(
        'http://image.baidu.com/search/index?tn=baiduimage&ps=1&ct=201326592&lm=-1&cl=2&nc=1&ie=utf-8&word=' + word)

# 调用函数
spider_pic(result.text, word)
