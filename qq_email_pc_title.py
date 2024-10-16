import time,re,requests
from selenium import webdriver
from selenium.webdriver.common.by import By

#登陆163网站
mail_url='https://mail.163.com/'
date_list=[]

#用户名和密码，用户名不需要加@163.com
email='18874812697'
password='Zby147258'


driver = webdriver.Chrome()
driver.get(mail_url)

# 因为登录位置处于iframe中,所以要切换进去
iframe=driver.find_element(By.TAG_NAME, "iframe")
driver.switch_to.frame(iframe)

#selenium模拟登录
driver.find_element(By.NAME, "email").send_keys(email)
driver.find_element(By.NAME, "password").send_keys(password)
driver.find_element(By.ID, "dologin").click()
time.sleep(5)

driver.switch_to.default_content()  # 切回默认
time.sleep(4)

# 获取cookies
cookies_list = driver.get_cookies()

'''
# 打印 cookies
for cookie in cookies_list:
    print(cookie)
'''

# 将 cookies 转换为字典
cookies = {}
for cookie in cookies_list:
    cookies[cookie['name']] = cookie['value']
#print(cookies)

# 获取当前页面的HTML内容
html_content = driver.page_source

#处理数据内容（re），提取sid，获取所有匹配项并保存到列表中
pattern = r",sid:'(.*?)',"
sid = re.findall(pattern, html_content)
#print(sid)
#-------------------------------------------#
#使用requests的post请求获取收件箱的邮件
url='https://mail.163.com/js6/s'

#Query String Parameters参数
query_params = {
    "sid": sid[0],
    "func": "mbox:listMessages"
}

#Form Data参数：var
form_data = {
    "var": '<?xml version="1.0"?><object><int name="fid">1</int><string name="order">date</string><boolean name="desc">true</boolean><int name="limit">15</int><int name="start">0</int><boolean name="skipLockedFolders">false</boolean><string name="topFlag">top</string><boolean name="returnTag">true</boolean><boolean name="returnTotal">true</boolean></object>'
}

#请求头的其他参数
headers={
            'Accept': "text/javascript",
            'Accept-Language': "zh-CN,zh;q=0.9",
            'Origin': "http://mail.163.com/",
            'Referer': "http://mail.163.com/js6/main.jsp?sid="+sid[0]+"&df=mail163_letter",#https://mail.163.com/js6/main.jsp?sid=CDcsCuHHEgrIOlgECiHHBCyAVPrmbvLH&df=mail163_letter
            'User-Agent':"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
}

#requests的post请求代入params，data，headers和cookies
response = requests.post(url, params=query_params, data=form_data,headers=headers,cookies=cookies)
response.close()
#-------------------------------------------#

# 处理响应结果
if response.status_code == 200:
    pattern_mail =r"{.*?'from':'\"(.*?)\" <(.*?)>'.*?'subject':'(.*?)'.*?'hmid':'<.*?>'}"#r"{.*?,'from':'\"(.*?)\" <(.*?)>'.*?'subject':'(.*?)'.*?'hmid':'<.*?>'}"# r'"title":"(.+?)","focus_date":".+?","url":"(.+?)","image":.+?,"brief":"(.+?)"'
    matched_mails = re.findall(pattern_mail, response.text,re.DOTALL)  # 获取所有匹配项并保存到列表中
    for mail in matched_mails:
        print("发件人：",mail[0])
        print("发件人邮箱：",mail[1])
        print("邮件内容：",mail[2])
        print('\r\n')
    #print(response.text)
else:
    print("请求失败:", response.status_code, response.reason)

# 关闭浏览器
driver.quit()
