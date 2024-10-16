import imapclient
import pyzmail
import pprint

#ACCOUNT = 'm18874812697@163.com'
#PASSWORD = 'MUsFD4baurHpuD6d'

# 登录服务器
imapObj = imapclient.IMAPClient('imap.163.com', ssl=True)
imapObj.login('m18874812697@163.com', 'MUsFD4baurHpuD6d')
# 打印邮件文件夹列表
pprint.pprint(imapObj.list_folders())
# 查询文件夹
imapObj.id_({"name": "IMAPClient", "version": "2.1.0"})
imapObj.select_folder('INBOX', readonly=False)

# 查询标记
UIDs = imapObj.search(['SEEN'])
print(UIDs)
# 获取邮件原始内容
rawMessages = imapObj.fetch([1719043385], ['BODY[]', 'FLAGS'])
#pprint.pprint(rawMessages)

# 删除邮件
# imapObj.delete_messages([1657986860])
# imapObj.expunge()
# 获取电子邮件地址
message = pyzmail.PyzMessage.factory(rawMessages[1719043385][b'BODY[]'])
message.get_subject()
print(message.get_addresses('from'))
print(message.get_addresses('to'))
print(message.get_addresses('cc'))
print(message.get_addresses('bcc'))
# 获取正文
if message.text_part != None:
    print(message.text_part.get_payload().decode(message.text_part.charset))
#if message.html_part != None:
#    message.html_part.get_payload().decode(message.html_part.charset)
# 从服务器断开
imapObj.logout()
