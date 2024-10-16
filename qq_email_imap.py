
import imapclient
import pyzmail
import pprint

# 登录服务器
imapObj = imapclient.IMAPClient('imap.qq.com', ssl=True)
imapObj.login('2925657548@qq.com', 'nrwmqlbpncxjdgeb')

# 打印邮件文件夹列表
pprint.pprint(imapObj.list_folders())

# 查询文件夹
imapObj.id_({"name": "IMAPClient", "version": "2.1.0"})
imapObj.select_folder('INBOX', readonly=True)  # 只读模式

# 查询所有已读邮件的UID
UIDs = imapObj.search(['SEEN'])
print(f'找到的邮件UID: {UIDs}')

# 遍历每个UID，获取邮件信息
for uid in UIDs:
    # 获取邮件原始内容和接收时间
    rawMessages = imapObj.fetch([uid], ['BODY[]', 'FLAGS', 'INTERNALDATE'])

    # 使用pyzmail解析邮件
    message = pyzmail.PyzMessage.factory(rawMessages[uid][b'BODY[]'])

    # 打印邮件主题
    subject = message.get_subject()
    print(f'邮件主题: {subject}')

    # 打印发件人、收件人、抄送和密送地址
    print(f'发件人: {message.get_addresses("from")}')
    print(f'收件人: {message.get_addresses("to")}')
    print(f'抄送: {message.get_addresses("cc")}')
    print(f'密送: {message.get_addresses("bcc")}')

    # 获取并打印邮件正文
    if message.text_part:
        print(f'正文: {message.text_part.get_payload().decode(message.text_part.charset)}')
    #if message.html_part:
    #    print(f'HTML 内容: {message.html_part.get_payload().decode(message.html_part.charset)}')

    # 获取邮件接收时间
    internal_date = rawMessages[uid][b'INTERNALDATE']
    print(f'接收时间: {internal_date}')

    print('-' * 40)  # 分隔线，便于阅读

# 从服务器断开
imapObj.logout()
