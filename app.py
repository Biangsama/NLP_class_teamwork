from flask import Flask, render_template, request, jsonify
import imapclient
import pyzmail

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract_emails', methods=['POST'])
def extract_emails():
    email = request.json['email']
    password = request.json['password']
    email_type = request.json['emailType']

    # 根据邮箱类型设置IMAP服务器
    if email_type == 'qq':
        imap_server = 'imap.qq.com'
    elif email_type == '163':
        imap_server = 'imap.163.com'
    elif email_type == 'gmail':
        imap_server = 'imap.gmail.com'
    elif email_type == 'outlook':
        imap_server = 'outlook.office.com'
    else:
        return jsonify({'error': '不支持的邮箱类型'})

    try:
        print("email:",email)
        print("password:", password)
        # 登录服务器
        imapObj = imapclient.IMAPClient(imap_server, ssl=True)
        imapObj.login(email, password)

        # 选择INBOX文件夹
        imapObj.select_folder('INBOX', readonly=True)

        # 查询所有已读邮件的UID
        UIDs = imapObj.search(['SEEN'])
        print(UIDs)
        emails = []

        # 遍历每个UID，获取邮件信息
        for uid in UIDs:
            rawMessages = imapObj.fetch([uid], ['BODY[]', 'INTERNALDATE'])
            message = pyzmail.PyzMessage.factory(rawMessages[uid][b'BODY[]'])
            print(f'发件人: {message.get_addresses("from")}')
            print(f'收件人: {message.get_addresses("to")}')
            if message.text_part:
                print(f'正文: {message.text_part.get_payload().decode(message.text_part.charset)}')
            email_info = {
                'sender': message.get_addresses("from"),
                'subject': message.get_subject(),
                'date': rawMessages[uid][b'INTERNALDATE'].strftime('%Y-%m-%d %H:%M:%S')
            }
            emails.append(email_info)

        imapObj.logout()
        return jsonify({'emails': emails})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
