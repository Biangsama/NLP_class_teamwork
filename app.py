from flask import Flask, render_template, request, jsonify
import imapclient
import pyzmail
import pprint
import chardet

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
        # 登录服务器
        imapObj = imapclient.IMAPClient(imap_server, ssl=True)
        imapObj.login(email, password)

        # 选择INBOX文件夹
        imapObj.id_({"name": "IMAPClient", "version": "2.1.0"})
        imapObj.select_folder('INBOX', readonly=True)

        # 查询所有已读邮件的UID
        UIDs = imapObj.search(['SEEN'])
        emails = []

        # 遍历每个UID，获取邮件信息
        for uid in UIDs:
            rawMessages = imapObj.fetch([uid], ['BODY[]', 'INTERNALDATE'])
            message = pyzmail.PyzMessage.factory(rawMessages[uid][b'BODY[]'])

            # 获取邮件正文
            email_body = None
            if message.text_part:
                raw_payload = message.text_part.get_payload()
                detected = chardet.detect(raw_payload)
                charset = detected['encoding'] if detected['encoding'] else 'utf-8'
                try:
                    email_body = raw_payload.decode(charset, errors='replace')
                except Exception:
                    # 如果 chardet 检测失败，尝试使用其他常见编码
                    try:
                        email_body = raw_payload.decode('utf-8', errors='replace')
                    except Exception:
                        email_body = raw_payload.decode('latin-1', errors='replace')
            elif message.html_part:
                raw_payload = message.html_part.get_payload()
                detected = chardet.detect(raw_payload)
                charset = detected['encoding'] if detected['encoding'] else 'utf-8'
                try:
                    email_body = raw_payload.decode(charset, errors='replace')
                except Exception:
                    # 如果 chardet 检测失败，尝试使用其他常见编码
                    try:
                        email_body = raw_payload.decode('utf-8', errors='replace')
                    except Exception:
                        email_body = raw_payload.decode('latin-1', errors='replace')

            email_info = {
                'sender': message.get_addresses("from"),
                'subject': message.get_subject(),
                'date': rawMessages[uid][b'INTERNALDATE'].strftime('%Y-%m-%d %H:%M:%S'),
                'body': email_body,
            }
            emails.append(email_info)

        imapObj.logout()
        return jsonify({'emails': emails})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
