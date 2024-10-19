import os
import re
from concurrent.futures import ThreadPoolExecutor
from mmpi import mmpi  # 确保已经导入你所使用的邮件解析库
from tqdm import tqdm  # 需要安装tqdm库
from threading import Lock  # 用于线程安全写入文件

# 去掉非中文字符
def clean_str(string):
    if isinstance(string, bytes):
        string = string.decode('utf-8')  # 根据你的数据使用合适的编码方式
    string = re.sub(r"[^\u4e00-\u9fff]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

# 从指定路径读取邮件文件内容信息
def get_data_in_a_file(emp, original_path):
    emp.parse(original_path)
    report = emp.get_report()
    # 如果可以解析到邮件头信息
    if report.get('headers') is not None:
        return clean_str(report['headers'][0]['Subject'])
    return None

# 线程安全写入文件
def write_to_file(output_file, text, label, lock):
    with lock:
        with open(output_file, "a+", encoding='utf-8') as f:
            f.write(f"{text}\t{label}\n")

# 处理单个邮件
def process_email(line, output_file, lock, emp):
    if not line.strip():
        return  # 直接返回，不处理空行
    str_list = line.split(" ")
    # 设置垃圾邮件的标签为0
    if str_list[0] == 'spam':
        label = '0'
    # 设置正常邮件标签为1
    elif str_list[0] == 'ham':
        label = '1'
    else:
        return None  # 如果标签不明则返回None
    try:
        text = get_data_in_a_file(emp, 'trec06c/full/' + str_list[1].strip())
        if text is not None:
            write_to_file(output_file, text, label, lock)  # 实时写入文件
        return (text, label) if text is not None else None
        # 继续处理逻辑
    except Exception as e:
        print(f"Error processing line: {line}, error: {str(e)}")


# 主函数
def main():
    output_file = "all_email.txt"
    lock = Lock()  # 创建一个锁对象，确保线程安全

    # 创建 mmpi 实例
    emp = mmpi()

    # 读取标签文件信息
    with open('trec06c/full/index', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 使用线程池并行处理邮件，设置128个线程
    with ThreadPoolExecutor(max_workers=64) as executor:
        # 使用tqdm显示进度条
        list(tqdm(executor.map(lambda line: process_email(line, output_file, lock, emp), lines), total=len(lines), desc="Processing Emails"))

if __name__ == "__main__":
    main()
