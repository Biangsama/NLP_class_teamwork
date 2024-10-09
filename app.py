from flask import Flask
from flask import request, redirect, url_for, session,flash
from flask_mysqldb import MySQL
import pymysql
import MySQLdb.cursors
import re
import socket
import struct
import threading
import numpy as np
import time
from flask import Flask, render_template, Response

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return render_template('home.html')


if __name__ == '__main__':
    app.run()
