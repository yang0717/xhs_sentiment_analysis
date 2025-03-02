import sys
import json
import os
import traceback
import csv
from PyQt5.QtWidgets import (
    QApplication, QWidget, QGridLayout, QPushButton, QLabel, QTextEdit,
    QLineEdit, QMessageBox, QFileDialog, QTableWidgetItem, QTableWidget,
    QSizePolicy, QHBoxLayout, QVBoxLayout, QProgressBar, QSpacerItem
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap,QFont
from wordcloud import WordCloud
from io import BytesIO
from PIL import Image
from loguru import logger
from predict import predict_  # 确保predict_.py在同一目录下且已正确实现
from xhs搜索 import Xhs
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.font_manager import FontProperties

# 文件存储路径
USER_DATA_FILE = 'users.json'

# Logo路径
LOGO_PATH = 'logo.png'  # 请确保logo.png在同一目录下
#cookie_str
cookie_str = 'abRequestId=62e93589-1432-52ef-87c3-925e59b523fc; webBuild=4.46.0; xsecappid=xhs-pc-web; a1=193a0032d2do9z69de9irp2hq4hpsery0sqivcy2g50000310429; webId=e3b8bcb78161edb27a86d1de9d225ff3; websectiga=29098a4cf41f76ee3f8db19051aaa60c0fc7c5e305572fec762da32d457d76ae; sec_poison_id=e7f78449-2fcc-477d-b032-92c2cfefc265; acw_tc=0a0bb4a317335563830492472e5530b6b68df3982afe4d3337778b29fbb1e9; gid=yjq088qqJJ8Dyjq088qJfMSvJfljuKjfdj3FAJkUJxA4x328UMdFvh888qy84Jj88i2f4Yj2; web_session=040069b3f1bd0fd4f202d0b066354b6454432c; unread={%22ub%22:%22672f0dc3000000001a01cbc1%22%2C%22ue%22:%2267483f79000000000601670e%22%2C%22uc%22:28}'


# 加载用户数据
def load_users():
    if not os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, 'w') as f:
            json.dump({}, f)
    with open(USER_DATA_FILE, 'r') as f:
        return json.load(f)

# 保存用户数据
def save_users(users):
    with open(USER_DATA_FILE, 'w') as f:
        json.dump(users, f, indent=4)

class LoginWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('登录')
        self.setGeometry(600, 300, 400, 400)
        self.init_ui()

    def init_ui(self):
        # self.setStyleSheet("background-color: lightgreen;")  # 设置背景色
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(50, 50, 50, 50)  # 设置上下左右的边距
        main_layout.setSpacing(20)  # 设置各控件之间的垂直间距

        # 添加标题标签
        self.label_title = QLabel('小红书情感分析系统', self)
        title_font = QFont()
        title_font.setPointSize(20)  # 设置字体大小
        title_font.setBold(True)  # 设置字体加粗
        self.label_title.setFont(title_font)
        self.label_title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.label_title)

        # 创建一个网格布局用于用户名和密码输入
        form_layout = QGridLayout()
        form_layout.setSpacing(10)  # 设置网格布局中控件之间的间距


        # 设置列伸缩以居中表单
        form_layout.setColumnStretch(0, 1)  # 左侧弹性空间
        form_layout.setColumnStretch(1, 0)  # 标签
        form_layout.setColumnStretch(2, 8)  # 输入框
        form_layout.setColumnStretch(3, 1)  # 右侧弹性空间

        # 用户名标签和输入框
        self.label_username = QLabel('用户名:', self)
        self.input_username = QLineEdit(self)
        form_layout.addWidget(self.label_username, 0, 1, alignment=Qt.AlignRight)
        form_layout.addWidget(self.input_username, 0, 2, alignment=Qt.AlignLeft)

        # 密码标签和输入框
        self.label_password = QLabel('密码:', self)
        self.input_password = QLineEdit(self)
        self.input_password.setEchoMode(QLineEdit.Password)  # 密码隐藏显示
        form_layout.addWidget(self.label_password, 1, 1, alignment=Qt.AlignRight)
        form_layout.addWidget(self.input_password, 1, 2, alignment=Qt.AlignLeft)

        # 将表单布局包裹在水平布局中，以实现居中
        form_outer_layout = QHBoxLayout()
        form_outer_layout.addStretch()
        form_outer_layout.addLayout(form_layout)
        form_outer_layout.addStretch()

        main_layout.addLayout(form_outer_layout)

        # 添加按钮的水平布局
        button_layout = QHBoxLayout()
        button_layout.setSpacing(20)  # 设置按钮之间的水平间距

        # 登录按钮
        self.button_login = QPushButton('登录', self)
        self.button_login.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        button_layout.addWidget(self.button_login)

        # 注册按钮
        self.button_register = QPushButton('注册', self)
        self.button_register.setStyleSheet("""
            QPushButton {
                background-color: #008CBA;
                color: white;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #007bb5;
            }
        """)
        button_layout.addWidget(self.button_register)

        main_layout.addLayout(button_layout)

        # 添加Logo
        self.add_logo(main_layout)
        # 添加弹性空间以确保内容居中
        main_layout.addStretch()

        self.setLayout(main_layout)


        # 连接按钮点击事件
        self.button_login.clicked.connect(self.handle_login)
        self.button_register.clicked.connect(self.open_register_window)

    def add_logo(self, layout):
        if os.path.exists(LOGO_PATH):
            self.logo_label = QLabel(self)
            pixmap = QPixmap(LOGO_PATH).scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)  # 调整Logo大小
            self.logo_label.setPixmap(pixmap)
            self.logo_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(self.logo_label, alignment=Qt.AlignCenter)
        else:
            print(f"Logo file not found: {LOGO_PATH}")

    def handle_login(self):
        username = self.input_username.text()
        password = self.input_password.text()

        if not username or not password:
            QMessageBox.warning(self, '警告', '请输入用户名和密码')
            return

        users = load_users()
        print(f"已加载用户数据: {users}")  # 调试语句

        # 这里使用硬编码的用户名和密码进行验证，可以根据需要修改为从文件或数据库读取
        if username in users and users[username] == password:
            print("登录成功，跳转到功能选择界面")  # 调试语句
            self.open_function_selection_window()
        else:
            print("登录失败")  # 调试语句
            QMessageBox.warning(self, '错误', '用户名或密码错误')

    def open_function_selection_window(self):
        print("打开功能选择界面")  # 调试语句
        self.function_selection_window = FunctionSelectionWindow()
        self.function_selection_window.show()
        self.close()  # 关闭登录窗口

    def open_register_window(self):
        self.register_window = RegisterWindow()
        self.register_window.show()
        self.close()  # 关闭登录窗口



class RegisterWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('注册')
        self.setGeometry(600, 300, 400, 450)  # 调整窗口大小以适应新布局
        self.init_ui()

    def init_ui(self):
        # 创建主垂直布局
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(50, 50, 50, 50)  # 设置上下左右的边距
        main_layout.setSpacing(20)  # 设置各控件之间的垂直间距

        # 添加标题标签
        self.label_title = QLabel('小红书情感分析系统', self)
        title_font = QFont()
        title_font.setPointSize(20)  # 设置字体大小
        title_font.setBold(True)      # 设置字体加粗
        self.label_title.setFont(title_font)
        self.label_title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.label_title)

        # 创建一个网格布局用于用户名、密码和确认密码输入
        form_layout = QGridLayout()
        form_layout.setSpacing(10)  # 设置网格布局中控件之间的间距

        # 设置列伸缩以居中表单
        form_layout.setColumnStretch(0, 1)  # 左侧弹性空间
        form_layout.setColumnStretch(1, 0)  # 标签
        form_layout.setColumnStretch(2, 2)  # 输入框
        form_layout.setColumnStretch(3, 1)  # 右侧弹性空间

        # 用户名标签和输入框
        self.label_username = QLabel('用户名:', self)
        self.input_username = QLineEdit(self)
        self.input_username.setMinimumWidth(150)  # 设置最小宽度
        form_layout.addWidget(self.label_username, 0, 1, alignment=Qt.AlignRight)
        form_layout.addWidget(self.input_username, 0, 2, alignment=Qt.AlignLeft)

        # 密码标签和输入框
        self.label_password = QLabel('密码:', self)
        self.input_password = QLineEdit(self)
        self.input_password.setEchoMode(QLineEdit.Password)
        self.input_password.setMinimumWidth(150)  # 设置最小宽度
        form_layout.addWidget(self.label_password, 1, 1, alignment=Qt.AlignRight)
        form_layout.addWidget(self.input_password, 1, 2, alignment=Qt.AlignLeft)

        # 确认密码标签和输入框
        self.label_confirm_password = QLabel('确认密码:', self)
        self.input_confirm_password = QLineEdit(self)
        self.input_confirm_password.setEchoMode(QLineEdit.Password)
        self.input_confirm_password.setMinimumWidth(150)  # 设置最小宽度
        form_layout.addWidget(self.label_confirm_password, 2, 1, alignment=Qt.AlignRight)
        form_layout.addWidget(self.input_confirm_password, 2, 2, alignment=Qt.AlignLeft)

        main_layout.addLayout(form_layout)

        # 添加按钮的水平布局
        button_layout = QHBoxLayout()
        button_layout.setSpacing(20)  # 设置按钮之间的水平间距

        # 注册按钮
        self.button_register = QPushButton('注册', self)
        self.button_register.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        button_layout.addWidget(self.button_register)

        # 返回登录按钮
        self.button_back = QPushButton('返回登录', self)
        self.button_back.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        button_layout.addWidget(self.button_back)

        main_layout.addLayout(button_layout)

        # 添加Logo
        self.add_logo(main_layout)

        # 设置主布局的对齐方式
        main_layout.addStretch()  # 添加弹性空间以确保内容居中

        self.setLayout(main_layout)

        # 连接按钮点击事件
        self.button_register.clicked.connect(self.handle_register)
        self.button_back.clicked.connect(self.back_to_login)

    def add_logo(self, layout):
        if os.path.exists(LOGO_PATH):
            self.logo_label = QLabel(self)
            pixmap = QPixmap(LOGO_PATH).scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)  # 调整Logo大小
            self.logo_label.setPixmap(pixmap)
            self.logo_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(self.logo_label, alignment=Qt.AlignCenter)
        else:
            print(f"Logo file not found: {LOGO_PATH}")

    def handle_register(self):
        username = self.input_username.text().strip()
        password = self.input_password.text().strip()
        confirm_password = self.input_confirm_password.text().strip()

        print(f"注册尝试 - 用户名: {username}, 密码: {password}")  # 调试语句

        if not username or not password or not confirm_password:
            QMessageBox.warning(self, '警告', '请填写所有字段')
            return

        if password != confirm_password:
            QMessageBox.warning(self, '错误', '密码和确认密码不匹配')
            return

        users = load_users()

        if username in users:
            QMessageBox.warning(self, '错误', '用户名已存在')
            return

        # 注册新用户
        users[username] = password
        save_users(users)
        QMessageBox.information(self, '成功', '注册成功！请登录。')
        self.back_to_login()

    def back_to_login(self):
        self.login_window = LoginWindow()
        self.login_window.show()
        self.close()  # 关闭注册窗口


class FunctionSelectionWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('功能选择')
        self.setGeometry(600, 300, 400, 400)  # 调整窗口大小以适应新布局
        self.init_ui()

    def init_ui(self):
        # 创建主垂直布局
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(50, 50, 50, 50)  # 设置上下左右的边距
        main_layout.setSpacing(20)  # 设置各控件之间的垂直间距

        # 添加标题标签
        self.label_title = QLabel('小红书情感分析系统', self)
        title_font = QFont()
        title_font.setPointSize(20)  # 设置字体大小
        title_font.setBold(True)      # 设置字体加粗
        self.label_title.setFont(title_font)
        self.label_title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.label_title)

        # 创建一个网格布局用于功能按钮
        button_layout = QVBoxLayout()
        button_layout.setSpacing(15)  # 设置功能按钮之间的垂直间距

        # 文本输入分类按钮
        self.button_text_classification = QPushButton('文本输入分类', self)
        self.button_text_classification.setStyleSheet("""
            QPushButton {
                background-color: #008CBA;
                color: white;
                padding: 15px;
                border-radius: 10px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #007bb5;
            }
        """)
        button_layout.addWidget(self.button_text_classification)

        # 文档分类按钮
        self.button_document_classification = QPushButton('文档分类', self)
        self.button_document_classification.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                padding: 15px;
                border-radius: 10px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        button_layout.addWidget(self.button_document_classification)


        # 爬虫功能按钮
        self.button_scraper = QPushButton('爬虫功能', self)
        self.button_scraper.setStyleSheet("""
            QPushButton {
                background-color: #FFA500;
                color: white;
                padding: 15px;
                border-radius: 10px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #FF8C00;
            }
        """)
        button_layout.addWidget(self.button_scraper)

        # 退出按钮
        self.button_exit = QPushButton('退出', self)
        self.button_exit.setStyleSheet("""
            QPushButton {
                background-color: #e7e7e7;
                color: black;
                padding: 15px;
                border-radius: 10px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #d6d6d6;
            }
        """)
        button_layout.addWidget(self.button_exit)

        main_layout.addLayout(button_layout)

        # 添加Logo
        self.add_logo(main_layout)

        # 设置主布局的对齐方式
        main_layout.addStretch()  # 添加弹性空间以确保内容居中

        self.setLayout(main_layout)

        # 连接按钮点击事件
        self.button_text_classification.clicked.connect(self.open_text_classification_window)
        self.button_document_classification.clicked.connect(self.open_document_classification_window)
        self.button_scraper.clicked.connect(self.open_scraper_window)
        self.button_exit.clicked.connect(self.close_application)

    def add_logo(self, layout):
        if os.path.exists(LOGO_PATH):
            self.logo_label = QLabel(self)
            pixmap = QPixmap(LOGO_PATH).scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)  # 调整Logo大小
            self.logo_label.setPixmap(pixmap)
            self.logo_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(self.logo_label, alignment=Qt.AlignCenter)
        else:
            print(f"Logo file not found: {LOGO_PATH}")

    def open_text_classification_window(self):
        self.text_classification_window = MainWindow()
        self.text_classification_window.show()
        self.close()  # 关闭功能选择窗口

    def open_document_classification_window(self):
        self.document_classification_window = DocumentClassificationWindow()
        self.document_classification_window.show()
        self.close()  # 关闭功能选择窗口

    def open_scraper_window(self):
        self.scraper_window = ScraperWindow()
        self.scraper_window.show()
        self.close()  # 关闭功能选择窗口

    def close_application(self):
        QApplication.quit()


class ScraperWorker(QThread):
    progress = pyqtSignal(int)
    result = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, keyword, parent=None):
        super().__init__(parent)
        self.keyword = keyword
        self.scraper = Xhs(cookie_str)

    def run(self):
        try:
            # Collect results here
            results = []
            total_pages = 11
            # For simplicity, we use pages 1 to 11
            for page in range(1, total_pages+1):
                logger.info(f"Scraping page {page} for keyword '{self.keyword}'")
                search = self.scraper.search(self.keyword, page, sort=1, note_type=0)
                if 'data' not in search or 'items' not in search['data']:
                    continue
                for note in search['data']['items']:
                    data = self.scraper.search_(note, data={'搜索关键字': self.keyword})
                    if not data:
                        continue
                    data = self.scraper.feed_(data=data)
                    if data:
                        results.append({
                            '笔记id': data.get('笔记id', ''),
                            '标题': data.get('标题', ''),
                            '描述': data.get('描述', ''),
                            '笔记类型': data.get('笔记类型', '')
                        })
                progress_percent = int((page / total_pages) * 100)
                self.progress.emit(progress_percent)
                if not search['data']['has_more']:
                    break
            self.result.emit(results)
        except Exception as e:
            logger.error(f"爬虫出错: {e}")
            self.error.emit(str(e))

class ScraperWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('爬虫功能')
        self.setGeometry(500, 200, 800, 600)
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(10)

        # Input fields
        input_layout = QGridLayout()
        input_layout.setSpacing(10)

        self.label_keyword = QLabel('关键词:', self)
        self.input_keyword = QLineEdit(self)
        self.input_keyword.setPlaceholderText('请输入搜索关键词')
        input_layout.addWidget(self.label_keyword, 0, 0, alignment=Qt.AlignRight)
        input_layout.addWidget(self.input_keyword, 0, 1)

        main_layout.addLayout(input_layout)

        # Scraping button
        self.button_start_scrape = QPushButton('开始爬取', self)
        self.button_start_scrape.setStyleSheet("""
            QPushButton {
                background-color: #FFA500;
                color: white;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #FF8C00;
            }
        """)
        main_layout.addWidget(self.button_start_scrape)

        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        # Table to display results
        self.table_results = QTableWidget(self)
        self.table_results.setColumnCount(4)
        self.table_results.setHorizontalHeaderLabels(['笔记id', '标题', '描述', '笔记类型'])
        self.table_results.horizontalHeader().setStretchLastSection(True)
        main_layout.addWidget(self.table_results)

        # Buttons layout
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(20)

        self.button_export_csv = QPushButton('导出CSV', self)
        self.button_export_csv.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.button_export_csv.setEnabled(False)  # Initially disabled
        buttons_layout.addWidget(self.button_export_csv)

        self.button_back = QPushButton('返回功能选择', self)
        self.button_back.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        buttons_layout.addWidget(self.button_back)

        main_layout.addLayout(buttons_layout)

        # 添加Logo
        self.add_logo(main_layout)

        self.setLayout(main_layout)

        # Connect buttons
        self.button_start_scrape.clicked.connect(self.start_scraping)
        self.button_export_csv.clicked.connect(self.export_csv)
        self.button_back.clicked.connect(self.back_to_function_selection)

    def add_logo(self, layout):
        if os.path.exists(LOGO_PATH):
            logo_label = QLabel(self)
            pixmap = QPixmap(LOGO_PATH).scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(pixmap)
            logo_label.setAlignment(Qt.AlignRight | Qt.AlignBottom)
            layout.addWidget(logo_label, alignment=Qt.AlignRight | Qt.AlignBottom)
        else:
            print(f"Logo file not found: {LOGO_PATH}")

    def start_scraping(self):
        keyword = self.input_keyword.text().strip()

        if not keyword:
            QMessageBox.warning(self, '警告', '请输入关键词')
            return

        self.button_start_scrape.setEnabled(False)
        self.button_export_csv.setEnabled(False)
        self.table_results.setRowCount(0)
        self.progress_bar.setValue(0)

        # Start the scraper in a separate thread
        self.worker = ScraperWorker(keyword)
        self.worker.progress.connect(self.update_progress)
        self.worker.result.connect(self.display_results)
        self.worker.error.connect(self.handle_error)
        self.worker.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def display_results(self, results):
        self.table_results.setRowCount(len(results))
        for row, data in enumerate(results):
            self.table_results.setItem(row, 0, QTableWidgetItem(data.get('笔记id', '')))
            self.table_results.setItem(row, 1, QTableWidgetItem(data.get('标题', '')))
            self.table_results.setItem(row, 2, QTableWidgetItem(data.get('描述', '')))
            self.table_results.setItem(row, 3, QTableWidgetItem(data.get('笔记类型', '')))
        self.button_start_scrape.setEnabled(True)
        if results:
            self.button_export_csv.setEnabled(True)
            QMessageBox.information(self, '完成', f'爬取完成，共 {len(results)} 条记录')
        else:
            QMessageBox.information(self, '完成', '未爬取到任何记录')

    def handle_error(self, error_msg):
        QMessageBox.critical(self, '错误', f'爬取过程中发生错误:\n{error_msg}')
        self.button_start_scrape.setEnabled(True)

    def export_csv(self):
        if self.table_results.rowCount() == 0:
            QMessageBox.warning(self, '警告', '没有数据可导出')
            return

        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "保存CSV文件",
            "",
            "CSV Files (*.csv);;All Files (*)",
            options=options
        )
        if file_name:
            try:
                with open(file_name, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    # Write headers
                    headers = ['笔记id', '标题', '描述', '笔记类型']
                    writer.writerow(headers)
                    # Write data
                    for row in range(self.table_results.rowCount()):
                        row_data = []
                        for column in range(self.table_results.columnCount()):
                            item = self.table_results.item(row, column)
                            row_data.append(item.text() if item else '')
                        writer.writerow(row_data)
                QMessageBox.information(self, '成功', '数据已成功导出到CSV文件')
            except Exception as e:
                QMessageBox.critical(self, '错误', f'导出过程中发生错误:\n{e}')

    def back_to_function_selection(self):
        self.function_selection_window = FunctionSelectionWindow()
        self.function_selection_window.show()
        self.close()  # 关闭爬虫功能窗口


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('文本情感分类')
        self.setGeometry(300, 300, 500, 500)
        self.init_ui()

    def init_ui(self):
        # self.setStyleSheet("background-color: lightgreen;")  # 设置背景色
        self.layout = QGridLayout(self)

        # 输入文本标签和文本框
        self.label_input = QLabel('请输入文本:', self)
        self.text_input = QTextEdit(self)
        self.layout.addWidget(self.label_input, 0, 0, 1, 1)
        self.layout.addWidget(self.text_input, 1, 0, 3, 4)

        # 运行按钮
        self.button_run = QPushButton('运行', self)
        self.layout.addWidget(self.button_run, 4, 0, 1, 1)

        # 预测结果标签和显示
        self.label_predict_result = QLabel('识别结果:', self)
        self.label_predict_result_display = QLabel('', self)
        self.layout.addWidget(self.label_predict_result, 5, 0, 1, 1)
        self.layout.addWidget(self.label_predict_result_display, 5, 1, 1, 3)

        # 预测概率标签和显示
        self.label_predict_acc = QLabel('识别概率:', self)
        self.label_predict_acc_display = QLabel('', self)
        self.layout.addWidget(self.label_predict_acc, 6, 0, 1, 1)
        self.layout.addWidget(self.label_predict_acc_display, 6, 1, 1, 3)

        # 返回功能选择按钮
        self.button_back = QPushButton('返回功能选择', self)
        self.layout.addWidget(self.button_back, 7, 0, 1, 4)

        # 添加Logo
        self.add_logo(self.layout, 8, 3)  # 添加到第9行，第4列
        self.setLayout(self.layout)

        # 连接按钮点击事件
        self.button_run.clicked.connect(self.run)
        self.button_back.clicked.connect(self.back_to_function_selection)

    def add_logo(self, layout, row, column):
        if os.path.exists(LOGO_PATH):
            self.logo_label = QLabel(self)
            pixmap = QPixmap(LOGO_PATH).scaled(38, 138, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.logo_label.setPixmap(pixmap)
            self.logo_label.setAlignment(Qt.AlignRight | Qt.AlignBottom)
            layout.addWidget(self.logo_label, row, column, alignment=Qt.AlignRight | Qt.AlignBottom)
        else:
            print(f"Logo file not found: {LOGO_PATH}")

    def run(self):
        # 获取输入的文本
        text = self.text_input.toPlainText()
        if text.strip():
            try:
                # 调用预测函数
                label, probability = predict_(text)
                self.label_predict_result_display.setText(label)
                self.label_predict_acc_display.setText(f"{probability:.4f}")
            except Exception as e:
                QMessageBox.critical(self, '错误', f'预测过程中发生错误:\n{e}')
                self.label_predict_result_display.setText('错误')
                self.label_predict_acc_display.setText('0.0000')
        else:
            QMessageBox.warning(self, '警告', '请输入文本进行预测')
            self.label_predict_result_display.setText('未输入文本')
            self.label_predict_acc_display.setText('0.0000')

    def back_to_function_selection(self):
        self.function_selection_window = FunctionSelectionWindow()
        self.function_selection_window.show()
        self.close()  # 关闭文本分类窗口

class DocumentClassificationWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('文档分类')
        self.setGeometry(300, 100, 1200, 800)  # 调整窗口大小以适应新布局
        self.init_ui()

    def init_ui(self):
        # 创建主垂直布局
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)  # 设置上下左右的边距
        main_layout.setSpacing(10)  # 设置各控件之间的垂直间距

        # 选择文件按钮
        self.button_select_file = QPushButton('选择TXT文件', self)
        self.button_select_file.setStyleSheet(self.button_style())
        main_layout.addWidget(self.button_select_file, alignment=Qt.AlignLeft)

        # 显示文件路径标签
        self.label_file_path = QLabel('文件路径:', self)
        main_layout.addWidget(self.label_file_path, alignment=Qt.AlignLeft)
        self.display_file_path = QLabel('', self)
        self.display_file_path.setStyleSheet("font-weight: bold;")
        main_layout.addWidget(self.display_file_path, alignment=Qt.AlignLeft)

        # 运行按钮
        self.button_run = QPushButton('运行', self)
        self.button_run.setStyleSheet(self.button_style())
        main_layout.addWidget(self.button_run, alignment=Qt.AlignLeft)

        # 结果表格
        self.table_results = QTableWidget(self)
        self.table_results.setColumnCount(3)
        self.table_results.setHorizontalHeaderLabels(['行号', '文本内容', '识别结果'])
        self.table_results.horizontalHeader().setStretchLastSection(True)
        self.table_results.setStyleSheet("""
            QTableWidget {
                background-color: #f0f0f0;
                alternate-background-color: #e0e0e0;
                gridline-color: #cccccc;
            }
            QHeaderView::section {
                background-color: #dcdcdc;
                padding: 4px;
                border: 1px solid #cccccc;
            }
        """)
        main_layout.addWidget(self.table_results)

        # 图表区域（初始为空，分类完成后动态添加）
        self.charts_layout = QHBoxLayout()
        self.charts_layout.setSpacing(20)  # 设置词云和柱状图之间的水平间距
        main_layout.addLayout(self.charts_layout)

        # 返回功能选择按钮
        self.button_back = QPushButton('返回功能选择', self)
        self.button_back.setStyleSheet(self.button_style())
        main_layout.addWidget(self.button_back, alignment=Qt.AlignLeft)

        # 添加Logo
        self.add_logo(main_layout)

        self.setLayout(main_layout)

        # 连接按钮点击事件
        self.button_select_file.clicked.connect(self.select_file)
        self.button_run.clicked.connect(self.run_classification)
        self.button_back.clicked.connect(self.back_to_function_selection)

        # 初始化图表相关属性
        self.wordcloud_container = None  # 初始未创建
        self.barchart_container = None   # 初始未创建

    def button_style(self):
        return """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """

    def add_logo(self, layout):
        if os.path.exists(LOGO_PATH):
            logo_label = QLabel(self)
            pixmap = QPixmap(LOGO_PATH).scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(pixmap)
            logo_label.setAlignment(Qt.AlignLeft | Qt.AlignBottom)

            # 创建一个水平布局，将Logo放在左下角
            logo_layout = QHBoxLayout()
            logo_layout.addWidget(logo_label, alignment=Qt.AlignLeft | Qt.AlignBottom)

            # 添加一个水平伸缩项，推动Logo到左下角
            spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
            logo_layout.addItem(spacer)

            layout.addLayout(logo_layout)
        else:
            print(f"Logo file not found: {LOGO_PATH}")

    def select_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "选择TXT文件",
            "",
            "Text Files (*.txt);;All Files (*)",
            options=options
        )
        if file_name:
            self.selected_file = file_name
            self.display_file_path.setText(file_name)
            self.table_results.setRowCount(0)  # 清空表格
            self.clear_wordcloud()  # 清空词云
            self.clear_barchart()  # 清空柱状图

    def run_classification(self):
        if not hasattr(self, 'selected_file') or not self.selected_file:
            QMessageBox.warning(self, '警告', '请先选择一个TXT文件')
            return

        try:
            with open(self.selected_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # 获取所有文本内容用于词云
            full_text = ' '.join([line.strip() for line in lines if line.strip()])

            # 初始化计数
            positive_count = 0
            negative_count = 0

            self.table_results.setRowCount(len(lines))
            for idx, line in enumerate(lines, 1):
                text = line.strip()
                if text:
                    label, probability = predict_(text)
                else:
                    label, probability = '空行', 0.0

                # 设置行号
                item_row = QTableWidgetItem(str(idx))
                item_row.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                self.table_results.setItem(idx-1, 0, item_row)

                # 设置文本内容
                item_text = QTableWidgetItem(text)
                item_text.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                self.table_results.setItem(idx-1, 1, item_text)

                # 设置识别结果
                if label != '空行':
                    result_text = f"{label} ({probability:.4f})"
                    if label == '正面':
                        positive_count += 1
                    elif label == '负面':
                        negative_count += 1
                else:
                    result_text = label
                item_result = QTableWidgetItem(result_text)
                item_result.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                self.table_results.setItem(idx-1, 2, item_result)

            # 生成并显示词云
            if full_text:
                print("生成词云...")  # 调试语句
                wordcloud_pixmap = self.generate_wordcloud(full_text)
                self.add_wordcloud(wordcloud_pixmap)
            else:
                self.add_wordcloud_text('文档为空，无法生成词云。')

            # 生成并显示柱状图
            self.generate_barchart(positive_count, negative_count)

            QMessageBox.information(self, '完成', '文档分类完成')
        except Exception as e:
            traceback_str = traceback.format_exc()
            QMessageBox.critical(self, '错误', f'分类过程中发生错误:\n{e}\n{traceback_str}')

    def add_wordcloud(self, pixmap):
        """
        动态添加词云展示区域
        """
        if not self.wordcloud_container:
            # 创建词云容器
            self.wordcloud_container = QVBoxLayout()
            self.label_wordcloud = QLabel('文档内容词云:', self)
            self.label_wordcloud.setStyleSheet("font-weight: bold;")
            self.wordcloud_container.addWidget(self.label_wordcloud, alignment=Qt.AlignCenter)

            self.wordcloud_label = QLabel(self)
            self.wordcloud_label.setAlignment(Qt.AlignCenter)
            self.wordcloud_label.setFixedSize(600, 400)  # 调整词云显示区域大小
            self.wordcloud_container.addWidget(self.wordcloud_label, alignment=Qt.AlignCenter)

            # 将词云容器添加到图表布局
            self.charts_layout.addLayout(self.wordcloud_container)

        # 设置词云图片
        self.wordcloud_label.setPixmap(pixmap.scaled(
            self.wordcloud_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))

    def add_wordcloud_text(self, text):
        """
        动态添加词云展示区域的文本信息
        """
        if not self.wordcloud_container:
            # 创建词云容器
            self.wordcloud_container = QVBoxLayout()
            self.label_wordcloud = QLabel('文档内容词云:', self)
            self.label_wordcloud.setStyleSheet("font-weight: bold;")
            self.wordcloud_container.addWidget(self.label_wordcloud, alignment=Qt.AlignCenter)

            self.wordcloud_label = QLabel(self)
            self.wordcloud_label.setAlignment(Qt.AlignCenter)
            self.wordcloud_label.setFixedSize(600, 400)  # 调整词云显示区域大小
            self.wordcloud_container.addWidget(self.wordcloud_label, alignment=Qt.AlignCenter)

            # 将词云容器添加到图表布局
            self.charts_layout.addLayout(self.wordcloud_container)

        # 设置词云文本
        self.wordcloud_label.setText(text)

    def generate_wordcloud(self, text):
        """
        生成词云并返回 QPixmap 对象
        """
        try:
            # 指定支持中文的字体路径
            # 请根据您的系统调整字体路径
            # Windows 示例：
            font_path = 'C:\\Windows\\Fonts\\simhei.ttf'  # 请根据实际情况修改

            if not os.path.exists(font_path):
                QMessageBox.warning(self, '警告', f'指定的字体文件不存在：{font_path}')
                return QPixmap()

            wordcloud = WordCloud(
                font_path=font_path,  # 指定中文字体路径
                width=800,
                height=400,
                background_color='white',
                max_words=200,
                collocations=False,  # 禁止重复单词
                stopwords=None,  # 可以根据需要添加停用词
                mode='RGB'
            ).generate(text)

            image = wordcloud.to_image()

            # 将 PIL Image 转换为 QPixmap
            buffer = BytesIO()
            image.save(buffer, format='PNG')
            pixmap = QPixmap()
            pixmap.loadFromData(buffer.getvalue())
            return pixmap
        except Exception as e:
            print(f"生成词云时出错: {e}")  # 调试语句
            QMessageBox.critical(self, '错误', f'生成词云时发生错误:\n{e}')
            return QPixmap()

    def generate_barchart(self, positive, negative):
        """
        生成柱状图并显示在界面上
        """
        try:
            # 检查是否已经创建了柱状图容器
            if not self.barchart_container:
                # 创建柱状图容器
                self.barchart_container = QVBoxLayout()
                self.label_barchart = QLabel('识别结果统计:', self)
                self.label_barchart.setStyleSheet("font-weight: bold;")
                self.barchart_container.addWidget(self.label_barchart, alignment=Qt.AlignCenter)

                self.barchart_canvas = FigureCanvas(plt.Figure(figsize=(6, 4)))
                self.barchart_container.addWidget(self.barchart_canvas, alignment=Qt.AlignCenter)

                # 将柱状图容器添加到图表布局
                self.charts_layout.addLayout(self.barchart_container)
            else:
                # 如果柱状图已经存在，清空之前的内容
                ax = self.barchart_canvas.figure.subplots()
                ax.clear()

            # 设置中文字体
            # 请根据您的系统调整font_path
            if os.name == 'nt':  # Windows
                font_path = 'C:\\Windows\\Fonts\\simhei.ttf'
            elif sys.platform == 'darwin':  # macOS
                font_path = '/System/Library/Fonts/STHeiti Medium.ttc'
            else:  # Linux
                font_path = '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc'

            if not os.path.exists(font_path):
                QMessageBox.warning(self, '警告', f'指定的字体文件不存在：{font_path}')
                return

            font = FontProperties(fname=font_path, size=12)

            # 数据
            categories = ['正面', '负面']
            counts = [positive, negative]
            colors = ['#4CAF50', '#F44336']  # 绿色和红色

            # 绘制柱状图
            ax = self.barchart_canvas.figure.subplots()
            ax.clear()
            bars = ax.bar(categories, counts, color=colors)

            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontproperties=font)

            # 设置标题和标签
            ax.set_title('情感识别结果统计', fontproperties=font)
            ax.set_ylabel('数量', fontproperties=font)
            ax.set_xlabel('类别', fontproperties=font)  # 如果需要

            # 设置刻度标签字体
            ax.set_xticklabels(categories, fontproperties=font)
            ax.set_yticklabels([str(int(i)) for i in ax.get_yticks()], fontproperties=font)

            # 调整布局
            self.barchart_canvas.figure.tight_layout()

            # 绘制
            self.barchart_canvas.draw()
        except Exception as e:
            traceback_str = traceback.format_exc()
            QMessageBox.critical(self, '错误', f'生成柱状图时发生错误:\n{e}\n{traceback_str}')

    def clear_wordcloud(self):
        """
        清空词云展示区域
        """
        if self.wordcloud_container:
            # 从charts_layout中移除词云容器
            while self.wordcloud_container.count():
                child = self.wordcloud_container.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            self.charts_layout.removeItem(self.wordcloud_container)
            self.wordcloud_container = None

    def clear_barchart(self):
        """
        清空柱状图
        """
        if self.barchart_container:
            # 从charts_layout中移除柱状图容器
            while self.barchart_container.count():
                child = self.barchart_container.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            self.charts_layout.removeItem(self.barchart_container)
            self.barchart_container = None

    def back_to_function_selection(self):
        self.function_selection_window = FunctionSelectionWindow()
        self.function_selection_window.show()
        self.close()  # 关闭文档分类窗口


if __name__ == '__main__':
    app = QApplication(sys.argv)
    login = LoginWindow()
    login.show()
    sys.exit(app.exec_())
