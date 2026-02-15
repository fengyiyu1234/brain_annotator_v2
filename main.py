# main.py
import sys
from PyQt5.QtWidgets import QApplication, QDialog
from src.widgets import SetupDialog
from src.annotator import BrainAnnotator

def main():
    app = QApplication(sys.argv)
    
    # 1. 弹出配置对话框
    setup = SetupDialog()
    if setup.exec_() == QDialog.Accepted:
        # 2. 获取配置参数 (现在包含 ontology path)
        params = setup.data()
        
        # 3. 启动主窗口
        viewer = BrainAnnotator(*params)
        viewer.show()
        
        sys.exit(app.exec_())

if __name__ == "__main__":
    main()