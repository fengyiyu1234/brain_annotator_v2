import sys
import os
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog
from src.widgets import SetupDialog
from src.simple_annotator import SimpleAnnotator
from src.coordinate_system import CoordinateSystem
from src.ontology import Ontology

def main():
    app = QApplication(sys.argv)
    
    # 1. 弹出配置 (或者你直接写死路径)
    setup = SetupDialog()
    if setup.exec_() == QDialog.Accepted:
        params = setup.data() 
        # params: ontology_path, xml_path, red, green, csv, nav_sample, nav_atlas, save
        
        ontology_path = params[0]
        xml_path = params[1]
        root_red = params[2]
        root_green = params[3]
        csv_root = params[4]
        # skip nav_sample (params[5])
        nav_atlas_path = params[6]
        save_root = params[7]
        
        # 确保保存目录存在
        os.makedirs(os.path.join(save_root, 'images'), exist_ok=True)
        os.makedirs(os.path.join(save_root, 'labels'), exist_ok=True)
        
        # 初始化基础数据
        print("Initializing Systems...")
        coord = CoordinateSystem(xml_path)
        onto = Ontology(ontology_path)
        
        # 启动极简标注器
        annotator = SimpleAnnotator(
            coord_sys=coord,
            ontology=onto,
            root_red=root_red,
            root_green=root_green,
            csv_root=csv_root,
            nav_atlas_path=nav_atlas_path,
            save_root=save_root
        )
        annotator.show()
        
        sys.exit(app.exec_())

if __name__ == "__main__":
    main()