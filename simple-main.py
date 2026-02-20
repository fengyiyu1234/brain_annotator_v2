import sys
import os
from PyQt5.QtWidgets import QApplication

# 引入我们写好的模块
from src.simple_annotator import SimpleAnnotator
from src.coordinate_system import CoordinateSystem
from src.ontology import Ontology

# ================= 配置区域 (请确认你的默认路径) =================

XML_PATH = "Z:/Ryan/Light_Sheet_Imaged_Brains/04-08-2025_FW_2/561nm/xml_merging.xml"
ONTOLOGY_PATH = "Z:/Fengyi/brain_analysis_6sample/CCF_v3_ontology.json" 
ROOT_RED = "Z:/Ryan/Light_Sheet_Imaged_Brains/04-08-2025_FW_2/561nm"
ROOT_GREEN = "Z:/Ryan/Light_Sheet_Imaged_Brains/04-08-2025_FW_2/numorph_align/aligned/GFP"

# 存放 Detection (9列) 结果的文件夹
DET_CSV_DIR = "Z:/Fengyi/brain_analysis_6sample/detection_results/fw2/detection_results" 

# 存放 Registration (包含 graph order 的全局坐标) 的文件夹
REG_CSV_DIR = r"Z:\Fengyi\brain_analysis_6sample\clearmap_p5_trimmed_atlas\fw2\cell_registration"

# 重新分类后，保存 Patch 图像和 YOLO txt 的根目录
SAVE_ROOT = "Z:/Fengyi/brain_analysis_6sample/re_classify/fw2"

# ===================================================================

def main():
    app = QApplication(sys.argv)

    # 1. 检查关键文件是否存在
    if not os.path.exists(XML_PATH):
        print(f"Error: XML not found at {XML_PATH}")
        return
    if not os.path.exists(ONTOLOGY_PATH):
        print(f"Error: Ontology not found at {ONTOLOGY_PATH}")
        return

    print("--- Loading Metadata ---")
    
    # 2. 初始化基础数据 (坐标系统和脑区树)
    coord_sys = CoordinateSystem(XML_PATH)
    ontology = Ontology(ONTOLOGY_PATH)

    print("--- Starting Annotator ---")
    
    # 3. 启动主程序 (将所有路径直接喂给全新的 Annotator)
    window = SimpleAnnotator(
        coord_sys=coord_sys,
        ontology=ontology,
        det_dir=DET_CSV_DIR,
        reg_dir=REG_CSV_DIR,
        save_root=SAVE_ROOT,
        root_red=ROOT_RED,
        root_green=ROOT_GREEN
    )
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()