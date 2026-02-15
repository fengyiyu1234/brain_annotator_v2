from PyQt5.QtWidgets import QWidget, QDialog, QGridLayout, QPushButton, QLabel, QLineEdit, QFileDialog, QDialogButtonBox
from PyQt5.QtCore import pyqtSignal
import os
class TileMapWidget(QWidget):
    tile_clicked = pyqtSignal(int)
    def __init__(self, tiles_meta):
        super().__init__()
        self.tiles_meta = tiles_meta
        self.buttons = {}
        self.init_ui()
        
    def init_ui(self):
        layout = QGridLayout(self)
        layout.setSpacing(1)
        if not self.tiles_meta: return

        rows = [t['row'] for t in self.tiles_meta]
        cols = [t['col'] for t in self.tiles_meta]
        min_r, min_c = min(rows), min(cols)
        
        for i, t in enumerate(self.tiles_meta):
            r, c = t['row'] - min_r, t['col'] - min_c
            btn = QPushButton(f"{i}")
            btn.setFixedSize(22, 22)
            btn.setStyleSheet("background-color: #ddd; border: 1px solid #999; font-size: 10px;")
            btn.clicked.connect(lambda _, idx=i: self.tile_clicked.emit(idx))
            layout.addWidget(btn, r, c)
            self.buttons[i] = btn

    def update_visuals(self, current_idx, cached_keys):
        for i, btn in self.buttons.items():
            if i == current_idx:
                btn.setStyleSheet("background-color: #ff3333; color: white; font-weight: bold; border: 2px solid black; font-size: 10px;")
            elif i in cached_keys:
                btn.setStyleSheet("background-color: #66cc66; color: white; border: 1px solid #336633; font-size: 10px;")
            else:
                btn.setStyleSheet("background-color: #ddd; color: black; border: 1px solid #999; font-size: 10px;")

class SetupDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Brain Annotator Setup")
        self.init_ui()
        
    def init_ui(self):
        l = QGridLayout(self)
        self.idx = 0
        
        # --- 新增 Ontology 选项 ---
        self.ont = self.row(l, "Ontology (*.json):", "Z:/Fengyi/brain_analysis_6sample/CCF_v3_ontology.json", "f", "*.json")
        self.xml = self.row(l, "XML:", "Z:/Ryan/Light_Sheet_Imaged_Brains/04-08-2025_FW_2/561nm/xml_merging.xml", "f", "*.xml")
        self.red = self.row(l, "Red Root:", "Z:/Ryan/Light_Sheet_Imaged_Brains/04-08-2025_FW_2/561nm", "d")
        self.grn = self.row(l, "Green Root:", "Z:/Ryan/Light_Sheet_Imaged_Brains/04-08-2025_FW_2/numorph_align/aligned/GFP", "d")
        self.csv = self.row(l, "CSV Root:", "Z:/Fengyi/brain_analysis_6sample/detection_results/fw2/detection_results", "d")
        self.nav = self.row(l, "Nav Sample:", "Z:/Fengyi/brain_analysis_6sample/clearmap_results/fw2/resampled.tif", "f", "*.tif")
        self.atl = self.row(l, "Nav Atlas:", "Z:/Fengyi/brain_analysis_6sample/clearmap_results/fw2/volume/result.mhd", "f", "*.mhd")
        self.sav = self.row(l, "Save Dir:", "Z:/Fengyi/brain_analysis_6sample/re_classify/fw2", "d")
        
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        l.addWidget(btns, self.idx, 0, 1, 3)

    def row(self, l, label, default, mode, filter=""):
        l.addWidget(QLabel(label), self.idx, 0)
        le = QLineEdit(default)
        l.addWidget(le, self.idx, 1)
        btn = QPushButton("...")
        btn.clicked.connect(lambda: self.browse(le, mode, filter))
        l.addWidget(btn, self.idx, 2)
        self.idx += 1
        return le

    def browse(self, le, mode, filter):
        if mode == "d":
            path = QFileDialog.getExistingDirectory(self, "Select Directory", le.text())
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select File", le.text(), filter)
        if path: le.setText(path)

    def data(self):
        # 返回参数列表，确保这里的顺序与 BrainAnnotator.__init__ 的参数顺序完全一致
        return (
            self.ont.text(),  self.xml.text(),  self.red.text(), self.grn.text(), self.csv.text(), self.nav.text(), self.atl.text(), self.sav.text()
        )

from PyQt5.QtWidgets import (QTreeWidget, QTreeWidgetItem, QHeaderView, 
                             QLineEdit, QVBoxLayout, QWidget, QLabel)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QBrush
class OntologyTreeWidget(QWidget):
    # 当用户点击某个脑区时发送信号 (region_id, region_name)
    region_selected = pyqtSignal(int, str)

    def __init__(self, ontology):
        super().__init__()
        self.ontology = ontology
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 1. 顶部增加一个简单的过滤器
        self.txt_filter = QLineEdit()
        self.txt_filter.setPlaceholderText("Filter regions...")
        self.txt_filter.textChanged.connect(self.filter_tree)
        layout.addWidget(self.txt_filter)

        # 2. 树状视图
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemClicked.connect(self.on_item_clicked)
        layout.addWidget(self.tree)
        
        # 3. 填充数据
        self.populate_tree()

    def populate_tree(self):
        root_node = self.ontology.tree_data
        root_item = self.create_item(root_node)
        self.tree.addTopLevelItem(root_item)
        root_item.setExpanded(True)

    def create_item(self, node_data):
        # 显示格式: "Name (Acronym)"
        display_text = f"{node_data['name']} ({node_data['acronym']})"
        item = QTreeWidgetItem([display_text])
        
        # 存储 Atlas ID 到 UserRole，以便点击时获取
        item.setData(0, Qt.UserRole, node_data['id'])
        
        # 设置颜色 (JSON里通常有 color_hex_triplet: "AABBCC")
        if 'color_hex_triplet' in node_data:
            hex_color = f"#{node_data['color_hex_triplet']}"
            # 设置前面小方块的颜色或者文字颜色
            item.setForeground(0, QBrush(QColor(hex_color)))
            # 也可以设置字体加粗
            font = QFont()
            font.setBold(True)
            item.setFont(0, font)

        # 递归添加子节点
        if 'children' in node_data:
            for child in node_data['children']:
                child_item = self.create_item(child)
                item.addChild(child_item)
                
        return item

    def on_item_clicked(self, item, col):
        region_id = item.data(0, Qt.UserRole)
        name = item.text(0)
        self.region_selected.emit(region_id, name)

    def filter_tree(self, text):
        # 简单的过滤逻辑：隐藏不匹配的节点
        # 注意：树状结构过滤比较复杂，通常只要父节点匹配就显示，
        # 这里做一个简单的实现：展开并高亮，或者简单地利用 Qt 的 match
        # 为了性能和简单，这里仅作为保留接口，暂不实现复杂的递归过滤显示
        pass