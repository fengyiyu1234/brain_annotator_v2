from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem, 
                             QTreeWidgetItemIterator, QGridLayout, QPushButton) # <--- [新增] QTreeWidgetItemIterator
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont
import os

# =========================================================
# TileMapWidget (保持不变)
# =========================================================
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
        if not rows or not cols: return
        
        min_r, min_c = min(rows), min(cols)
        
        for i, t in enumerate(self.tiles_meta):
            r, c = t['row'] - min_r, t['col'] - min_c
            btn = QPushButton(f"{i}")
            btn.setFixedSize(22, 22)
            btn.setStyleSheet("background-color: #ddd; border: 1px solid #999; font-size: 10px;")
            btn.clicked.connect(lambda _, idx=i: self.tile_clicked.emit(idx))
            layout.addWidget(btn, r, c)
            self.buttons[i] = btn

    def update_visuals(self, current_idx):
        for idx, btn in self.buttons.items():
            if idx == current_idx:
                btn.setStyleSheet("background-color: yellow; border: 2px solid red; font-weight: bold;")
            else:
                btn.setStyleSheet("background-color: #ddd; border: 1px solid #999;")

# =========================================================
# OntologyTreeWidget (添加了 update_counts 方法)
# =========================================================
class OntologyTreeWidget(QWidget):
    def __init__(self, ontology):
        super().__init__()
        self.ontology = ontology
        
        # 布局初始化
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # 树控件
        self.tree = QTreeWidget()
        self.tree.setHeaderLabel("Brain Regions")
        self.layout.addWidget(self.tree)
        
        self.populate_tree()
        
    def populate_tree(self):
        """初始化树结构"""
        root_node = self.ontology.tree_data
        # 处理 tree_data 可能是列表或单个字典的情况
        if isinstance(root_node, list):
            for node in root_node:
                root_item = self.create_item(node)
                self.tree.addTopLevelItem(root_item)
                root_item.setExpanded(True)
        else:
            root_item = self.create_item(root_node)
            self.tree.addTopLevelItem(root_item)
            root_item.setExpanded(True)

    def create_item(self, node_data):
        """递归创建树节点"""
        # 显示格式: "Name (Acronym)"
        name = node_data.get('name', 'Unknown')
        acronym = node_data.get('acronym', '')
        display_text = f"{name} ({acronym})"
        
        item = QTreeWidgetItem([display_text])
        
        # 存储 Atlas ID 到 UserRole，以便点击时获取
        item.setData(0, Qt.UserRole, node_data['id'])
        
        # 设置颜色 (JSON里通常有 color_hex_triplet: "AABBCC")
        if 'color_hex_triplet' in node_data:
            hex_color = f"#{node_data['color_hex_triplet']}"
            item.setForeground(0, QBrush(QColor(hex_color)))
            # 设置字体加粗
            font = QFont()
            font.setBold(True)
            item.setFont(0, font)

        # 递归添加子节点
        if 'children' in node_data:
            for child in node_data['children']:
                child_item = self.create_item(child)
                item.addChild(child_item)
                
        return item

    # --- [新增] 核心修复方法 ---
    def update_counts(self, region_map):
        """
        根据匹配结果更新树状图的显示 (Name + Count)
        region_map: dict { graph_order: [cell_list] }
        """
        # 必须使用 QTreeWidgetItemIterator 遍历所有节点（包括折叠的）
        iterator = QTreeWidgetItemIterator(self.tree)
        while iterator.value():
            item = iterator.value()
            node_id = item.data(0, Qt.UserRole)
            
            # 从 Ontology 对象反查节点详细信息，获取 graph_order
            node = self.ontology.id_map.get(node_id)
            
            count = 0
            if node and 'graph_order' in node:
                go = node['graph_order']
                if go in region_map:
                    count = len(region_map[go])
            
            # 更新文本
            if node:
                base_name = node.get('name', 'Unknown')
                acronym = node.get('acronym', '')
                
                # 如果有细胞匹配到该区域，显示数量并高亮
                if count > 0:
                    item.setText(0, f"{base_name} ({acronym}) [{count}]")
                    # 也可以选择自动展开
                    # item.setExpanded(True)
                else:
                    # 如果没有细胞，恢复原始文本
                    item.setText(0, f"{base_name} ({acronym})")
            
            iterator += 1