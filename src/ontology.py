# src/ontology.py
import json

class Ontology:
    def __init__(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # CCF v3 JSON 通常包裹在 msg 列表中
        if 'msg' in data:
            self.tree_data = data['msg'][0]
        else:
            self.tree_data = data # 兼容不同格式
            
        self.id_map = {}
        self.name_map = {}
        self.parent_map = {} # 用于记录层级关系
        
        # 扁平化索引以便快速查找，同时保留树结构引用
        self._build_index(self.tree_data)

    def _build_index(self, node):
        nid = node['id']
        name = node['name']
        acronym = node['acronym']
        
        self.id_map[nid] = node
        self.name_map[name.lower()] = nid
        self.name_map[acronym.lower()] = nid # 支持缩写搜索
        
        if 'children' in node:
            for child in node['children']:
                self.parent_map[child['id']] = nid
                self._build_index(child)

    def get_name(self, ontology_id):
        if ontology_id in self.id_map:
            return self.id_map[ontology_id]['name']
        return "Unknown"

    def get_all_descendant_ids(self, root_id):
        """
        获取某个ID下属的所有子孙ID（包括自己）。
        用于：选中 'Cerebrum' 时，能加载它下面所有的具体脑区数据。
        """
        if root_id not in self.id_map:
            return []
        
        result = [root_id]
        node = self.id_map[root_id]
        
        if 'children' in node:
            for child in node['children']:
                result.extend(self.get_all_descendant_ids(child['id']))
        
        return result