# src/ontology.py
import json

class Ontology:
    def __init__(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if 'msg' in data:
            self.tree_data = data['msg'][0]
        else:
            self.tree_data = data
            
        self.id_map = {}
        self.name_map = {}
        self.parent_map = {}
        
        # 新增：建立 Graph Order 和 ID 的双向映射
        self.go_to_id = {}
        self.id_to_go = {}
        
        self._build_index(self.tree_data)

    def _build_index(self, node):
        nid = node['id']
        name = node['name']
        acronym = node['acronym']
        
        self.id_map[nid] = node
        self.name_map[name.lower()] = nid
        self.name_map[acronym.lower()] = nid
        
        # 提取 graph_order
        go = node.get('graph_order')
        if go is not None:
            self.go_to_id[go] = nid
            self.id_to_go[nid] = go
        
        if 'children' in node:
            for child in node['children']:
                self.parent_map[child['id']] = nid
                self._build_index(child)

    def get_name(self, ontology_id):
        if ontology_id in self.id_map:
            return self.id_map[ontology_id]['name']
        return "Unknown"

    def get_all_descendant_ids(self, root_id):
        """递归获取某个ID下属的所有子脑区 ID 列表"""
        ids = [root_id]
        node = self.id_map.get(root_id)
        if node and 'children' in node:
            for child in node['children']:
                ids.extend(self.get_all_descendant_ids(child['id']))
        return ids