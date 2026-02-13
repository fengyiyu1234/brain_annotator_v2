from PyQt5.QtWidgets import QWidget, QDialog, QGridLayout, QPushButton, QLabel, QLineEdit, QFileDialog, QDialogButtonBox
from PyQt5.QtCore import pyqtSignal

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
        self.setWindowTitle("Config")
        self.resize(700, 300)
        l = QGridLayout(self)
        self.idx=0
        # Default paths for convenience
        self.xml = self.row(l, "XML:", "Z:/Ryan/Light_Sheet_Imaged_Brains/04-08-2025_FW_2/561nm/xml_merging.xml", "f", "*.xml")
        self.red = self.row(l, "Red Root:", "Z:/Ryan/Light_Sheet_Imaged_Brains/04-08-2025_FW_2/561nm", "d")
        self.grn = self.row(l, "Green Root:", "Z:/Ryan/Light_Sheet_Imaged_Brains/04-08-2025_FW_2/numorph_align/aligned/GFP", "d")
        self.csv = self.row(l, "CSV Root:", "Z:/Fengyi/brain_analysis_6sample/detection_results/fw2/detection_results", "d")
        self.nav = self.row(l, "Nav Sample:", "Z:/Fengyi/brain_analysis_6sample/clearmap_results/fw2/resampled.tif", "f", "*.tif")
        self.atl = self.row(l, "Nav Atlas:", "Z:/Fengyi/brain_analysis_6sample/clearmap_results/fw2/volume/result.mhd", "f", "*.mhd")
        self.sav = self.row(l, "Save Dir:", "Z:/Fengyi/brain_analysis_6sample/re_classify/fw2", "d")
        
        btns = QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        l.addWidget(btns, self.idx, 0, 1, 3)
    def row(self, l, txt, df, m, flt=""):
        l.addWidget(QLabel(txt), self.idx, 0)
        e = QLineEdit(df); l.addWidget(e, self.idx, 1)
        b = QPushButton("Browse"); l.addWidget(b, self.idx, 2)
        b.clicked.connect(lambda: e.setText(QFileDialog.getOpenFileName(self, "", "", flt)[0] if m=="f" else QFileDialog.getExistingDirectory(self, "")))
        self.idx+=1
        return e
    def data(self): return (self.xml.text(), self.red.text(), self.grn.text(), self.csv.text(), self.nav.text(), self.atl.text(), self.sav.text())
