import sys
import os
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.gradcam_core import DeepfakeEngine, MODEL_CONFIG, MODEL_KEYS, ENSEMBLE_KEY
from utils.save_report import save_analysis_report, RESULTS_DIR

import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QAction, QComboBox, QMessageBox,
    QProgressBar, QFrame, QStackedWidget, QSizePolicy, QShortcut, QScrollArea,
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QKeySequence
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QEvent

# ── 디자인 토큰 ──
C_BG = "#f0f4f8"
C_CARD = "#ffffff"
C_BORDER = "#e2e8f0"
C_PRIMARY = "#3b82f6"
C_PRIMARY_H = "#2563eb"
C_TEXT = "#1e293b"
C_MUTED = "#64748b"
C_FAKE = "#ef4444"
C_REAL = "#22c55e"

APP_STYLE = f"""
QMainWindow {{ background-color: {C_BG}; }}
QWidget#centralRoot {{ background-color: {C_BG}; }}
QMenuBar {{
    background-color: {C_CARD};
    border-bottom: 1px solid {C_BORDER};
    padding: 4px 8px;
    color: {C_TEXT};
}}
QMenuBar::item {{ padding: 6px 12px; border-radius: 6px; }}
QMenuBar::item:selected {{ background-color: #eff6ff; color: {C_PRIMARY}; }}
QMenu {{
    background-color: {C_CARD};
    border: 1px solid {C_BORDER};
    border-radius: 8px;
    padding: 4px;
}}
QMenu::item {{ padding: 8px 24px; border-radius: 4px; }}
QMenu::item:selected {{ background-color: #eff6ff; color: {C_PRIMARY}; }}
QStatusBar {{
    background-color: {C_CARD};
    border-top: 1px solid {C_BORDER};
    color: {C_MUTED};
    padding: 2px 8px;
}}
QPushButton {{
    background-color: {C_CARD};
    border: 1px solid {C_BORDER};
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 13px;
    color: {C_TEXT};
}}
QPushButton:hover {{ background-color: #f8fafc; border-color: #cbd5e1; }}
QPushButton:pressed {{ background-color: #f1f5f9; }}
QPushButton:disabled {{
    color: #94a3b8;
    background-color: #f8fafc;
    border-color: {C_BORDER};
}}
QPushButton#primaryBtn {{
    background-color: {C_PRIMARY};
    color: white;
    border: none;
    font-weight: bold;
}}
QPushButton#primaryBtn:hover {{ background-color: {C_PRIMARY_H}; }}
QPushButton#primaryBtn:disabled {{ background-color: #93c5fd; color: #eff6ff; }}
QPushButton#zoomBtn {{
    min-width: 36px;
    max-width: 36px;
    padding: 6px;
    font-size: 15px;
    font-weight: bold;
}}
QComboBox {{
    background-color: {C_CARD};
    border: 1px solid {C_BORDER};
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 13px;
    color: {C_TEXT};
}}
QComboBox:hover {{ border-color: #cbd5e1; }}
QComboBox::drop-down {{ border: none; width: 28px; }}
QComboBox QAbstractItemView {{
    border: 1px solid {C_BORDER};
    border-radius: 8px;
    selection-background-color: #eff6ff;
    selection-color: {C_PRIMARY};
    padding: 4px;
}}
QProgressBar {{
    border: none;
    border-radius: 4px;
    background: #e2e8f0;
    max-height: 4px;
}}
QProgressBar::chunk {{ background-color: {C_PRIMARY}; border-radius: 4px; }}
QScrollArea {{ border: none; background: transparent; }}
"""


class CardPanel(QFrame):
    """둥근 카드형 패널."""

    def __init__(self, title: str, subtitle: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("cardPanel")
        self.setStyleSheet(f"""
            QFrame#cardPanel {{
                background-color: {C_CARD};
                border: 1px solid {C_BORDER};
                border-radius: 14px;
            }}
        """)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 16, 18, 18)
        outer.setSpacing(10)

        header = QVBoxLayout()
        header.setSpacing(2)
        self.title_label = QLabel(title)
        self.title_label.setFont(QFont("맑은 고딕", 13, QFont.Bold))
        self.title_label.setStyleSheet(f"color: {C_TEXT}; border: none;")
        header.addWidget(self.title_label)

        if subtitle:
            sub = QLabel(subtitle)
            sub.setStyleSheet(f"color: {C_MUTED}; font-size: 11px; border: none;")
            header.addWidget(sub)

        self.content = QWidget()
        self.content.setStyleSheet("border: none; background: transparent;")
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(8)

        outer.addLayout(header)
        outer.addWidget(self.content, stretch=1)


class PhotoDropArea(QFrame):
    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: #f8fafc;
                border: 2px dashed #cbd5e1;
                border-radius: 10px;
            }}
        """)
        self.on_open_dialog = None
        self.on_file_path = None
        self.on_wheel_zoom = None

    @classmethod
    def is_image_path(cls, path: str) -> bool:
        return os.path.splitext(path)[1].lower() in cls.IMAGE_EXTS

    def set_handlers(self, open_dialog, file_path, wheel_zoom=None):
        self.on_open_dialog = open_dialog
        self.on_file_path = file_path
        self.on_wheel_zoom = wheel_zoom

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.on_open_dialog:
            self.on_open_dialog()
        super().mousePressEvent(event)

    def wheelEvent(self, event):
        if self.on_wheel_zoom and event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self.on_wheel_zoom(1.12)
            elif delta < 0:
                self.on_wheel_zoom(0.89)
            event.accept()
            return
        super().wheelEvent(event)

    def dragEnterEvent(self, event):
        if self._extract_path(event):
            self.setStyleSheet(f"""
                QFrame {{
                    background-color: #eff6ff;
                    border: 2px dashed {C_PRIMARY};
                    border-radius: 10px;
                }}
            """)
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.setStyleSheet(f"""
            QFrame {{
                background-color: #f8fafc;
                border: 2px dashed #cbd5e1;
                border-radius: 10px;
            }}
        """)
        event.accept()

    def dropEvent(self, event):
        self.dragLeaveEvent(event)
        path = self._extract_path(event)
        if path and self.on_file_path:
            self.on_file_path(path)
            event.acceptProposedAction()
        else:
            event.ignore()

    def _extract_path(self, event):
        if not event.mimeData().hasUrls():
            return None
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path and self.is_image_path(path):
                return path
        return None


class EngineInitWorker(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def run(self):
        try:
            self.finished.emit(DeepfakeEngine())
        except Exception as exc:
            self.error.emit(str(exc))


class AnalysisWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, engine, image_path, model_key):
        super().__init__()
        self.engine = engine
        self.image_path = image_path
        self.model_key = model_key

    def run(self):
        try:
            real_prob, fake_prob, logit, heatmap, _, temperature, ensemble_details = (
                self.engine.process_image(self.image_path, self.model_key)
            )
            self.finished.emit({
                "real_prob": real_prob,
                "fake_prob": fake_prob,
                "logit": logit,
                "heatmap": heatmap,
                "model_key": self.model_key,
                "temperature": temperature,
                "ensemble_details": ensemble_details,
            })
        except Exception as exc:
            self.error.emit(str(exc))


class DeepfakeDashboard(QMainWindow):
    IMAGE_BOX_SIZE = 440
    GRADCAM_BOX_SIZE = 440
    ZOOM_MIN, ZOOM_MAX = 0.5, 2.5

    def __init__(self):
        super().__init__()
        self.engine = None
        self.current_image_path = None
        self.last_heatmap = None
        self.last_analysis = None
        self._zoom = 1.0
        self._zoom_labels: list[QLabel] = []
        self._init_worker = None
        self._analysis_worker = None
        self.initUI()
        self._setup_shortcuts()
        self._start_engine_init()

    def initUI(self):
        self.setWindowTitle("딥페이크 탐지 시스템")
        self.setMinimumSize(1000, 700)
        self.resize(1080, 760)
        self.setStyleSheet(APP_STYLE)

        self.create_menu_bar()
        self._zoom_status_label = QLabel("100%")
        self._zoom_status_label.setStyleSheet(f"color: {C_MUTED}; padding: 0 8px;")
        self.statusBar().addPermanentWidget(self._zoom_status_label)
        self.statusBar().showMessage("모델 로딩 중...")

        central = QWidget()
        central.setObjectName("centralRoot")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(16)

        # ── 상단 헤더 ──
        header = QHBoxLayout()
        title_block = QVBoxLayout()
        title_block.setSpacing(2)
        app_title = QLabel("딥페이크 탐지")
        app_title.setFont(QFont("맑은 고딕", 20, QFont.Bold))
        app_title.setStyleSheet(f"color: {C_TEXT};")
        app_sub = QLabel("Deepfake Detection · Grad-CAM")
        app_sub.setStyleSheet(f"color: {C_MUTED}; font-size: 12px;")
        title_block.addWidget(app_title)
        title_block.addWidget(app_sub)
        header.addLayout(title_block)
        header.addStretch()

        shortcut_hint = QLabel("Ctrl+휠 또는 Ctrl+± 확대 · Ctrl+0 초기화 · F5 분석")
        shortcut_hint.setStyleSheet(f"color: {C_MUTED}; font-size: 11px;")
        header.addWidget(shortcut_hint, alignment=Qt.AlignVCenter)
        root.addLayout(header)

        body = QHBoxLayout()
        body.setSpacing(20)
        body.addLayout(self._build_left_panel(), stretch=1)
        body.addLayout(self._build_right_panel(), stretch=1)
        root.addLayout(body, stretch=1)

    def _make_zoom_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setSpacing(6)

        btn_out = QPushButton("−")
        btn_out.setObjectName("zoomBtn")
        btn_out.setToolTip("축소 (Ctrl+-)")
        btn_out.clicked.connect(lambda: self.adjust_zoom(0.87))

        self._zoom_label = QLabel("100%")
        self._zoom_label.setAlignment(Qt.AlignCenter)
        self._zoom_label.setFixedWidth(52)
        self._zoom_label.setStyleSheet(f"color: {C_MUTED}; font-size: 12px; border: none;")
        self._zoom_labels.append(self._zoom_label)

        btn_in = QPushButton("+")
        btn_in.setObjectName("zoomBtn")
        btn_in.setToolTip("확대 (Ctrl++)")
        btn_in.clicked.connect(lambda: self.adjust_zoom(1.15))

        btn_reset = QPushButton("100%")
        btn_reset.setToolTip("줌 초기화 (Ctrl+0)")
        btn_reset.setFixedWidth(56)
        btn_reset.setStyleSheet("font-size: 11px; padding: 6px 8px;")
        btn_reset.clicked.connect(self.reset_zoom)

        bar.addStretch()
        bar.addWidget(btn_out)
        bar.addWidget(self._zoom_label)
        bar.addWidget(btn_in)
        bar.addWidget(btn_reset)
        return bar

    def _build_left_panel(self) -> QVBoxLayout:
        layout = QVBoxLayout()
        layout.setSpacing(12)

        photo_panel = CardPanel("입력 이미지", "클릭 · 드래그 · Ctrl+O")
        photo_panel.setMinimumHeight(self.IMAGE_BOX_SIZE + 80)

        self.photo_drop = PhotoDropArea()
        self.photo_drop.set_handlers(
            open_dialog=self.load_image_event,
            file_path=self.set_image_path,
            wheel_zoom=self.adjust_zoom,
        )
        photo_drop_layout = QVBoxLayout(self.photo_drop)
        photo_drop_layout.setContentsMargins(8, 8, 8, 8)

        self.photo_stack = QStackedWidget()
        self.photo_stack.setMinimumHeight(self.IMAGE_BOX_SIZE)

        empty_page = QWidget()
        empty_layout = QVBoxLayout(empty_page)
        empty_layout.addStretch()
        icon_lbl = QLabel("🖼")
        icon_lbl.setAlignment(Qt.AlignCenter)
        icon_lbl.setStyleSheet("font-size: 36px; border: none;")
        hint = QLabel("사진을 불러오세요")
        hint.setAlignment(Qt.AlignCenter)
        hint.setFont(QFont("맑은 고딕", 12))
        hint.setStyleSheet(f"color: {C_MUTED}; border: none;")
        sub_hint = QLabel("Ctrl+O  ·  드래그 앤 드롭")
        sub_hint.setAlignment(Qt.AlignCenter)
        sub_hint.setStyleSheet(f"color: #94a3b8; font-size: 11px; border: none;")
        self.btn_load = QPushButton("사진 불러오기")
        self.btn_load.setObjectName("primaryBtn")
        self.btn_load.setFixedWidth(180)
        self.btn_load.setCursor(Qt.PointingHandCursor)
        self.btn_load.clicked.connect(self.load_image_event)
        empty_layout.addWidget(icon_lbl)
        empty_layout.addWidget(hint)
        empty_layout.addWidget(sub_hint)
        empty_layout.addSpacing(12)
        empty_layout.addWidget(self.btn_load, alignment=Qt.AlignCenter)
        empty_layout.addStretch()

        image_page = QWidget()
        image_layout = QVBoxLayout(image_page)
        image_layout.setContentsMargins(0, 0, 0, 0)
        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setMinimumSize(self.IMAGE_BOX_SIZE, self.IMAGE_BOX_SIZE)
        self.img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.img_label.setStyleSheet("border: none; background: transparent;")
        image_layout.addWidget(self.img_label)

        self.photo_stack.addWidget(empty_page)
        self.photo_stack.addWidget(image_page)
        photo_drop_layout.addWidget(self.photo_stack)
        photo_panel.content_layout.addWidget(self.photo_drop)
        photo_panel.content_layout.addLayout(self._make_zoom_bar())
        layout.addWidget(photo_panel, stretch=1)

        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(10)

        self.btn_run = QPushButton("▶  결과 보기")
        self.btn_run.setObjectName("primaryBtn")
        self.btn_run.setEnabled(False)
        self.btn_run.setCursor(Qt.PointingHandCursor)
        self.btn_run.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_run.setMinimumHeight(44)
        self.btn_run.clicked.connect(self.run_analysis)

        model_wrap = QVBoxLayout()
        model_wrap.setSpacing(4)
        model_lbl = QLabel("모델 선택")
        model_lbl.setStyleSheet(f"color: {C_MUTED}; font-size: 11px;")
        self.combo_model = QComboBox()
        self.combo_model.setMinimumHeight(44)
        for key in MODEL_KEYS:
            self.combo_model.addItem(MODEL_CONFIG[key]["label"], key)
        self.combo_model.addItem(MODEL_CONFIG[ENSEMBLE_KEY]["label"], ENSEMBLE_KEY)
        model_wrap.addWidget(model_lbl)
        model_wrap.addWidget(self.combo_model)

        ctrl_row.addWidget(self.btn_run, stretch=2)
        ctrl_row.addLayout(model_wrap, stretch=3)
        layout.addLayout(ctrl_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        return layout

    def _build_right_panel(self) -> QVBoxLayout:
        layout = QVBoxLayout()
        layout.setSpacing(12)

        result_panel = CardPanel("분류 결과")
        result_panel.setMinimumHeight(180)
        result_panel.setMaximumHeight(340)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        result_body = QWidget()
        result_body_layout = QVBoxLayout(result_body)
        result_body_layout.setContentsMargins(4, 4, 4, 4)

        self.result_badge = QLabel("—")
        self.result_badge.setAlignment(Qt.AlignCenter)
        self.result_badge.setFont(QFont("맑은 고딕", 28, QFont.Bold))
        self.result_badge.setStyleSheet(f"color: {C_MUTED}; border: none; padding: 8px 0;")

        self.result_detail = QLabel("사진을 불러온 뒤 [결과 보기]를 눌러주세요.")
        self.result_detail.setAlignment(Qt.AlignCenter)
        self.result_detail.setWordWrap(True)
        self.result_detail.setFont(QFont("맑은 고딕", 10))
        self.result_detail.setStyleSheet(f"color: {C_MUTED}; border: none; line-height: 1.5;")

        result_body_layout.addWidget(self.result_badge)
        result_body_layout.addWidget(self.result_detail)
        scroll.setWidget(result_body)
        result_panel.content_layout.addWidget(scroll)
        layout.addWidget(result_panel)

        gradcam_panel = CardPanel("Grad-CAM", "모델 판단 근거 시각화")
        gradcam_panel.setMinimumHeight(self.GRADCAM_BOX_SIZE + 80)

        self.gradcam_container = QFrame()
        self.gradcam_container.setStyleSheet(f"""
            QFrame {{
                background-color: #f8fafc;
                border: 1px solid {C_BORDER};
                border-radius: 10px;
            }}
        """)
        gradcam_inner = QVBoxLayout(self.gradcam_container)
        gradcam_inner.setContentsMargins(8, 8, 8, 8)

        self.gradcam_label = QLabel("분석 후 히트맵이 표시됩니다")
        self.gradcam_label.setAlignment(Qt.AlignCenter)
        self.gradcam_label.setMinimumSize(self.GRADCAM_BOX_SIZE, self.GRADCAM_BOX_SIZE)
        self.gradcam_label.setStyleSheet(f"color: {C_MUTED}; border: none;")
        self.gradcam_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.gradcam_label.installEventFilter(self)

        gradcam_inner.addWidget(self.gradcam_label)
        gradcam_panel.content_layout.addWidget(self.gradcam_container)
        gradcam_panel.content_layout.addLayout(self._make_zoom_bar())
        layout.addWidget(gradcam_panel, stretch=1)

        return layout

    def eventFilter(self, obj, event):
        if obj is self.gradcam_label and event.type() == QEvent.Wheel:
            if event.modifiers() & Qt.ControlModifier:
                delta = event.angleDelta().y()
                if delta > 0:
                    self.adjust_zoom(1.12)
                elif delta < 0:
                    self.adjust_zoom(0.89)
                return True
        return super().eventFilter(obj, event)

    def _setup_shortcuts(self):
        def _add(key, slot):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ApplicationShortcut)
            sc.activated.connect(slot)

        _add("Ctrl+O", self.load_image_event)
        _add("Ctrl+S", self.save_gradcam)
        _add("Ctrl+Z", self.undo_analysis)
        _add("F5", self.run_analysis)
        _add("Return", self.run_analysis)
        _add("Ctrl++", lambda: self.adjust_zoom(1.15))
        _add("Ctrl+=", lambda: self.adjust_zoom(1.15))
        _add("Ctrl+-", lambda: self.adjust_zoom(0.87))
        _add("Ctrl+0", self.reset_zoom)

    def create_menu_bar(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("파일")
        self._add_action(file_menu, "불러오기", self.load_image_event, "Ctrl+O")
        self._add_action(file_menu, "저장", self.save_gradcam, "Ctrl+S")
        file_menu.addAction("저장 기록 폴더 열기", self.open_results_folder)

        edit_menu = menubar.addMenu("편집")
        self._add_action(edit_menu, "되돌리기", self.undo_analysis, "Ctrl+Z")

        view_menu = menubar.addMenu("보기")
        self._add_action(view_menu, "확대", lambda: self.adjust_zoom(1.15), "Ctrl++")
        self._add_action(view_menu, "축소", lambda: self.adjust_zoom(0.87), "Ctrl+-")
        self._add_action(view_menu, "줌 초기화", self.reset_zoom, "Ctrl+0")

        analyze_menu = menubar.addMenu("분석")
        self._add_action(analyze_menu, "결과 보기", self.run_analysis, "F5")

    @staticmethod
    def _add_action(menu, text, slot, shortcut=None):
        action = QAction(text, menu)
        action.triggered.connect(slot)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        menu.addAction(action)

    def _update_zoom_display(self):
        pct = f"{self._zoom * 100:.0f}%"
        for lbl in self._zoom_labels:
            lbl.setText(pct)
        self._zoom_status_label.setText(f"줌 {pct}")

    def _display_size(self, base: int) -> QSize:
        return QSize(int(base * self._zoom), int(base * self._zoom))

    def adjust_zoom(self, factor: float = None, delta: float = None):
        if delta is not None:
            self._zoom = max(self.ZOOM_MIN, min(self.ZOOM_MAX, self._zoom + delta))
        elif factor is not None:
            self._zoom = max(self.ZOOM_MIN, min(self.ZOOM_MAX, self._zoom * factor))
        if self.current_image_path:
            self._refresh_image()
        if self.last_heatmap is not None:
            self._show_heatmap(self.last_heatmap)
        self._update_zoom_display()
        self.statusBar().showMessage(f"확대/축소: {self._zoom * 100:.0f}%")

    def reset_zoom(self):
        self._zoom = 1.0
        if self.current_image_path:
            self._refresh_image()
        if self.last_heatmap is not None:
            self._show_heatmap(self.last_heatmap)
        self._update_zoom_display()
        self.statusBar().showMessage("줌 100%로 초기화")

    def _refresh_image(self):
        if not self.current_image_path:
            return
        size = self._display_size(self.IMAGE_BOX_SIZE)
        self.img_label.setPixmap(
            QPixmap(self.current_image_path).scaled(
                size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def _show_heatmap(self, heatmap):
        h, w, ch = heatmap.shape
        rgb = np.ascontiguousarray(heatmap)
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        size = self._display_size(self.GRADCAM_BOX_SIZE)
        self.gradcam_label.setPixmap(
            QPixmap.fromImage(qt_img).scaled(
                size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )
        self.gradcam_label.setStyleSheet("border: none;")

    def load_image_event(self, event=None):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "사진 선택", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if file_path:
            self.set_image_path(file_path)

    def set_image_path(self, file_path: str):
        if not file_path or not PhotoDropArea.is_image_path(file_path):
            QMessageBox.warning(self, "경고", "지원하는 이미지 파일이 아닙니다.")
            return

        self.current_image_path = file_path
        self.photo_stack.setCurrentIndex(1)
        self.photo_drop.setStyleSheet(f"""
            QFrame {{
                background-color: {C_CARD};
                border: 1px solid {C_BORDER};
                border-radius: 10px;
            }}
        """)
        self._refresh_image()
        self._clear_analysis()
        if self.engine:
            self.btn_run.setEnabled(True)
        self.statusBar().showMessage(f"사진 불러옴: {os.path.basename(file_path)}")

    def _clear_analysis(self):
        self.last_heatmap = None
        self.last_analysis = None
        self.result_badge.setText("—")
        self.result_badge.setStyleSheet(f"color: {C_MUTED}; border: none; padding: 8px 0;")
        self.result_detail.setText("사진을 불러온 뒤 [결과 보기]를 눌러주세요.")
        self.gradcam_label.clear()
        self.gradcam_label.setText("분석 후 히트맵이 표시됩니다")
        self.gradcam_label.setStyleSheet(f"color: {C_MUTED}; border: none;")

    def undo_analysis(self):
        self._clear_analysis()
        self.statusBar().showMessage("분석 결과를 되돌렸습니다.")

    def _start_engine_init(self):
        self._init_worker = EngineInitWorker()
        self._init_worker.finished.connect(self._on_engine_ready)
        self._init_worker.error.connect(self._on_engine_error)
        self._init_worker.start()

    def _on_engine_ready(self, engine):
        self.engine = engine
        self.statusBar().showMessage("모델 준비 완료 · Ctrl+O 불러오기 · F5 분석")
        if self.current_image_path:
            self.btn_run.setEnabled(True)

    def _on_engine_error(self, message):
        self.statusBar().showMessage("모델 로드 실패")
        QMessageBox.critical(
            self, "모델 로드 오류",
            f"models/ 폴더의 .pth 파일을 확인하세요.\n\n{message}",
        )

    def run_analysis(self):
        if not self.engine:
            QMessageBox.warning(self, "대기", "모델 로딩이 완료될 때까지 기다려 주세요.")
            return
        if not self.current_image_path:
            QMessageBox.warning(self, "경고", "이미지를 먼저 불러오세요.")
            return

        model_key = self.combo_model.currentData()
        model_label = MODEL_CONFIG[model_key]["label"]

        self.btn_run.setEnabled(False)
        self.progress.setVisible(True)
        self.statusBar().showMessage(f"{model_label} 분석 중...")

        self._analysis_worker = AnalysisWorker(
            self.engine, self.current_image_path, model_key
        )
        self._analysis_worker.finished.connect(self._on_analysis_done)
        self._analysis_worker.error.connect(self._on_analysis_error)
        self._analysis_worker.start()

    def _on_analysis_done(self, payload: dict):
        self.progress.setVisible(False)
        self.btn_run.setEnabled(True)

        real_prob = payload["real_prob"]
        fake_prob = payload["fake_prob"]
        heatmap = payload["heatmap"]
        model_key = payload["model_key"]
        temperature = payload["temperature"]
        ensemble_details = payload.get("ensemble_details")

        self.last_heatmap = heatmap
        result = DeepfakeEngine.classify(real_prob, fake_prob)
        model_label = MODEL_CONFIG[model_key]["label"]

        self.last_analysis = {
            "model_key": model_key,
            "model_label": model_label,
            "model_file": MODEL_CONFIG[model_key]["file"],
            "real_prob": real_prob,
            "fake_prob": fake_prob,
            "logit": payload["logit"],
            "temperature": temperature,
            "verdict_label": result["label"],
            "confidence": result["confidence"],
            "is_fake": result["is_fake"],
            "ensemble_details": ensemble_details,
        }

        if result["is_fake"]:
            self.result_badge.setText("위조 (Fake)")
            self.result_badge.setStyleSheet(f"color: {C_FAKE}; border: none; padding: 8px 0;")
        else:
            self.result_badge.setText("원본 (Real)")
            self.result_badge.setStyleSheet(f"color: {C_REAL}; border: none; padding: 8px 0;")

        detail_lines = [
            f"모델: {model_label}",
            f"온도 보정 (T={temperature:.1f})",
            f"Real {real_prob * 100:.1f}%  ·  Fake {fake_prob * 100:.1f}%",
            f"판정 신뢰도: {result['confidence'] * 100:.1f}%",
        ]
        if ensemble_details:
            detail_lines.append("")
            wi = ensemble_details.get("weight_info", {})
            if wi:
                detail_lines.append(
                    f"[앙상블 · {wi.get('best_model_label', '최고')} 모델 기준]"
                )
            detail_lines.append(DeepfakeEngine.format_ensemble_weights_text())
            detail_lines.append("")
            for m in ensemble_details["members"]:
                tag = "Fake" if m["is_fake"] else "Real"
                detail_lines.append(
                    f"  {m['model_label']}: {tag} Fake {m['fake_prob'] * 100:.1f}% "
                    f"(w={m['weight'] * 100:.1f}%)"
                )

        self.result_detail.setText("\n".join(detail_lines))
        self._show_heatmap(heatmap)
        self.statusBar().showMessage(
            f"{model_label} — {result['label']} (Fake {fake_prob * 100:.1f}%)"
        )

    def _on_analysis_error(self, message):
        self.progress.setVisible(False)
        self.btn_run.setEnabled(True)
        self.statusBar().showMessage("분석 실패")
        QMessageBox.critical(self, "분석 오류", message)

    def save_gradcam(self):
        if self.last_heatmap is None or self.last_analysis is None:
            QMessageBox.warning(self, "경고", "저장할 분석 결과가 없습니다.\n먼저 [결과 보기]를 실행하세요.")
            return
        if not self.current_image_path:
            QMessageBox.warning(self, "경고", "원본 이미지 경로를 찾을 수 없습니다.")
            return

        try:
            a = self.last_analysis
            save_dir = save_analysis_report(
                image_path=self.current_image_path,
                heatmap=self.last_heatmap,
                model_key=a["model_key"],
                model_label=a["model_label"],
                model_file=a["model_file"],
                real_prob=a["real_prob"],
                fake_prob=a["fake_prob"],
                logit=a["logit"],
                temperature=a["temperature"],
                verdict_label=a["verdict_label"],
                confidence=a["confidence"],
                is_fake=a["is_fake"],
                ensemble_details=a.get("ensemble_details"),
            )
        except Exception as exc:
            QMessageBox.critical(self, "저장 오류", str(exc))
            return

        QMessageBox.information(
            self, "저장 완료",
            f"분석 기록이 저장되었습니다.\n\n폴더: {save_dir}",
        )
        self.statusBar().showMessage(f"저장 완료: {save_dir}")

    def open_results_folder(self):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(RESULTS_DIR)
        elif sys.platform == "darwin":
            subprocess.run(["open", RESULTS_DIR], check=False)
        else:
            subprocess.run(["xdg-open", RESULTS_DIR], check=False)
        self.statusBar().showMessage(f"저장 기록 폴더: {RESULTS_DIR}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("맑은 고딕", 10))
    window = DeepfakeDashboard()
    window.show()
    sys.exit(app.exec_())
