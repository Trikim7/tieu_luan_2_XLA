import sys, os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QFileDialog, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QComboBox, QSpinBox, QGroupBox, QMessageBox, QAction, QCheckBox,
    QFrame, QSplitter, QScrollArea
)
from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QFont, QPalette, QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from . import filters as F
from . import enhancement as E
from . import utils as U

def np_to_qpixmap(img_rgb: np.ndarray) -> QPixmap:
    h, w, ch = img_rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

class HistCanvas(FigureCanvas):
    def __init__(self, title="", parent=None):
        fig = Figure(figsize=(5, 3), tight_layout=True, facecolor='#1e1e1e')
        super().__init__(fig)
        self.ax = fig.add_subplot(111)
        self.ax.set_facecolor('#1e1e1e')
        self.title = title
        self.setParent(parent)

    def plot_hist(self, img_rgb: np.ndarray):
        self.ax.clear()
        self.ax.set_facecolor('#1e1e1e')
        
        if img_rgb.ndim == 3:
            r, g, b = img_rgb[...,0].ravel(), img_rgb[...,1].ravel(), img_rgb[...,2].ravel()
            self.ax.hist(r, bins=256, alpha=0.7, color='#ff6b6b', label='Red', density=True)
            self.ax.hist(g, bins=256, alpha=0.7, color='#4ecdc4', label='Green', density=True)
            self.ax.hist(b, bins=256, alpha=0.7, color='#45b7d1', label='Blue', density=True)
        else:
            self.ax.hist(img_rgb.ravel(), bins=256, alpha=0.8, color='#95a5a6', label='Gray', density=True)
        
        self.ax.set_title(self.title if self.title else 'Histogram', color='white', fontsize=10, fontweight='bold')
        self.ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
        self.ax.tick_params(colors='white', labelsize=8)
        self.ax.grid(True, alpha=0.3, color='#555')
        self.ax.set_xlim(0, 255)
        self.draw_idle()

class ImageLabel(QLabel):
    def __init__(self, on_drop_callback=None, title="", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)
        self.on_drop_callback = on_drop_callback
        self.title = title
        self.setMinimumHeight(300)
        self.setAlignment(Qt.AlignCenter)
        
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("""
                background-color: #2d3748; 
                border: 2px dashed #4299e1; 
                border-radius: 12px; 
                color: #4299e1;
                font-size: 14px;
                font-weight: bold;
            """)
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.setStyleSheet("""
            background-color: #1a202c; 
            border: 2px dashed #4a5568; 
            border-radius: 12px; 
            color: #a0aec0;
            font-size: 14px;
        """)

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isfile(path) and os.path.splitext(path)[1].lower() in {'.png','.jpg','.jpeg','.bmp'}:
                if self.on_drop_callback:
                    self.on_drop_callback(path)
                break
        self.setStyleSheet("""
            background-color: #1a202c; 
            border: 2px dashed #4a5568; 
            border-radius: 12px; 
            color: #a0aec0;
            font-size: 14px;
        """)

class PhotoEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('🎨 Mini Photo Editor - PyQt5')
        self.resize(1400, 900)
        self.img = None
        self.img_original = None
        self.view_mode = 'fit'
        self.scale = 1.0
        self.build_ui()
        self.apply_style()
        self.setMinimumSize(1200, 700)

    def build_ui(self):
        # Menu bar
        open_act = QAction('📁 Open', self); open_act.triggered.connect(self.open_image)
        save_act = QAction('💾 Save As', self); save_act.triggered.connect(self.save_image)
        reset_act = QAction('🔄 Reset Image', self); reset_act.triggered.connect(self.reset_image)
        menubar = self.menuBar(); file_menu = menubar.addMenu('File')
        for a in (open_act, save_act, reset_act): file_menu.addAction(a)

        # Image display area
        self.lbl_original = ImageLabel(on_drop_callback=self.load_path, title="Original")
        self.lbl_result   = ImageLabel(on_drop_callback=self.load_path, title="Result")
        
        # Set default text and style for image labels
        for lbl, name in [(self.lbl_original,'📸 Original'),(self.lbl_result,'✨ Result')]:
            lbl.setText(f'{name}\n\n🖱️ Drag & drop image here\nor click Upload button')
            lbl.setStyleSheet("""
                background-color: #1a202c; 
                border: 2px dashed #4a5568; 
                border-radius: 12px; 
                color: #a0aec0;
                font-size: 14px;
                padding: 20px;
            """)

        # Create splitter for resizable panels
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Left panel for images
        left_panel = QWidget()
        img_layout = QHBoxLayout(left_panel)
        img_layout.setSpacing(10)
        img_layout.addWidget(self.lbl_original, 1)
        img_layout.addWidget(self.lbl_result, 1)
        
        # Right panel for controls
        right_panel = QWidget()
        right_panel.setFixedWidth(350)
        
        # Controls section
        self.combo_op = QComboBox()
        operations = [
            '🔹 Mean Blur', '🔹 Gaussian Blur', '🔹 Median Blur', '🔹 Bilateral',
            '⚡ Sharpen (Unsharp)', '⚡ Sharpen (Laplacian)',
            '🔍 Edge: Sobel', '🔍 Edge: Prewitt', '🔍 Edge: Laplacian', '🔍 Edge: Canny',
            '📊 HistEq (global)', '📊 CLAHE'
        ]
        self.combo_op.addItems(operations)
        self.combo_op.setStyleSheet("""
            QComboBox {
                padding: 8px 12px;
                font-size: 13px;
                border-radius: 6px;
                background-color: #2d3748;
                border: 1px solid #4a5568;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #a0aec0;
                margin-right: 5px;
            }
        """)
        
        # Parameter controls with improved styling
        self.spin_kernel = QSpinBox()
        self.spin_kernel.setRange(1,99)
        self.spin_kernel.setValue(5)
        self.spin_kernel.setSingleStep(2)
        
        self.spin_sigma = QSpinBox()
        self.spin_sigma.setRange(0,50)
        self.spin_sigma.setValue(1)
        
        for spin in [self.spin_kernel, self.spin_sigma]:
            spin.setStyleSheet("""
                QSpinBox {
                    padding: 6px 8px;
                    border-radius: 6px;
                    background-color: #2d3748;
                    border: 1px solid #4a5568;
                    font-size: 13px;
                }
            """)
        
        self.slider_thresh1 = QSlider(Qt.Horizontal)
        self.slider_thresh1.setRange(0,255)
        self.slider_thresh1.setValue(100)
        
        self.slider_thresh2 = QSlider(Qt.Horizontal)
        self.slider_thresh2.setRange(0,255)
        self.slider_thresh2.setValue(200)
        
        for slider in [self.slider_thresh1, self.slider_thresh2]:
            slider.setStyleSheet("""
                QSlider::groove:horizontal {
                    border: 1px solid #4a5568;
                    height: 8px;
                    background: #2d3748;
                    border-radius: 4px;
                }
                QSlider::handle:horizontal {
                    background: #4299e1;
                    border: 2px solid #2b6cb0;
                    width: 18px;
                    margin: -5px 0;
                    border-radius: 9px;
                }
                QSlider::handle:horizontal:hover {
                    background: #63b3ed;
                }
            """)
        
        self.chk_hist = QCheckBox('📈 Hiển thị Histogram')
        self.chk_hist.setStyleSheet("""
            QCheckBox {
                font-size: 13px;
                font-weight: bold;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 3px;
                border: 2px solid #4a5568;
                background-color: #2d3748;
            }
            QCheckBox::indicator:checked {
                background-color: #4299e1;
                border-color: #2b6cb0;
            }
        """)

        # Buttons with modern styling
        buttons_data = [
            ('📁 Upload', self.open_image, '#22c55e'),
            ('✨ Apply', self.apply_operation, '#3b82f6'),
            ('💾 Save', self.save_image, '#8b5cf6')
        ]
        
        action_buttons = []
        for text, callback, color in buttons_data:
            btn = QPushButton(text)
            btn.clicked.connect(callback)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color};
                    color: white;
                    border: none;
                    padding: 10px 16px;
                    font-size: 13px;
                    font-weight: bold;
                    border-radius: 8px;
                    min-height: 16px;
                }}
                QPushButton:hover {{
                    background-color: {color}dd;
                    transform: translateY(-1px);
                }}
                QPushButton:pressed {{
                    background-color: {color}bb;
                }}
            """)
            action_buttons.append(btn)

        # View control buttons
        view_buttons_data = [
            ('🔍 Fit', lambda: self.set_view_mode('fit')),
            ('🔍+', lambda: self.zoom_by(1.25)),
            ('🔍-', lambda: self.zoom_by(0.8)),
            ('100%', self.zoom_reset),
            ('🔄', self.reset_view)
        ]
        
        view_buttons = []
        for text, callback in view_buttons_data:
            btn = QPushButton(text)
            btn.clicked.connect(callback)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #374151;
                    color: white;
                    border: 1px solid #4b5563;
                    padding: 8px 12px;
                    font-size: 12px;
                    border-radius: 6px;
                    min-height: 14px;
                }
                QPushButton:hover {
                    background-color: #4b5563;
                    border-color: #6b7280;
                }
            """)
            view_buttons.append(btn)

        # Layout for controls
        ctrl_layout = QVBoxLayout()
        ctrl_layout.setSpacing(15)
        
        # Operation selection
        op_group = QGroupBox('🎛️ Operation')
        op_layout = QVBoxLayout()
        op_layout.addWidget(self.combo_op)
        op_group.setLayout(op_layout)
        
        # Parameters
        param_group = QGroupBox('⚙️ Parameters')
        param_layout = QVBoxLayout()
        
        kernel_layout = QHBoxLayout()
        kernel_layout.addWidget(QLabel('Kernel Size:'))
        kernel_layout.addWidget(self.spin_kernel)
        
        sigma_layout = QHBoxLayout()
        sigma_layout.addWidget(QLabel('Sigma:'))
        sigma_layout.addWidget(self.spin_sigma)
        
        thresh1_layout = QVBoxLayout()
        thresh1_layout.addWidget(QLabel('Threshold 1:'))
        thresh1_layout.addWidget(self.slider_thresh1)
        
        thresh2_layout = QVBoxLayout()
        thresh2_layout.addWidget(QLabel('Threshold 2:'))
        thresh2_layout.addWidget(self.slider_thresh2)
        
        param_layout.addLayout(kernel_layout)
        param_layout.addLayout(sigma_layout)
        param_layout.addLayout(thresh1_layout)
        param_layout.addLayout(thresh2_layout)
        param_group.setLayout(param_layout)
        
        # Actions
        action_group = QGroupBox('🚀 Actions')
        action_layout = QVBoxLayout()
        action_layout.setSpacing(8)
        for btn in action_buttons:
            action_layout.addWidget(btn)
        action_group.setLayout(action_layout)
        
        # View controls
        view_group = QGroupBox('👁️ View Controls')
        view_layout = QHBoxLayout()
        view_layout.setSpacing(5)
        for btn in view_buttons:
            view_layout.addWidget(btn)
        view_group.setLayout(view_layout)
        
        # Add histogram checkbox
        hist_layout = QVBoxLayout()
        hist_layout.addWidget(self.chk_hist)
        
        # Add all groups to control layout
        ctrl_layout.addWidget(op_group)
        ctrl_layout.addWidget(param_group)
        ctrl_layout.addLayout(hist_layout)
        ctrl_layout.addWidget(action_group)
        ctrl_layout.addWidget(view_group)
        ctrl_layout.addStretch()
        
        right_panel.setLayout(ctrl_layout)

        # Histogram panel
        self.hist_panel = QGroupBox('📊 Histogram Analysis')
        vhist = QHBoxLayout()
        self.hist_canvas_before = HistCanvas('Before', self)
        self.hist_canvas_after  = HistCanvas('After', self)
        vhist.addWidget(self.hist_canvas_before, 1)
        vhist.addWidget(self.hist_canvas_after, 1)
        self.hist_panel.setLayout(vhist)
        self.hist_panel.setVisible(False)
        self.chk_hist.stateChanged.connect(lambda s: self.toggle_hist(s == Qt.Checked))

        # Main layout
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([1000, 350])
        
        # Create main container with histogram panel
        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.addWidget(main_splitter, 1)
        main_layout.addWidget(self.hist_panel)
        
        self.setCentralWidget(container)

    def apply_style(self):
        self.setStyleSheet('''
            QMainWindow { 
                background-color: #0f172a; 
                color: #e2e8f0;
                font-family: "Segoe UI", Arial, sans-serif;
            }
            
            QLabel { 
                font-size: 13px; 
                color: #cbd5e1;
                font-weight: 500;
            }
            
            QGroupBox { 
                background-color: #1e293b;
                border: 2px solid #334155; 
                border-radius: 12px; 
                margin-top: 12px; 
                padding: 12px;
                font-weight: bold;
                font-size: 14px;
            }
            
            QGroupBox::title { 
                subcontrol-origin: margin; 
                left: 15px; 
                padding: 4px 8px; 
                color: #f1f5f9;
                background-color: #1e293b;
                border-radius: 6px;
            }
            
            QPushButton { 
                background-color: #3730a3; 
                color: white; 
                padding: 8px 16px; 
                border: none;
                border-radius: 8px; 
                font-size: 13px;
                font-weight: 600;
                min-height: 16px;
            }
            
            QPushButton:hover { 
                background-color: #4338ca;
                border: 2px solid #6366f1;
            }
            
            QPushButton:pressed {
                background-color: #312e81;
            }
            
            QComboBox, QSpinBox { 
                color: #e2e8f0;
                background-color: #334155;
                border: 1px solid #475569;
                border-radius: 6px;
                padding: 6px 8px;
                font-size: 13px;
            }
            
            QComboBox:hover, QSpinBox:hover {
                border-color: #64748b;
                background-color: #475569;
            }
            
            QSlider::groove:horizontal {
                border: 1px solid #475569;
                height: 6px;
                background: #334155;
                border-radius: 3px;
            }
            
            QSlider::handle:horizontal {
                background: #6366f1;
                border: 2px solid #4f46e5;
                width: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }
            
            QSlider::handle:horizontal:hover {
                background: #8b5cf6;
                border-color: #7c3aed;
            }
            
            QCheckBox { 
                color: #e2e8f0;
                font-size: 13px;
                font-weight: 500;
            }
            
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 3px;
                border: 2px solid #475569;
                background-color: #334155;
            }
            
            QCheckBox::indicator:checked {
                background-color: #6366f1;
                border-color: #4f46e5;
            }
            
            QMenuBar { 
                background-color: #1e293b; 
                color: #e2e8f0;
                border-bottom: 1px solid #334155;
                padding: 4px;
            }
            
            QMenuBar::item:selected { 
                background-color: #334155;
                border-radius: 4px;
            }
            
            QMenu {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 4px;
            }
            
            QMenu::item {
                padding: 8px 16px;
                border-radius: 4px;
            }
            
            QMenu::item:selected {
                background-color: #334155;
            }
            
            QSplitter::handle {
                background-color: #334155;
                width: 2px;
            }
            
            QSplitter::handle:hover {
                background-color: #475569;
            }
        ''')

    def set_view_mode(self, mode: str):
        self.view_mode = mode
        if mode == 'fit':
            self.scale = 1.0
        self.update_views()

    def zoom_by(self, factor: float):
        self.view_mode = 'zoom'
        self.scale = max(0.1, min(8.0, self.scale * factor))
        self.update_views()

    def zoom_reset(self):
        self.view_mode = 'zoom'
        self.scale = 1.0
        self.update_views()

    def reset_view(self):
        self.view_mode = 'fit'
        self.scale = 1.0
        self.update_views()

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Images (*.png *.jpg *.jpeg *.bmp)')
        if path: self.load_path(path)

    def load_path(self, path: str):
        try:
            img = U.read_image(path, as_gray=False)
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Không thể mở ảnh:\n{e}')
            return
        self.img_original = img.copy()
        self.img = img.copy()
        self.update_views()

    def save_image(self):
        if self.img is None:
            QMessageBox.warning(self, 'Warning', 'No image to save')
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Save Image', '', 'PNG (*.png);;JPEG (*.jpg *.jpeg)')
        if not path: return
        U.save_image(path, self.img)

    def reset_image(self):
        if self.img_original is not None:
            self.img = self.img_original.copy()
            self.update_views()

    def _scaled_pixmap(self, img_rgb: np.ndarray, target_label: QLabel) -> QPixmap:
        pix = np_to_qpixmap(img_rgb)
        if self.view_mode == 'fit':
            return pix.scaled(target_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        else:
            w = int(img_rgb.shape[1] * self.scale)
            h = int(img_rgb.shape[0] * self.scale)
            return pix.scaled(w, h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

    def update_views(self):
        if self.img_original is not None:
            self.lbl_original.setPixmap(self._scaled_pixmap(self.img_original, self.lbl_original))
        if self.img is not None:
            self.lbl_result.setPixmap(self._scaled_pixmap(self.img, self.lbl_result))
        if self.hist_panel.isVisible():
            if self.img_original is not None: self.hist_canvas_before.plot_hist(self.img_original)
            if self.img is not None: self.hist_canvas_after.plot_hist(self.img)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_views()

    def toggle_hist(self, show: bool):
        self.hist_panel.setVisible(show)
        if show: self.update_views()

    def apply_operation(self):
        if self.img is None:
            QMessageBox.warning(self, 'Warning', 'Please open an image first.')
            return
        
        op_text = self.combo_op.currentText()
        # Remove emoji from operation text for processing
        op = op_text.split(' ', 1)[-1] if ' ' in op_text else op_text
        
        k = int(self.spin_kernel.value())
        sigma = int(self.spin_sigma.value())
        t1 = int(self.slider_thresh1.value())
        t2 = int(self.slider_thresh2.value())

        img_rgb = self.img_original.copy()
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        if op == 'Mean Blur':
            out = F.mean_filter(img_rgb, k)
        elif op == 'Gaussian Blur':
            out = F.gaussian_filter(img_rgb, k, sigma)
        elif op == 'Median Blur':
            out = F.median_filter(img_rgb, k)
        elif op == 'Bilateral':
            out = F.bilateral_filter(img_rgb, d=max(3, k), sigmaColor=75, sigmaSpace=75)
        elif op == 'Sharpen (Unsharp)':
            out = E.unsharp_mask(img_rgb, k, sigma, amount=1.5, threshold=0)
        elif op == 'Sharpen (Laplacian)':
            sharp = E.laplacian_sharpen(img_gray)
            out = cv2.cvtColor(sharp, cv2.COLOR_GRAY2RGB)
        elif op == 'Edge: Sobel':
            _, _, mag = F.sobel(img_gray)
            out = cv2.cvtColor(mag, cv2.COLOR_GRAY2RGB)
        elif op == 'Edge: Prewitt':
            _, _, mag = F.prewitt(img_gray)
            out = cv2.cvtColor(mag, cv2.COLOR_GRAY2RGB)
        elif op == 'Edge: Laplacian':
            lap = F.laplacian(img_gray)
            out = cv2.cvtColor(lap, cv2.COLOR_GRAY2RGB)
        elif op == 'Edge: Canny':
            can = F.canny(img_gray, t1, t2)
            out = cv2.cvtColor(can, cv2.COLOR_GRAY2RGB)
        elif op == 'HistEq (global)':
            he = E.hist_equalization(img_gray)
            out = cv2.cvtColor(he, cv2.COLOR_GRAY2RGB)
        elif op == 'CLAHE':
            he = E.clahe_equalization(img_gray, clip_limit=2.0, tile_grid_size=(8,8))
            out = cv2.cvtColor(he, cv2.COLOR_GRAY2RGB)
        else:
            out = img_rgb

        self.img = out
        self.update_views()

def main():
    app = QApplication(sys.argv)
    w = PhotoEditor()
    w.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()