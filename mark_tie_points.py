import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import os

from rpc import RPCModelParameterTorch,load_rpc
from shapely.geometry import Polygon, Point, MultiPoint
from typing import List,Tuple,Optional,Dict
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !! 用户需要实现这个函数 (You need to implement this function)
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def find_windows(image_shapes:List[np.ndarray],rpcs:List[RPCModelParameterTorch],heights:List[np.ndarray],margin = 3000,size=500):
    H,W = image_shapes[0][:2]
    print(H,W)
    res = []
    lines = np.arange(margin,H - margin - size,size)
    samps = np.arange(margin,W - margin - size,size)
    lines,samps = np.meshgrid(lines,samps,indexing='ij')
    linesamps = np.stack([lines.ravel(),samps.ravel()],axis=-1)
    h = heights[0][linesamps[:,0],linesamps[:,1]]
    x,y = rpcs[0].RPC_LINESAMP2XY(linesamps[:,0],linesamps[:,1],h,'numpy')
    for rpc in rpcs:
        tl = np.stack(rpc.RPC_XY2LINESAMP(x,y,h,'numpy'),axis=-1).astype(int)
        br = tl + [size,size]
        window = np.stack([tl,br],axis=1).astype(int)
        res.append(window)
    return np.stack(res,axis=0) # (img_num,win_num,2,2)
    

def get_windows(k: int, windows:np.ndarray):
    """
    根据窗口索引 k 获取每个影像的窗口范围。
    Args:
        k (int): 当前窗口的索引，从0开始。
        windows:(img_num,win_num,2,2)
    Returns:
        np.ndarray: 一个 (N, 2, 2) 的numpy数组，N是影像数量。
                    每张影像的 [ [[r1, c1], [r2, c2]], ... ]
                    其中 (r1, c1) 是左上角行列号, (r2, c2) 是右下角行列号。
                    如果某个影像没有对应的窗口，可以返回一个无效的窗口，
                    例如 [[-1,-1],[-1,-1]]，程序会处理这种情况。
    """
    return windows[:,k]

    # ----- 结束: 用户实现区域 (End: User Implementation Area) -----
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !! 上述 get_windows 函数需要你来实现
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

class ImageViewer(tk.Canvas):
    """
    一个支持平移和缩放图像的高级画布。
    """
    def __init__(self, master, app_controller=None, **kwargs):
        super().__init__(master, **kwargs)
        self.app = app_controller # 保存主应用的引用
        self.image_tk = None
        self.pil_image = None
        self.scale = 1.0
        self.view_x = 0  # 视图左上角在图像坐标系中的x坐标
        self.view_y = 0  # 视图左上角在图像坐标系中的y坐标
        self._drag_data = {"x": 0, "y": 0, "item": None}

        self.bind("<ButtonPress-3>", self._on_right_press)
        self.bind("<B3-Motion>", self._on_right_motion)
        self.bind("<MouseWheel>", self._on_mouse_wheel)
        self.bind("<Configure>", self._on_resize)

    def set_image(self, pil_image: Image.Image):
        """设置要显示的新图像。"""
        self.pil_image = pil_image
        self.scale = 1.0
        self.view_x = 0
        self.view_y = 0
        self._fit_to_screen()

    def _fit_to_screen(self):
        """计算缩放比例和位置以使图像适应画布。"""
        if not self.pil_image:
            self._redraw()
            return
        
        canvas_w = self.winfo_width()
        canvas_h = self.winfo_height()
        img_w, img_h = self.pil_image.size

        if img_w == 0 or img_h == 0 or canvas_w == 0 or canvas_h == 0:
            return

        scale_w = canvas_w / img_w
        scale_h = canvas_h / img_h
        self.scale = min(scale_w, scale_h)

        self.view_x = (img_w - canvas_w / self.scale) / 2
        self.view_y = (img_h - canvas_h / self.scale) / 2
        self._redraw()

    def _redraw(self):
        """根据当前视图在画布上重绘图像。"""
        self.delete("all")
        if not self.pil_image:
            return

        canvas_w = self.winfo_width()
        canvas_h = self.winfo_height()
        
        if canvas_w <= 0 or canvas_h <= 0:
            return

        # BUG修复: 使用仿射变换进行高精度重绘，消除抖动
        # 计算从画布坐标到源图像坐标的变换矩阵
        # x_src = (1/scale) * x_canvas + view_x
        # y_src = (1/scale) * y_canvas + view_y
        transform_matrix = (1/self.scale, 0, self.view_x, 0, 1/self.scale, self.view_y)

        # 使用仿射变换生成精确的视图
        # Image.BICUBIC 提供了较好的重采样质量
        disp_img = self.pil_image.transform(
            (canvas_w, canvas_h),
            Image.AFFINE,
            transform_matrix,
            Image.BICUBIC  # 使用双三次插值以获得更好的视觉效果
        )
        
        self.image_tk = ImageTk.PhotoImage(disp_img)
        self.create_image(0, 0, anchor=tk.NW, image=self.image_tk, tags="image")

    def _on_right_press(self, event):
        """处理右键拖动的开始事件。"""
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

    def _on_right_motion(self, event):
        """处理右键拖动以平移图像。"""
        dx = event.x - self._drag_data["x"]
        dy = event.y - self._drag_data["y"]
        self.view_x -= dx / self.scale
        self.view_y -= dy / self.scale
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y
        self._redraw()
        if self.app:
            self.app._redraw_all_points()

    def _on_mouse_wheel(self, event):
        """处理鼠标滚轮滚动以缩放图像。"""
        if not self.pil_image: return
        # 缩放前获取鼠标指针下的图像坐标
        img_x, img_y = self.canvas_to_image_coords(event.x, event.y)

        # 确定缩放因子
        if event.delta > 0:
            zoom_factor = 1.1
        else:
            zoom_factor = 0.9
        
        self.scale *= zoom_factor

        # 调整视图，使鼠标下的点保持在原位
        self.view_x = img_x - event.x / self.scale
        self.view_y = img_y - event.y / self.scale
        
        self._redraw()
        if self.app:
            self.app._redraw_all_points()

    def _on_resize(self, event):
        """处理画布尺寸调整事件。"""
        self._fit_to_screen()
        if self.app:
            self.app._redraw_all_points()

    def canvas_to_image_coords(self, canvas_x, canvas_y):
        """将画布坐标转换为原始图像坐标。"""
        img_x = self.view_x + canvas_x / self.scale
        img_y = self.view_y + canvas_y / self.scale
        return img_x, img_y

    def image_to_canvas_coords(self, img_x, img_y):
        """将原始图像坐标转换为画布坐标。"""
        canvas_x = (img_x - self.view_x) * self.scale
        canvas_y = (img_y - self.view_y) * self.scale
        return canvas_x, canvas_y

class TiePointPickerApp:
    def __init__(self, master):
        self.master = master
        master.title("优化版遥感影像刺点工具")
        master.geometry("1400x900")

        # --- 数据存储 ---
        self.image_paths = []
        self.full_loaded_images = [] # BUG修复1: 缓存加载后的完整影像
        self.loaded_image_patches = {} # {影像索引: PIL 图像}
        self.patch_origin_coords = {} # {影像索引: (r1, c1)}
        self.image_shapes = []
        self.num_images = 0
        self.windows = np.array([])
        self.image_rpcs = []
        self.heights = []
        self.point_files = []
        
        # --- 刺点数据 ---
        self.saved_points: List[List[Tuple[float, float]]] = []
        self.current_group_points: List[Optional[Tuple[float, float]]] = []
        self.selected_point_info: Optional[Dict] = None
        self.tie_point_group_counter = 1
        
        # --- UI 状态 ---
        self.current_window_k = tk.IntVar(value=0)
        self.img_idx_left = tk.IntVar(value=0)
        self.img_idx_right = tk.IntVar(value=1)

        self._setup_ui()
        self._bind_shortcuts()
        self._update_ui_state()

    def _setup_ui(self):
        # --- 主框架 ---
        top_frame = ttk.Frame(self.master, padding=10)
        top_frame.pack(fill=tk.X)
        display_frame = ttk.Frame(self.master, padding=10)
        display_frame.pack(fill=tk.BOTH, expand=True)
        status_frame = ttk.Frame(self.master, padding=10)
        status_frame.pack(fill=tk.X)

        # --- 顶部控件 ---
        ttk.Button(top_frame, text="加载影像", command=self._load_images).pack(side=tk.LEFT, padx=5)
        ttk.Label(top_frame, text="窗口 K:").pack(side=tk.LEFT, padx=(10, 0))
        self.k_spinbox = ttk.Spinbox(top_frame, from_=0, to=0, textvariable=self.current_window_k, width=5, command=self._on_window_k_change, state=tk.DISABLED)
        self.k_spinbox.pack(side=tk.LEFT, padx=5)

        # --- 左侧影像面板 ---
        left_panel = ttk.Frame(display_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        left_controls = ttk.Frame(left_panel)
        left_controls.pack(fill=tk.X)
        ttk.Label(left_controls, text="左侧影像 I:").pack(side=tk.LEFT)
        self.img_left_spinbox = ttk.Spinbox(left_controls, from_=0, to=0, textvariable=self.img_idx_left, width=3, command=self._on_img_selection_change, state=tk.DISABLED)
        self.img_left_spinbox.pack(side=tk.LEFT, padx=5)
        self.img_left_label = ttk.Label(left_controls, text="影像: -")
        self.img_left_label.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        # BUG修复4: 添加 takefocus=True 使画布可以接收键盘焦点
        self.viewer_left = ImageViewer(left_panel, app_controller=self, bg="gray", takefocus=True)
        self.viewer_left.pack(fill=tk.BOTH, expand=True)
        self.viewer_left.bind("<Button-1>", lambda event: self._on_canvas_click(event, self.viewer_left, 0))

        # --- 右侧影像面板 ---
        right_panel = ttk.Frame(display_frame)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        right_controls = ttk.Frame(right_panel)
        right_controls.pack(fill=tk.X)
        ttk.Label(right_controls, text="右侧影像 J:").pack(side=tk.LEFT)
        self.img_right_spinbox = ttk.Spinbox(right_controls, from_=0, to=0, textvariable=self.img_idx_right, width=3, command=self._on_img_selection_change, state=tk.DISABLED)
        self.img_right_spinbox.pack(side=tk.LEFT, padx=5)
        self.img_right_label = ttk.Label(right_controls, text="影像: -")
        self.img_right_label.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        self.viewer_right = ImageViewer(right_panel, app_controller=self, bg="gray", takefocus=True)
        self.viewer_right.pack(fill=tk.BOTH, expand=True)
        self.viewer_right.bind("<Button-1>", lambda event: self._on_canvas_click(event, self.viewer_right, 1))

        # --- 状态与保存 ---
        self.status_label = ttk.Label(status_frame, text="状态: 未加载影像", anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.save_button = ttk.Button(status_frame, text="保存当前组 (Ctrl+S)", command=self._save_current_group, state=tk.DISABLED)
        self.save_button.pack(side=tk.RIGHT, padx=5)
        self.undo_button = ttk.Button(status_frame, text="撤销上个点 (Ctrl+Z)", command=self._undo_last_point, state=tk.DISABLED)
        self.undo_button.pack(side=tk.RIGHT, padx=5)

    def _bind_shortcuts(self):
        """为主窗口绑定键盘快捷键。"""
        self.master.bind("<Control-s>", lambda event: self._save_current_group())
        self.master.bind("<Control-z>", lambda event: self._undo_last_point())
        # BUG修复3: 微调的步长是屏幕上的1个像素
        self.master.bind("<Up>", lambda event: self._finetune_point(-1, 0))
        self.master.bind("<Down>", lambda event: self._finetune_point(1, 0))
        self.master.bind("<Left>", lambda event: self._finetune_point(0, -1))
        self.master.bind("<Right>", lambda event: self._finetune_point(0, 1))

    def _load_images(self):
        paths = filedialog.askopenfilenames(
            title="选择PNG影像文件",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if not paths: return

        self._reset_state()
        self.image_paths = list(paths)
        self.num_images = len(self.image_paths)
        
        try:
            for p in self.image_paths:
                # BUG修复1: 将影像读入内存
                img = cv2.imread(p, cv2.IMREAD_COLOR)
                if img is None: raise ValueError(f"无法读取影像: {p}")
                self.full_loaded_images.append(img)
                self.image_shapes.append(img.shape)
                
                rpc_path = os.path.splitext(p)[0] + '.rpc'
                if not os.path.exists(rpc_path): raise FileNotFoundError(f"RPC 文件未找到: {rpc_path}")
                self.image_rpcs.append(load_rpc(rpc_path))

                height_path = os.path.splitext(p)[0] + '_height.npy'
                if not os.path.exists(height_path): raise FileNotFoundError(f"高度文件未找到: {height_path}")
                height_data = np.load(height_path, mmap_mode='r')
                self.heights.append(height_data[0] if height_data.ndim == 3 else height_data)

                base, _ = os.path.splitext(p)
                self.point_files.append(f"{base}_points.txt")

            self.windows = find_windows(self.image_shapes, self.image_rpcs, self.heights, 1000, 500)
            if self.windows.ndim < 2 or self.windows.shape[1] == 0:
                messagebox.showwarning("无窗口", "未能根据参数生成任何窗口。请检查find_windows函数和输入数据。")

        except Exception as e:
            messagebox.showerror("加载错误", f"加载影像或关联文件失败: {e}")
            self._reset_state()
            return
        
        self._reset_point_data()
        self._load_saved_points()

        # 更新UI
        if self.windows.ndim > 1:
            self.k_spinbox.config(to=max(0, self.windows.shape[1] - 1))
        self.img_left_spinbox.config(to=max(0, self.num_images - 1))
        self.img_right_spinbox.config(to=max(0, self.num_images - 1))

        if self.num_images > 0: self.img_idx_left.set(0)
        if self.num_images > 1: self.img_idx_right.set(1)
        else: self.img_idx_right.set(0)

        self.current_window_k.set(0)
        self._on_window_k_change()
        self._update_ui_state()

    def _reset_state(self):
        """重置整个应用程序的状态。"""
        self.image_paths = []
        self.num_images = 0
        self.full_loaded_images = [] # BUG修复1: 重置缓存
        self._reset_point_data()
        self.viewer_left.set_image(None)
        self.viewer_right.set_image(None)
        self._update_ui_state()
        self._update_status_label()

    def _reset_point_data(self):
        """仅重置与点相关的数据结构。"""
        self.saved_points = [[] for _ in range(self.num_images)]
        self.current_group_points = [None] * self.num_images
        self.selected_point_info = None
        self.tie_point_group_counter = 1

    def _load_saved_points(self):
        """从每个影像的 .txt 文件加载点。"""
        for i, filepath in enumerate(self.point_files):
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                r, c = float(parts[0]), float(parts[1])
                                self.saved_points[i].append((r, c))
                except Exception as e:
                    print(f"读取点文件 {filepath} 时出错: {e}")
        print("已加载的保存点:", self.saved_points)

    def _on_window_k_change(self):
        if self.num_images == 0 or self.windows.ndim < 2 or self.windows.shape[1] == 0: return
        k = self.current_window_k.get()
        
        try:
            window_coords = get_windows(k, self.windows)
        except Exception as e:
            messagebox.showerror("窗口错误", f"调用 get_windows({k}) 出错: {e}")
            return

        self.loaded_image_patches.clear()
        self.patch_origin_coords.clear()

        for i in range(self.num_images):
            coords = window_coords[i]
            r1, c1 = coords[0]
            r2, c2 = coords[1]

            if not (r1 < r2 and c1 < c2):
                continue
            
            # BUG修复1: 直接从内存中的完整影像裁切图块
            full_img = self.full_loaded_images[i]
            patch_bgr = full_img[r1:r2, c1:c2]
            if patch_bgr.size == 0: continue

            patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
            self.loaded_image_patches[i] = Image.fromarray(patch_rgb)
            self.patch_origin_coords[i] = (r1, c1)

        self._display_images()

    def _on_img_selection_change(self):
        if self.num_images == 0: return
        self._display_images()

    def _display_images(self):
        """更新查看器中显示的图像。"""
        # 左侧查看器
        idx_l = self.img_idx_left.get()
        img_l = self.loaded_image_patches.get(idx_l)
        self.viewer_left.set_image(img_l)
        if img_l:
            self.img_left_label.config(text=f"影像 {idx_l}: {os.path.basename(self.image_paths[idx_l])}")
        else:
            self.img_left_label.config(text=f"影像 {idx_l}: (无内容)")

        # 右侧查看器
        idx_r = self.img_idx_right.get()
        img_r = self.loaded_image_patches.get(idx_r)
        self.viewer_right.set_image(img_r)
        if img_r:
            self.img_right_label.config(text=f"影像 {idx_r}: {os.path.basename(self.image_paths[idx_r])}")
        else:
            self.img_right_label.config(text=f"影像 {idx_r}: (无内容)")
        
        self._redraw_all_points()
        self._update_status_label()

    def _redraw_all_points(self):
        """在两个画布上重绘点。"""
        idx_l = self.img_idx_left.get()
        self._draw_points_on_viewer(self.viewer_left, idx_l)
        
        idx_r = self.img_idx_right.get()
        self._draw_points_on_viewer(self.viewer_right, idx_r)

    def _draw_points_on_viewer(self, viewer: ImageViewer, image_idx: int):
        """在单个查看器上绘制所有相关的点。"""
        viewer.delete("point") 
        
        patch_origin = self.patch_origin_coords.get(image_idx)
        if not patch_origin or not viewer.pil_image:
            return
        
        patch_r1, patch_c1 = patch_origin
        patch_img = self.loaded_image_patches.get(image_idx)
        if not patch_img: return
        patch_h, patch_w = patch_img.height, patch_img.width
        patch_r2, patch_c2 = patch_r1 + patch_h, patch_c1 + patch_w

        # 绘制已保存的点 (绿色)
        if image_idx < len(self.saved_points):
            for r, c in self.saved_points[image_idx]:
                if patch_r1 <= r < patch_r2 and patch_c1 <= c < patch_c2:
                    self._draw_single_point(viewer, r, c, patch_r1, patch_c1, "green")

        # 绘制当前组的点 (蓝色或红色)
        if image_idx < len(self.current_group_points):
            point_coords = self.current_group_points[image_idx]
            if point_coords:
                r, c = point_coords
                color = "blue"
                if self.selected_point_info and self.selected_point_info["image_idx"] == image_idx:
                    color = "red"
                
                if patch_r1 <= r < patch_r2 and patch_c1 <= c < patch_c2:
                    self._draw_single_point(viewer, r, c, patch_r1, patch_c1, color)
    
    def _draw_single_point(self, viewer: ImageViewer, abs_r, abs_c, patch_r1, patch_c1, color):
        """根据绝对坐标绘制单个点的辅助函数。"""
        patch_y = abs_r - patch_r1
        patch_x = abs_c - patch_c1
        
        canvas_x, canvas_y = viewer.image_to_canvas_coords(patch_x, patch_y)
        
        size = 4
        viewer.create_oval(canvas_x - size, canvas_y - size, canvas_x + size, canvas_y + size,
                           fill=color, outline="white", width=1, tags="point")

    def _on_canvas_click(self, event, viewer: ImageViewer, panel_id: int):
        if self.num_images == 0 or not viewer.pil_image: return

        image_idx = self.img_idx_left.get() if panel_id == 0 else self.img_idx_right.get()
        
        if self.current_group_points[image_idx] is not None:
            messagebox.showwarning("刺点限制", f"影像 {image_idx} 在当前组中已经标过点了。\n请先撤销或保存。")
            return

        patch_x, patch_y = viewer.canvas_to_image_coords(event.x, event.y)
        patch_r1, patch_c1 = self.patch_origin_coords[image_idx]

        abs_c = patch_c1 + patch_x
        abs_r = patch_r1 + patch_y

        img_h, img_w = self.image_shapes[image_idx][:2]
        if not (0 <= abs_r < img_h and 0 <= abs_c < img_w):
            messagebox.showwarning("超出边界", "标记点超出了影像原始边界。")
            return

        self.current_group_points[image_idx] = (abs_r, abs_c)
        self.selected_point_info = {
            "image_idx": image_idx,
            "panel_id": panel_id
        }
        
        viewer.focus_set() # BUG修复4: 点击画布后，将焦点设置到该画布
        print(f"面板 {panel_id}, 影像 {image_idx}: 点击 -> 绝对坐标 ({abs_r:.2f},{abs_c:.2f})")

        self._redraw_all_points()
        self._update_status_label()
        self._update_ui_state()

    def _undo_last_point(self):
        """撤销当前组中最近标记的一个点。"""
        if self.selected_point_info is None:
            messagebox.showinfo("提示", "当前组中没有可以撤销的点。")
            return
        
        image_idx_to_undo = self.selected_point_info["image_idx"]
        self.current_group_points[image_idx_to_undo] = None
        self.selected_point_info = None # 清除选择

        for i in range(self.num_images - 1, -1, -1):
            if self.current_group_points[i] is not None:
                panel_id = 0 if self.img_idx_left.get() == i else 1 
                self.selected_point_info = {"image_idx": i, "panel_id": panel_id}
                break

        print(f"已撤销影像 {image_idx_to_undo} 的刺点")
        self._redraw_all_points()
        self._update_status_label()
        self._update_ui_state()

    def _finetune_point(self, dr_screen, dc_screen):
        """使用箭头键按屏幕像素微调所选点的位置。"""
        # BUG修复4: 检查焦点，只有当焦点在图像视图上时才执行微调
        focused_widget = self.master.focus_get()
        if focused_widget not in [self.viewer_left, self.viewer_right]:
            return
            
        if self.selected_point_info is None: return
        
        panel_id = self.selected_point_info["panel_id"]
        viewer = self.viewer_left if panel_id == 0 else self.viewer_right
        
        if not viewer.pil_image: return

        idx = self.selected_point_info["image_idx"]
        r, c = self.current_group_points[idx]
        
        # BUG修复2&3: 根据缩放比例计算在绝对坐标系中的移动量
        dr_abs = dr_screen / viewer.scale
        dc_abs = dc_screen / viewer.scale
        
        new_r, new_c = r + dr_abs, c + dc_abs
        
        img_h, img_w = self.image_shapes[idx][:2]
        if not (0 <= new_r < img_h and 0 <= new_c < img_w):
            return 
            
        self.current_group_points[idx] = (new_r, new_c)
        self._redraw_all_points()
        self._update_status_label()

    def _save_current_group(self):
        if any(p is None for p in self.current_group_points):
            messagebox.showerror("错误", "当前组刺点未完成，请为所有影像标点。")
            return

        try:
            for img_idx, coords in enumerate(self.current_group_points):
                filepath = self.point_files[img_idx]
                with open(filepath, 'a') as f:
                    r, c = coords
                    # 保存时四舍五入为整数
                    f.write(f"{int(round(r))} {int(round(c))}\n")
                self.saved_points[img_idx].append(coords)
            
            messagebox.showinfo("保存成功", f"第 {self.tie_point_group_counter} 组刺点已保存。")

            self.tie_point_group_counter += 1
            self.current_group_points = [None] * self.num_images
            self.selected_point_info = None
            
            self._redraw_all_points()
            self._update_status_label()
            self._update_ui_state()

        except Exception as e:
            messagebox.showerror("保存失败", f"保存刺点数据失败: {e}")

    def _update_status_label(self):
        if self.num_images == 0:
            self.status_label.config(text="状态: 未加载影像")
            return
        
        status_parts = []
        for i in range(self.num_images):
            coords = self.current_group_points[i]
            if coords:
                status_parts.append(f"影像{i}: ({coords[0]:.1f}, {coords[1]:.1f})")
            else:
                status_parts.append(f"影像{i}: 未标")
        
        status_text = f"第 {self.tie_point_group_counter} 组 | K={self.current_window_k.get()} | " + " | ".join(status_parts)
        self.status_label.config(text=status_text)

    def _update_ui_state(self):
        """根据当前状态启用/禁用UI组件。"""
        has_images = self.num_images > 0
        has_windows = self.windows.ndim > 1 and self.windows.shape[1] > 0
        
        self.k_spinbox.config(state=tk.NORMAL if has_images and has_windows else tk.DISABLED)
        self.img_left_spinbox.config(state=tk.NORMAL if has_images else tk.DISABLED)
        self.img_right_spinbox.config(state=tk.NORMAL if self.num_images > 1 else tk.DISABLED)
        
        can_save = self.num_images > 0 and all(p is not None for p in self.current_group_points)
        self.save_button.config(state=tk.NORMAL if can_save else tk.DISABLED)
        
        can_undo = self.selected_point_info is not None
        self.undo_button.config(state=tk.NORMAL if can_undo else tk.DISABLED)

if __name__ == '__main__':
    root = tk.Tk()
    app = TiePointPickerApp(root)
    root.mainloop()
