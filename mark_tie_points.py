import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import os

from rpc import RPCModelParameterTorch,load_rpc
from shapely.geometry import Polygon, Point, MultiPoint
from typing import List,Tuple,Optional,Dict

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
        self.view_y = (img_h - canvas_w / self.scale) / 2
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

        transform_matrix = (1/self.scale, 0, self.view_x, 0, 1/self.scale, self.view_y)

        # 【修改 1】: 将图像插值方式改为最近邻，实现像素放大效果
        # Change interpolation method to NEAREST for pixelated zoom effect
        disp_img = self.pil_image.transform(
            (canvas_w, canvas_h),
            Image.AFFINE,
            transform_matrix,
            Image.NEAREST  # 原为 Image.BICUBIC
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
        master.title("优化版遥感影像刺点工具 (v2.0 带编辑功能)")
        master.geometry("1600x900") # 增加了宽度以容纳新面板

        # --- 数据存储 ---
        self.image_paths = []
        self.full_loaded_images = [] 
        self.loaded_image_patches = {} # {影像索引: PIL 图像}
        self.patch_origin_coords = {} # {影像索引: (r1, c1)}
        self.image_shapes = []
        self.num_images = 0
        self.windows = np.array([])
        self.image_rpcs = []
        self.heights = []
        self.point_files = []
        
        # --- 刺点数据 ---
        self.saved_points: List[List[Tuple[int, int]]] = []
        self.current_group_points: List[Optional[Tuple[int, int]]] = []
        self.selected_point_info: Optional[Dict] = None
        self.tie_point_group_counter = 1
        
        # --- UI 状态 ---
        self.current_window_k = tk.IntVar(value=0)
        self.img_idx_left = tk.IntVar(value=0)
        self.img_idx_right = tk.IntVar(value=1)
        
        # --- 【新增】编辑模式状态 ---
        self.edit_mode = False
        self.editing_group_index: Optional[int] = None

        self._setup_ui()
        self._bind_shortcuts()
        self._update_ui_state()

    def _setup_ui(self):
        # --- 主框架 ---
        top_frame = ttk.Frame(self.master, padding=10)
        top_frame.pack(fill=tk.X)
        
        # 【修改】将 display_frame 分为三栏
        display_frame = ttk.Frame(self.master, padding=(10, 0, 10, 10))
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        status_frame = ttk.Frame(self.master, padding=10)
        status_frame.pack(fill=tk.X)

        # --- 顶部控件 ---
        ttk.Button(top_frame, text="加载影像", command=self._load_images).pack(side=tk.LEFT, padx=5)
        ttk.Label(top_frame, text="窗口 K:").pack(side=tk.LEFT, padx=(10, 0))
        self.k_spinbox = ttk.Spinbox(top_frame, from_=0, to=0, textvariable=self.current_window_k, width=5, command=self._on_window_k_change, state=tk.DISABLED)
        self.k_spinbox.pack(side=tk.LEFT, padx=5)

        # --- 左侧影像面板 (不变) ---
        left_panel = ttk.Frame(display_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        left_controls = ttk.Frame(left_panel)
        left_controls.pack(fill=tk.X)
        ttk.Label(left_controls, text="左侧影像 I:").pack(side=tk.LEFT)
        self.img_left_spinbox = ttk.Spinbox(left_controls, from_=0, to=0, textvariable=self.img_idx_left, width=3, command=self._on_img_selection_change, state=tk.DISABLED)
        self.img_left_spinbox.pack(side=tk.LEFT, padx=5)
        self.img_left_label = ttk.Label(left_controls, text="影像: -")
        self.img_left_label.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        self.viewer_left = ImageViewer(left_panel, app_controller=self, bg="gray", takefocus=True)
        self.viewer_left.pack(fill=tk.BOTH, expand=True)
        self.viewer_left.bind("<Button-1>", lambda event: self._on_canvas_click(event, self.viewer_left, 0))

        # --- 右侧影像面板 (不变) ---
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

        # --- 【新增】已标注点列表面板 ---
        point_list_frame = ttk.Frame(display_frame, width=250)
        point_list_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        point_list_frame.pack_propagate(False) # 固定宽度

        ttk.Label(point_list_frame, text="已标注点 (影像0坐标)", anchor=tk.W).pack(fill=tk.X, pady=(0, 5))

        tree_frame = ttk.Frame(point_list_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        self.point_list_tree = ttk.Treeview(tree_frame, columns=("coords",), show="tree headings", height=10)
        self.point_list_tree.heading("#0", text="组 ID")
        self.point_list_tree.heading("coords", text="坐标 (r, c)")
        self.point_list_tree.column("#0", width=80, stretch=False)
        self.point_list_tree.column("coords", width=120, stretch=True)

        # 添加滚动条
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.point_list_tree.yview)
        self.point_list_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.point_list_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.point_list_tree.bind("<<TreeviewSelect>>", self._on_point_list_select)

        self.edit_button = ttk.Button(point_list_frame, text="启动编辑", command=self._toggle_edit_mode, state=tk.DISABLED)
        self.edit_button.pack(fill=tk.X, pady=5)

        # --- 状态与保存 ---
        self.status_label = ttk.Label(status_frame, text="状态: 未加载影像", anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.save_button = ttk.Button(status_frame, text="保存新组 (Ctrl+S)", command=self._save_current_group, state=tk.DISABLED)
        self.save_button.pack(side=tk.RIGHT, padx=5)
        self.undo_button = ttk.Button(status_frame, text="撤销上个点 (Ctrl+Z)", command=self._undo_last_point, state=tk.DISABLED)
        self.undo_button.pack(side=tk.RIGHT, padx=5)

    def _bind_shortcuts(self):
        """为主窗口绑定键盘快捷键。"""
        self.master.bind("<Control-s>", lambda event: self._save_current_group())
        self.master.bind("<Control-z>", lambda event: self._undo_last_point())
        
        # 【修改 4】: 将微调功能改为按原图整像素移动
        # Change finetuning to move by one original image pixel
        self.master.bind("<Up>", lambda event: self._finetune_point(-1, 0))
        self.master.bind("<Down>", lambda event: self._finetune_point(1, 0))
        self.master.bind("<Left>", lambda event: self._finetune_point(0, -1))
        self.master.bind("<Right>", lambda event: self._finetune_point(0, 1))

    # --- 【新增】功能函数 ---

    def _populate_point_list(self):
        """清空并重新填充已保存点的列表。"""
        for item in self.point_list_tree.get_children():
            self.point_list_tree.delete(item)

        if not self.saved_points or self.num_images == 0:
            return

        try:
            num_groups = len(self.saved_points[0])
            for group_index in range(num_groups):
                # 使用影像0的坐标作为参考
                r, c = self.saved_points[0][group_index]
                # 关键：iid 存储 group_index，text 显示 group_id
                self.point_list_tree.insert("", "end", iid=str(group_index), text=f"Group {group_index + 1}", values=(f"({r}, {c})",))
        except IndexError:
            print("警告: saved_points 列表为空或结构不一致。")
        except Exception as e:
            print(f"填充列表时出错: {e}")

    def _on_point_list_select(self, event=None):
        """处理列表点击：非编辑模式下跳转，并激活编辑按钮。"""
        selected_item_id = self.point_list_tree.focus()
        if not selected_item_id:
            self.edit_button.config(state=tk.DISABLED)
            return

        self.edit_button.config(state=tk.NORMAL)
        
        if self.edit_mode:
            # 编辑模式下，不执行跳转（因为列表点击已被解绑，理论上不会到这里）
            return

        # 非编辑模式，执行跳转功能
        try:
            group_index = int(selected_item_id)
            r, c = self.saved_points[0][group_index]

            # 查找包含该点 (影像0) 的窗口 K
            for k in range(self.windows.shape[1]):
                r1, c1 = self.windows[0, k, 0] # 影像0, 窗口k, 左上角
                r2, c2 = self.windows[0, k, 1] # 影像0, 窗口k, 右下角
                if r1 <= r < r2 and c1 <= c < c2:
                    if self.current_window_k.get() != k:
                        self.current_window_k.set(k) # 会自动触发 _on_window_k_change
                    else:
                        # 如果已经在正确的K，手动重绘点（例如切换组时）
                        self._redraw_all_points()
                    return
            
            messagebox.showinfo("未找到窗口", f"Group {group_index + 1} 的点 (影像0: {r},{c}) 不在任何自动生成的窗口内。")

        except Exception as e:
            print(f"跳转到点时出错: {e}")

    def _toggle_edit_mode(self):
        """切换“启动编辑”和“完成编辑”的状态。"""
        
        # --- 情况 1: 启动编辑 ---
        if not self.edit_mode:
            selected_item_id = self.point_list_tree.focus()
            if not selected_item_id:
                messagebox.showwarning("未选择", "请先从右侧列表中选择一个点组。")
                return

            if any(p is not None for p in self.current_group_points):
                messagebox.showwarning("操作冲突", "请先保存或撤销当前正在标注的新点，然后再开始编辑。")
                return
            
            group_index = int(selected_item_id)

            # 1. 进入编辑模式
            self.edit_mode = True
            self.editing_group_index = group_index

            # 2. 加载数据到缓冲区
            try:
                point_group_to_edit = [self.saved_points[i][group_index] for i in range(self.num_images)]
                self.current_group_points = list(point_group_to_edit)
            except Exception as e:
                messagebox.showerror("加载错误", f"加载点组 {group_index+1} 失败: {e}")
                self.edit_mode = False
                self.editing_group_index = None
                return

            # 3. 激活微调
            left_img_idx = self.img_idx_left.get()
            self.selected_point_info = {"image_idx": left_img_idx, "panel_id": 0}
            if left_img_idx < len(self.current_group_points):
                self.viewer_left.focus_set()
            
            # 4. 更新UI
            self.edit_button.config(text="完成编辑")
            # 【修复】通过解绑事件来禁用 Treeview
            self.point_list_tree.unbind("<<TreeviewSelect>>") # 编辑时禁用列表点击

            print(f"已启动编辑模式: Group {group_index + 1}")

        # --- 情况 2: 完成编辑 ---
        else:
            if any(p is None for p in self.current_group_points):
                messagebox.showerror("错误", "编辑未完成，组内存在空点（可能被撤销）。请重新标点。")
                return
            
            group_index = self.editing_group_index
            if group_index is None: return # 安全检查

            try:
                # 1. 更新内存
                for i in range(self.num_images):
                    self.saved_points[i][group_index] = self.current_group_points[i]
                
                # 2. 更新文件
                self._rewrite_all_point_files()

                # 3. 退出编辑模式
                self.edit_mode = False
                self.editing_group_index = None
                
                # 4. 清理缓冲区
                self.current_group_points = [None] * self.num_images
                self.selected_point_info = None
                
                # 5. 更新UI
                self.edit_button.config(text="启动编辑", state=tk.NORMAL)
                # 【修复】通过重新绑定事件来启用 Treeview
                self.point_list_tree.bind("<<TreeviewSelect>>", self._on_point_list_select) # 解禁列表点击
                
                # 6. 刷新列表显示
                self._populate_point_list()
                
                messagebox.showinfo("保存成功", f"Group {group_index + 1} 已更新。")
            
            except Exception as e:
                messagebox.showerror("保存失败", f"保存编辑失败: {e}")

        # 统一更新
        self._redraw_all_points()
        self._update_status_label()
        self._update_ui_state()

    def _rewrite_all_point_files(self):
        """
        【新增】使用内存中的 saved_points 覆盖重写所有 .txt 文件。
        """
        print("正在重写所有点文件...")
        try:
            for img_idx in range(self.num_images):
                filepath = self.point_files[img_idx]
                points_to_save = self.saved_points[img_idx]
                
                with open(filepath, 'w') as f: # 'w' 模式 = 覆盖
                    for r, c in points_to_save:
                        f.write(f"{r} {c}\n")
        except Exception as e:
            print(f"重写文件失败: {e}")
            messagebox.showerror("文件写入错误", f"无法重写点文件: {e}")

    # --- 现有函数修改 ---

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
        
        # 【修改】加载点后，填充列表
        self._populate_point_list()

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
        self.full_loaded_images = []
        self._reset_point_data()
        
        # 【修改】重置时清空列表
        if hasattr(self, 'point_list_tree'):
             self._populate_point_list()
             
        # 【修改】重置编辑状态
        self.edit_mode = False
        self.editing_group_index = None

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
        min_points = float('inf')
        for i, filepath in enumerate(self.point_files):
            points = []
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                r, c = int(parts[0]), int(parts[1])
                                points.append((r, c))
                except Exception as e:
                    print(f"读取点文件 {filepath} 时出错: {e}")
            self.saved_points[i] = points
            min_points = min(min_points, len(points))
        
        if min_points == float('inf'): min_points = 0
            
        # 确保所有列表长度一致
        if any(len(p) != min_points for p in self.saved_points):
            messagebox.showwarning("数据不一致", "点文件中的点数量不一致。将截断为最短长度。")
            self.saved_points = [p[:min_points] for p in self.saved_points]

        # 更新组计数器
        self.tie_point_group_counter = min_points + 1
        print(f"已加载 {min_points} 组保存点。")

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
            
            full_img = self.full_loaded_images[i]
            patch_bgr = full_img[r1:r2, c1:c2]
            if patch_bgr.size == 0: continue

            patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
            self.loaded_image_patches[i] = Image.fromarray(patch_rgb)
            self.patch_origin_coords[i] = (r1, c1)

        self._display_images()

    def _on_img_selection_change(self):
        if self.num_images == 0: return
        
        # 【新增】更新微调时的焦点
        if self.edit_mode and self.selected_point_info:
            panel_id = self.selected_point_info["panel_id"]
            if panel_id == 0:
                self.selected_point_info["image_idx"] = self.img_idx_left.get()
            else:
                self.selected_point_info["image_idx"] = self.img_idx_right.get()
        
        self._display_images()

    def _display_images(self):
        """更新查看器中显示的图像。"""
        idx_l = self.img_idx_left.get()
        img_l = self.loaded_image_patches.get(idx_l)
        self.viewer_left.set_image(img_l)
        if img_l:
            self.img_left_label.config(text=f"影像 {idx_l}: {os.path.basename(self.image_paths[idx_l])}")
        else:
            self.img_left_label.config(text=f"影像 {idx_l}: (无内容)")

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
            for group_idx, (r, c) in enumerate(self.saved_points[image_idx]):
                
                # 【修改】如果点正在被编辑，则跳过（由下面的逻辑绘制）
                if self.edit_mode and self.editing_group_index == group_idx:
                    continue
                
                if patch_r1 <= r < patch_r2 and patch_c1 <= c < patch_c2:
                    self._draw_single_point(viewer, r, c, patch_r1, patch_c1, "green")

        # 绘制当前组的点 (蓝色或红色)
        if image_idx < len(self.current_group_points):
            point_coords = self.current_group_points[image_idx]
            if point_coords:
                r, c = point_coords
                color = "blue"
                if self.selected_point_info and self.selected_point_info["image_idx"] == image_idx:
                    color = "red" # 红色高亮
                
                if patch_r1 <= r < patch_r2 and patch_c1 <= c < patch_c2:
                    self._draw_single_point(viewer, r, c, patch_r1, patch_c1, color)
    
    def _draw_single_point(self, viewer: ImageViewer, abs_r, abs_c, patch_r1, patch_c1, color):
        """
        【修改 2.2】: (重写) 根据绝对像素坐标高亮单个像素并添加标记。
        (Rewritten) Highlight a single pixel and add a marker based on absolute pixel coordinates.
        """
        # abs_r 和 abs_c 是像素的整数坐标
        patch_y = abs_r - patch_r1
        patch_x = abs_c - patch_c1
        
        # 获取像素左上角和右下角在画布上的坐标
        cx1, cy1 = viewer.image_to_canvas_coords(patch_x, patch_y)
        cx2, cy2 = viewer.image_to_canvas_coords(patch_x + 1, patch_y + 1)
        
        # 绘制像素的轮廓
        viewer.create_rectangle(cx1, cy1, cx2, cy2, outline=color, width=1, tags="point")

        # 为了更醒目，在中心绘制一个十字标记 (固定屏幕尺寸)
        ccx, ccy = (cx1 + cx2) / 2, (cy1 + cy2) / 2
        cross_size = 5 # 十字标记在屏幕上的大小 (单位: 像素)
        viewer.create_line(ccx - cross_size, ccy, ccx + cross_size, ccy, fill=color, width=1, tags="point")
        viewer.create_line(ccx, ccy - cross_size, ccx, ccy + cross_size, fill=color, width=1, tags="point")

    def _on_canvas_click(self, event, viewer: ImageViewer, panel_id: int):
        if self.num_images == 0 or not viewer.pil_image: return

        image_idx = self.img_idx_left.get() if panel_id == 0 else self.img_idx_right.get()
        
        # 【修改】允许在编辑模式下覆盖
        if self.current_group_points[image_idx] is not None and not self.edit_mode:
            messagebox.showwarning("刺点限制", f"影像 {image_idx} 在当前组中已经标过点了。\n请先撤销或保存。")
            return

        patch_x, patch_y = viewer.canvas_to_image_coords(event.x, event.y)
        
        patch_origin = self.patch_origin_coords.get(image_idx)
        if not patch_origin:
            print(f"警告: 影像 {image_idx} 没有加载切片，无法刺点。")
            return
            
        patch_r1, patch_c1 = patch_origin

        abs_c = patch_c1 + patch_x
        abs_r = patch_r1 + patch_y
        
        # 【修改 2.1】: 选择最近的整数像素坐标
        # Select the nearest integer pixel coordinate
        selected_abs_r = int(round(abs_r))
        selected_abs_c = int(round(abs_c))

        img_h, img_w = self.image_shapes[image_idx][:2]
        if not (0 <= selected_abs_r < img_h and 0 <= selected_abs_c < img_w):
            messagebox.showwarning("超出边界", "标记点超出了影像原始边界。")
            return

        self.current_group_points[image_idx] = (selected_abs_r, selected_abs_c)
        self.selected_point_info = {
            "image_idx": image_idx,
            "panel_id": panel_id
        }
        
        viewer.focus_set() # 激活画布以接收键盘事件
        print(f"面板 {panel_id}, 影像 {image_idx}: 点击 -> 绝对坐标 ({selected_abs_r},{selected_abs_c})")

        self._redraw_all_points()
        self._update_status_label()
        self._update_ui_state()

    def _undo_last_point(self):
        """撤销当前组中最近标记的一个点。"""
        if self.selected_point_info is None:
            # 【修改】编辑模式下，可能没有 selected_point_info，但仍可撤销
            if not self.edit_mode:
                messagebox.showinfo("提示", "当前组中没有可以撤销的点。")
                return
            else:
                # 在编辑模式下，如果缓冲区有内容，但没有焦点，则默认撤销最后一个
                for i in range(self.num_images - 1, -1, -1):
                    if self.current_group_points[i] is not None:
                        self.current_group_points[i] = None
                        print(f"已撤销影像 {i} 的刺点 (编辑模式)")
                        self._redraw_all_points()
                        self._update_status_label()
                        self._update_ui_state()
                        return
                messagebox.showinfo("提示", "编辑组为空，无法撤销。")
                return
        
        image_idx_to_undo = self.selected_point_info["image_idx"]
        self.current_group_points[image_idx_to_undo] = None
        self.selected_point_info = None # 清除选择

        # 寻找上一个点并设为焦点
        for i in range(self.num_images - 1, -1, -1):
            if self.current_group_points[i] is not None:
                panel_id = 0 
                if self.img_idx_left.get() == i:
                    panel_id = 0
                elif self.img_idx_right.get() == i:
                    panel_id = 1
                # (如果两个都不匹配，默认焦点给左侧)
                
                self.selected_point_info = {"image_idx": i, "panel_id": panel_id}
                break

        print(f"已撤销影像 {image_idx_to_undo} 的刺点")
        self._redraw_all_points()
        self._update_status_label()
        self._update_ui_state()

    def _finetune_point(self, dr_abs, dc_abs):
        """
        【修改 5】: (重构) 使用箭头键按原始影像的一个像素微调所选点的位置。
        """
        focused_widget = self.master.focus_get()
        if focused_widget not in [self.viewer_left, self.viewer_right]:
            return
            
        if self.selected_point_info is None: return
        
        idx = self.selected_point_info["image_idx"]
        
        # 确保索引有效
        if idx >= len(self.current_group_points) or self.current_group_points[idx] is None:
            return
            
        r, c = self.current_group_points[idx]
        
        # 直接对整数坐标进行加减
        new_r, new_c = r + dr_abs, c + dc_abs
        
        img_h, img_w = self.image_shapes[idx][:2]
        if not (0 <= new_r < img_h and 0 <= new_c < img_w):
            return 
            
        self.current_group_points[idx] = (new_r, new_c)
        self._redraw_all_points()
        self._update_status_label()

    def _save_current_group(self):
        # 【修改】编辑模式下不应保存新组
        if self.edit_mode:
            messagebox.showwarning("模式错误", "您正处于编辑模式。请点击“完成编辑”来保存修改。")
            return

        if any(p is None for p in self.current_group_points):
            messagebox.showerror("错误", "当前组刺点未完成，请为所有影像标点。")
            return

        try:
            for img_idx, coords in enumerate(self.current_group_points):
                filepath = self.point_files[img_idx]
                with open(filepath, 'a') as f: # 'a' = 追加
                    r, c = coords
                    f.write(f"{r} {c}\n")
                self.saved_points[img_idx].append(coords)
            
            messagebox.showinfo("保存成功", f"第 {self.tie_point_group_counter} 组刺点已保存。")

            self.tie_point_group_counter += 1
            self.current_group_points = [None] * self.num_images
            self.selected_point_info = None
            
            self._redraw_all_points()
            self._update_status_label()
            self._update_ui_state()
            
            # 【修改】保存新组后刷新列表
            self._populate_point_list()

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
                status_parts.append(f"影像{i}: ({coords[0]}, {coords[1]})")
            else:
                status_parts.append(f"影像{i}: 未标")
        
        # 【修改】根据模式显示不同标题
        if self.edit_mode:
            group_id = self.editing_group_index + 1 if self.editing_group_index is not None else '?'
            mode_text = f"【编辑中】组 {group_id}"
        else:
            mode_text = f"【新增】组 {self.tie_point_group_counter}"
            
        status_text = f"{mode_text} | K={self.current_window_k.get()} | " + " | ".join(status_parts)
        self.status_label.config(text=status_text)

    def _update_ui_state(self):
        """【修改】根据当前状态启用/禁用UI组件。"""
        has_images = self.num_images > 0
        has_windows = self.windows.ndim > 1 and self.windows.shape[1] > 0
        
        self.k_spinbox.config(state=tk.NORMAL if has_images and has_windows and not self.edit_mode else tk.DISABLED)
        self.img_left_spinbox.config(state=tk.NORMAL if has_images else tk.DISABLED)
        self.img_right_spinbox.config(state=tk.NORMAL if self.num_images > 1 else tk.DISABLED)
        
        # 撤销按钮：只要缓冲区有内容或有焦点，就可撤销
        can_undo = self.selected_point_info is not None or (self.edit_mode and any(p is not None for p in self.current_group_points))
        self.undo_button.config(state=tk.NORMAL if can_undo else tk.DISABLED)
        
        # 保存新组按钮：缓冲区满 且 *非* 编辑模式
        can_save_new = self.num_images > 0 and all(p is not None for p in self.current_group_points)
        self.save_button.config(state=tk.NORMAL if can_save_new and not self.edit_mode else tk.DISABLED)
        
        # 编辑按钮
        if hasattr(self, 'edit_button'):
            has_selection = bool(self.point_list_tree.focus())
            if self.edit_mode:
                self.edit_button.config(state=tk.NORMAL) # “完成编辑”按钮始终可用
            else:
                self.edit_button.config(state=tk.NORMAL if has_selection else tk.DISABLED) # “启动编辑”需选中
        
        # 【修复】删除导致错误的代码块


if __name__ == '__main__':
    root = tk.Tk()
    app = TiePointPickerApp(root)
    root.mainloop()

