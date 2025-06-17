import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
from rpc import RPCModelParameterTorch

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !! 用户需要实现这个函数 (You need to implement this function)
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def get_windows(k: int, num_images: int, image_shapes: list):
    """
    根据窗口索引 k 获取每个影像的窗口范围。
    Args:
        k (int): 当前窗口的索引，从0开始。
        num_images (int): 总影像数量。
        image_shapes (list): 包含每个影像形状 (height, width, channels) 的列表。
    Returns:
        np.ndarray: 一个 (N, 2, 2) 的numpy数组，N是影像数量。
                    每张影像的 [ [[r1, c1], [r2, c2]], ... ]
                    其中 (r1, c1) 是左上角行列号, (r2, c2) 是右下角行列号。
                    如果某个影像没有对应的窗口，可以返回一个无效的窗口，
                    例如 [[-1,-1],[-1,-1]]，程序会处理这种情况。
    """
    # ----- 开始: 用户实现区域 (Start: User Implementation Area) -----
    # 这是一个示例实现，它为每个图像返回一个基于k移动的固定大小窗口
    # 请你根据你的实际需求替换这部分逻辑
    print(f"调用: get_windows(k={k}, num_images={num_images})")
    windows = np.zeros((num_images, 2, 2), dtype=int)
    window_size = 300 # 假设窗口大小为 300x300

    for i in range(num_images):
        img_h, img_w = image_shapes[i][:2]

        # 简单示例：窗口在影像内平移，你可以设计更复杂的逻辑
        # 例如，如果k代表的是特定的特征区域，那么这里的逻辑会完全不同
        offset_r = (k * 100) % max(1, img_h - window_size)
        offset_c = (k * 100) % max(1, img_w - window_size)

        r1 = offset_r
        c1 = offset_c
        r2 = min(img_h, r1 + window_size)
        c1 = min(img_w, c1 + window_size) #修正笔误 c2->c1

        # 修正：确保 r2 和 c2 的计算正确
        r2 = min(img_h -1 , r1 + window_size -1)
        c2 = min(img_w -1, c1 + window_size -1)


        if img_h < window_size or img_w < window_size: # 如果影像太小
            r1, c1 = 0, 0
            r2, c2 = img_h -1 , img_w -1


        windows[i] = [[r1, c1], [r2, c2]]
        # print(f"  影像 {i} (形状: {img_h}x{img_w}): 窗口 k={k} -> [{r1},{c1}] - [{r2},{c2}]")

    if num_images == 0: # 处理没有影像的情况
        return np.array([]).reshape(0,2,2)

    return windows
    # ----- 结束: 用户实现区域 (End: User Implementation Area) -----
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !! 上述 get_windows 函数需要你来实现
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

class TiePointPickerApp:
    def __init__(self, master):
        self.master = master
        master.title("遥感影像刺点工具 (Tie-Point Picker)")
        master.geometry("1200x800")

        self.image_paths = []
        self.loaded_images = [] # 存储cv2读取的原始影像
        self.image_shapes = [] # 存储原始影像的形状
        self.num_images = 0

        # 窗口控制
        self.current_window_k = tk.IntVar(value=0)
        self.window_coords_for_k = None # 存储 get_windows(k) 的结果

        # 影像选择器 (用于左右两个显示区域)
        self.img_idx_left = tk.IntVar(value=0)
        self.img_idx_right = tk.IntVar(value=1)

        # 刺点数据
        self.current_tie_point_group_abs_coords = [] # 存储当前组各影像上的绝对坐标 [ (r,c) or None, ... ]
        self.tie_point_group_counter = 1 # 当前是第几组刺点
        self.point_files = [] # 每个影像对应的txt文件路径

        # 标记点暂存 (用于在canvas上绘制)
        self.drawn_points_on_canvas_left = [] # (canvas_x, canvas_y, point_id)
        self.drawn_points_on_canvas_right = []

        self._setup_ui()
        self._update_ui_state()

    def _setup_ui(self):
        # --- 主框架 ---
        top_frame = ttk.Frame(self.master, padding=10)
        top_frame.pack(fill=tk.X)

        display_frame = ttk.Frame(self.master, padding=10)
        display_frame.pack(fill=tk.BOTH, expand=True)

        status_frame = ttk.Frame(self.master, padding=10)
        status_frame.pack(fill=tk.X)

        # --- 顶部控制区域 ---
        ttk.Button(top_frame, text="加载影像 (Load Images)", command=self._load_images).pack(side=tk.LEFT, padx=5)

        # 窗口k控制
        ttk.Label(top_frame, text="窗口 K:").pack(side=tk.LEFT, padx=(10,0))
        self.k_spinbox = ttk.Spinbox(top_frame, from_=0, to=0, textvariable=self.current_window_k, width=5, command=self._on_window_k_change, state=tk.DISABLED)
        self.k_spinbox.pack(side=tk.LEFT, padx=5)
        # ttk.Button(top_frame, text="应用K (Apply K)", command=self._on_window_k_change).pack(side=tk.LEFT, padx=5) # Spinbox command handles this

        # --- 影像显示区域 ---
        # 左侧影像
        left_panel = ttk.Frame(display_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        left_controls = ttk.Frame(left_panel)
        left_controls.pack(fill=tk.X)
        ttk.Label(left_controls, text="左侧影像 I:").pack(side=tk.LEFT)
        self.img_left_spinbox = ttk.Spinbox(left_controls, from_=0, to=0, textvariable=self.img_idx_left, width=3, command=self._on_img_selection_change, state=tk.DISABLED)
        self.img_left_spinbox.pack(side=tk.LEFT, padx=5)
        self.img_left_label = ttk.Label(left_controls, text="影像: -")
        self.img_left_label.pack(side=tk.LEFT, padx=5)
        ttk.Button(left_controls, text="撤销此点 (Undo Pt)", command=lambda: self._undo_point(0)).pack(side=tk.RIGHT, padx=5)

        self.canvas_left = tk.Canvas(left_panel, bg="gray", width=500, height=500)
        self.canvas_left.pack(fill=tk.BOTH, expand=True)
        self.canvas_left.bind("<Button-1>", lambda event: self._on_canvas_click(event, 0)) # 0 for left panel

        # 右侧影像
        right_panel = ttk.Frame(display_frame)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        right_controls = ttk.Frame(right_panel)
        right_controls.pack(fill=tk.X)
        ttk.Label(right_controls, text="右侧影像 J:").pack(side=tk.LEFT)
        self.img_right_spinbox = ttk.Spinbox(right_controls, from_=0, to=0, textvariable=self.img_idx_right, width=3, command=self._on_img_selection_change, state=tk.DISABLED)
        self.img_right_spinbox.pack(side=tk.LEFT, padx=5)
        self.img_right_label = ttk.Label(right_controls, text="影像: -")
        self.img_right_label.pack(side=tk.LEFT, padx=5)
        ttk.Button(right_controls, text="撤销此点 (Undo Pt)", command=lambda: self._undo_point(1)).pack(side=tk.RIGHT, padx=5)

        self.canvas_right = tk.Canvas(right_panel, bg="gray", width=500, height=500)
        self.canvas_right.pack(fill=tk.BOTH, expand=True)
        self.canvas_right.bind("<Button-1>", lambda event: self._on_canvas_click(event, 1)) # 1 for right panel

        # --- 状态和保存区域 ---
        self.status_label = ttk.Label(status_frame, text="状态: 未加载影像 (Status: No images loaded)")
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.save_button = ttk.Button(status_frame, text="保存当前组并开始下一组 (Save Group & Next)", command=self._save_current_group, state=tk.DISABLED)
        self.save_button.pack(side=tk.RIGHT, padx=5)

        # PIL PhotoImage references (to prevent garbage collection)
        self.photo_left = None
        self.photo_right = None

    def _load_images(self):
        paths = filedialog.askopenfilenames(
            title="选择PNG影像文件 (Select PNG Image Files)",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if not paths:
            return

        self.image_paths = list(paths)
        self.loaded_images = []
        self.image_shapes = []
        self.point_files = []

        try:
            for p in self.image_paths:
                img = cv2.imread(p, cv2.IMREAD_COLOR) # cv2 reads as BGR
                if img is None:
                    raise ValueError(f"无法读取影像: {p}")
                self.loaded_images.append(img)
                self.image_shapes.append(img.shape)
                base, ext = os.path.splitext(p)
                self.point_files.append(f"{base}_points.txt")

        except Exception as e:
            messagebox.showerror("加载错误 (Load Error)", f"加载影像失败 (Failed to load images): {e}")
            self.image_paths = []
            self.loaded_images = []
            self.image_shapes = []
            self.point_files = []
            return

        self.num_images = len(self.loaded_images)
        if self.num_images == 0:
            self._update_ui_state()
            return

        # 初始化刺点数据结构
        self.current_tie_point_group_abs_coords = [None] * self.num_images
        self.tie_point_group_counter = 1

        # 更新Spinbox范围
        self.k_spinbox.config(to=max(0, 1000), state=tk.NORMAL) # 假设最多1000个窗口，你可以调整
        self.img_left_spinbox.config(to=max(0, self.num_images - 1), state=tk.NORMAL if self.num_images > 0 else tk.DISABLED)
        self.img_right_spinbox.config(to=max(0, self.num_images - 1), state=tk.NORMAL if self.num_images > 1 else tk.DISABLED)

        if self.num_images > 0:
            self.img_idx_left.set(0)
        if self.num_images > 1:
            self.img_idx_right.set(1)
        else: # 如果只有一张图，右边也显示这张图（或者禁用）
            self.img_idx_right.set(0)
            self.img_right_spinbox.config(state=tk.DISABLED)


        self.current_window_k.set(0) # 重置k为0
        self._on_window_k_change() # 这会调用 get_windows 和 _display_images
        self._update_status_label()
        self._update_ui_state()

    def _on_window_k_change(self):
        if not self.loaded_images:
            return
        k = self.current_window_k.get()
        try:
            self.window_coords_for_k = get_windows(k, self.num_images, self.image_shapes)
            if self.window_coords_for_k is None or self.window_coords_for_k.shape[0] != self.num_images:
                messagebox.showerror("窗口错误 (Window Error)", f"get_windows({k}) 返回了无效的数据 (returned invalid data).")
                # 提供一个默认的空窗口，避免后续崩溃
                self.window_coords_for_k = np.full((self.num_images, 2, 2), -1, dtype=int)

        except Exception as e:
            messagebox.showerror("窗口错误 (Window Error)", f"调用 get_windows({k}) 出错 (Error calling get_windows): {e}")
            self.window_coords_for_k = np.full((self.num_images, 2, 2), -1, dtype=int) # Fallback
            return

        self._display_images()
        self._update_status_label() # k改变时，当前点在窗口内的位置可能变了

    def _on_img_selection_change(self):
        if not self.loaded_images:
            return
        # 确保左右不选同一张 (如果需要强制不同)
        # if self.num_images > 1 and self.img_idx_left.get() == self.img_idx_right.get():
        #     messagebox.showwarning("提示", "左右两侧选择了相同的影像。")
        #     # 你可以决定是否要自动调整一个，或者允许这样做
        self._display_images()

    def _display_images(self):
        if not self.loaded_images or self.window_coords_for_k is None:
            self.canvas_left.delete("all")
            self.canvas_right.delete("all")
            self.img_left_label.config(text="影像: -")
            self.img_right_label.config(text="影像: -")
            return

        # 显示左侧影像
        idx_l = self.img_idx_left.get()
        self._display_patch_on_canvas(self.canvas_left, idx_l, 0, store_ref='photo_left')
        self.img_left_label.config(text=f"影像: {idx_l} ({os.path.basename(self.image_paths[idx_l])})")

        # 显示右侧影像
        if self.num_images > 0 : # 至少有一张图
            idx_r = self.img_idx_right.get()
            if idx_r >= self.num_images: # 防御性编程，如果spinbox设置不当
                idx_r = 0
                self.img_idx_right.set(0)
            self._display_patch_on_canvas(self.canvas_right, idx_r, 1, store_ref='photo_right')
            self.img_right_label.config(text=f"影像: {idx_r} ({os.path.basename(self.image_paths[idx_r])})")
        else:
            self.canvas_right.delete("all")
            self.img_right_label.config(text="影像: -")


    def _display_patch_on_canvas(self, canvas: tk.Canvas, image_idx: int, panel_id: int, store_ref: str):
        canvas.delete("all") # 清除旧内容，包括旧的 "image_on_canvas" 和 "point"
        self.drawn_points_on_canvas_left = [] if panel_id == 0 else self.drawn_points_on_canvas_left
        self.drawn_points_on_canvas_right = [] if panel_id == 1 else self.drawn_points_on_canvas_right

        if image_idx < 0 or image_idx >= self.num_images or self.window_coords_for_k is None:
            return

        coords = self.window_coords_for_k[image_idx]
        r1, c1 = coords[0]
        r2, c2 = coords[1]

        if r1 == -1 and c1 == -1 and r2 == -1 and c2 == -1: # get_windows指示此影像无有效窗口
            canvas.create_text(canvas.winfo_width()//2, canvas.winfo_height()//2, text="当前窗口无此影像内容\n(No content for this image\n in current window)", anchor=tk.CENTER, fill="white")
            setattr(self, store_ref, None) # 清除旧的photo reference
            return

        # 确保r2 > r1, c2 > c1
        if r2 <= r1 or c2 <= c1:
            canvas.create_text(canvas.winfo_width()//2, canvas.winfo_height()//2, text="无效窗口范围\n(Invalid window extent)", anchor=tk.CENTER, fill="white")
            # print(f"警告: 影像 {image_idx} 窗口范围无效 [{r1},{c1}]-[{r2},{c2}]")
            setattr(self, store_ref, None)
            return


        img_patch_bgr = self.loaded_images[image_idx][r1:r2+1, c1:c2+1] # OpenCV切片是 exclusive for end

        if img_patch_bgr.size == 0:
            canvas.create_text(canvas.winfo_width()//2, canvas.winfo_height()//2, text="影像块为空\n(Image patch is empty)", anchor=tk.CENTER, fill="white")
            setattr(self, store_ref, None)
            return

        img_patch_rgb = cv2.cvtColor(img_patch_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_patch_rgb)

        # 缩放以适应Canvas (保持宽高比)
        canvas_w, canvas_h = canvas.winfo_width(), canvas.winfo_height()
        img_w, img_h = pil_img.size

        if img_w == 0 or img_h == 0 : # 防止除零
             setattr(self, store_ref, None)
             return

        scale_w = canvas_w / img_w
        scale_h = canvas_h / img_h
        scale = min(scale_w, scale_h)

        if scale <= 0: # 如果canvas太小或图片太大导致缩放为0或负，则不显示
             setattr(self, store_ref, None)
             return

        new_w, new_h = int(img_w * scale), int(img_h * scale)

        if new_w <=0 or new_h <=0: # 再次检查
            setattr(self, store_ref, None)
            return

        resized_pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        photo_img = ImageTk.PhotoImage(resized_pil_img)

        setattr(self, store_ref, photo_img) # 保存引用

        # 在Canvas中心显示图片
        x_offset = (canvas_w - new_w) // 2
        y_offset = (canvas_h - new_h) // 2
        canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=photo_img, tags="image_on_canvas")
        canvas.image_scale = scale # 存储缩放比例和偏移，用于坐标转换
        canvas.image_offset = (x_offset, y_offset)
        canvas.patch_origin_coords = (r1,c1) # 存储当前patch在原图的左上角坐标

        # 重绘该影像上已有的刺点 (如果它在当前窗口内)
        point_abs_coords = self.current_tie_point_group_abs_coords[image_idx]
        if point_abs_coords:
            abs_r, abs_c = point_abs_coords
            # 检查点是否在当前patch内
            if r1 <= abs_r <= r2 and c1 <= abs_c <= c2:
                # 转换为patch内相对坐标
                patch_r = abs_r - r1
                patch_c = abs_c - c1
                # 转换为canvas坐标
                canvas_x = int(patch_c * scale + x_offset)
                canvas_y = int(patch_r * scale + y_offset)
                self._draw_point_on_canvas(canvas, canvas_x, canvas_y, panel_id, is_new_point=False)

    def _on_canvas_click(self, event, panel_id: int): # panel_id: 0 for left, 1 for right
        if not self.loaded_images or self.window_coords_for_k is None:
            return

        canvas = self.canvas_left if panel_id == 0 else self.canvas_right
        image_idx = self.img_idx_left.get() if panel_id == 0 else self.img_idx_right.get()

        if not hasattr(canvas, 'image_scale') or not hasattr(canvas, 'image_offset') or not hasattr(canvas, 'patch_origin_coords'):
            # print("Canvas not ready for click (no image attributes)")
            return # 图片未正确加载到canvas

        # 检查是否已为该影像在此组中标点
        if self.current_tie_point_group_abs_coords[image_idx] is not None:
            messagebox.showwarning("刺点限制 (Marking Limit)",
                                 f"影像 {image_idx} 在当前组中已经标过点了。\n"
                                 f"(Image {image_idx} has already been marked in the current group.)\n"
                                 "请先撤销该点，或完成后保存当前组。 (Please undo or save the group.)")
            return

        # 获取canvas上的图片缩放比例和偏移
        scale = canvas.image_scale
        offset_x, offset_y = canvas.image_offset
        patch_r1_abs, patch_c1_abs = canvas.patch_origin_coords

        # 计算在patch上的坐标 (相对于patch左上角)
        patch_x = (event.x - offset_x) / scale
        patch_y = (event.y - offset_y) / scale

        # 获取patch在原图的左上角坐标 (r1, c1)
        # (已在 _display_patch_on_canvas 中存为 canvas.patch_origin_coords)

        # 计算在原图上的绝对坐标
        abs_col = int(round(patch_c1_abs + patch_x))
        abs_row = int(round(patch_r1_abs + patch_y))

        # 边界检查，确保点在原图内
        img_h, img_w = self.image_shapes[image_idx][:2]
        if not (0 <= abs_row < img_h and 0 <= abs_col < img_w):
            messagebox.showwarning("超出边界 (Out of Bounds)", "标记点超出了影像原始边界。 (Point is outside original image bounds.)")
            return

        # 存储绝对坐标
        self.current_tie_point_group_abs_coords[image_idx] = (abs_row, abs_col)
        # print(f"Panel {panel_id}, Img {image_idx}: Click ({event.x},{event.y}) -> Patch ({patch_x:.1f},{patch_y:.1f}) -> Abs ({abs_row},{abs_col})")

        # 在Canvas上绘制点 (使用event.x, event.y，因为这是用户实际点击的位置)
        self._draw_point_on_canvas(canvas, event.x, event.y, panel_id, is_new_point=True)
        self._update_status_label()
        self._check_and_enable_save()

    def _draw_point_on_canvas(self, canvas, x, y, panel_id, is_new_point=True, color="red", size=3):
        # 如果是新点，先清除旧点 (因为每张图每组只允许一个点)
        drawn_points_list = self.drawn_points_on_canvas_left if panel_id == 0 else self.drawn_points_on_canvas_right
        if is_new_point:
            for pt_id in drawn_points_list:
                canvas.delete(pt_id)
            drawn_points_list.clear()

        # 画一个圆圈代表点
        pt_id = canvas.create_oval(x - size, y - size, x + size, y + size, fill=color, outline=color, tags="point")
        drawn_points_list.append(pt_id)


    def _undo_point(self, panel_id: int):
        if not self.loaded_images: return

        image_idx = self.img_idx_left.get() if panel_id == 0 else self.img_idx_right.get()
        canvas = self.canvas_left if panel_id == 0 else self.canvas_right
        drawn_points_list = self.drawn_points_on_canvas_left if panel_id == 0 else self.drawn_points_on_canvas_right

        if self.current_tie_point_group_abs_coords[image_idx] is not None:
            self.current_tie_point_group_abs_coords[image_idx] = None
            for pt_id in drawn_points_list:
                canvas.delete(pt_id)
            drawn_points_list.clear()
            # print(f"撤销了影像 {image_idx} 的刺点 (Undid point for image {image_idx})")
            self._update_status_label()
            self._check_and_enable_save()
        else:
            messagebox.showinfo("提示 (Info)", f"影像 {image_idx} 在当前组中尚未标点。(Image {image_idx} is not marked in the current group.)")


    def _update_status_label(self):
        if not self.loaded_images:
            self.status_label.config(text="状态: 未加载影像 (Status: No images loaded)")
            return

        status_parts = []
        all_marked_for_current_group = True
        for i in range(self.num_images):
            coords = self.current_tie_point_group_abs_coords[i]
            if coords:
                status_parts.append(f"影像{i}: ({coords[0]},{coords[1]})")
            else:
                status_parts.append(f"影像{i}: 未标点 (Pending)")
                all_marked_for_current_group = False

        self.status_label.config(text=f"第 {self.tie_point_group_counter} 组刺点 (Group {self.tie_point_group_counter}) | K={self.current_window_k.get()} | " + " | ".join(status_parts))
        return all_marked_for_current_group

    def _check_and_enable_save(self):
        all_marked = True
        if not self.current_tie_point_group_abs_coords: # 还未加载影像
             all_marked = False
        else:
            for coords in self.current_tie_point_group_abs_coords:
                if coords is None:
                    all_marked = False
                    break
        self.save_button.config(state=tk.NORMAL if all_marked and self.num_images > 0 else tk.DISABLED)

    def _save_current_group(self):
        if not self._update_status_label(): # 确保所有点都已标记
            messagebox.showerror("错误 (Error)", "当前组刺点未完成，请为所有影像标点。 (Current group is incomplete. Mark points on all images.)")
            return

        confirm_save = messagebox.askyesno("确认保存 (Confirm Save)",
                                           f"是否保存第 {self.tie_point_group_counter} 组刺点？\n"
                                           f"(Save tie-point group {self.tie_point_group_counter}?)")
        if not confirm_save:
            return

        try:
            for img_idx in range(self.num_images):
                filepath = self.point_files[img_idx]
                coords = self.current_tie_point_group_abs_coords[img_idx]
                if coords is None: # 应该不会发生，因为有_check_and_enable_save
                    messagebox.showerror("内部错误 (Internal Error)", f"影像 {img_idx} 坐标丢失。(Coordinates missing for image {img_idx})")
                    return

                r, c = coords
                with open(filepath, 'a') as f: # 追加模式
                    f.write(f"{r} {c}\n")
            messagebox.showinfo("保存成功 (Save Successful)", f"第 {self.tie_point_group_counter} 组刺点已保存。 (Group {self.tie_point_group_counter} points saved.)")

            # 重置进行下一组
            self.tie_point_group_counter += 1
            self.current_tie_point_group_abs_coords = [None] * self.num_images
            # 清除画布上的点
            for pt_id in self.drawn_points_on_canvas_left: self.canvas_left.delete(pt_id)
            self.drawn_points_on_canvas_left.clear()
            for pt_id in self.drawn_points_on_canvas_right: self.canvas_right.delete(pt_id)
            self.drawn_points_on_canvas_right.clear()

            self._update_status_label()
            self._check_and_enable_save() # 会禁用保存按钮
            self._display_images() # 刷新显示，清除上一组的点

        except Exception as e:
            messagebox.showerror("保存失败 (Save Failed)", f"保存刺点数据失败 (Failed to save tie-points): {e}")


    def _update_ui_state(self):
        """根据当前状态启用/禁用UI组件"""
        has_images = self.num_images > 0
        self.k_spinbox.config(state=tk.NORMAL if has_images else tk.DISABLED)
        self.img_left_spinbox.config(state=tk.NORMAL if has_images else tk.DISABLED)
        self.img_right_spinbox.config(state=tk.NORMAL if self.num_images > 1 else tk.DISABLED)

        # 保存按钮由 _check_and_enable_save 控制
        if not has_images:
            self.save_button.config(state=tk.DISABLED)
            self.canvas_left.delete("all")
            self.canvas_right.delete("all")
            self.img_left_label.config(text="影像: -")
            self.img_right_label.config(text="影像: -")
            self.status_label.config(text="状态: 未加载影像")


if __name__ == '__main__':
    root = tk.Tk()
    app = TiePointPickerApp(root)
    root.mainloop()