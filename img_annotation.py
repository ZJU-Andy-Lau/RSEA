import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os

class ImageAnnotator:
    """
    一个用于图像标注的GUI应用程序。
    支持打开图像、使用滚轮缩放、右键拖拽、左键添加标注点、
    撤销和保存标注点等功能。
    """
    def __init__(self, master):
        self.master = master
        self.master.title("图像标注工具")
        self.master.geometry("1000x800")

        # --------------- 内部状态变量 ---------------
        self.image_path = None
        self.pil_image = None
        self.tk_image = None
        
        # 点的坐标 (row, col)
        self.saved_points = []  # 从文件加载或已保存的点 (绿色)
        self.new_points = []    # 本次会话中新增的点 (红色)
        
        # 视图控制变量
        self.scale = 1.0
        self.view_x = 0
        self.view_y = 0
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.point_radius = 5  # 标注点的显示半径

        # --------------- UI 控件 ---------------
        # 顶部按钮框架
        top_frame = tk.Frame(self.master)
        top_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        self.btn_open = tk.Button(top_frame, text="打开图像 (Ctrl+O)", command=self.open_image)
        self.btn_open.pack(side=tk.LEFT, padx=5)

        self.btn_save = tk.Button(top_frame, text="保存 (Ctrl+S)", command=self.save_annotations)
        self.btn_save.pack(side=tk.LEFT, padx=5)

        self.btn_undo = tk.Button(top_frame, text="撤销 (Ctrl+Z)", command=self.undo_annotation)
        self.btn_undo.pack(side=tk.LEFT, padx=5)
        
        # 信息标签
        self.info_label = tk.Label(top_frame, text="无图像加载")
        self.info_label.pack(side=tk.RIGHT, padx=10)

        # 主画布
        self.canvas = tk.Canvas(self.master, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # --------------- 事件绑定 ---------------
        # 鼠标事件
        self.canvas.bind("<ButtonPress-1>", self.on_left_click)      # 左键点击 (标注)
        self.canvas.bind("<ButtonPress-3>", self.on_right_press)   # 右键按下 (拖拽开始)
        self.canvas.bind("<B3-Motion>", self.on_right_drag)         # 右键拖拽 (平移)
        
        # 滚轮事件 (跨平台兼容)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)       # Windows & macOS
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)         # Linux (向上滚)
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)         # Linux (向下滚)

        # 键盘快捷键
        self.master.bind("<Control-o>", lambda event: self.open_image())
        self.master.bind("<Control-s>", lambda event: self.save_annotations())
        self.master.bind("<Control-z>", lambda event: self.undo_annotation())
    
    def open_image(self):
        """打开一张新图像，并重置所有状态"""
        path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
        )
        if not path:
            return

        self.image_path = path
        try:
            self.pil_image = Image.open(self.image_path)
        except Exception as e:
            messagebox.showerror("打开失败", f"无法加载图像文件: {e}")
            return
        
        # 重置状态
        self.scale = 1.0
        self.view_x = 0
        self.view_y = 0
        self.saved_points = []
        self.new_points = []
        
        # 尝试加载已有的标注文件
        self.load_annotations()
        
        self.info_label.config(text=f"图像: {os.path.basename(self.image_path)}")
        self.redraw_canvas()

    def load_annotations(self):
        """从同名txt文件中加载标注点"""
        if not self.image_path:
            return
        
        txt_path = os.path.splitext(self.image_path)[0] + ".txt"
        if os.path.exists(txt_path):
            try:
                with open(txt_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) == 2:
                            row, col = int(parts[0]), int(parts[1])
                            self.saved_points.append((row, col))
                print(f"成功加载 {len(self.saved_points)} 个标注点。")
            except Exception as e:
                messagebox.showerror("加载标注失败", f"读取标注文件失败: {e}")

    def save_annotations(self):
        """将所有标注点保存到txt文件"""
        if not self.image_path:
            messagebox.showwarning("保存失败", "没有加载图像，无法保存。")
            return

        # 将新点合并到旧点中
        self.saved_points.extend(self.new_points)
        self.new_points = []

        if not self.saved_points:
            print("没有标注点需要保存。")
            self.redraw_canvas() # 即使不保存也重绘，以更新颜色
            return

        txt_path = os.path.splitext(self.image_path)[0] + ".txt"
        try:
            with open(txt_path, 'w') as f:
                for row, col in self.saved_points:
                    f.write(f"{row},{col}\n")
            messagebox.showinfo("成功", f"标注已保存到:\n{txt_path}")
        except Exception as e:
            messagebox.showerror("保存失败", f"写入文件时发生错误: {e}")
        
        self.redraw_canvas()

    def undo_annotation(self):
        """撤销最后一个添加的新标注点"""
        if self.new_points:
            self.new_points.pop()
            self.redraw_canvas()
        else:
            print("没有可撤销的新标注。")

    def redraw_canvas(self):
        """根据当前缩放、平移和标注点重绘整个画布"""
        self.canvas.delete("all")
        if not self.pil_image:
            return
        
        # 根据当前缩放比例计算显示尺寸
        disp_w = int(self.pil_image.width * self.scale)
        disp_h = int(self.pil_image.height * self.scale)
        
        # 使用NEAREST以获得最快的重绘速度，适用于频繁缩放
        # 如果追求高质量缩放，可以换成 Image.Resampling.BILINEAR 或 BICUBIC
        resized_pil_image = self.pil_image.resize((disp_w, disp_h), Image.Resampling.NEAREST)
        
        # 必须保持对tk_image的引用，否则会被垃圾回收
        self.tk_image = ImageTk.PhotoImage(resized_pil_image)
        
        # 在画布上绘制图像
        self.canvas.create_image(self.view_x, self.view_y, anchor="nw", image=self.tk_image)
        
        # 绘制标注点
        # 绘制已保存的点 (绿色)
        for row, col in self.saved_points:
            cx, cy = self.image_to_canvas(row, col)
            self.draw_point(cx, cy, "green")
            
        # 绘制未保存的点 (红色)
        for row, col in self.new_points:
            cx, cy = self.image_to_canvas(row, col)
            self.draw_point(cx, cy, "red")

    def draw_point(self, cx, cy, color):
        """在画布上绘制一个点"""
        r = self.point_radius
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=color, outline="white", width=1)

    # --------------- 坐标转换 ---------------
    def canvas_to_image(self, canvas_x, canvas_y):
        """将画布坐标转换为图像坐标 (row, col)"""
        col = (canvas_x - self.view_x) / self.scale
        row = (canvas_y - self.view_y) / self.scale
        return int(row), int(col)

    def image_to_canvas(self, row, col):
        """将图像坐标 (row, col) 转换为画布坐标"""
        canvas_x = col * self.scale + self.view_x
        canvas_y = row * self.scale + self.view_y
        return canvas_x, canvas_y

    # --------------- 事件处理器 ---------------
    def on_left_click(self, event):
        """处理左键点击事件，添加新标注点"""
        if not self.pil_image:
            return
            
        row, col = self.canvas_to_image(event.x, event.y)
        
        # 检查点击是否在图像范围内
        if 0 <= row < self.pil_image.height and 0 <= col < self.pil_image.width:
            self.new_points.append((row, col))
            self.redraw_canvas()

    def on_right_press(self, event):
        """记录右键按下的初始位置"""
        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def on_right_drag(self, event):
        """处理右键拖拽事件，平移视图"""
        if not self.pil_image:
            return
        
        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y
        
        self.view_x += dx
        self.view_y += dy
        
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        
        self.redraw_canvas()

    def on_mouse_wheel(self, event):
        """处理滚轮事件，以鼠标为中心进行缩放"""
        if not self.pil_image:
            return
        
        # --- 确定缩放因子 ---
        scale_factor = 1.1  # 缩放灵敏度
        # Linux系统下event.delta是空的, 但event.num存在
        if event.num == 5 or event.delta < 0: # 向下滚，缩小
            self.scale /= scale_factor
        elif event.num == 4 or event.delta > 0: # 向上滚，放大
            self.scale = min(self.scale * scale_factor, 20) # 限制最大缩放倍数

        # --- 以鼠标为中心进行缩放的关键逻辑 ---
        # 1. 获取鼠标在画布上的位置
        mouse_x, mouse_y = event.x, event.y
        
        # 2. 计算缩放前，鼠标指向的图像上的点
        img_row_before, img_col_before = self.canvas_to_image(mouse_x, mouse_y)
        
        # 3. 更新视图的偏移量
        self.view_x = mouse_x - img_col_before * self.scale
        self.view_y = mouse_y - img_row_before * self.scale
        
        self.redraw_canvas()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageAnnotator(root)
    root.mainloop()