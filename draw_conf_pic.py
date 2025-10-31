import cv2
import numpy as np
import torch
from model.encoder_dino_0927 import EncoderDino
import os
from torchvision import transforms
import argparse
from utils import get_current_time

def draw_confidence_heatmap(image: np.ndarray, 
                            confidence_map: np.ndarray, 
                            output_path: str,
                            heatmap_alpha: float = 0.4):
    """
    根据置信度图在原图上绘制平滑的红-绿热力图。

    低置信度 (0.0) 映射为红色，高置信度 (1.0) 映射为绿色。

    参数:
    image (np.ndarray): 
        输入的 BGR 图像，形状为 (H, W, 3)，数据类型为 np.uint8。
    confidence_map (np.ndarray): 
        置信度图，形状为 (h, w)，数据类型为 float，值范围 0.0 ~ 1.0。
    output_path (str): 
        叠加后图像的保存路径。
    heatmap_alpha (float, 可选): 
        热力图的不透明度，范围 0.0 ~ 1.0。默认为 0.4。
    """
    
    # --- 步骤 0: 输入校验 ---
    if not (0.0 <= heatmap_alpha <= 1.0):
        raise ValueError(f"heatmap_alpha 必须在 [0, 1] 范围内, 但得到了 {heatmap_alpha}")
        
    if image.dtype != np.uint8:
        print(f"警告: 输入图像的数据类型为 {image.dtype}，将强制转换为 np.uint8。")
        image = image.astype(np.uint8)
        
    if confidence_map.min() < 0.0 or confidence_map.max() > 1.0:
        print(f"警告: 置信度图的值超出了 [0, 1] 范围。将进行裁剪。")
        confidence_map = np.clip(confidence_map, 0.0, 1.0)

    # --- 步骤一：上采样与平滑 ---
    
    # 1.1. 获取原图的目标尺寸 (H, W)
    H, W = image.shape[:2]
    
    # 1.2. 使用双三次插值 (INTER_CUBIC) 将置信度图 resize 到原图大小
    #      OpenCV 的 resize 函数需要 (W, H) 格式的尺寸
    #      这步操作实现了 "上采样" 和 "平滑连续"
    try:
        heatmap_resized = cv2.resize(confidence_map, (W, H), interpolation=cv2.INTER_CUBIC)
    except Exception as e:
        print(f"错误: OpenCV resize 失败 - {e}")
        print("请检查输入图像和置信度图是否有效。")
        return

    # 1.3. 归一化/裁剪
    #      插值过程可能会产生略微超出 [0, 1] 范围的值，我们将其裁剪回来
    heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)

    # --- 步骤二：色彩映射 (Colormapping) ---
    
    # 我们需要一个从 0.0 (红) 到 1.0 (绿) 的渐变
    # 在 BGR 色彩空间中:
    # 红色 = (B=0, G=0, R=255)
    # 绿色 = (B=0, G=255, R=0)
    
    # 2.1. 计算 R (红色) 通道
    #      v=0.0 (低置信度) -> R=255
    #      v=1.0 (高置信度) -> R=0
    R_channel = (1.0 - heatmap_resized) * 255.0
    
    # 2.2. 计算 G (绿色) 通道
    #      v=0.0 (低置信度) -> G=0
    #      v=1.0 (高置信度) -> G=255
    G_channel = heatmap_resized * 255.0
    
    # 2.3. B (蓝色) 通道始终为 0
    B_channel = np.zeros_like(heatmap_resized)
    
    # 2.4. 合并三个通道 (注意 OpenCV 的 BGR 顺序)
    heatmap_color = np.stack([B_channel, G_channel, R_channel], axis=-1)
    
    # 2.5. 转换数据类型为 np.uint8 (0-255 整数)，以便进行图像融合
    heatmap_color = heatmap_color.astype(np.uint8)

    # --- 步骤三：图像融合 (Blending) ---
    
    # 3.1. 计算原图的权重
    image_alpha = 1.0 - heatmap_alpha
    
    # 3.2. 使用 cv2.addWeighted 进行带权重的图像叠加
    #      公式: blended = image * image_alpha + heatmap * heatmap_alpha + gamma
    #      我们设置 gamma (额外亮度) 为 0
    try:
        blended_image = cv2.addWeighted(image, image_alpha, heatmap_color, heatmap_alpha, 0)
    except cv2.error as e:
        print(f"错误: 图像融合失败 (cv2.addWeighted) - {e}")
        print(f"原图 尺寸: {image.shape}, Dtype: {image.dtype}")
        print(f"热力图 尺寸: {heatmap_color.shape}, Dtype: {heatmap_color.dtype}")
        return

    # --- 步骤四：保存输出 ---
    
    try:
        cv2.imwrite(output_path, blended_image)
        print(f"成功: 叠加了热力图的图像已保存至 {output_path}")
    except Exception as e:
        print(f"错误: 无法保存图像到 {output_path} - {e}")

def main():
    # 步骤 1: Argparse
    parser = argparse.ArgumentParser(description="""
    图像对特征匹配与可视化 (V2 - 支持仿射变换).
    1. 随机生成仿射变换 (N 次).
    2. Warp 图像到 (1024, 1024).
    3. 提取特征 (by get_feature).
    4. 计算特征图上每个点对应的原图坐标.
    5. 找到原图坐标距离最小的最佳匹配对.
    6. 保存 K 个最佳结果的可视化.
    """)
    parser.add_argument("--img_path", type=str, help="输入图片1的路径")
    parser.add_argument("--output_dir", type=str, help="输出结果的目录")
    parser.add_argument("--n_crops", type=int, default=100, help="随机裁切的总次数 (N)")
    parser.add_argument("--k_top", type=int, default=5, help="保存前 K 个最佳匹配结果")
    parser.add_argument("--downsample_s", type=int, default=16, help="特征提取器的下采样率 (s)")
    parser.add_argument('--encoder_path',type=str)
    
    args = parser.parse_args()
    os.makedirs(args.output_dir)
    img = cv2.imread(args.img_path)
    encoder = EncoderDino(
            'weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',
            upsample_times=int(np.log2(16 // args.downsample_s)),
            use_adapter=True,
            use_conf = True
        )
    encoder.load_adapter(os.path.join(args.encoder_path,'adapter.pth'))
    encoder.cuda()
    encoder.eval()
    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) 
                        ])
    img_tensor = transform(img)[None].cuda()

    _,conf = encoder(img_tensor)
    conf = conf.squeeze().cpu().numpy()
    draw_confidence_heatmap(img,conf,os.path.join(args.output_dir,f'conf_vis_{get_current_time()}.png'))

if __name__ == '__main__':
    main()