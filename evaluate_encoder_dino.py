import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# 导入你的模型
from model_new import EncoderDino
from utils import get_current_time
import argparse
import cv2
import kornia.augmentation as K

# 设置Matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 'SimHei' 是黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# --- 0. 辅助函数 ---

def extract_features_at_coords(feature_map, coords, device='cpu'):
    """
    从特征图中根据坐标精确提取特征向量。
    使用grid_sample进行亚像素级别的精确插值。

    Args:
        feature_map (Tensor): Encoder输出的特征图, shape [1, C, H, W]
        coords (Tensor): 归一化到[-1, 1]的坐标, shape [1, N, 1, 2] for (x,y)
    
    Returns:
        Tensor: 提取出的特征向量, shape [N, C]
    """
    features = nn.functional.grid_sample(
        feature_map, 
        coords, 
        mode='bilinear', 
        padding_mode='border', 
        align_corners=False
    ) # 输出 shape [1, C, N, 1]
    return features.squeeze().T 

# --- 1. 清晰度与双射性诊断 ---

def run_clarity_and_bijectivity_diagnostics(encoder, image, output_dir, device='cpu'):
    """运行清晰度和双射性诊断。"""
    print("\n--- 开始诊断: 特征清晰度与双射性 ---")
    
    # --- 1.1 t-SNE 可视化 ---
    print("正在生成 t-SNE 可视化图...")
    
    H, W = image.shape[-2:]
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, H - 1, steps=32),
        torch.linspace(0, W - 1, steps=32),
        indexing='ij'
    )
    coords_flat = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    
    coords_normalized = coords_flat.clone()
    coords_normalized[:, 0] = 2.0 * coords_normalized[:, 0] / (W - 1) - 1.0
    coords_normalized[:, 1] = 2.0 * coords_normalized[:, 1] / (H - 1) - 1.0
    coords_normalized = coords_normalized.unsqueeze(0).unsqueeze(2).to(device)
    
    with torch.no_grad():
        feature_map = encoder(image)
        # 检查encoder是否有downsample_factor属性
        if not hasattr(encoder, 'SAMPLE_FACTOR'):
            raise AttributeError("Encoder模型必须包含 'SAMPLE_FACTOR' 属性。")

        features = extract_features_at_coords(feature_map, coords_normalized, device=device)

    features_np = features.cpu().numpy()
    
    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=42)
    features_2d = tsne.fit_transform(features_np)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"t-SNE 可视化 - {encoder.__class__.__name__}", fontsize=16)

    sc1 = ax1.scatter(features_2d[:, 0], features_2d[:, 1], c=coords_flat[:, 0].numpy(), cmap='viridis')
    ax1.set_title("根据原始X坐标着色")
    ax1.set_xlabel("t-SNE 维度1")
    ax1.set_ylabel("t-SNE 维度2")
    fig.colorbar(sc1, ax=ax1)
    
    sc2 = ax2.scatter(features_2d[:, 0], features_2d[:, 1], c=coords_flat[:, 1].numpy(), cmap='viridis')
    ax2.set_title("根据原始Y坐标着色")
    ax2.set_xlabel("t-SNE 维度1")
    fig.colorbar(sc2, ax=ax2)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(output_dir, "tsne_visualization.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"t-SNE 可视化图已保存至: {save_path}")

    # --- 1.2 逆向探针量化 ---
    print("\n正在训练逆向探针以量化特征信息...")
    
    class InverseProbeNet(nn.Module):
        def __init__(self, feature_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(feature_dim, 64), nn.ReLU(),
                nn.Linear(64, 2)
            )
        def forward(self, x):
            return self.net(x)

    probe_features = features.detach()
    probe_targets = coords_flat.clone()
    probe_targets[:, 0] /= (W - 1)
    probe_targets[:, 1] /= (H - 1)
    probe_targets = probe_targets.to(device)

    feature_dim = probe_features.shape[1]
    if probe_features.numel() == 0 or feature_dim == 0:
        print("特征维度为0或为空，无法训练逆向探针。")
        return
        
    probe_net = InverseProbeNet(feature_dim).to(device)
    probe_optimizer = optim.Adam(probe_net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    print("开始训练探针...")
    for step in tqdm(range(1000), desc="训练逆向探针"):
        probe_optimizer.zero_grad()
        pred_coords = probe_net(probe_features)
        loss = loss_fn(pred_coords, probe_targets)
        loss.backward()
        probe_optimizer.step()
    
    final_loss = loss.item()
    print(f"逆向探针训练完成。最终均方误差 (MSE Loss): {final_loss:.6f}")
    print("解读: 损失越低，说明特征中包含的坐标信息越丰富、越清晰。")

# --- 2. 平滑性诊断 ---

def run_smoothness_diagnostics(encoder, image, output_dir, device='cpu'):
    """运行平滑性诊断。"""
    print("\n--- 开始诊断: 特征平滑性 ---")
    
    H, W = image.shape[-2:]
    
    # --- 2.1 特征轨迹线可视化 ---
    print("正在生成特征轨迹线可视化图...")
    
    path_y = H // 2
    path_x = torch.arange(0, W)
    path_coords = torch.stack([path_x, torch.full_like(path_x, path_y)], dim=1)
    
    path_coords_normalized = path_coords.clone().float()
    path_coords_normalized[:, 0] = 2.0 * path_coords_normalized[:, 0] / (W - 1) - 1.0
    path_coords_normalized[:, 1] = 2.0 * path_coords_normalized[:, 1] / (H - 1) - 1.0
    path_coords_normalized = path_coords_normalized.unsqueeze(0).unsqueeze(2).to(device)

    with torch.no_grad():
        feature_map = encoder(image)
        trajectory_features = extract_features_at_coords(feature_map, path_coords_normalized, device)
        trajectory_features_np = trajectory_features.cpu().numpy()
    
    if trajectory_features_np.shape[0] > 2 and trajectory_features_np.shape[1] > 2:
        pca = PCA(n_components=2)
        trajectory_2d = pca.fit_transform(trajectory_features_np)
    else:
        trajectory_2d = trajectory_features_np[:, :2] if trajectory_features_np.shape[1] > 2 else trajectory_features_np

    fig = plt.figure(figsize=(8, 8))
    plt.plot(trajectory_2d[:, 0], trajectory_2d[:, 1], marker='.', markersize=4)
    plt.title(f"特征轨迹线 (PCA降维) - {encoder.__class__.__name__}")
    plt.xlabel("主成分 1")
    plt.ylabel("主成分 2")
    plt.grid(True)
    save_path = os.path.join(output_dir, "feature_trajectory.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"特征轨迹线图已保存至: {save_path}")

    # --- 2.2 局部敏感度量化 ---
    print("\n正在计算局部敏感度...")
    
    num_samples = 10000
    c1_x = torch.randint(0, W - 2, (num_samples,))
    c1_y = torch.randint(0, H - 1, (num_samples,))
    c1 = torch.stack([c1_x, c1_y], dim=1)
    c2 = c1.clone() + torch.tensor([1, 0])
    coords_all = torch.cat([c1, c2], dim=0)
    
    coords_all_normalized = coords_all.clone().float()
    coords_all_normalized[:, 0] = 2.0 * coords_all_normalized[:, 0] / (W - 1) - 1.0
    coords_all_normalized[:, 1] = 2.0 * coords_all_normalized[:, 1] / (H - 1) - 1.0
    coords_all_normalized = coords_all_normalized.unsqueeze(0).unsqueeze(2).to(device)

    R = np.array([0])
    mean_r, var_r = float('nan'), float('nan')
    with torch.no_grad():
        all_features = extract_features_at_coords(feature_map, coords_all_normalized, device)
        f1, f2 = all_features.chunk(2, dim=0)
        dist_feat = torch.linalg.norm(f1 - f2, dim=1)
        R = dist_feat.cpu().numpy()
        mean_r, var_r = np.mean(R), np.var(R)

    print(f"局部敏感度分析完成。")
    print(f"相邻特征距离的均值 (Mean of R): {mean_r:.6f}")
    print(f"相邻特征距离的方差 (Variance of R): {var_r:.6f}")
    print("解读: 均值和方差越小，说明特征空间对于坐标空间越平滑。")

    fig = plt.figure(figsize=(8, 5))
    plt.hist(R, bins=50)
    plt.title(f"局部敏感度比率(R)分布 - {encoder.__class__.__name__}")
    plt.xlabel("相邻特征距离")
    plt.ylabel("频数")
    plt.grid(True)
    save_path = os.path.join(output_dir, "sensitivity_distribution.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"局部敏感度分布图已保存至: {save_path}")

# --- 3. 主执行函数 ---
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_path',type=str)
    parser.add_argument('--img_path',type=str)
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {DEVICE}")

    # 创建输出目录
    OUTPUT_DIR = f"./vis/diagnostic_results_{get_current_time()}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"所有诊断结果将保存在 '{OUTPUT_DIR}' 文件夹中。")

    transform = nn.Sequential(
            #for-swt
            K.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]), 
                std=torch.tensor([0.229, 0.224, 0.225])
            )
            #for-dino
            # K.Normalize(
            #     mean=torch.tensor([0.430, 0.411, 0.296]), 
            #     std=torch.tensor([0.213, 0.156, 0.143])
            # )
        ).eval().to(DEVICE)

    img = cv2.imread(args.img_path)
    img = cv2.resize(img,(1024,1024))
    img = torch.from_numpy(img).float().unsqueeze(0).permute(0,3,1,2).to(DEVICE) / 255.0
    img = transform(img)

    # 实例化你的Encoder
    try:
        encoder = EncoderDino(dino_weight_path = os.path.join(args.encoder_path,'dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth')).to(DEVICE)
        encoder.load_adapter(os.path.join(args.encoder_path,'adapter.pth'))
        encoder.eval() # 设置为评估模式
    except Exception as e:
        print(f"实例化或加载模型时出错: {e}")
        print("请确保 'model_new.py' 文件存在且其中包含一个有效的 'Encoder' 类。")
        exit()

    print("\n" + "="*50)
    print(f"正在诊断 Encoder: {encoder.__class__.__name__}")
    print("="*50)
    
    # 运行所有诊断
    run_clarity_and_bijectivity_diagnostics(encoder, img, OUTPUT_DIR, DEVICE)
    run_smoothness_diagnostics(encoder, img, OUTPUT_DIR, DEVICE)
    
    print("\n" + "="*50)
    print(f"Encoder {encoder.__class__.__name__} 诊断结束。")
    print("="*50)
