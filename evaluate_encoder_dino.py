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
    print("\n--- Start Diagnosis: Feature Clarity & Bijectivity ---")
    
    # --- 1.1 t-SNE 可视化 ---
    print("Generating t-SNE visualization...")
    
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
        feature_map,_ = encoder(image)
        # 检查encoder是否有SAMPLE_FACTOR属性
        if not hasattr(encoder, 'SAMPLE_FACTOR'):
            raise AttributeError("Encoder model must have a 'SAMPLE_FACTOR' attribute.")

        features = extract_features_at_coords(feature_map, coords_normalized, device=device)

    features_np = features.cpu().numpy()
    
    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=42)
    features_2d = tsne.fit_transform(features_np)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"t-SNE Visualization - {encoder.__class__.__name__}", fontsize=16)

    sc1 = ax1.scatter(features_2d[:, 0], features_2d[:, 1], c=coords_flat[:, 0].numpy(), cmap='viridis')
    ax1.set_title("Colored by Original X-coordinate")
    ax1.set_xlabel("t-SNE Dimension 1")
    ax1.set_ylabel("t-SNE Dimension 2")
    fig.colorbar(sc1, ax=ax1)
    
    sc2 = ax2.scatter(features_2d[:, 0], features_2d[:, 1], c=coords_flat[:, 1].numpy(), cmap='viridis')
    ax2.set_title("Colored by Original Y-coordinate")
    ax2.set_xlabel("t-SNE Dimension 1")
    fig.colorbar(sc2, ax=ax2)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(output_dir, "tsne_visualization.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"t-SNE visualization saved to: {save_path}")

    # --- 1.2 逆向探针量化 ---
    print("\nTraining inverse probe to quantify feature information...")
    
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
        print("Feature dimension is 0 or empty, cannot train inverse probe.")
        return
        
    probe_net = InverseProbeNet(feature_dim).to(device)
    probe_optimizer = optim.Adam(probe_net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    print("Starting probe training...")
    for step in tqdm(range(1000), desc="Training Inverse Probe"):
        probe_optimizer.zero_grad()
        pred_coords = probe_net(probe_features)
        loss = loss_fn(pred_coords, probe_targets)
        loss.backward()
        probe_optimizer.step()
    
    final_loss = loss.item()
    print(f"Inverse probe training finished. Final Mean Squared Error (MSE Loss): {final_loss:.6f}")
    print("Interpretation: A lower loss indicates that the features contain richer and clearer coordinate information.")

# --- 2. 平滑性诊断 ---

def run_smoothness_diagnostics(encoder, image, output_dir, device='cpu'):
    """运行平滑性诊断。"""
    print("\n--- Start Diagnosis: Feature Smoothness ---")
    
    H, W = image.shape[-2:]
    
    # --- 2.1 特征轨迹线可视化 ---
    print("Generating feature trajectory visualization...")
    
    path_y = H // 2
    path_x = torch.arange(0, W)
    path_coords = torch.stack([path_x, torch.full_like(path_x, path_y)], dim=1)
    
    path_coords_normalized = path_coords.clone().float()
    path_coords_normalized[:, 0] = 2.0 * path_coords_normalized[:, 0] / (W - 1) - 1.0
    path_coords_normalized[:, 1] = 2.0 * path_coords_normalized[:, 1] / (H - 1) - 1.0
    path_coords_normalized = path_coords_normalized.unsqueeze(0).unsqueeze(2).to(device)

    with torch.no_grad():
        feature_map,_ = encoder(image)
        trajectory_features = extract_features_at_coords(feature_map, path_coords_normalized, device)
        trajectory_features_np = trajectory_features.cpu().numpy()
    
    if trajectory_features_np.shape[0] > 2 and trajectory_features_np.shape[1] > 2:
        pca = PCA(n_components=2)
        trajectory_2d = pca.fit_transform(trajectory_features_np)
    else:
        trajectory_2d = trajectory_features_np[:, :2] if trajectory_features_np.shape[1] > 2 else trajectory_features_np

    fig = plt.figure(figsize=(8, 8))
    plt.plot(trajectory_2d[:, 0], trajectory_2d[:, 1], marker='.', markersize=4)
    plt.title(f"Feature Trajectory (PCA Reduced) - {encoder.__class__.__name__}")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    save_path = os.path.join(output_dir, "feature_trajectory.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Feature trajectory plot saved to: {save_path}")

    # --- 2.2 局部敏感度量化 ---
    print("\nCalculating local sensitivity...")
    
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

    print(f"Local sensitivity analysis complete.")
    print(f"Mean of adjacent feature distances (Mean of R): {mean_r:.6f}")
    print(f"Variance of adjacent feature distances (Variance of R): {var_r:.6f}")
    print("Interpretation: Smaller mean and variance indicate a smoother feature space with respect to the coordinate space.")

    fig = plt.figure(figsize=(8, 5))
    plt.hist(R, bins=50)
    plt.title(f"Local Sensitivity Ratio (R) Distribution - {encoder.__class__.__name__}")
    plt.xlabel("Distance between Adjacent Features")
    plt.ylabel("Frequency")
    plt.grid(True)
    save_path = os.path.join(output_dir, "sensitivity_distribution.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Local sensitivity distribution plot saved to: {save_path}")

# --- 3. 主执行函数 ---
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_path',type=str)
    parser.add_argument('--img_path',type=str)
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # 创建输出目录
    OUTPUT_DIR = f"./vis/diagnostic_results_{get_current_time()}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"All diagnostic results will be saved in the '{OUTPUT_DIR}' folder.")

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
        print(f"Error instantiating or loading model: {e}")
        print("Please ensure 'model_new.py' exists and contains a valid 'Encoder' class.")
        exit()

    print("\n" + "="*50)
    print(f"Diagnosing Encoder: {encoder.__class__.__name__}")
    print("="*50)
    
    # 运行所有诊断
    run_clarity_and_bijectivity_diagnostics(encoder, img, OUTPUT_DIR, DEVICE)
    run_smoothness_diagnostics(encoder, img, OUTPUT_DIR, DEVICE)
    
    print("\n" + "="*50)
    print(f"Encoder {encoder.__class__.__name__} diagnosis finished.")
    print("="*50)

