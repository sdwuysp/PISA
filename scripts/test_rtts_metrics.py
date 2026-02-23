import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import pyiqa
import pandas as pd
from tqdm import tqdm
import glob
from PIL import Image

# ================= 配置区域 =================
# 待测试的图片文件夹路径
img_folder = 'results/rtts' 

# 图片扩展名
exts = ['*.png', '*.jpg', '*.jpeg', '*.bmp']

# 需要计算的指标列表 (对应 PyIQA 的模型名称)
# 注意：PI 是越小越好，其他通常是越大越好
metrics_to_run = [
    'musiq',      # Multi-Scale Image Quality Transformer
    'pi',         # Perceptual Index (越小越好)
    'maniqa',     # Multi-dimension Attention Network
    'clipiqa',    # CLIP-based IQA
    'topiq_iaa',  # TOPIQ (Top-down IQA)
    'qalign',     # Q-Align (注意：这个模型很大，首次运行会自动下载约几GB权重)
]

# 结果保存文件名
output_csv = 'metrics_rtts_results.csv'
# ===========================================

def get_image_paths(folder):
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder, ext)))
    # 排序以保证顺序一致
    return sorted(files)

def main():
    # 1. 准备环境
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"🚀 使用设备: {device}")
    
    img_paths = get_image_paths(img_folder)
    if not img_paths:
        print(f"❌ 错误: 在 {img_folder} 未找到图片")
        return

    print(f"📂 找到 {len(img_paths)} 张图片，准备计算...")

    # 初始化一个 DataFrame 用来存所有结果，先存文件名
    # 提取纯文件名（不带路径）作为索引
    file_names = [os.path.basename(p) for p in img_paths]
    df = pd.DataFrame({'Filename': file_names})
    df.set_index('Filename', inplace=True)

    # 2. 逐个指标进行计算 (Metric-Major Loop)
    # 这种方式为了节省显存：加载一个模型 -> 跑完所有图 -> 删模型 -> 下一个
    for metric_name in metrics_to_run:
        print(f"\n======== 正在处理指标: {metric_name.upper()} ========")
        
        try:
            # 2.1 加载模型
            # PyIQA 会自动下载预训练权重到缓存文件夹
            iqa_model = pyiqa.create_metric(metric_name, device=device)
            
            # 如果是 PI 指标，它包含 NIQE 和 Ma，通常不需要 gradients
            # Q-Align 等大模型建议开启 eval 模式
            if hasattr(iqa_model, 'eval'):
                iqa_model.eval()

            scores = []
            
            # 2.2 遍历所有图片
            with torch.no_grad(): # 禁用梯度计算，节省显存
                for img_path in tqdm(img_paths, desc=f"计算 {metric_name}"):
                    # PyIQA 处理图片读取和预处理
                    # 注意：PyIQA 内部会自动将路径转为 Tensor
                    try:
                        score = iqa_model(img_path)
                        # score 通常是一个 tensor，取数值
                        scores.append(score.item())
                    except Exception as e:
                        print(f"⚠️ 图片 {os.path.basename(img_path)} 计算出错: {e}")
                        scores.append(None)
            
            # 2.3 将结果写入 DataFrame
            df[metric_name] = scores
            
            # 2.4 释放显存
            del iqa_model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ 指标 {metric_name} 初始化或计算失败: {e}")
            print("可能是显存不足或网络问题导致权重下载失败。")
            continue

    # 3. 统计平均值并保存
    print("\n======== 计算完成，统计结果 ========")
    
    # 计算平均分 (忽略 NaN)
    mean_scores = df.mean(numeric_only=True)
    
    # 打印平均分
    print(mean_scores)
    
    # 追加一行平均值到表格最后
    df.loc['AVERAGE'] = mean_scores
    
    # 保存
    df.to_csv(output_csv)
    print(f"\n✅ 详细结果已保存至: {os.path.abspath(output_csv)}")

if __name__ == '__main__':
    main()