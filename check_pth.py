import torch

# 修改为你权重的实际路径
ckpt_path = './RS5M_Pretrain.pth'
checkpoint = torch.load(ckpt_path, map_location='cpu')
state_dict = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint

# 打印前 10 个键名，看看前缀到底是什么
print("--- 权重文件中的前 10 个 Key ---")
for i, key in enumerate(list(state_dict.keys())[:10]):
    print(f"{i}: {key}")