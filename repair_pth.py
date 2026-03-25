import torch
import pickle
import types


# 1. 究极无敌黑洞类：吸收一切报错，伪装成所有数据结构！
class Dummy:
    def __init__(self, *args, **kwargs): pass

    def __setattr__(self, key, value): self.__dict__[key] = value

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)

    # 伪装成列表 (List) 吸收操作
    def append(self, *args, **kwargs): pass

    def extend(self, *args, **kwargs): pass

    def insert(self, *args, **kwargs): pass

    # 伪装成集合/字典吸收操作
    def add(self, *args, **kwargs): pass

    def update(self, *args, **kwargs): pass

    # 🚀 伪装成字典进行索引赋值 (解决 item assignment 报错)
    def __setitem__(self, key, value): pass

    def __getitem__(self, key): return Dummy()  # 如果它想读取，继续返回一个黑洞

    # 如果代码试图调用任何不存在的方法，直接返回一个什么都不做的空函数
    def __getattr__(self, item):
        return lambda *args, **kwargs: Dummy()


# 2. 安全解包器
class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        safe_modules = ['collections', 'builtins', '_codecs', 'typing', 'numpy', 'numpy.core.multiarray', 'argparse']
        if module.startswith('torch') or module in safe_modules:
            try:
                return super().find_class(module, name)
            except Exception:
                return Dummy
        return Dummy


fake_pickle = types.ModuleType('fake_pickle')
fake_pickle.Unpickler = SafeUnpickler
fake_pickle.load = lambda f, **kw: SafeUnpickler(f, **kw).load()

print("🚀 正在启动究极暴力破解，强行提取原版权重...")
try:
    # 绕过一切报错，强行加载！
    ckpt = torch.load('RS5M_Pretrain.pth', map_location='cpu', pickle_module=fake_pickle)

    # 定位字典
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        sd = ckpt['state_dict']
    elif isinstance(ckpt, dict) and 'model' in ckpt:
        sd = ckpt['model']
    else:
        sd = ckpt

    pure_sd = {}
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            # 统一剥离冗余前缀，确保最纯净
            new_k = k
            if new_k.startswith('module.'): new_k = new_k[7:]
            if new_k.startswith('model.'): new_k = new_k[6:]
            if new_k.startswith('clip.'): new_k = new_k[5:]
            pure_sd[new_k] = v

    print(f"✅ 破解成功！提取出 {len(pure_sd)} 个纯净的权重张量。")
    torch.save(pure_sd, 'pure_RS5M_Pretrain.pth')
    print("🎉 已保存为终极纯净版：pure_RS5M_Pretrain.pth")
    print("接下来只需在 train.sh 中使用 pure_RS5M_Pretrain.pth 即可！")
except Exception as e:
    print("❌ 提取失败，错误信息：", e)