import torch
import pickle


# 1. 能“吃掉”所有初始化参数的伪装列表
class DummyList(list):
    def __new__(cls, *args, **kwargs):
        return list.__new__(cls)

    def __init__(self, *args, **kwargs): pass


# 2. 能“吃掉”所有初始化参数的伪装字典
class DummyDict(dict):
    def __new__(cls, *args, **kwargs):
        return dict.__new__(cls)

    def __init__(self, *args, **kwargs): pass


# 3. 伪装浮点数 (拦截 ScalarFloat，避免数值报错)
class DummyFloat(float):
    def __new__(cls, *args, **kwargs):
        return float.__new__(cls, args[0] if args else 0.0)

    def __init__(self, *args, **kwargs): pass


# 4. 伪装字符串 (拦截 ScalarString)
class DummyString(str):
    def __new__(cls, *args, **kwargs):
        return str.__new__(cls, args[0] if args else "")

    def __init__(self, *args, **kwargs): pass


# 5. ⭐️ 终极无底洞对象，能吃掉任何初始化参数，并允许随意设置属性
class DummyObj:
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # 🟢 绿名单：只允许基础类型和 PyTorch 原生张量正常加载
        if module.startswith('torch') or module.startswith('numpy') or module in ['builtins', 'collections', '_codecs']:
            return super().find_class(module, name)

        print(f"[*] 拦截并伪装: {module}.{name}")

        # 🟡 根据名字动态分配高级伪装类
        name_lower = name.lower()
        if 'seq' in name_lower or 'list' in name_lower:
            return DummyList
        if 'map' in name_lower or 'dict' in name_lower:
            return DummyDict
        if 'float' in name_lower:
            return DummyFloat
        if 'string' in name_lower or 'str' in name_lower:
            return DummyString

        # 🔴 其他所有千奇百怪的类，全部扔进无底洞对象
        return DummyObj


class SafePickleModule:
    Unpickler = SafeUnpickler


def main():
    input_file = 'RS5M_Pretrain.pth'
    output_file = 'clean_RS5M_Pretrain.pth'

    print(f"🚀 正在启动究极版安全净化程序，读取: {input_file}")

    try:
        # 使用我们自定义的究极安全模块强行加载
        checkpoint = torch.load(input_file, map_location='cpu', pickle_module=SafePickleModule)
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return

    print("\n✅ 成功绕过所有属性冲突，读取完成！")

    # 剥离多余外壳，提取真正的纯净权重字典
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
        print("🔍 成功提取干净的模型权重 (state_dict)！已将原作者带入的配置毒瘤全部丢弃。")
    else:
        state_dict = checkpoint
        print("🔍 未找到 'model' 键，默认全文件为权重字典...")

    print(f"💾 正在保存纯净版权重至: {output_file}")
    # 重新保存时，只存最干净的字典，不带任何第三方模块的私货！
    torch.save(state_dict, output_file)
    print("🎉 净化彻底完毕！请修改 train.sh 后安全起飞！")


if __name__ == "__main__":
    main()