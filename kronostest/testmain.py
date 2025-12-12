import pickle

# 替换为你的实际文件路径
file_path = 'predictions-2.pkl'

try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # 打印数据类型以确认加载成功
    print(f"加载成功！数据类型: {type(data)}")

    # 如果是字典，可以查看键
    if isinstance(data, dict):
        print(f"包含的键: {data.keys()}")
    # 如果是列表，查看前几项
    elif isinstance(data, list):
        print(f"数据预览: {data[:2]}")

except Exception as e:
    print(f"加载失败: {e}")