import os
import scipy.io
import h5py
import numpy as np

# 1. 指定源目录和目标目录
source_dir = '/media/aetly/Data/BaiduNetdiskDownload/sEMG_DeepLearning-master/datah5/'
target_dir = '/media/aetly/Data/BaiduNetdiskDownload/sEMG_DeepLearning-master/datah5/'

# 2. 获取源目录下的所有 .mat 文件列表
mat_files = [f for f in os.listdir(source_dir) if f.endswith('.mat')]

# 检查是否找到 .mat 文件
if not mat_files:
    print("在指定目录中未找到任何 .mat 文件。")
else:
    print(f"在目录 {source_dir} 中找到 {len(mat_files)} 个 .mat 文件。")

# 3. 遍历每个 .mat 文件并进行转换
for mat_file in mat_files:
    # 构建完整的文件路径
    mat_file_path = os.path.join(source_dir, mat_file)

    # 打印当前处理的文件
    print(f"正在转换文件: {mat_file_path}")

    # 4. 加载 .mat 文件
    mat = scipy.io.loadmat(mat_file_path)

    # 5. 提取所需的数据变量
    # 请根据实际的变量名替换以下内容
    # 假设变量名为 'emgData' 和 'emgLabel'
    if 'emg' in mat and 'restimulus' in mat:
        emg_data = mat['emg']
        emg_label = mat['restimulus']
    else:
        print(f"文件 {mat_file} 中未找到 'emgData' 或 'emgLabel' 变量，跳过该文件。")
        continue  # 跳过此文件，继续下一个

    # 6. 数据类型转换（可选）
    emg_data = emg_data.astype(np.float32)
    emg_label = emg_label.astype(np.int64)

    # 7. 构建目标 .h5 文件的路径和名称
    h5_file_name = os.path.splitext(mat_file)[0] + '.h5'
    h5_file_path = os.path.join(target_dir, h5_file_name)

    # 8. 创建并保存 .h5 文件
    with h5py.File(h5_file_path, 'w') as hf:
        hf.create_dataset('emg', data=emg_data)
        hf.create_dataset('restimulus', data=emg_label)

    print(f"文件 {mat_file} 转换完成，保存为 {h5_file_name}")

print("所有文件转换完成。")
