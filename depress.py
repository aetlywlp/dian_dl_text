import gzip
import shutil
import os


def decompress_gz(file_path, output_path):
    """
    解压 .gz 文件到指定路径，生成没有 .gz 后缀的文件。

    :param file_path: .gz 文件的完整路径
    :param output_path: 解压后的文件路径
    """
    try:
        # 确保文件存在
        if not os.path.isfile(file_path):
            print(f"文件不存在: {file_path}")
            return

        # 检查是否已经解压
        if os.path.isfile(output_path):
            print(f"文件已解压: {output_path}")
            return

        # 解压文件
        with gzip.open(file_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"成功解压: {output_path}")

    except Exception as e:
        print(f"解压 {file_path} 时出错: {e}")


def main():
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 构建 raw_dir 的绝对路径
    raw_dir = os.path.join(script_dir, 'data', 'FashionMNIST', 'raw')

    # 检查 raw_dir 是否存在
    if not os.path.isdir(raw_dir):
        print(f"目录不存在: {raw_dir}")
        return

    # 遍历 raw_dir 下的所有文件
    for file_name in os.listdir(raw_dir):
        if file_name.endswith('.gz'):
            file_path = os.path.join(raw_dir, file_name)
            output_file = os.path.join(raw_dir, file_name[:-3])  # 移除 .gz 后缀
            decompress_gz(file_path, output_file)


if __name__ == '__main__':
    main()
