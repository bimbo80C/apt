def read_first_10_lines(file_path, output_file):
    min_value = float('inf')  # 初始化最小值为正无穷大
    with open(file_path, 'r', encoding='utf-8') as file, open(output_file, 'w', encoding='utf-8') as output:
        for line in file:
            line = line.strip()
            if '\t' in line:
                last_part = line.rsplit('\t',1)[-1]  # 通过 rsplit 从右侧分割，取最后一部分
            else:
                last_part = line  # 如果没有空格，直接取整行
            current_value = float(last_part)
            if current_value < min_value:
                min_value = current_value
            output.write(last_part + '\n')  # 写入到输出文件
    print(f"最小值是: {min_value}")
# 使用示例
file_path = '../dataset/trace/attr_event.txt'  # 将此处替换为你的txt文件路径
output_path = './dataset/mini_attr_time.txt'
read_first_10_lines(file_path,output_path)
