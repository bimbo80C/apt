def read_first_10_lines(file_path, output_file):
    with open(file_path, 'r', encoding='utf-8') as file, open(output_file, 'w', encoding='utf-8') as output:
        for i in range(10):
            line = file.readline()
            if not line:
                break
            output.write(line)

# 使用示例
file_path = '../dataset/trace/attr_subject.txt'  # 将此处替换为你的txt文件路径
output_path = './dataset/mini_attr_subject.txt'
read_first_10_lines(file_path,output_path)
