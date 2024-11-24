def read_first_10_lines(file_path, output_file):
    with open(file_path, 'r', encoding='utf-8') as file, open(output_file, 'w', encoding='utf-8') as output:
        for line in file:
            line = line.strip().split('\t')
            src_ip = line[3]  
            dst_ip = line[5]
            output.write(src_ip + '\n')
            output.write(dst_ip + '\n')
# 使用示例
file_path = '../dataset/trace/attr_netflow.txt' 
output_path = './dataset/mini_attr_ip.txt'
read_first_10_lines(file_path,output_path)
