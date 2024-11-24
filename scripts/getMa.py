def record_min_max(file_path, output_file):
    min_value = float('inf')  # 初始化最小值为正无穷大
    max_value = float('-inf')  # 初始化最大值为负无穷大
    
    with open(file_path, 'r', encoding='utf-8') as file, open(output_file, 'w', encoding='utf-8') as output:
        for line in file:
            line = line.strip().split('\t')
            ma = float(line[3])
            
            # 更新最小值和最大值
            min_value = min(min_value, ma)
            max_value = max(max_value, ma)
            
            # 写入当前值到输出文件
            output.write(f'{ma}\n')
        
        # 写入最大值和最小值到输出文件
        output.write(f'\nMin Value: {min_value}\n')
        output.write(f'Max Value: {max_value}\n')

# 使用示例
file_path = '../dataset/trace/attr_memory.txt' 
output_path = './dataset/mini_attr_memory.txt'
record_min_max(file_path, output_path)
