# 文件路径
input_file = '../dataset/trace/attr_subject.txt'
output_file = 'subject_cid_catalogue.txt'

# 用于存储唯一的第三个元素
unique_third_elements = set()
# 用于存储新出现的第三个元素
new_third_elements = []

# 读取文件并提取每行的第三个元素
with open(input_file, 'r') as file:
    for line in file:
        # 使用 \t 分割每行
        elements = line.strip().split('\t')
        
        # 检查是否有足够的元素，避免索引错误
        if len(elements) >= 3:
            third_element = elements[5]
            # 检查是否已经出现过
            if third_element not in unique_third_elements:
                unique_third_elements.add(third_element)  # 加入集合以跟踪
                new_third_elements.append(third_element)  # 添加到新出现的元素列表

# 输出新出现的第三个元素到控制台
for element in new_third_elements:
    print(element)

# 输出总共的新出现的元素数量
print(f"\n总共的新出现的第三个元素数量: {len(new_third_elements)}")

# 可选：将结果写入新的文件
with open(output_file, 'w') as file:
    for element in new_third_elements:
        file.write(element + '\n')

print(f"\n提取的新出现的第三个元素已写入 {output_file}")
