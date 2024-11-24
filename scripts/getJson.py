import json

def extract_first_10_items(file_path, output_file):

    # 读取整个 JSON 文件到字典
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)  # 假设文件内容为字典
    
    # 提取前10个键值对
    first_10_items = {k: data[k] for i, (k, v) in enumerate(data.items()) if i < 22}
    
    # 将前10个键值对写入新文件
    with open(output_file, 'w', encoding='utf-8') as output:
        json.dump(first_10_items, output, ensure_ascii=False, indent=4)
    

# 使用示例
file_path = '../dataset/trace/cnt_record_map.json'  # 输入文件路径
output_path = './dataset/mini_cnt_record_map.json'  # 输出文件路径
extract_first_10_items(file_path, output_path)
