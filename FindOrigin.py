import re
import os
datasets = ["cadets"]
for dataset in datasets:
    with open(f'./groundtruth/{dataset}.txt', 'r') as f:
        uuids = set(line.strip() for line in f)
    # 定义正则表达式模式
    pattern_uuid = re.compile(r'uuid":"(.*?)"')
    # 获取目录下的所有 .json 文件
    json_dir = f'./dataset/{dataset}/origin_json'
    json_files = [os.path.join(json_dir, f) for f in os.listdir(json_dir)]
    # 打开输出文件
    with open(f'./output_{dataset}.txt', 'w') as output_file:
        for json_file in json_files:
            # 处理每个 JSON 文件
            print(json_file)
            with open(json_file, 'r') as source_file:
                for line in source_file:
                    # 在每行中搜索 UUID
                    match = pattern_uuid.search(line)
                    if match:
                        uuid = match.group(1)
                        if uuid in uuids:
                            output_file.write(f"File: {json_file}\n")
                            output_file.write(line)

    print(f'匹配完成,已经写入output_{dataset}.txt')


