import json
import numpy as np
import sys
# 设置要读取的行数
lines_to_read = 5
data_list = []
filename = sys.argv[1]
# 打开文件并逐行读取
with open(filename, 'r') as f:
    for i, line in enumerate(f):
        if i >= lines_to_read:
            break
        # 将每行 JSON 数据解析为 Python 对象
        data = json.loads(line.strip())
        data_list.append(data)
print(data)
