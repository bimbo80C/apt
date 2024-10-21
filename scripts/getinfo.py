import re

# 原始字符串
attr_event = " attr_memory = str(uuid) + '\t' + str(record) + '\t' + str(epoch) + '\t' + str(memory_address) + '\t' + str(tgid) + '\t' + str(size) + '\n'"

# 使用正则表达式提取 str() 内部的内容
field_names = re.findall(r'str\((.*?)\)', attr_event)

# 将字段名称转换为字符串，用制表符分隔
result = '\t'.join(field_names)

print(result)
