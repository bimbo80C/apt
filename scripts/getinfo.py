import re

attr_event_set = {
" attr_memory = str(uuid) + '\t' + str(record) + '\t' + str(epoch) + '\t' + str(memory_address) + '\t' + str(tgid) + '\t' + str(size) + '\n'",
"                    attr_event = str(uuid) + '\t' + str(record) + '\t' + str(event_type) + '\t' + str(seq) + '\t' + str(thread_id) + '\t' + str(src) + '\t' + str(dst1) + '\t' + str(dst2) + '\t' + str(size) + '\t' + str(time) + '\n'",
"                    attr_unnamed = str(uuid) + '\t' + str(record) + '\t' + str(epoch) + '\t' + str(pid) + '\t' + str(source_file_descriptor) + '\t' + str(sink_file_descriptor) + '\n'",
"                    attr_netflow = str(uuid) + '\t' + str(record) + '\t' + str(epoch) + '\t' + str(local_address) + '\t' + str(local_port) + '\t' + str(remote_address) + '\t' + str(remote_port) + '\t' + str(ip_protocol) + '\n'",
"                    attr_file = str(uuid) + '\t' + str(record) + '\t' + str(file_type) + '\t' + str(epoch) + '\t' + str(permission) + '\t' + str(path) + '\n'",
"                    attr_subject = str(uuid) + '\t' + str(record) + '\t' + str(subject_type) + '\t' + str(parent) + '\t' + str(local_principal) + '\t' + str(cid) + '\t' + str(start_time) + '\t' + str(unit_id) + '\t' + str(cmdline) + '\n'",
"                    attr_principal = str(uuid) + '\t' + str(record) + '\t' + str(principal_type) + '\t' + str(user_id) + '\t' + str(group_ids) + '\t' + str(euid) + '\n'",
"                    attr_src = str(uuid) + '\t' + str(record) + '\t' + str(epoch) + '\t' + str(pid) + '\t' + str(fileDescriptor) + '\n'",
} 

with open('./info/memo.txt', 'w') as file:
    for attr_event in attr_event_set:
        field_names = re.findall(r'str\((.*?)\)', attr_event)
        result = '\t'.join(field_names)
        file.write(result + '\n')
