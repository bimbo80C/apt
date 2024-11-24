import re
import ipaddress
ipv4_ints = []
ipv6_ints = []
def extract_ip_addresses(file_path):
    ipv4_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    ipv6_pattern = r'\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b'
    with open(file_path, 'r') as file:
        for line in file:
            # 查找IPv4地址
            ipv4_matches = re.findall(ipv4_pattern, line)
            for ipv4 in ipv4_matches:
                ipv4_int = int(ipaddress.IPv4Address(ipv4))
                ipv4_ints.append(ipv4_int)
                print(f"IPv4 Address: {ipv4} -> Integer: {ipv4_int}")
            
            # 查找IPv6地址
            ipv6_matches = re.findall(ipv6_pattern, line)
            for ipv6 in ipv6_matches:
                ipv6_int = int(ipaddress.IPv6Address(ipv6))
                ipv6_ints.append(ipv6_int)
                print(f"IPv6 Address: {ipv6} -> Integer: {ipv6_int}")


file_path = './dataset/mini_attr_ip.txt'
ipv4_addresses, ipv6_addresses = extract_ip_addresses(file_path)
