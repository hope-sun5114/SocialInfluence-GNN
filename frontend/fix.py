import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(BASE_DIR, "data", "graph.json")
output_file = os.path.join(BASE_DIR, "data", "graph_fixed.json")

print("读取路径：", input_file)

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# ---- 提取节点 ----
nodes = []
for obj in data.get("nodes", []):
    if "id" in obj:
        nodes.append({"id": obj["id"]})

# ---- 提取边 ----
links = []
for obj in data.get("edges", []):
    if "source" in obj and "target" in obj:
        links.append({
            "source": obj["source"],
            "target": obj["target"]
        })

# ---- 输出标准格式 ----
fixed = {
    "nodes": nodes,
    "links": links
}

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(fixed, f, indent=2, ensure_ascii=False)

print("修复完成！")
print("节点数：", len(nodes))
print("边数：", len(links))
print("输出文件：", output_file)
