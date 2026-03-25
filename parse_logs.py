import os
import json
import pandas as pd
from pathlib import Path


def extract_metrics(log_root="./checkpoints/rsitmd/"):
    all_data = []
    # 搜索目录下所有的 log.txt 文件
    log_files = list(Path(log_root).rglob("*-log.txt"))

    if not log_files:
        print(f"未在 {log_root} 找到任何日志文件！")
        return

    for log_path in log_files:
        print(f"正在处理: {log_path}")
        with open(log_path, 'r') as f:
            for line in f:
                try:
                    # 跳过非 JSON 格式的行（比如 best epoch 打印）
                    if not line.strip().startswith('{'):
                        continue

                    data = json.loads(line)
                    # 只提取包含测试指标的行
                    if "test_r_mean" in data:
                        metrics = {
                            "Epoch": data.get("epoch"),
                            "IR@1": data.get("test_img_r1"),
                            "IR@5": data.get("test_img_r5"),
                            "IR@10": data.get("test_img_r10"),
                            "TR@1": data.get("test_txt_r1"),
                            "TR@5": data.get("test_txt_r5"),
                            "TR@10": data.get("test_txt_r10"),
                            "mR": data.get("test_r_mean"),
                            "Path": log_path.parent.name
                        }
                        all_data.append(metrics)
                except json.JSONDecodeError:
                    continue

    if all_data:
        df = pd.DataFrame(all_data)
        # 按照 mR 从高到低排序，让你一眼看到最高分
        df = df.sort_values(by="mR", ascending=False)

        # 保存为 CSV 方便在 Excel 中查看
        output_file = "experiment_results.csv"
        df.to_csv(output_file, index=False)
        print(f"\n✅ 提取完成！结果已保存至: {output_file}")
        print("\n--- 前 5 名最佳表现 ---")
        print(df.head(5).to_string(index=False))
    else:
        print("未能从日志中提取到有效指标。")


if __name__ == "__main__":
    # 如果你跑的是 rsicd，把这里改一下
    extract_metrics("./checkpoints/rsitmd/")