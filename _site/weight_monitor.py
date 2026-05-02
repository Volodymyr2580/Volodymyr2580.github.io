import os
import csv
from datetime import date
import tkinter as tk
from tkinter import simpledialog, messagebox

def load_data(path):
    data = []
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    day = int(row.get("day", "").strip())
                    weight = float(row.get("weight_kg", "").strip())
                    d = (row.get("date") or "").strip()
                    data.append({"day": day, "weight_kg": weight, "date": d})
                except:
                    pass
    return data

def upsert(data, day, weight, d):
    for r in data:
        if r["day"] == day:
            r["weight_kg"] = weight
            r["date"] = d
            return True
    data.append({"day": day, "weight_kg": weight, "date": d})
    return False

def save_data(data, path):
    data_sorted = sorted(data, key=lambda r: r["day"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["day", "weight_kg", "date"])
        writer.writeheader()
        writer.writerows(data_sorted)

def plot_data(data, out_path):
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    data_sorted = sorted(data, key=lambda r: r["day"])
    days = [r["day"] for r in data_sorted]
    weights = [r["weight_kg"] for r in data_sorted]
    fig, ax = plt.subplots(figsize=(9, 5), dpi=120)
    ax.plot(days, weights, marker="o", color="#1f77b4")
    ax.set_title("体重变化曲线")
    ax.set_xlabel("第n天")
    ax.set_ylabel("体重(kg)")
    ax.grid(True, linestyle="--", alpha=0.4)
    if days:
        ax.scatter([days[-1]], [weights[-1]], color="#d62728")
        ax.annotate(f"{weights[-1]:.1f} kg", xy=(days[-1], weights[-1]), xytext=(10, -10), textcoords="offset points", color="#d62728")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "weights.csv")
    plot_path = os.path.join(base_dir, "weight_progress.png")
    root = tk.Tk()
    root.withdraw()
    n = simpledialog.askinteger("输入", "今天是 i^2 计划减肥的第几天？", minvalue=1)
    if n is None:
        messagebox.showinfo("已取消", "未输入天数，程序已退出。")
        return
    w = simpledialog.askfloat("输入", "请输入今日体重(kg)：", minvalue=0.0)
    if w is None:
        messagebox.showinfo("已取消", "未输入体重，程序已退出。")
        return
    today_str = date.today().isoformat()
    data = load_data(data_path)
    existed = upsert(data, n, w, today_str)
    save_data(data, data_path)
    try:
        plot_data(data, plot_path)
    except ImportError:
        messagebox.showerror("缺少依赖", "未检测到 matplotlib，请安装后再运行。")
        return
    messagebox.showinfo("完成", f"已保存数据并更新图像。\nCSV: {data_path}\n图像: {plot_path}\n" + (f"已更新第 {n} 天数据" if existed else "已新增今日数据"))
    try:
        os.startfile(plot_path)
    except:
        pass

if __name__ == "__main__":
    main()