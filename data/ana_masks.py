import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter


def count_pixels_in_image(img_path):
    """
    读取灰度图并统计每个像素值的数量
    返回:
        dict: {pixel_value: count}
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    unique, counts = np.unique(img, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist()))


def main():
    # =========================
    # 1. 修改这里
    # =========================
    ROOT = Path(r"./Testing/")
    # =========================

    segs_dir = ROOT / "segs"
    bony_dir = ROOT / "bony_masks"

    if not segs_dir.exists():
        raise FileNotFoundError(f"不存在文件夹: {segs_dir}")
    if not bony_dir.exists():
        raise FileNotFoundError(f"不存在文件夹: {bony_dir}")

    # 支持常见图片格式
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    seg_files = {p.name: p for p in segs_dir.iterdir() if p.is_file() and p.suffix.lower() in exts}
    bony_files = {p.name: p for p in bony_dir.iterdir() if p.is_file() and p.suffix.lower() in exts}

    common_names = sorted(set(seg_files.keys()) & set(bony_files.keys()))

    print(f"segs 中图片数: {len(seg_files)}")
    print(f"bony_masks 中图片数: {len(bony_files)}")
    print(f"同名图片数: {len(common_names)}")

    per_image_rows = []
    summary_counter = {
        "segs": Counter(),
        "bony_masks": Counter()
    }

    for name in common_names:
        seg_path = seg_files[name]
        bony_path = bony_files[name]

        # 统计 segs
        seg_counts = count_pixels_in_image(seg_path)
        for pixel_value, count in seg_counts.items():
            per_image_rows.append({
                "folder": "segs",
                "image_name": name,
                "pixel_value": pixel_value,
                "count": count
            })
            summary_counter["segs"][pixel_value] += count

        # 统计 bony_masks
        bony_counts = count_pixels_in_image(bony_path)
        for pixel_value, count in bony_counts.items():
            per_image_rows.append({
                "folder": "bony_masks",
                "image_name": name,
                "pixel_value": pixel_value,
                "count": count
            })
            summary_counter["bony_masks"][pixel_value] += count

    # 保存每张图的统计结果
    per_image_df = pd.DataFrame(per_image_rows)
    per_image_out = ROOT / "pixel_value_stats_per_image.csv"
    per_image_df.to_csv(per_image_out, index=False, encoding="utf-8-sig")

    # 保存汇总统计结果
    summary_rows = []
    for folder_name, counter in summary_counter.items():
        for pixel_value in sorted(counter.keys()):
            summary_rows.append({
                "folder": folder_name,
                "pixel_value": pixel_value,
                "total_count": counter[pixel_value]
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_out = ROOT / "pixel_value_stats_summary.csv"
    summary_df.to_csv(summary_out, index=False, encoding="utf-8-sig")

    print(f"\n每张图像统计已保存到: {per_image_out}")
    print(f"汇总统计已保存到: {summary_out}")


if __name__ == "__main__":
    main()