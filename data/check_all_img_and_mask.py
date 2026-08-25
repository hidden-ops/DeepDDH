import os
import numpy as np
from PIL import Image

# =========================
# 修改这里：你的数据集根目录
# =========================
ROOT_DIR = r"Testing"

IMG_DIR = os.path.join(ROOT_DIR, "imgs")
SEG_DIR = os.path.join(ROOT_DIR, "segs")
BONY_DIR = os.path.join(ROOT_DIR, "bony_masks")

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_image_files(folder):
    files = {}
    for fn in os.listdir(folder):
        ext = os.path.splitext(fn)[1].lower()
        if ext in VALID_EXTS:
            stem = os.path.splitext(fn)[0]
            files[stem] = fn
    return files


def read_image_keep_original(path):
    """
    使用 PIL.Image 读取图像，保留原始通道信息
    返回:
        img_np: numpy数组
        mode: PIL图像模式，如 L / RGB / RGBA
    """
    try:
        img = Image.open(path)
    except Exception as e:
        raise ValueError(f"无法读取图像: {path}, 错误: {e}")

    img_np = np.array(img)
    return img_np, img.mode


def pixel_distribution_original(img_np):
    """
    保留原图分布进行统计：
    - 单通道图: 统计每个像素值的出现次数
    - 多通道图: 统计每种像素组合的出现次数
    """
    if img_np.ndim == 2:
        unique_vals, counts = np.unique(img_np, return_counts=True)
        distribution = [(int(v), int(c)) for v, c in zip(unique_vals, counts)]
        return "single", distribution

    elif img_np.ndim == 3:
        h, w, c = img_np.shape
        pixels = img_np.reshape(-1, c)
        unique_pixels, counts = np.unique(pixels, axis=0, return_counts=True)
        distribution = [(tuple(map(int, pix)), int(cnt)) for pix, cnt in zip(unique_pixels, counts)]
        return "multi", distribution

    else:
        raise ValueError(f"不支持的图像维度: {img_np.shape}")


def print_distribution(title, img_np, mode, dist_type, distribution):
    total = img_np.shape[0] * img_np.shape[1]

    print(f"\n[{title}]")
    print(f"mode: {mode}")
    print(f"shape: {img_np.shape}")
    print(f"总像素数: {total}")
    print(f"不同像素取值/组合个数: {len(distribution)}")

    for value, count in distribution:
        ratio = count / total * 100
        print(f"  像素值 {value}: {count} ({ratio:.4f}%)")


def main():
    imgs_files = list_image_files(IMG_DIR)
    segs_files = list_image_files(SEG_DIR)
    bony_files = list_image_files(BONY_DIR)

    imgs_names = set(imgs_files.keys())
    segs_names = set(segs_files.keys())
    bony_names = set(bony_files.keys())

    common_names = sorted(imgs_names & segs_names & bony_names)

    print("========== 文件匹配检查 ==========")
    print(f"imgs 数量: {len(imgs_names)}")
    print(f"segs 数量: {len(segs_names)}")
    print(f"bony_masks 数量: {len(bony_names)}")
    print(f"三者同名匹配数量: {len(common_names)}")

    if len(common_names) == 0:
        print("未找到三者都存在的同名图像。")
        return

    print("\n========== 开始逐张统计原图分布 ==========")

    for name in common_names:
        img_path = os.path.join(IMG_DIR, imgs_files[name])
        seg_path = os.path.join(SEG_DIR, segs_files[name])
        bony_path = os.path.join(BONY_DIR, bony_files[name])

        try:
            img, img_mode = read_image_keep_original(img_path)
            seg, seg_mode = read_image_keep_original(seg_path)
            bony, bony_mode = read_image_keep_original(bony_path)
        except Exception as e:
            print(f"\n[跳过] {name} 读取失败: {e}")
            continue

        try:
            img_type, img_dist = pixel_distribution_original(img)
            seg_type, seg_dist = pixel_distribution_original(seg)
            bony_type, bony_dist = pixel_distribution_original(bony)
        except Exception as e:
            print(f"\n[跳过] {name} 统计失败: {e}")
            continue

        # print("\n" + "=" * 80)
        print(f"图像名称: {name}", "max", np.max(img), "min", np.min(img), "std", np.std(img))
        # print(f"imgs 路径: {img_path}")
        # print(f"segs 路径: {seg_path}")
        # print(f"bony_masks 路径: {bony_path}")

        # print_distribution("imgs", img, img_mode, img_type, img_dist)
        # print_distribution("segs", seg, seg_mode, seg_type, seg_dist)
        # print_distribution("bony_masks", bony, bony_mode, bony_type, bony_dist)

    print("\n========== 统计结束 ==========")


if __name__ == "__main__":
    main()