import shutil, re
from pathlib import Path

BASE_DIR = Path("/home/artem/Documents/TechnoStrelka")
COMBINED_DIR = BASE_DIR / "combined_dataset"
AOD4_DIR = BASE_DIR / "Airborne-Object-Detection-4-AOD4-2"
DATASET_DIRS = [BASE_DIR / str(i) for i in range(1, 8)]

TARGET_CLASSES = {
    "aerial-object": 0, "airplane": 1, "bird": 2, "drone": 3, "helicopter": 4
}
KEYWORDS = {
    "aerial-object": ["aerial-object", "aerialobject"],
    "airplane": ["airplane", "aircraft", "plane", "aeroplane"],
    "bird": ["bird"],
    "drone": ["drone", "uav", "quadcopter", "multirotor", "0"],
    "helicopter": ["helicopter", "chopper"]
}

def read_data_yaml(dataset_dir):
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists(): return None
    content = yaml_path.read_text()
    m = re.search(r"names\s*:\s*\[([^\]]+)\]", content)
    if m:
        return [n.strip().strip("'\"") for n in m.group(1).split(",")]
    m = re.search(r"names\s*:\s*\n((?:\s*-[^\n]+\n?)+)", content)
    if m:
        return re.findall(r"-\s*(.+)", m.group(1))
    return None

def build_remap_and_keep(source_names):
    remap, keep = {}, set()
    for idx, name in enumerate(source_names):
        norm = name.lower().strip()
        for target, kws in KEYWORDS.items():
            if any(kw in norm for kw in kws):
                remap[idx] = TARGET_CLASSES[target]
                keep.add(idx)
                break
    return remap, keep

def copy_filtered(src_dir, dst_dir, remap, keep):
    for subset in ["train", "valid"]:
        src_lbl = src_dir / subset / "labels"
        src_img = src_dir / subset / "images"
        dst_lbl = dst_dir / subset / "labels"
        dst_img = dst_dir / subset / "images"
        if not src_lbl.exists() or not src_img.exists(): continue
        dst_lbl.mkdir(parents=True, exist_ok=True)
        dst_img.mkdir(parents=True, exist_ok=True)
        for txt in src_lbl.glob("*.txt"):
            lines = txt.read_text().splitlines()
            new = []
            has = False
            for line in lines:
                parts = line.strip().split()
                if not parts: continue
                cls = int(parts[0])
                if cls in keep:
                    has = True
                    if cls in remap: parts[0] = str(remap[cls])
                    new.append(" ".join(parts))
            if not has: continue
            img_name = txt.stem
            img_path = None
            for ext in [".jpg",".jpeg",".png",".bmp",".tif"]:
                p = src_img / (img_name + ext)
                if p.exists():
                    img_path = p
                    break
            if img_path is None: continue
            shutil.copy2(img_path, dst_img / img_path.name)
            (dst_lbl / txt.name).write_text("\n".join(new))

def create_yaml():
    yaml = f"path: {COMBINED_DIR}\ntrain: train/images\nval: valid/images\n\nnames:\n"
    for name, idx in sorted(TARGET_CLASSES.items(), key=lambda x: x[1]):
        yaml += f"  {idx}: {name}\n"
    (COMBINED_DIR / "data.yaml").write_text(yaml)

# ── main ──
if __name__ == "__main__":
    if COMBINED_DIR.exists():
        shutil.rmtree(COMBINED_DIR)
    COMBINED_DIR.mkdir(parents=True)

    print("Копирую AOD-4...")
    copy_filtered(AOD4_DIR, COMBINED_DIR, remap={}, keep=set(range(len(TARGET_CLASSES))))

    for d in DATASET_DIRS:
        if not d.exists(): continue
        print(f"\nОбрабатываю {d.name}...")
        names = read_data_yaml(d)
        if names is None: continue
        print(f"  Классы: {names}")
        remap, keep = build_remap_and_keep(names)
        if not keep:
            print("  Нет подходящих классов, пропуск.")
            continue
        copy_filtered(d, COMBINED_DIR, remap, keep)

    create_yaml()
    print("\nГотово! Датасет собран в", COMBINED_DIR)