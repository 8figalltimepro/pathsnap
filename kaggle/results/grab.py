import glob, os, shutil
from pathlib import Path
work = Path("/kaggle/working")
work.mkdir(parents=True, exist_ok=True)
patterns = ["**/results/*.json", "**/results/*.csv", "**/results/*.npz", "**/mlp_model.pt"]
copied = []
for pat in patterns:
    for src in glob.glob("/kaggle/input/**/" + pat.split("/", 1)[1], recursive=True):
        name = Path(src).name
        if "dataset_build" in name:
            continue
        dst = work / name
        shutil.copy2(src, dst)
        copied.append(str(dst))
print("copied:", sorted(set(copied)))
print("tree:")
for root, dirs, files in os.walk("/kaggle/input"):
    depth = root.count(os.sep) - 2
    if depth > 2:
        continue
    print(root, len(files))
