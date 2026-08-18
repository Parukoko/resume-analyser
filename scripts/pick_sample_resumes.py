"""Randomly picks real resume PDFs from the Kaggle "Resume Dataset" (snehaanbhawal)
and copies them into data/picked/, for exercising the /analyze-resume pipeline
end to end against real (if generic/template-like) resumes across a spread of fit.

Setup (manual — Kaggle requires a login, so this isn't scripted):
    1. https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset -> Download
    2. Save the zip as data/raw/kaggle_resume_dataset/archive.zip

Usage:
    python scripts/pick_sample_resumes.py --strong 1 --medium 1 --weak 1 --seed 42

data/ is gitignored: nothing this script reads or writes gets committed. These
are real people's resumes (Kaggle's dataset card describes them as anonymized) —
don't redistribute the output beyond local testing.
"""

import argparse
import random
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_PATH = REPO_ROOT / "data" / "raw" / "kaggle_resume_dataset" / "archive.zip"
OUT_DIR = REPO_ROOT / "data" / "picked"

# Kaggle categories (data/data/<CATEGORY>/*.pdf) grouped by expected fit against
# the AI & Data Solution Intern role. The dataset has no dedicated "Data Science"
# category, so INFORMATION-TECHNOLOGY/ENGINEERING stand in as the closest fit.
POOLS = {
    "strong": ["INFORMATION-TECHNOLOGY", "ENGINEERING"],
    "medium": ["CONSULTANT", "BUSINESS-DEVELOPMENT", "FINANCE", "BANKING", "DIGITAL-MEDIA"],
    "weak": [
        "SALES", "TEACHER", "FITNESS", "APPAREL", "CHEF", "AVIATION", "ARTS",
        "AGRICULTURE", "CONSTRUCTION", "ADVOCATE", "AUTOMOBILE", "HEALTHCARE",
        "HR", "PUBLIC-RELATIONS", "BPO", "ACCOUNTANT",
    ],
}


def list_pdfs_by_category(zf: zipfile.ZipFile) -> dict[str, list[str]]:
    by_category: dict[str, list[str]] = {}
    for name in zf.namelist():
        # e.g. "data/data/INFORMATION-TECHNOLOGY/12345678.pdf"
        parts = name.split("/")
        if len(parts) == 4 and parts[0] == "data" and parts[1] == "data" and name.endswith(".pdf"):
            by_category.setdefault(parts[2], []).append(name)
    return by_category


def pick_pool(by_category: dict[str, list[str]], categories: list[str], n: int, rng: random.Random) -> list[str]:
    candidates = [name for cat in categories for name in by_category.get(cat, [])]
    if not candidates:
        return []
    return rng.sample(candidates, k=min(n, len(candidates)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strong", type=int, default=1, help="Resumes to pick from the strong-match pool")
    parser.add_argument("--medium", type=int, default=1, help="Resumes to pick from the medium-match pool")
    parser.add_argument("--weak", type=int, default=1, help="Resumes to pick from the weak-match pool")
    parser.add_argument("--seed", type=int, default=None, help="Random seed, for reproducible picks")
    parser.add_argument("--archive", type=Path, default=ARCHIVE_PATH, help="Path to the Kaggle archive.zip")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="Output directory")
    args = parser.parse_args()

    if not args.archive.exists():
        raise SystemExit(
            f"Archive not found at {args.archive}.\n"
            "Download it from https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset "
            f"and save it to {args.archive}"
        )

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.archive) as zf:
        by_category = list_pdfs_by_category(zf)
        print(f"Found {sum(len(v) for v in by_category.values())} resumes across {len(by_category)} categories")

        for pool, categories in POOLS.items():
            count = getattr(args, pool)
            picked = pick_pool(by_category, categories, count, rng)
            if not picked:
                print(f"[{pool}] no matching resumes found, skipping")
                continue
            for name in picked:
                category = name.split("/")[2]
                resume_id = Path(name).stem
                out_name = f"{pool}_{category.lower()}_{resume_id}.pdf"
                out_path = args.out / out_name
                out_path.write_bytes(zf.read(name))
                print(f"[{pool}] {category} -> {out_path.relative_to(REPO_ROOT)}")

    print(
        f"\nDone. These are real resumes with no additional scrubbing applied beyond "
        f"what the Kaggle dataset already did. Don't commit or redistribute "
        f"{args.out.relative_to(REPO_ROOT)} (data/ is gitignored)."
    )


if __name__ == "__main__":
    main()
