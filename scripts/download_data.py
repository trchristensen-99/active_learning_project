#!/usr/bin/env python3
import argparse
import csv
import hashlib
import os
import pathlib
import sys
import urllib.request


def compute_sha256(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_md5(path: str) -> str:
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def safe_mkdir(path: str) -> None:
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def download(url: str, dest_path: str) -> None:
    tmp_path = dest_path + ".part"
    urllib.request.urlretrieve(url, tmp_path)
    os.replace(tmp_path, dest_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    safe_mkdir(args.out)

    with open(args.manifest, newline="") as f:
        reader = csv.DictReader(filter(lambda row: not row.strip().startswith("#"), f))
        for row in reader:
            url = row["url"].strip()
            checksum_md5 = (row.get("md5") or "").strip()
            checksum_sha256 = (row.get("sha256") or "").strip()
            filename = row.get("filename") or os.path.basename(url)
            dest = os.path.join(args.out, filename)

            print(f"Downloading {url} -> {dest}")
            download(url, dest)

            # Prefer md5 if provided, else use sha256 if provided
            if checksum_md5:
                actual = compute_md5(dest)
                if actual != checksum_md5:
                    raise SystemExit(
                        f"MD5 mismatch for {filename}: {actual} != {checksum_md5}"
                    )
            elif checksum_sha256:
                actual = compute_sha256(dest)
                if actual != checksum_sha256:
                    raise SystemExit(
                        f"SHA256 mismatch for {filename}: {actual} != {checksum_sha256}"
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


