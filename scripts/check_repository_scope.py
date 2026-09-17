"""Reject research data/results in the Git index; allow synthetic test fixtures."""
from pathlib import PurePosixPath
import subprocess
import sys


def forbidden(name):
    path = PurePosixPath(name)
    if name.startswith("tests/fixtures/"):
        return False
    if any(part in {"input", "output", "outputs"} for part in path.parts):
        return True
    if name.startswith("data/") and path.name != "README.md":
        return True
    if name.startswith("manifests/") and path.name != "README.md":
        return not name.endswith(".example.csv")
    return path.suffix.lower() in {
        ".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".feather",
        ".set", ".fdt", ".edf", ".bdf", ".mat", ".h5", ".hdf5", ".pt", ".pth",
    }


def main():
    names = subprocess.check_output(["git", "ls-files", "-z"]).decode("utf-8").split("\0")
    rejected = [name for name in names if name and forbidden(name)]
    if rejected:
        print("Research data/results must stay local:\n" + "\n".join(rejected))
        return 1
    print("Repository scope check passed (path policy; not a content/privacy audit).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
