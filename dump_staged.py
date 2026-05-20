import subprocess
from pathlib import Path

diff = subprocess.run(
    ["git", "diff", "--cached"],
    capture_output=True,
    text=True,
    check=True,
)

out = Path(__file__).parent / "staged_changes.txt"
out.write_text(diff.stdout, encoding="utf-8")
print(f"Written {len(diff.stdout)} chars → {out}")
