import main
import json
from pathlib import Path

evals = Path("evals")
for file in Path("convs").iterdir():
    if not file.is_file(): continue
    eval_file = evals / file.name 
    if eval_file.is_file(): continue

    print("judging conv " + file.name)
    with file.open() as f:
        conv = json.load(f)

    with eval_file.open("w") as f:
        json.dump(main.judge(conv), f, indent=2)