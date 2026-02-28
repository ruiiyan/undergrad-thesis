import pandas as pd
import re
import json

# 1️⃣ Load Excel
file_path = "../ENGG_REFLECTIONS.xlsx"   # adjust if in different directory
df = pd.read_excel(file_path)

# Access the 'reflection_text' column directly
reflections = df["reflection_text"].dropna().astype(str)

# 2️⃣ Strictly match only fully formatted reflections
strict_pattern = re.compile(
    r"SITUATION:\s+.*?TASK/ACTION:\s+.*?RESULT:\s+.*",
    flags=re.DOTALL
)
strict_reflections = reflections[reflections.str.match(strict_pattern)]

# 3️⃣ Take 100 (or fewer if not available)
df_subset = strict_reflections.sample(
    n=min(100, len(strict_reflections)), random_state=42
).reset_index(drop=True)

print(f"✅ Found {len(df_subset)} reflections that strictly follow SITUATION → TASK/ACTION → RESULT format")

# 4️⃣ Parsing function
def parse_star_strict(text: str):
    if not isinstance(text, str) or not text.strip():
        return {"situation": "", "task_action": "", "result": ""}
    text = re.sub(r"\s+", " ", text.strip())
    try:
        situation_part = text.split("TASK/ACTION:")[0].replace("SITUATION:", "").strip()
        taskaction_part = text.split("TASK/ACTION:")[1].split("RESULT:")[0].strip()
        result_part = text.split("RESULT:")[1].strip()
    except (IndexError, ValueError):
        situation_part, taskaction_part, result_part = "", "", ""
    return {
        "situation": situation_part,
        "task_action": taskaction_part,
        "result": result_part
    }

# 5️⃣ Parse and save
parsed = [parse_star_strict(txt) for txt in df_subset]
for i, p in enumerate(parsed):
    p["id"] = f"Ref_{i+1:04d}"

with open("star_reflections_strict_100.json", "w", encoding="utf-8") as f:
    json.dump(parsed, f, indent=2, ensure_ascii=False)

print("🎉 Done — saved to star_reflections_strict_100.json")
