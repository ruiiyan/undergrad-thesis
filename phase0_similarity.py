# Phase 0 - Similarity Score
# Using 100, 10/10 scored reflections, we define a centroid made up of these good reflections
# for us to compare future reflections against.
# We do this by embedding the new reflections, and defining their cosine similarity score.

import pandas as pd
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1.load the benchmark reflection
file_path = "utils/star_reflections_strict_100.json" 
with open(file_path, "r", encoding="utf-8") as f:
    reflections = json.load(f)

# 2. combine the reflections back (doing combined for now, but could test out if doing it per
#.   part of the framework, and averaging those out makes a diff or not) 
df_good = pd.DataFrame(reflections)
df_good["combined"] = (
    df_good["situation"].astype(str) + " " +
    df_good["task_action"].astype(str) + " " +
    df_good["result"].astype(str)
)

#3. We then compute the embeddings for each of the benchmark reflections
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df_good["combined"].tolist(), normalize_embeddings=True, show_progress_bar=True)
df_good["embedding"] = embeddings.tolist()

#4. we then compute the benchmark centroid, for us to compare against later
#.  currently the centroid is just an average of all the embeddings.
embedding_matrix = np.vstack(df_good["embedding"].values)
centroid = np.mean(embedding_matrix, axis=0)

# 5. For each benchmark reflection, we then compare it's similarity to the centroid
#    we do this to validate that our group of 100 reflections, are tightly grouped or not. 
#    This also allows us to define the multiple similarity cutoffs for us to 'score' new reflections

df_good["similarity_to_centroid"] = cosine_similarity(
    df_good["embedding"].tolist(), [centroid]
).flatten()

print(df_good["similarity_to_centroid"].describe())

# p10, p25, p50 = np.percentile(df_good["similarity_to_centroid"], [10, 25, 50])
# print(f"Similarity thresholds:\n P10={p10:.3f}  P25={p25:.3f}  P50={p50:.3f}")

# 6. helper function that takes a new reflection, and computes its similarity score and qualitative label

def score_reflection(reflection_obj):
    """
    Takes a dict like:
    {
      'situation': '...',
      'task_action': '...',
      'result': '...'
    }
    Returns similarity score and qualitative label.
    """
    text = f"{reflection_obj['situation']} {reflection_obj['task_action']} {reflection_obj['result']}"
    emb = model.encode([text])[0]
    sim = cosine_similarity([emb], [centroid])[0][0]
    label = (
        "Excellent" if sim >= 0.70 else
        "Good" if sim >= 0.60 else
        "Needs improvement" if sim >= 0.50 else
        "Poor"
    )
    return {"similarity_to_centroid": float(sim), "quality_label": label}

# test it out for 3 example reflections

five_reflection = {
  "situation": "We need group discussion to plan and build a model.",
  "task_action": "Divide a task into manageable pieces and assign one to each group member. when a member encounters a problem with their task, they can raise questions, and we then discuss the problem with each person contributing their own arguments in support of or opposition to the viewpoint.",
  "result": "We can get a solution that is relatively acceptable to everyone, and that results in relatively satisfactory results.",
}

eight_reflection_short = {
  "situation": "In week 9 during our breakout room, our tutor, Maya introduced a new member, Asma to our team. It was quite exciting to get to know a new person from our tutor group but that meant that we had to make adjustments to our plan and design of the RGM project.",
  "task_action": "As a group, we introduced ourselves and tried to make a comfortable space for Asma to contribute in. We were urged to start brainstorming ideas for new events to satisfy the criteria where each students should have two action events. We had to go back to our timeline with all the drawings and diagrams of our RGM project and find out where we could add new events without changing any of the previous ones. With maximum participation from every group member, we made a successful judgment upon the final decision and were able to create two more events that fit with our timeline of events to allocate to Asma. To ensure these events were reliable, we used critical thinking before executing our new events. We made diagrams and labels for each event including, issues we might face and solutions that will help. For example, one new event involved a cup getting tipped over and a pin would fall out and pop a balloon. We analysed an issue where the pin might not pop the balloon, which was solved by sticking two pins together to make it more sharp and blowing up the balloon more so that the surface is thinner and easy to pop. A few days after this workshop, we tested the event and it was successful only 14 out of 20 times. We continued to make further adjustments in order to achieve a 100% reliability. We also applied the same steps with all of our events to maximise our reliability tests.",
  "result": "We were able to achieve '100%' reliability for most of our events through critical thinking, where we identify issues, analyse and find possible solutions to resolve it. This process allowed us to make judgements whether an event is worth executing and if it will actually work, which in this case worked and is now a part of our final RGM project.",
}

eight_reflection_long = {
  "situation": "As part of ENGG1050, the students have been tasked with building a functioning Rube-Goldberg Machine (RGM) to perform a simple task. The machine is broken up into a number of sections, depending on the number of people in the group, so that each member is tasked with designing and constructing part of the contraption. There are a number of aims associated with this project, one of which is that the RGM must be reliable.",
  "task_action": "To ensure the RGM that I am a part of is reliable, my group has set a minimum standard that each component of each section must perform as required 9/10 times. If a component can not be modified to meet this degree of reliability, it must be removed from the machine and replaced with another, more reliable, action event. This guarantees that the machine as a whole is at least '90%' reliable. o meet this standard of reliability, each member of the group conducted testing of every component of their section 3 times. Each test consisted of 10 runs of the relevant event. The first test was conducted before any optimisation changes had been made. The second was conducted after making one adjustment to the component, and the third was conducted after all possible optimisation improvements were implemented. The number of times the action event was successful in each test was recorded and tabulated to make a graph of the change in reliability.",
  "result": "As a result of this rigorous testing approach, data was produced that greatly influenced the decision to keep or replace components in each section of the RGM. In each successive test for every group member, the reliability increased. In all but two components, the optimisation changes made allowed the event to function as intended at least 9/10 times. The data for the two components that did not reach 90% reliability drove us to replace these events. Further testing was conducted on the new actions to determine they performed at least 9/10 times. Overall, the RGM is now proven to be '90%' reliable."
}

ten_reflection = {
  "situation": "I decided to design, manufacture and program a macropad for my personal computer.",
  "task_action": "During the design and programming process I needed to learn two brand new applications both of which took quiet a while to get a hand of and required perseverance. In order to complete the project, it required investing hundreds of hours of my spare time in order to complete and deliver the project by the due date, exhibiting my urgency and willingness to deliver. Some resourcefulness was needed during early prototyping in the manufacturing process using home made methods to create makeshift rigid PCBs and further flexibility with receiving final designs from the manufacturer and ordering components to be soldered from suppliers.",
  "result": "Due to my perseverance, urgency and will to deliver, and resourcefulness/flexibility. I completed the project in time to work out unexpected issues and deliver it in near perfect condition to what I had envisioned.",
}


# Interesting. a eight reflection, got a higher score, compared to a 10 reflection
# Tried it with a different size reflection and instead, got an even worse score xD
# Possible reason: The one that scored higher, is SEMANTICALLY more similar to the benchmark. 
# 
print(score_reflection(eight_reflection_short))

# # First, precompute each benchmark reflection’s similarity to the centroid
# df_good["similarity_to_centroid"] = [
#     float(cosine_similarity([emb], [centroid])[0][0])
#     for emb in df_good["embedding"]
# ]

# def get_nearest_reflections(test_reflection, df_good, model, top_n=5):
#     """Return top-N most semantically similar benchmark reflections, including their centroid similarity."""
#     text = f"{test_reflection['situation']} {test_reflection['task_action']} {test_reflection['result']}"
#     emb = model.encode([text], normalize_embeddings=True)[0]
#     emb_matrix = np.vstack(df_good["embedding"].values)
#     sims = cosine_similarity([emb], emb_matrix).flatten()

#     top_idx = np.argsort(sims)[::-1][:top_n]
#     results = []
#     for rank, i in enumerate(top_idx, start=1):
#         row = df_good.iloc[i]
#         results.append({
#             "Rank": rank,
#             "Similarity_to_Test": float(sims[i]),
#             "Benchmark_ID": i,
#             "Benchmark_Text": row["combined"],
#             "Benchmark_Similarity_to_Centroid": float(row["similarity_to_centroid"]),
#             "Benchmark_Situation": row.get("situation", ""),
#             "Benchmark_Task_Action": row.get("task_action", ""),
#             "Benchmark_Result": row.get("result", "")
#         })
#     return pd.DataFrame(results)

# # --- Generate tables ---
# nearest_short = get_nearest_reflections(eight_reflection_short, df_good, model, top_n=10)
# nearest_long  = get_nearest_reflections(eight_reflection_long, df_good, model, top_n=10)

# # --- Add metadata for context ---
# meta_info = pd.DataFrame([
#     {
#         "Reflection_Type": "Short",
#         "Similarity_to_Centroid": float(cosine_similarity(
#             model.encode([f"{eight_reflection_short['situation']} {eight_reflection_short['task_action']} {eight_reflection_short['result']}"],
#                          normalize_embeddings=True),
#             [centroid]
#         )[0][0])
#     },
#     {
#         "Reflection_Type": "Long",
#         "Similarity_to_Centroid": float(cosine_similarity(
#             model.encode([f"{eight_reflection_long['situation']} {eight_reflection_long['task_action']} {eight_reflection_long['result']}"],
#                          normalize_embeddings=True),
#             [centroid]
#         )[0][0])
#     }
# ])

# # --- Export everything to Excel ---
# output_path = "nearest_reflections_with_centroid.xlsx"
# with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
#     meta_info.to_excel(writer, sheet_name="Summary", index=False)
#     nearest_short.to_excel(writer, sheet_name="Short_Reflection", index=False)
#     nearest_long.to_excel(writer, sheet_name="Long_Reflection", index=False)

# print(f"✅ Exported to {output_path}")