import matplotlib.pyplot as plt
import numpy as np

# Data
rounds = np.arange(1, 51)  # 50 rounds
azure_stt_times = np.random.normal(70, 5, size=50)  # simulate ~70s with slight variation
motion_times = np.random.normal(215, 10, size=50)   # simulate ~215s
azure_openai_times = np.random.normal(0.45, 0.05, size=50)
qwen_times = np.random.normal(4.5, 0.3, size=50)

# Average values
avg_times = {
    "Azure Speech-to-Text": np.mean(azure_stt_times),
    "Motion Generation & Normalizer": np.mean(motion_times),
    "Azure OpenAI": np.mean(azure_openai_times),
    "Local Qwen2.5": np.mean(qwen_times)
}

# 1. Bar chart (average times)
plt.figure(figsize=(8,6))
plt.bar(avg_times.keys(), avg_times.values(), color=['blue','orange','green','red'])
plt.ylabel("Time (seconds)")
plt.title("Average Processing Time per Component")
plt.show()

# 2. Line chart (time per round for STT and Motion)
plt.figure(figsize=(10,6))
plt.plot(rounds, azure_stt_times, label="Azure STT (~70s)")
plt.plot(rounds, motion_times, label="Motion Gen & Normalizer (~215s)")
plt.xlabel("Round")
plt.ylabel("Time (seconds)")
plt.title("Processing Time Across 50 Rounds")
plt.legend()
plt.show()

# 3. Stacked bar chart (total pipeline per round)
pipeline_total = azure_stt_times + motion_times + azure_openai_times + qwen_times
plt.figure(figsize=(10,6))
plt.bar(rounds, azure_stt_times, label="Azure STT")
plt.bar(rounds, motion_times, bottom=azure_stt_times, label="Motion Gen & Normalizer")
plt.bar(rounds, azure_openai_times, bottom=azure_stt_times+motion_times, label="Azure OpenAI")
plt.bar(rounds, qwen_times, bottom=azure_stt_times+motion_times+azure_openai_times, label="Qwen2.5 Local")
plt.xlabel("Round")
plt.ylabel("Time (seconds)")
plt.title("Stacked Total Pipeline Time per Round")
plt.legend()
plt.show()

# 4. Pie chart (percentage contribution)
plt.figure(figsize=(7,7))
plt.pie(avg_times.values(), labels=avg_times.keys(), autopct='%1.1f%%', startangle=140)
plt.title("Average Contribution to Total Processing Time")
plt.show()

# 5. Total Average Time
total_avg = sum(avg_times.values())
print(f"🔹 Total Average Pipeline Time: {total_avg:.2f} seconds")
