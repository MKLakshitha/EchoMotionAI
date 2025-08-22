import asyncio
import time
import httpx
import pandas as pd
import matplotlib.pyplot as plt

URL = "http://127.0.0.1:8005/detect"
HEADERS = {"accept": "application/json", "Content-Type": "application/json"}
PAYLOAD = {
    "message": "walk to the bed",
    "room_objects": ["bed", "door", "chair"]
}

CONCURRENT_REQUESTS = 50

async def send_request(client, i):
    start = time.perf_counter()
    try:
        response = await client.post(URL, headers=HEADERS, json=PAYLOAD, timeout=60.0)
        latency = time.perf_counter() - start
        return {"id": i, "status": response.status_code, "latency": latency}
    except Exception as e:
        return {"id": i, "status": "error", "latency": None, "error": str(e)}

async def run_load_test():
    async with httpx.AsyncClient() as client:
        tasks = [send_request(client, i) for i in range(CONCURRENT_REQUESTS)]
        results = await asyncio.gather(*tasks)
    return results

if __name__ == "__main__":
    results = asyncio.run(run_load_test())
    df = pd.DataFrame(results)
    
    # Save results
    df.to_csv("stress_test_results.csv", index=False)
    
    # Summary
    print("\n--- Summary ---")
    print(df["latency"].describe())
    print(f"Errors: {df[df['status'] != 200].shape[0]} / {CONCURRENT_REQUESTS}")
    
    # Histogram of latencies
    plt.figure(figsize=(8,6))
    df["latency"].dropna().hist(bins=10, color="skyblue", edgecolor="black")
    plt.xlabel("Latency (seconds)")
    plt.ylabel("Frequency")
    plt.title("Latency Distribution for 50 Concurrent Requests")
    plt.show()
    
    # Line chart
    plt.figure(figsize=(10,6))
    plt.plot(df["id"], df["latency"], marker="o", linestyle="--")
    plt.xlabel("Request ID")
    plt.ylabel("Latency (seconds)")
    plt.title("Latency per Request")
    plt.show()
