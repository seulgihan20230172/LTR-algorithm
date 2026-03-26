import subprocess

scripts = [
    "LambdaMART.py",
    "lambdaRank.py",
    "RankNet.py",
    "ListNet.py",
    "ListMLE.py",
    "XGBoost_Rank.py",
    "BM25.py"
]

for script in scripts:
    print(f"\n===== Running {script} =====\n")
    result = subprocess.run(["python", script])

    if result.returncode != 0:
        print(f"❌ Error in {script}")
        break
