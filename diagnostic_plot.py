import pandas as pd
import matplotlib.pyplot as plt

from strategy import RankingStrategy


# ----------------------------
# Load data
# ----------------------------

strategy = RankingStrategy()

df = strategy.load_data("multi_stock_dataset.csv")

# Normalize features
df = strategy.normalize_features(df)


# ----------------------------
# Train/test split
# ----------------------------

split_date = df["Datetime"].quantile(0.8)

train = df[df["Datetime"] <= split_date]
test = df[df["Datetime"] > split_date]


# ----------------------------
# Train model
# ----------------------------

strategy.train(train)


# ----------------------------
# Predict returns
# ----------------------------

test = strategy.predict(test)


# ----------------------------
# Create deciles
# ----------------------------

test["decile"] = pd.qcut(
    test["prediction"],
    5,
    labels=False,
    duplicates="drop"
)


# ----------------------------
# Compute realized returns
# ----------------------------

decile_returns = test.groupby("decile")["Target"].mean()


print("\nAverage Return by Prediction Decile")
print(decile_returns)


# ----------------------------
# Plot
# ----------------------------

plt.figure(figsize=(8,5))

decile_returns.plot(kind="bar")

plt.title("Prediction Decile vs Realized Return")
plt.xlabel("Prediction Decile (0 = worst prediction)")
plt.ylabel("Average Next-Day Return")

plt.grid(True)

plt.show()