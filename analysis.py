
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load dataset
DATA_PATH = Path("data/Telco-Customer-Churn.csv")
df = pd.read_csv(DATA_PATH)

# Clean TotalCharges
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].astype(str).str.strip(),
    errors="coerce"
)

# Basic dataset report
print("DATASET OVERVIEW")
print("-" * 40)
print("Rows and columns:", df.shape)
print("Duplicate rows:", df.duplicated().sum())
print("Duplicate customer IDs:", df["customerID"].duplicated().sum())
print("Missing values:\n", df.isnull().sum().sort_values(ascending=False).head(10))

# Churn distribution
print("\nCHURN DISTRIBUTION")
print(df["Churn"].value_counts())
print((df["Churn"].value_counts(normalize=True) * 100).round(2))

# Churn by contract
print("\nCHURN RATE BY CONTRACT (%)")
contract_churn = (
    df.assign(ChurnFlag=df["Churn"].eq("Yes").astype(int))
      .groupby("Contract")["ChurnFlag"]
      .mean()
      .mul(100)
      .sort_values(ascending=False)
)
print(contract_churn.round(2))

# Save churn distribution chart
Path("images").mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")

ax = sns.countplot(data=df, x="Churn", order=["No", "Yes"])
total = len(df)

for bar in ax.patches:
    count = int(bar.get_height())
    percentage = count / total * 100
    ax.annotate(
        f"{count:,}\n({percentage:.2f}%)",
        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
        ha="center",
        va="bottom",
        xytext=(0, 5),
        textcoords="offset points"
    )

plt.title("Customer Churn Distribution")
plt.xlabel("Customer Churn")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("images/churn_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

# Churn rate by contract type
df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)

contract_churn = (
    df.groupby("Contract")["ChurnFlag"]
      .mean()
      .mul(100)
      .sort_values(ascending=False)
)

print("\nChurn Rate by Contract Type (%)")
print(contract_churn.round(2))

# Create and save the chart
plt.figure(figsize=(9, 5))

ax = sns.barplot(
    x=contract_churn.index,
    y=contract_churn.values
)

for bar in ax.patches:
    ax.annotate(
        f"{bar.get_height():.1f}%",
        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
        ha="center",
        va="bottom",
        xytext=(0, 5),
        textcoords="offset points"
    )

plt.title("Customer Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate (%)")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(
    "images/churn_by_contract.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
