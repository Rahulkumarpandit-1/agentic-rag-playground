import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    "hours": [2, 4, 5, 3, 8, 6, 1, 7, 4, 10],
    "attendance": [60, 75, 80, 65, 95, 85, 50, 90, 72, 98],
    "score": [40, 55, 65, 48, 88, 75, 35, 82, 58, 95],
    "city": [
        "Guwahati", "Delhi", "Guwahati", "Mumbai", "Delhi",
        "Guwahati", "Mumbai", "Delhi", "Guwahati", "Delhi"
    ]
}

df = pd.DataFrame(data)
print(df.corr(numeric_only=True))

# Correlation heatmap
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()


