# pylint: skip-file
# Shows color palette usage in Seaborn
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_palette("colorblind", 10)

# Pie plot with 10 random values
data = [0.1] * 10
plt.pie(data, labels=[f"Label {i}" for i in range(10)])

#plt.legend()
plt.show()