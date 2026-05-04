import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

sentence = ["I", "love", "deep", "learning"]

# Attention scores
attention_weights = torch.tensor([0.1, 0.3, 0.4, 0.2])

# Softmax normalization
attention_weights = F.softmax(attention_weights, dim=0).detach().numpy()

# Plot heatmap
sns.heatmap([attention_weights], annot=True, xticklabels=sentence, cmap="Blues")
plt.title("Attention Heatmap")
plt.show()
