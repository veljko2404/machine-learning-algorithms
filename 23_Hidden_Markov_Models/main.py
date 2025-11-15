import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import CategoricalHMM

# 0 = coffee in café
# 1 = umbrella
# 2 = walking outside

observations = np.array([[0,1,1,2,2,1,0,0,2,1]]).T

model = CategoricalHMM(n_components=2, random_state=42)
model.startprob_ = np.array([0.6, 0.4])
model.transmat_ = np.array([
    [0.7, 0.3], # Sun -> Sun (70%), Sun -> Rain (30%)
    [0.4, 0.6] # Rain -> Sun (40%), Rain -> Rain (60%)
])

model.emissionprob_ = np.array([
    [0.6, 0.1, 0.3], # Sun emits: coffee 60%, umbrella 10%, walk 30%
    [0.2, 0.7, 0.1] # Rain emits: coffee 20%, umbrella 70%, walk 10%
])

logprob, hidden_states = model.decode(observations, algorithm="viterbi")
print("Most likely hidden states:", hidden_states)

plt.figure(figsize=(10,2))
plt.plot(hidden_states, marker='o')
plt.yticks([0,1], ["Sunce", "Kiša"])
plt.title("Viterbi – Most Likely Hidden State Sequence")
plt.grid(True)
plt.show()
