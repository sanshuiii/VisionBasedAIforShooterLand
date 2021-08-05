import pickle
with open("data/episode_data.pkl","rb") as f:
    data = pickle.load(f)
for x in data:
    print(x[1])
    print(x[2])
    print(x[-1])
    print("--------")

print(len(data))