

count = 0
f = open("all_configs_rewards.txt", "r")

for line in f:
    state = line.split(",")[0]
    if state.count("1") == 4:
        print(state)
        count += 1
    
print(count)
