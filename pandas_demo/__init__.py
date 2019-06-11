import pandas as pd

food_info = pd.read_csv("food_info.csv")

print(food_info.tail(3))
print(food_info.shape)
print(food_info.dtypes)

# 取数 取第0条数据
print(food_info.loc[0])
print(food_info.loc[3:6])

# 注意用法
print(food_info.loc[[2, 7]])

print(food_info["Vit_D_IU"])

# 注意用法
print(food_info[["Vit_D_IU", "Vit_K_(mcg)"]])

print(food_info.columns.tolist())

print(food_info['Water_(g)'] / 1000)

print(food_info.shape)
food_info["Water_(mg)"] = food_info['Water_(g)'] / 1000
print(food_info.shape)

food_info.sort_values("Vit_B6_(mg)", inplace=True)
print(food_info["Vit_B6_(mg)"])

taitannike = pd.read_csv("titanic_train.csv")
print(taitannike.head())

print(taitannike[0:6])

print(pd.isnull(taitannike["Age"]))

print(sum(taitannike["Age"]))
print(len(taitannike["Age"]))
print(sum(taitannike["Age"]) / len(taitannike["Age"]))

print(taitannike["Age"].mean())
