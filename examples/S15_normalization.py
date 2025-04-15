from sklearn.preprocessing import StandardScaler

data = [[2, 0], [2, 0], [1, 1], [1, 1]]

sc = StandardScaler()

new_data = sc.fit_transform(data)
print(new_data)
