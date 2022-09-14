import os


print("start testing!")
print(os.path.exists("knn.py"))
assert os.path.exists("knn.py")
assert 5 == 6
print("done!")
