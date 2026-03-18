import numpy as np

True and True  # True
True and False  # False
False and True  # False
False and False  # False

[True] and [True]  # [True]
[True, True] and [True, False]  # [True, False]

print(np.array([True, True]))  # [ True  True]

np.array([True, False]) and np.array(
    [True, True]
)  # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
print(np.array([True, False]) & np.array([True, True]))  # [ True False]
...
