from sortedcontainers import SortedList
from bisect import bisect_right

nums = [1,3,3,2,2,1000000000,9989,10234,9989]
print(len(nums))
all_nums = SortedList(range(max(nums)+len(nums)))

for i in range(len(nums)):
    if nums[i] not in all_nums:
        nums[i] = all_nums[bisect_right(all_nums, nums[i])]
    all_nums.remove(nums[i])

print(nums)
