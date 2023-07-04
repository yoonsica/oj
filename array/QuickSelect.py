# 第k个元素=k - 1个位置上的元素
def quick_Select(arr, s, e, k):
    l, r = s, e
    pivot = arr[s]
    while l < r:
        while l < r and arr[r] >= pivot:
            r -= 1
        if l < r:
            arr[l] = arr[r]
            l += 1
        while l < r and arr[l] <= pivot:
            l += 1
        if l < r:
            arr[r] = arr[l]
            r -= 1
    arr[l] = pivot
    if l < k - 1:
        quick_Select(arr, l + 1, e, k)
    if l > k - 1:
        quick_Select(arr, s, l - 1, k)
