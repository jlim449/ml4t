def quicksort(arr):

    if len(arr) < 2:
        return arr

    else:
        pivot = arr[0]
        smaller, larger = [], []

        for x in arr[1:]:
            if x < pivot:
                smaller.append(x)
            else:
                larger.append(x)
        return quicksort(smaller) + [pivot] + quicksort(larger)


if __name__ == '__main__':
    arr = [3, 6, 8, 10, 1, 2, 5]
    print("Unsorted array:", arr)

    sorted_arr = quicksort(arr)
    print("Sorted array:", sorted_arr)