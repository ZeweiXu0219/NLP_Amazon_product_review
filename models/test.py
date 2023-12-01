def answer(interval):
    interval.sort()
    result = [interval[0]]
    for i in range(1, len(interval)):
        new_left = interval[i][0]
        new_right = interval[i][1]
        last_end = result[-1][-1]
        if last_end <= new_right and last_end >= new_left:
            result[-1][-1] = new_right
        else:
            result.append(interval[i][:])
    return result

if __name__ == '__main__':
    print(answer([[1,3],[2,6],[8,10],[15,18]]))
    print(answer([[2,6],[1,3],[8,10],[15,18]]))