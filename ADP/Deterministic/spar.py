# keep monotonicity

def spar_update(bars, update_id):

    # from test import Nt
    Nt = 5

    basic_slope = bars[update_id-1]
    stop_id = update_id

    # Left search
    if (update_id-1) == 1:
        if bars[update_id-2] > basic_slope:
            sum_1 = sum(bars[update_id-2:update_id])
            avg_1 = sum_1 /2
            for i in range(update_id-2, update_id):
                bars[i] = avg_1
    else:
        for stop_id in range(update_id - 2, -1, -1):
            if bars[stop_id] <= basic_slope:
                break
            if stop_id == 0:
                stop_id = stop_id - 1

        sum_slope = sum(bars[stop_id + 1:update_id])
        avg1 = sum_slope / (update_id - stop_id - 1)
        for i in range(stop_id + 1, update_id):
            bars[i] = avg1


    # Right search

    right_id = update_id
    if update_id == (Nt-1):
        if bars[update_id] < basic_slope:
            sum_2 = sum(bars[update_id-1:update_id+1])
            avg_2 = sum_2 /2
            for i in range(update_id-1, update_id+1):
                bars[i] = avg_2
    else:
        for right_id in range(update_id,Nt):
            if bars[right_id] >= basic_slope:
                break
            if right_id == (Nt-1):
                right_id = right_id + 1

        sum_slope2 = sum(bars[update_id-1:right_id])
        avg2 = sum_slope2/(right_id-update_id+1)
        for j in range(update_id-1,right_id):
            bars[j] = avg2


    return bars