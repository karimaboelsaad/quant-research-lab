def generate_walk_forward_windows(number_of_rows, initial_train_size, validation_size, test_size):
    if number_of_rows<=0:
        raise ValueError(
            "number_of_rows must be positive."
        )

    if initial_train_size<=0:
        raise ValueError(
            "initial_train_size must be positive."
        )

    if validation_size<=0:
        raise ValueError(
            "validation_size must be positive."
        )

    if test_size<=0:
        raise ValueError(
            "test_size must be positive."
        )

    minimum_required=initial_train_size+validation_size+test_size

    if minimum_required>number_of_rows:
        raise ValueError(
            "Not enough rows for one complete window."
        )

    results=[]

    number_of_windows=(number_of_rows-initial_train_size-validation_size)//test_size

    for i in range(number_of_windows):
        train_end=initial_train_size+test_size*i
        validation_end=train_end+validation_size
        test_end=validation_end+test_size

        results.append({
            "train_end":train_end,
            "validation_end":validation_end,
            "test_end":test_end
        })

    return results


def calculate_common_training_start(lookbacks,train_end):
    if not lookbacks:
        raise ValueError(
            "Lookbacks cannot be empty."
        )

    if any(not isinstance(lookback,int) or isinstance(lookback,bool) or lookback<=0 for lookback in lookbacks):
        raise ValueError(
            "Lookbacks must contain positive integers."
        )

    common_start=max(lookbacks)

    if common_start>=train_end:
        raise ValueError(
            "Training period must extend beyond the largest lookback."
        )

    return common_start


def calculate_unused_walk_forward_rows(number_of_rows,initial_train_size,validation_size,test_size):
    windows=generate_walk_forward_windows(number_of_rows,initial_train_size,validation_size,test_size)

    return number_of_rows-windows[-1]["test_end"]
