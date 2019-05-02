def softmax_without_overflow(a):
    exp_a = np.exp(a)
    exp_a_sum = np.sum(exp_a)
    y = exp_a / exp_a_sum

    return y

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    exp_a_sum = np.sum(exp_a)
    y = exp_a / exp_a_sum

    return y
