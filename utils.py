import torch
import random
import numpy as np
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)

def compute_f1(gold, predicted, logger):
    c_predict = 0
    c_correct = 0
    c_gold = 0

    for g, p in zip(gold, predicted):
        if g != 0:
            c_gold += 1
        if p != 0:
            c_predict += 1
        if g != 0 and p != 0:
            c_correct += 1

    p = c_correct / (c_predict + 1e-100) if c_predict != 0 else .0
    r = c_correct / c_gold if c_gold != 0 else .0
    f = 2 * p * r / (p + r + 1e-100) if (r + p) > 1e-4 else .0

    logger.info("correct {}, predicted {}, golden {}".format(c_correct, c_predict, c_gold))

    return p, r, f




