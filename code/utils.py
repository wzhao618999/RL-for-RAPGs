def str_key(*args):
    new_arg = []
    for arg in args:
        if type(arg) in [tuple, list]:
            new_arg += [str(i) for i in arg]
        else:
            new_arg.append(str(arg))
    return "_".join(new_arg)


def set_dict(target_dict, value, *args):
    target_dict[str_key(*args)] = value


def set_prob(P, s, a, s1, p):
    set_dict(P, p, s, a, s1)


def get_prob(P, s, a, s1):
    return P.get(str_key(s, a, s1), 0)


def set_reward(R, s, a, s1, r):
    set_dict(R, r, s, a, s1)


def get_reward(R, s, a, s1):
    return R.get(str_key(s, a, s1), 0)


def display_dict(target_dict):
    for key in target_dict.keys():
        print("{}:{:.2f}".format(key, target_dict[key]))
    print("")


def set_pi(Pi, s, a, p=0.5):
    set_dict(Pi, p, s, a)


def get_pi(Pi, s, a):
    return Pi.get(str_key(s, a), 0)
