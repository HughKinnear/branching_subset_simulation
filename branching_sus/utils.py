import numpy as np
import matplotlib.pyplot as plt



def indicators_to_classifier(indis):
    def classifier(point):
        for i, indi in enumerate(indis):
            if indi(point) == 1:
                return i + 1
        return 0
    return classifier


def seeded_choice_no_replace(options, size, random_state=None):
    if random_state is None:
        indx = np.random.choice(len(options), size=size,replace=False)
    else:
        indx = random_state.choice(len(options), size=size,replace=False)
    return [options[i] for i in indx]


def seeded_choice(options, size, random_state=None):
    if random_state is None:
        indx = np.random.choice(len(options), size=size)
    else:
        indx = random_state.choice(len(options), size=size)
    return [options[i] for i in indx]


def contourf_plot(x_range, y_range, step, function, levels=None):
    x = np.arange(x_range[0], x_range[1], step)
    y = np.arange(y_range[0], y_range[1], step)
    xx, yy = np.meshgrid(x, y)
    flat = np.vstack([xx.ravel(), yy.ravel()])
    z = np.array([function(point) for point in flat.T]).reshape(xx.shape)
    plt.contourf(xx, yy, z, levels=levels)

def contour_plot(x_range, y_range, step, function, levels=None):
    x = np.arange(x_range[0], x_range[1], step)
    y = np.arange(y_range[0], y_range[1], step)
    xx, yy = np.meshgrid(x, y)
    flat = np.vstack([xx.ravel(), yy.ravel()])
    z = np.array([function(point) for point in flat.T]).reshape(xx.shape)
    plt.contour(xx, yy, z, levels=levels,colors='black')



def plotting_range(samples):
    all_array = np.array(samples).T
    marg = (all_array[0].max() - all_array[0].min()) * plt.margins()[1]
    x_range = (all_array[0].min() - marg, all_array[0].max() + marg)
    marg = (all_array[1].max() - all_array[1].min()) * plt.margins()[1]
    y_range = (all_array[1].min() - marg, all_array[1].max() + marg)
    return x_range, y_range



