import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy import special
from tqdm import tqdm
rng = np.random.default_rng(11)



DIRNAME = os.path.abspath("")
PLOTS_PATH = os.path.join(DIRNAME, "Plots")
DATA_DIR = os.path.join(os.path.dirname(DIRNAME), "FINO1Data")

os.makedirs(PLOTS_PATH, exist_ok=True)
datasets = {path.replace(".npy", ""): np.load(os.path.join(DATA_DIR, path), allow_pickle=True) for path in os.listdir(DATA_DIR) if path.endswith("npy")}


def lerp(data):
    ok = ~np.isnan(data)
    xp = ok.nonzero()[0]
    fp = data[ok]
    x = np.isnan(data).nonzero()[0]
    data[~ok] = np.interp(x, xp, fp)
    return data



datasets = {key: lerp(value) for key, value in datasets.items()}                    # linear interpolation
datasets = {key: dataset[1:] - dataset[:-1] for key, dataset in datasets.items()}   # Differentiation to make data stationary
dname = "2015-2017_90m"                                                             # Active dataset
dataset = datasets[dname]




class AR():

    def __init__(self, *, p, train_per):

        self.p = p
        self.train_per = train_per


    def get_stds(self, *, data):

        indices = np.arange(0, len(data)-self.train_per)
        indices_split = np.array_split(indices, 1000)

        stds = []
        for x in indices_split:
            x = x[:, None] + np.arange(self.train_per)
            stds.append(data[x].std(axis=1))
        
        stds = np.concatenate(stds)
        stds [~(stds > 1e-08)] += 1e-08

        return stds[:-self.p]
    
    def get_phis(self, *, data, stds):
        
        # TODO Implement substract mean
        #means = np.convolve(data, np.ones(train_per), "valid") / train_per
        #data = data[train_per: -train_per] - means
    
        z = [np.array(data[lag:] * data[:-lag])[:lag-(self.p+1)] for lag in range(1, self.p+1)]

        acfs = [(np.convolve(z[lag], np.ones(self.train_per-(lag+1)), "valid") / (self.train_per-(lag+1)))[:-(lag+1)] for lag in range(self.p)]
        acfs = [acf / (stds ** 2) for acf in acfs]

        acfs.insert(0, np.ones_like(acfs[0]))
        acfs = np.stack(acfs)
        
        matrix_mask = np.array([abs(-k+j) for k in range(self.p) for j in range(self.p)])
        matrices = acfs[matrix_mask, :].reshape(self.p,self.p,-1).T
        
        det_mask = np.linalg.det(matrices)
        zero_det = np.where(det_mask < 1e-8, True, False)
        matrices[zero_det, 0] += 1e-8   
        
        phis = np.linalg.solve(matrices, acfs[1:].T)

        return phis
    

    def forecast(self, *, data, phis, stds):

        indices = np.arange(self.train_per-1, len(data))[:, None] - np.arange(self.p)
        x = data[indices]

        predictions = x[:-(self.p+1)] * phis
        predictions = predictions.sum(axis=1)

        white_noise = rng.normal(scale=(stds**2))
        predictions += white_noise

        return predictions
    

    def get_integrals(self, *, predictions, stds, gust_th):

        z = (gust_th - predictions) / ((2*stds**2)**0.5)
        prediction_integrals = 0.5 * (1 - special.erf(z))
        
        return prediction_integrals
    


def get_performances(*, data, prediction_integral, p, train_per, gust_th):

    steps = np.linspace(0, 40 ,20)[::-1]
    quantiles = 1 / np.exp(steps)        
    performances = dict.fromkeys(quantiles)

    wind_gusts = np.where(data[train_per:-p] >= gust_th, 3, 0) # G(t)
    n_gusts = (wind_gusts / 3).sum()

    for quantile in quantiles:

        wind_gusts_pred = np.where(prediction_integral >= quantile, 1, 0) 

        true_positive = np.where(wind_gusts_pred + wind_gusts == 4, 1, 0)
        false_positive = np.where(wind_gusts_pred + wind_gusts == 1, 1, 0)
        
        true_positive_rate = true_positive.sum() / n_gusts
        false_positive_rate = false_positive.sum() / (len(wind_gusts) - n_gusts)

        performances[quantile] = (true_positive_rate, false_positive_rate)

    return performances



def plot_roc(*, performances, p, train_per, gust_th):

    colors = sns.color_palette("rocket_r", len(performances))
    _ = plt.figure(1, figsize=(5,5))

    i = 0
    for threshold, (true_positive_rate, false_positive_rate) in performances.items():
        plt.plot(false_positive_rate, true_positive_rate, "o", label=f"{(threshold*100):.4f}%", color=colors[i])
        i += 1

    plt.plot(np.arange(2), linestyle="--", color='#0f0f0f30')


    plt.grid(linewidth=0.4, alpha=0.8)
    plt.xlim(0,1)
    plt.ylim(0,1)

    plt.minorticks_on()
    plt.tick_params(direction="in", which="major", length=7, bottom=True, top=True, right=True)
    plt.tick_params(direction="in", which="minor", length=2.5, bottom=True, top=True, right=True)
    plt.xlabel("False positive rate", fontsize=14)
    plt.ylabel("True positive rate", fontsize=14)

    #plt.suptitle("Integral method", fontsize=16)
    plt.title(f"$AR({p})$, " + r"$\tau$" + f"={train_per}")
    plt.legend(title="Exceeding probability", fontsize=10, bbox_to_anchor=(1.02,1.1))

    plt.savefig(os.path.join(PLOTS_PATH, f"{dname}-p={p}-training_per={train_per}-gust_th={gust_th}.png"), format="png", dpi=300, bbox_inches="tight")
    plt.close()


training_pers = [90, 120, 150]
p_parameters = [4,5,6,7,8,9,10]
gust_th = 1.6

for p in p_parameters:
    for train_per in tqdm(training_pers):

        ar = AR(p=p, train_per=train_per)

        stds = ar.get_stds(data=dataset)
        phis = ar.get_phis(data=dataset, stds=stds)
        predictions = ar.forecast(data=dataset, phis=phis, stds=stds)
        prediction_integrals = ar.get_integrals(predictions=predictions, stds=stds, gust_th=gust_th)

        performances = get_performances(data=dataset, prediction_integral=prediction_integrals, p=p, train_per=train_per, gust_th=gust_th)
        plot_roc(performances=performances, p=p, train_per=train_per, gust_th=gust_th)    

