import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
#rng = np.random.default_rng(11)


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

def autocovariance(data, lag):
    cov = (data - data.mean())[lag:] * (data - data.mean())[:-lag]
    return np.mean(cov)


datasets = {key: lerp(value) for key, value in datasets.items()}                    # linear interpolation
datasets = {key: dataset[1:] - dataset[:-1] for key, dataset in datasets.items()}   # Differentiation to make data stationary
dname = "2015-2017_90m"                                                             # Active dataset
dataset = datasets[dname]

test_set_size = min(1000000, int(len(dataset) * 0.5))
train_data = dataset[:-test_set_size]
test_data = dataset[-test_set_size:]



class AR():

    def __init__(self, p):

        self.p = p
        self.lags = range(1,p+1)
        self.phis = None

    def get_phis(self, train_data):

        acf = [autocovariance(train_data, lag) / train_data.std()**2 for lag in self.lags]      # Autocorrelation function

        acf.insert(0, 1.)   
        r = [acf[abs(-k+j)] for k in range(self.p) for j in range(self.p)]  # Yule-Walker Estimation with autocorrelation function
        acf.pop(0)
        r_matrix = (np.array(r).reshape(self.p, self.p))
        r_inv_matrix = np.linalg.inv(r_matrix)
        phi_hats = r_inv_matrix @ np.array(acf).reshape(self.p, 1)
        phi_hats = [phi[0] for phi in phi_hats]
        self.phis = phi_hats


    def predict(self, test_data):

        buffer = len(self.phis)
        predictions = [0] * buffer

        for i, _ in enumerate(test_data):                           # Model predictions only 1 time step into the future
            prediction = sum([x*phi for x, phi in zip(test_data[i:], reversed(self.phis))])
            predictions.append(prediction)
        predictions = np.array(predictions[buffer:2*-buffer])       # predictions shape and test_data shape not equal

        return predictions
    
    def calculate_error(self, predictions, test_data):
        hypotheses = {}
        hypotheses.update({f"AR({self.p})": [(x-x_pred)**2 for x, x_pred in zip(predictions, test_data)]})         # Squared losses
        hypotheses = dict(map(lambda x: (x[0] ,np.mean(x[1])), hypotheses.items()))         # squared errors --> mean squared errors (mse)

        return hypotheses
    
    def get_performances(self, predictions, test_data):

        buffer = len(self.phis)
        test_data = np.array(test_data[buffer:-buffer])

        thresholds = np.round(np.linspace(0, .6, 20), 2)        # Model: Wind gust thresholds
        u_threshold = 1.5                                       # Test_data: Wind gust thresholds
        performances = dict.fromkeys(thresholds)
        
        for threshold in thresholds:         

            accuracies = []
            wind_gusts = np.where(test_data>=u_threshold, 3, 0)                  # Actual occurance: wind gust
            wind_gusts_pred = np.where(predictions>=threshold, 1, 0)             # Model prediction: wind gust

            true_positive = np.where(wind_gusts_pred + wind_gusts == 4, 1, 0)            # 3, 0
            false_positive = np.where(wind_gusts_pred + wind_gusts == 1, 1, 0)           # 1, 0
            wind_gusts = wind_gusts / 3

            assert wind_gusts.sum() > 0
            true_positive_rate = sum(true_positive) / sum(wind_gusts)
            false_positive_rate = sum(false_positive) / sum(np.where(wind_gusts == 0, 1, 0))
            
            accuracies.append(true_positive_rate)
            accuracies.append(false_positive_rate)
            performances.update({threshold: accuracies})

        return performances

def roc_plot(performances, colorname, p):

    _ = plt.figure(1, figsize=(5,5))
    colors = sns.color_palette(colorname, len(performances))

    i = 0
    for threshold, (true_positive_rate, false_positive_rate) in performances.items():
        plt.plot(false_positive_rate, true_positive_rate, "o", label=f"{threshold:.2f}", color=colors[i])
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

    plt.title(f"$AR({p})$", fontsize=14)
    plt.suptitle("Receiver-operating-characteristic", fontsize=16)
    plt.legend(title="Threshold (m/s)", fontsize=10, bbox_to_anchor=(1.02,1.1))

    plt.savefig(os.path.join(PLOTS_PATH, f"{dname}-AR_overview.png"), format="png", dpi=300, bbox_inches="tight")

ar_2 = AR(2)

ar_2.get_phis(train_data)
predictions = ar_2.predict(test_data)
performances = ar_2.get_performances(predictions, test_data)

color_palette = ["rocket_r", "dark:salmon_r", "dark:b_r", "dark:seagreen_r"]
roc_plot(performances, color_palette[0], 2)