import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy import special
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

def autocovariance(data, lag):
    cov = (data - data.mean())[lag:] * (data - data.mean())[:-lag]
    return np.mean(cov)


#datasets = {key: lerp(value) for key, value in datasets.items()}                    # linear interpolation
datasets = {key: dataset[1:] - dataset[:-1] for key, dataset in datasets.items()}   # Differentiation to make data stationary
dname = "2015-2017_90m"                                                             # Active dataset
dataset = datasets[dname]

test_set_size = min(1000000, int(len(dataset) * 0.5))
train_data = dataset[:-test_set_size]
test_data = dataset[-test_set_size:]



class ARMA():

    def __init__(self, *, p, q, train_per, gust_th=1.6):

        self.p = p
        self.q = q
        self.train_per = train_per
        self.gust_th = gust_th
        
        self.lags = range(1,p+1)
        self.phis = None


    def get_intervals(self, data):
        
        """
        This method divides up the dataset into data intervals of length tau. 
        """
        n = len(data)

        indices = np.arange(0, n-self.train_per, self.train_per)
        start_indices = indices[:, None] + np.arange(0, self.train_per+self.p)
        data_intervals = data[start_indices]
        
        return data_intervals
        


    def get_sma(self, data):

        ok = ~np.isnan(data)
        data[~ok] = .0
        denominators = np.convolve(ok, np.ones(self.q), "valid")
        #denominators[~(denominators>0)] += 1

        sma = np.convolve(data, np.ones(self.q), "valid")
        sma /= denominators
        return sma


    def get_phis(self, data):

        acf = [autocovariance(data, lag) / data.std()**2 for lag in self.lags]      # Autocorrelation function

        acf.insert(0, 1.)   
        r = [acf[abs(-k+j)] for k in range(self.p) for j in range(self.p)]  # Yule-Walker Estimation with autocorrelation function
        acf.pop(0)
        r_matrix = (np.array(r).reshape(self.p, self.p))
        r_inv_matrix = np.linalg.inv(r_matrix)
        phi_hats = r_inv_matrix @ np.array(acf).reshape(self.p, 1)
        phi_hats = [phi[0] for phi in phi_hats]
        self.phis = phi_hats



    def forecast(self, data):

        data_intervals = self.get_intervals(data)
        smoothed_data_intervals = self.get_intervals(self.get_sma(data))

        
        predictions = []
        self.get_phis(data_intervals[0]) # Pre-training

        for i, x in enumerate(smoothed_data_intervals[1:], start=1):
            prediction = np.convolve(x[:-1], self.phis, "valid")
            predictions.append(prediction)
            self.get_phis(data_intervals[i])

        return data_intervals,  np.array(predictions)
    


    def get_integrals(self, predictions, data_intervals):

        stds = data_intervals[:-1].std(axis=1)
        white_noise = rng.normal(size=data_intervals[:-1, self.p:].shape, scale=(stds**2)[:, None])
        predictions += white_noise 

        z = (self.gust_th - predictions) / ((2*stds**2)**0.5)[:, None]
        prediction_integrals = 0.5 * (1 - special.erf(z))
        
        return prediction_integrals.flatten()

    
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
    

    def get_performances2(self, prediction_integral, data_intervals):
    
        steps = np.linspace(0, 40 ,20)[::-1]
        quantiles = 1/np.exp(steps)        
        performances2 = dict.fromkeys(quantiles)

        wind_gusts = np.where(data_intervals[1:, self.p:].flatten() >= self.gust_th, 3, 0) # G(t)
        n_gusts = np.where(wind_gusts == 3, 1, 0).sum()

        for quantile in quantiles:

            wind_gusts_pred = np.where(prediction_integral >= quantile, 1, 0) 

            true_positive = np.where(wind_gusts_pred + wind_gusts == 4, 1, 0)
            false_positive = np.where(wind_gusts_pred + wind_gusts == 1, 1, 0)
            
            true_positive_rate = true_positive.sum() / n_gusts
            false_positive_rate = false_positive.sum() / (len(wind_gusts) - n_gusts)

            performances2[quantile] = (true_positive_rate, false_positive_rate)

        return performances2, quantiles
        

    def roc_plot(self, performances, colorname, p):

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

        #plt.savefig(os.path.join(PLOTS_PATH, f"{dname}-AR_overview.png"), format="png", dpi=300, bbox_inches="tight")



    def roc2(self, performances, quantiles):
        colors = sns.color_palette("rocket_r", len(performances))
        _ = plt.figure(1, figsize=(5,5))

        i = 0
        for threshold, (true_positive_rate, false_positive_rate) in performances.items():
            plt.plot(false_positive_rate, true_positive_rate, "o", label=f"{quantiles[i]*100}%", color=colors[i])
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

        plt.suptitle("Integral method", fontsize=16)
        plt.title(f"$ARMA({self.p},{self.q})$" + r"$\tau$" + f"={self.train_per}")
        plt.legend(title="Exceeding probability", fontsize=10, bbox_to_anchor=(1.02,1.1))

        plt.savefig(os.path.join(PLOTS_PATH, f"{dname}-training_period={self.train_per}-p={self.p}-q={self.q}.png"), format="png", dpi=300, bbox_inches="tight")
        plt.close()

for i in range(1, 10):
    arma = ARMA(p=i, q=2, train_per=2000, gust_th=1.6)
    data_intervals, predictions = arma.forecast(dataset)
    prediction_integrals = arma.get_integrals(predictions, data_intervals)
    performances2, quantiles = arma.get_performances2(prediction_integrals, data_intervals)
    arma.roc2(performances2, quantiles)

