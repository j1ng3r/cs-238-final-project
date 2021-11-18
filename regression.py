import numpy as np
from math import erf, sqrt
from main import run_simulation
from datetime import datetime

def z2p(z):
	return (1 + erf(z/sqrt(2)))/2

def train_params(num_params, timestep, num_samples_per_sim, num_sims):
	num_samples = num_samples_per_sim * num_sims
	targets = np.empty(num_samples)
	datas = np.empty((num_samples, num_params))
	sample_i = 0
	for sim_index in range(num_sims):
		print("Running training simulation:", sim_index + 1, "/", num_sims)
		simdata = run_simulation()
		for sample_index in range(num_samples_per_sim):
			target = simdata.get_final_price()
			data = np.log(simdata.get_sample(timestep, num_params))
			targets[sample_i] = target
			datas[sample_i] = data
			sample_i += 1

	params = np.linalg.lstsq(datas, targets)[0]
	return params

class Assets:
	def __init__(self, fiat, comm):
		self.fiat = fiat
		self.comm = comm
	def buy(self, price):
		return Assets(self.fiat - price, self.comm + 1)
	def sell(self, price):
		return Assets(self.fiat + price, self.comm - 1)

def test_params(params, timestep, num_sims):
	num_params = params.shape[0]
	profits = np.empty(num_sims)
	for sim_index in range(num_sims):
		print("Running testing simulation:", sim_index + 1, "/", num_sims)
		simdata = run_simulation()
		assets = Assets(0, 0)
		num_timesteps = simdata.get_timesteps()
		for t in range(num_params * timestep, num_timesteps - 1, timestep):
			data = np.log(simdata.get_data(t, timestep, num_params))
			predicted_final_price = np.dot(params, data)
			current_price = simdata.get_price_at(t)
			next_price = simdata.get_price_at(t+1)
			if current_price > predicted_final_price:
				assets = assets.sell(next_price)
			else:
				assets = assets.buy(next_price)
		
		final_price = simdata.get_final_price()
		profit = assets.fiat + final_price * assets.comm # forced to sell all commodities
		profits[sim_index] = profit
		print("Profit:", profit)
	
	avg_profit = np.average(profits)
	stddev_profit = np.std(profits, ddof=1)
	return avg_profit, stddev_profit

def main():
	start_time = datetime.now()
	num_params = 4
	timestep = 50
	num_samples_per_sim = 30
	num_training_sims = 50
	params = train_params(num_params, timestep, num_samples_per_sim, num_training_sims)
	print("Time to train:", datetime.now() - start_time)
	start_time = datetime.now()
	num_testing_sims = 10
	avg_profit, stddev_profit = test_params(params, timestep, num_testing_sims)
	print("Average profit:", avg_profit)
	print("Standard deviation of profit:", stddev_profit)
	zscore = sqrt(num_testing_sims) * avg_profit / stddev_profit
	print("Z-score:", zscore)
	
	
	print("Time to test:", datetime.now() - start_time)

if __name__ == '__main__':
	main()
