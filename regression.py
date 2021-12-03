import numpy as np
from math import erf, sqrt, floor
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
		simdata = run_simulation(seed = sim_index)
		assets = Assets(0, 0)
		num_timesteps = simdata.get_timesteps()
		for t in range(num_params * timestep, num_timesteps - 1, timestep):
			past_L_prices = np.log(simdata.get_data(t, timestep, num_params))
			predicted_final_L_price = np.dot(params, past_L_prices)
			current_L_price = np.log(simdata.get_price_at(t))
			next_price = simdata.get_price_at(t+1)
			if current_L_price > predicted_final_L_price:
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

def test_fn(get_final_assets, timestep, num_sims, sim_type):
	profits = np.empty(num_sims)
	max_profits = np.empty(num_sims)
	for sim_index in range(num_sims):
		print("Running", sim_type, "simulation:", sim_index + 1, "/", num_sims)
		simdata = run_simulation(seed = sim_index + 1000)
		num_timesteps = simdata.get_timesteps()
		final_assets = get_final_assets(simdata, timestep)
		final_price = simdata.get_final_price()
		profit = final_assets.fiat + final_price * final_assets.comm # forced to sell all commodities
		profits[sim_index] = profit
		print("Profit:", profit)
		max_assets = max_profit_get_final_assets(simdata, timestep)
		max_profit = max_assets.fiat + final_price * max_assets.comm # forced to sell all commodities
		max_profits[sim_index] = max_profit
	avg_profit = np.average(profits)
	stddev_profit = np.std(profits, ddof=1)
	return avg_profit, stddev_profit, profits, max_profits

def params2get_final_assets(params):
	num_params = params.shape[0]
	def get_final_assets(simdata, timestep):
		assets = Assets(0, 0)
		num_timesteps = simdata.get_timesteps()
		for t in range(num_params * timestep, num_timesteps - 1, timestep):
			past_L_prices = np.log(simdata.get_data(t, timestep, num_params))
			predicted_final_L_price = np.dot(params, past_L_prices)
			current_L_price = np.log(simdata.get_price_at(t))
			next_price = simdata.get_price_at(t+1)
			if current_L_price > predicted_final_L_price:
				assets = assets.sell(next_price)
			else:
				assets = assets.buy(next_price)
		return assets
	return get_final_assets

def max_profit_get_final_assets(simdata, timestep):
	assets = Assets(0, 0)
	num_timesteps = simdata.get_timesteps()
	final_price = simdata.get_final_price()
	for t in range(0, num_timesteps - 1, timestep):
		current_price = simdata.get_price_at(t)
		next_price = simdata.get_price_at(t+1)
		if current_price >final_price:
			assets = assets.sell(next_price)
		else:
			assets = assets.buy(next_price)
	return assets

def get_unweighted_slope(reverse_data, n):
	factor1 = n*(n-1)/2
	sumt0 = sum(data)
	sumt1 = np.dot(np.arange(n), data)
	numerator = n * sumt1 - factor1 * sumt0
	denominator = n**2*(n**2-1)/12
	slope = numerator / denominator
	return slope

def slope_is_postive(reverse_data, n):
	reduced_factor = (n-1)/2
	sumt0 = sum(reverse_data)
	sumt1 = np.dot(np.arange(n), reverse_data)
	return sumt1 < reduced_factor * sumt0

def slope_based_2_get_final_assets(short_term_memory, num_points):
	def get_final_assets(simdata, timestep):
		assets = Assets(0, 0)
		num_timesteps = simdata.get_timesteps()
		for t in range(short_term_memory, num_timesteps - 1, timestep):
			n = min(num_points, 1 + floor(t / short_term_memory))
			past_L_prices = np.log(simdata.get_data(t, short_term_memory, num_points))

			next_price = simdata.get_price_at(t+1)
			if not slope_is_postive(past_L_prices, num_points):
				assets = assets.sell(next_price)
			else:
				assets = assets.buy(next_price)
		return assets
	return get_final_assets


# test_fn(max_profit_get_final_assets, 30, 40, "max profit")
# test_fn(params2get_final_assets(params), 30, 40, "param testing")

def param_sims():
	start_time = datetime.now()
	num_params = 10
	timestep = 30
	num_samples_per_sim = 50
	num_training_sims = 20
	params = train_params(num_params, timestep, num_samples_per_sim, num_training_sims)
	print("Time to train:", datetime.now() - start_time)
	print("Params:",params)
	start_time = datetime.now()
	num_testing_sims = 10
	avg_profit, stddev_profit, _, _ = test_fn(params2get_final_assets(params), timestep, num_testing_sims, "param testing")
	print("Average profit:", avg_profit)
	print("Standard deviation of profit:", stddev_profit)
	zscore = sqrt(num_testing_sims) * avg_profit / stddev_profit
	print("Z-score:", zscore)	
	print("Time to test:", datetime.now() - start_time)

def mp_sims():
	timestep = 30
	print("Doing max profit sims")
	start_time = datetime.now()
	num_testing_sims = 10
	avg_profit, stddev_profit, _, _ = test_fn(max_profit_get_final_assets, timestep, num_testing_sims, "max profit")
	print("Average profit:", avg_profit)
	print("Standard deviation of profit:", stddev_profit)
	zscore = sqrt(num_testing_sims) * avg_profit / stddev_profit
	print("Z-score:", zscore)	
	print("Time to test:", datetime.now() - start_time)

def slope_sims():
	timestep = 30
	print("Doing slope based sims")
	start_time = datetime.now()
	num_testing_sims = 50
	short_term_memory = 10
	num_points = 2
	avg_profit, stddev_profit, profits, max_profits = test_fn(slope_based_2_get_final_assets(short_term_memory, num_points), timestep, num_testing_sims, "slope based profits")
	print("Average profit:", avg_profit)
	print("Standard deviation of profit:", stddev_profit)
	zscore = sqrt(num_testing_sims) * avg_profit / stddev_profit
	print("Z-score:", zscore)	
	print("Time to test:", datetime.now() - start_time)
	print("Profits:", profits)
	print("Max Profits", max_profits)

#-7.2687542692971165
def main():
	# param_sims()
	# mp_sims()
	slope_sims()


if __name__ == '__main__':
	main()






















