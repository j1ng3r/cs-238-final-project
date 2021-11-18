import numpy as np
import matplotlib.pyplot as plt

true_price = 100

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def arcsigmoid(z):
	return np.log(z / (1 - z))

class Buyer:
	sprice_stdev = 0.1
	aprice_stdev = 0.01
	aggressiveness = 0.01
	def __init__(self, rng):
		self.subjective_price = true_price * np.exp(rng.normal(0, Buyer.sprice_stdev))
		self.asking_price = self.subjective_price * np.exp(-rng.normal(0, np.sqrt(Buyer.aprice_stdev))**2)
		self.aggressiveness = Buyer.aggressiveness
	
	def update_asking_price(self, order_accepted: bool):
		if order_accepted:
			arg0 = arcsigmoid(self.asking_price / self.subjective_price)
			arg = arg0 - self.aggressiveness
			self.asking_price = self.subjective_price * sigmoid(arg)
		else:
			arg0 = arcsigmoid(self.asking_price / self.subjective_price)
			arg = arg0 + self.aggressiveness
			self.asking_price = self.subjective_price * sigmoid(arg)
	


class Seller:
	sprice_stdev = 0.1
	aprice_stdev = 0.01
	aggressiveness = 0.01
	
	def __init__(self, rng):
		self.subjective_price = true_price * np.exp(rng.normal(0, Seller.sprice_stdev))
		#TODO justify this / tweak as necessary
		self.asking_price = self.subjective_price * np.exp(rng.normal(0, np.sqrt(Seller.aprice_stdev))**2)
		self.aggressiveness = Seller.aggressiveness
		
	def update_asking_price(self, order_accepted: bool):
		if order_accepted:
			arg0 = np.log(self.asking_price / self.subjective_price - 1)
			arg = arg0 + self.aggressiveness
			self.asking_price = self.subjective_price * (1 + np.exp(arg))
		else:
			arg0 = np.log(self.asking_price / self.subjective_price - 1)
			arg = arg0 - self.aggressiveness
			self.asking_price = self.subjective_price * (1 + np.exp(arg))

class Market:
	def __init__(self):
		self.asking_price = 0
		self.decay_time = 80
		self.decayed_asking_price = 0
		self.decayed_volume_commodity = 0
		self.decayed_volume_fiat = 0
	
	def get_asking_price(self, buyer, seller):
		return (buyer.asking_price + seller.asking_price)/2
		
	def sort_buyers(self, buyers):
		return sorted(buyers, key = lambda buyer: buyer.asking_price, reverse = True)
		
	def sort_sellers(self, sellers):
		return sorted(sellers, key = lambda seller: seller.asking_price)
		
	def exchange_orders(self, buyers, sellers):
		sorted_buyers = self.sort_buyers(buyers)
		sorted_sellers = self.sort_sellers(sellers)
		matched = []
		for (buyer, seller) in zip(sorted_buyers, sorted_sellers):
			if buyer.asking_price >= seller.asking_price:
				buyer.update_asking_price(order_accepted = True)
				seller.update_asking_price(order_accepted = True)
				matched.append((buyer, seller))
			else: break
		
		self.asking_price = self.get_asking_price(*matched[-1]) if len(matched) > 0 else self.asking_price
		volume_commodity = len(matched)
		volume_fiat = self.asking_price * volume_commodity
		self.decayed_volume_commodity = volume_commodity + self.decayed_volume_commodity * np.exp( -1 / self.decay_time )
		self.decayed_volume_fiat = volume_fiat + self.decayed_volume_fiat * np.exp( -1 / self.decay_time )
		self.decayed_asking_price = self.decayed_volume_fiat / self.decayed_volume_commodity if self.decayed_volume_commodity > 0 else 0
		
		for j in range(len(matched), len(sorted_buyers)):
			sorted_buyers[j].update_asking_price(order_accepted = False)
		for j in range(len(matched), len(sorted_sellers)):
			sorted_sellers[j].update_asking_price(order_accepted = False)
				# Do something else
	def get_last_price(self):
		return self.asking_price
	def get_decayed_price(self):
		return self.decayed_asking_price


def time_step(buyers, sellers, market):
	market.exchange_orders(buyers, sellers)

class Simulation_Data:
	sampling_time_error = 5
	sampling_proportion = 0.25
	
	def __init__(self):
		self.times = None
		self.decayed_market_prices = None
		self.final_buy_sell_prices = None
	
	def get_data(self, t, timestep, num_datapoints):
		rand_t = Simulation_Data.sampling_time_error
		indices = np.maximum(t - np.arange(0, num_datapoints * timestep, timestep) + np.random.randint(-rand_t, rand_t + 1, size = num_datapoints), 0)
		indices[0] = t
		return np.array([self.decayed_market_prices[indices[i]] for i in range(num_datapoints)])
		
		# [t-150,t-100,t-50,t]
		# [t,t-50,t-100,t-150]
	
	def get_sample(self, timestep, num_datapoints):
		first_timestep = num_datapoints * timestep
		last_timestep = np.round(self.get_timesteps() * Simulation_Data.sampling_proportion)
		t = np.random.randint(first_timestep, last_timestep + 1)
		return self.get_data(t, timestep, num_datapoints)
	
	def get_final_price(self):
		return self.decayed_market_prices[-1]
	
	def get_price_at(self, t):
		return self.decayed_market_prices[t]
	
	def get_timesteps(self):
		return self.times.shape[0]

def run_simulation(seed = None):
	rng = np.random.RandomState(seed)
	TIMESTEPS = 10000
	
	buyer_subj_prices = []
	buyers = []
	for i in range(100):
		buyer = Buyer(rng)
		buyers.append(buyer)
		buyer_subj_prices.append(buyer.subjective_price)
		
	seller_subj_prices = []
	sellers = []
	for i in range(100):
		seller = Seller(rng)
		sellers.append(seller)
		seller_subj_prices.append(seller.subjective_price)
		
	buyer_subj_prices = sorted(buyer_subj_prices, reverse=True)
	seller_subj_prices = sorted(seller_subj_prices)
	tup = None
	for (buyer_price, seller_price) in zip(buyer_subj_prices, seller_subj_prices):
			if buyer_price >= seller_price:
				tup = (buyer_price, seller_price)
			else:
				break
	
	market = Market()
	
	decayed_market_prices = []
	market_prices = []
	for i in range(TIMESTEPS):
		time_step(buyers, sellers, market)
		market_prices.append(market.get_last_price())
		decayed_market_prices.append(market.get_decayed_price())
	
	times = np.arange(TIMESTEPS)
	
	simdata = Simulation_Data()
	simdata.times = times
	simdata.decayed_market_prices = decayed_market_prices
	simdata.final_buy_sell_prices = tup
	
	return simdata

def test_seed(seed):
	Buyer.aprice_stdev = 0.01
	Seller.aprice_stdev= 0.01
	simdata1 = run_simulation(seed = seed)
	tup = simdata1.final_buy_sell_prices
	
	Buyer.aprice_stdev = 0.1
	Seller.aprice_stdev= 0.1
	simdata2 = run_simulation(seed = seed)
	
	Buyer.aprice_stdev = 0.01
	Seller.aprice_stdev= 0.1
	simdata3 = run_simulation(seed = seed)
	
	Buyer.aprice_stdev = 0.1
	Seller.aprice_stdev= 0.01
	simdata4 = run_simulation(seed = seed)
	
	plt.plot(simdata1.times, simdata1.decayed_market_prices)
	plt.plot(simdata2.times, simdata2.decayed_market_prices)
	plt.plot(simdata3.times, simdata3.decayed_market_prices)
	plt.plot(simdata4.times, simdata4.decayed_market_prices)
	
	plt.axhline(y=tup[0], color='b', linestyle='-')
	plt.axhline(y=tup[1], color='cyan', linestyle='-')
	# plt.show()
	plt.savefig("parameter-0_001/apricevar-" + str(seed)+'.png')
	plt.clf()
	print(seed)

def main():
	test_seed(1731)
	test_seed(1722)
	test_seed(103)
	test_seed(44)
	test_seed(165)
	test_seed(171)
	test_seed(2013)
	test_seed(65538)
	test_seed(2863311532)
	test_seed(3735928561)

if __name__ == '__main__':
	main()
