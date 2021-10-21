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
	aggressiveness = 0.005
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
	aggressiveness = 0.005
	
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

def run(seed):
	rng = np.random.RandomState(seed)
	TIMESTEPS = 15000
	
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
	
	return (times, decayed_market_prices, tup)

def test_seed(seed):
	Buyer.aprice_stdev = 0.01
	Seller.aprice_stdev= 0.01
	(times1, decayed_market_prices1, tup) = run(seed = seed)
	Buyer.aprice_stdev = 0.1
	Seller.aprice_stdev= 0.1
	(times2, decayed_market_prices2, _) = run(seed = seed)
	Buyer.aprice_stdev = 0.01
	Seller.aprice_stdev= 0.1
	(times3, decayed_market_prices3, _) = run(seed = seed)
	Buyer.aprice_stdev = 0.1
	Seller.aprice_stdev= 0.01
	(times4, decayed_market_prices4, _) = run(seed = seed)
	plt.plot(times1, decayed_market_prices1)
	plt.plot(times2, decayed_market_prices2)
	plt.plot(times3, decayed_market_prices3)
	plt.plot(times4, decayed_market_prices4)
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
main()
