import numpy as np
import matplotlib.pyplot as plt

true_price = 100

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def arcsigmoid(z):
	return np.log(z / (1 - z))

class Buyer:
	def __init__(self):
		self.subjective_price = true_price * np.exp(np.random.normal(0, .2))
		self.asking_price = self.subjective_price * np.exp(-np.random.normal(0, .4)**2)
		self.aggressiveness = .01
	
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
	def __init__(self):
		self.subjective_price = true_price * np.exp(np.random.normal(0, .2))
		#TODO justify this / tweak as necessary
		self.asking_price = self.subjective_price * np.exp(np.random.normal(0, .4)**2)
		self.aggressiveness = .01
		
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
		self.decay_time = 20
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

def main():
	TIMESTEPS = 3000
	
	buyers = []
	for i in range(50):
		buyers.append(Buyer())
	
	sellers = []
	for i in range(50):
		sellers.append(Seller())
		
	market = Market()
	
	decayed_market_prices = []
	market_prices = []
	for i in range(TIMESTEPS):
		time_step(buyers, sellers, market)
		market_prices.append(market.get_last_price())
		decayed_market_prices.append(market.get_decayed_price())
	
	times = np.arange(TIMESTEPS)
	
	# plt.plot(times, market_prices)
	plt.plot(times, decayed_market_prices)
	plt.show()

main()
