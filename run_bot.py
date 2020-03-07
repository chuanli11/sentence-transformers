import os
import argparse
import pickle
import numpy as np

import torch
from sentence_transformers import SentenceTransformer, LoggingHandler

parser = argparse.ArgumentParser(description='Input argument parser.')


parser = argparse.ArgumentParser(description='Input argument parser.')

parser.add_argument('--model_name', type=str, help='name of model',
					default='roberta-large-nli-stsb-mean-tokens')

parser.add_argument('--database_path', type=str, help='path to the questions and answer dataset. Can be txt or pickle',
					default='examples.txt')

parser.add_argument('--batch_size', type=int, help='batch size',
					default=1)

parser.add_argument('--verbose', default=False, action="store_true" , help="Flag to display input questions")

args = parser.parse_args()


class Bot(object):
	def __init__(self, model_name, database_path):
		self.encoder = SentenceTransformer(model_name)
		self.db = self.database(database_path)

	def encode(self, q):
		enc_q = self.encoder.encode([q])[0]
		return enc_q

	def database(self, database_path):
		dtype = os.path.splitext(database_path)[1]

		db = []

		if dtype == '.txt':
			# create database from scratch
			with torch.no_grad():
				with open(database_path, 'r', encoding='utf8', errors='ignore') as fp:
					for cnt, line in enumerate(fp):
						d = line.split(' ||| ')
						
						enc_q = self.encode(d[0])

						db.append([enc_q, d[0], d[1]])

						print(d[0])

				pickle_path = os.path.splitext(database_path)[0] + '.pkl'
				


			with open(pickle_path, 'wb') as f:
				pickle.dump(db, f)
			
		elif dtype == '.pkl':
			# load pre-generated database
			with open(database_path, 'rb') as f:
				db = pickle.load(f)
		else:
			print('Database type {} has not been implemented'.format(dtype))

		return db

	def answer(self, q):
		
		with torch.no_grad():
			enc_q = self.encode(q)

		# Query database
		score = np.zeros(len(self.db))
		for cnt, d in enumerate(self.db):
			score[cnt] = np.dot(enc_q, d[0]) / (np.linalg.norm(enc_q) * np.linalg.norm(d[0]) + 0.000001)

		idx = np.argmax(score)
		return self.db[idx][2]

def main():
	bot_name = 'Victoria'
	print(bot_name + ': Welcome')

	bot = Bot(args.model_name, args.database_path)

	# chat 
	while True:
		user_input = input("You\t\t> ")
		if args.verbose:
			print("User input raw: {}".format(user_input))
		bot_answer = bot.answer(user_input)
		print("{}\t> {}".format(bot_name, bot_answer))


if __name__ == '__main__':
	main()