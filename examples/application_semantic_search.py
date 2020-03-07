"""
This is a simple application for sentence embeddings: semantic search

We have a corpus with various sentences. Then, for a given query sentence,
we want to find the most similar sentence in this corpus.

This script outputs for various queries the top 5 most similar sentences in the corpus.
"""
from sentence_transformers import SentenceTransformer
import scipy.spatial

#name = 'roberta-large-nli-stsb-mean-tokens'
name = 'roberta-large-nli-mean-tokens'

embedder = SentenceTransformer(name)

# Corpus with example sentences
corpus = ["How many GPUs does the Lambda Blade have?",
          "How many GPUs does the Lambda Quad have?",
          "How many GPUs does the TensorBook have?",
          "What's the wattage of the blade?",
          "What's the wattage of the quad?",
          "How fast do you usually ship?",
          "What are your lead times?",
          "What's the best GPU for the money?",
          "Do you ship to my country?",
          "Do you have academic/student/non-profit/government/startup discounts?",
          "What methods of payment do you accept?",
          "What if I have problems? What service options are there?",
          "I don't see the configuration I need. Can I customize it?",
          "Can I use your machines for gaming?",
          "How much power do your machines use? Can I plug it into a normal outlet?",
          "Do you provide on-site installation?",
          "I already have a GPU. Can I use it in combination with the preinstalled GPUs?",
          "What comes pre-installed on your machines?",
          "I want to buy a TensorBook with 512gb of storage. Can I upgrade it later if I need more?",
          "GPUs tend to run hot! How are your machines cooled?",
          "How do I install Lambda Stack?",
          "What sort of server rack do I need for the Lambda Blade?",
          "How loud do they run?",
          "Is there a free trial for to access Lambda's Deep Learning Cloud?",
          "What do your products cost?",
          "Can I buy my device locally? Do you have agents or resellers?",
          "How soon can I get my machine?",
          "What operating systems come on your devices?",
          "Can I mine cryptocurrency with these?",
          "Do you offer financing?",
          "How heavy is the TensorBook?",
          "Have a question you don't see here?",
          "Can I talk with a real human?",
          "What motherboard do you use in the Lambda Quad?",
          "What is the pricing for Neon Miner?",
          "How do I request support for my Lambda product?",
          "What is the Lambda Blade server size?",
          "Do you offer AMD solutions?",
          "Do you offer custom solutions?",
          "Do you offer custom builds?",
          "What are the specs on Lambda Tensorbook?",
          "I'm looking for additional specs and information about the TensorBook?",
          "I need to start training, but I cant commit buying a machine.",
          "What if the software included in the stack releases a new version?"]

corpus_embeddings = embedder.encode(corpus)

# Query sentences:
queries = ['How many GPUs in a Quad?',
           'How many GPUs in a Blade?',
           'How many GPUs in a TensorBook?',
           'Can I talk to your sales?',
           'Can I talk to your engineer?',
           'What GPUs do you use?',
           'Can I use my own GPUs on your machine?',
           'Do you ship to U.K.?',
           'How much power does a blade consume?',
           'How much power does a quad consume?',
           'How much power does a TensorBook consume?',
           'What is your best machine?',
           'What OS do you install?',
           'What operating systems do you install?',
           'What software do you install?']

query_embeddings = embedder.encode(queries)

# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
closest_n = 5
for query, query_embedding in zip(queries, query_embeddings):
    distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")

    for idx, distance in results[0:closest_n]:
        print(corpus[idx].strip(), "(Score: %.4f)" % (1-distance))



