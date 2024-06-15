#!/usr/bin/python3
import torch
from unixcoder import UniXcoder
from sentence_transformers import util

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UniXcoder("microsoft/unixcoder-base")
model.to(device)


# First step is extracting the embeddings for a code-snippet
def get_embeddings(text):
    tokens_ids = model.tokenize([text], max_length=512,mode="<encoder-only>")
    source_ids = torch.tensor(tokens_ids).to(device)
    tokens_embeddings, nl_embedding = model(source_ids)
    norm_nl_embedding = torch.nn.functional.normalize(nl_embedding, p=2, dim=1)
    norm_nl_embedding = norm_nl_embedding.detach().cpu().numpy()[0]
    return norm_nl_embedding

# Some code snippets we want to look up embeddings for.
code_corpus = [
"""
Customer c = context.current_customer()
transactions = c.list_transactions(2, time.MINUTES, sortOrder.DESC )
""",
"""
ticket = Tickes.from(description,prority.MEDIUM)
ticket.save()
""",
"""
Customer c = context.current_customer()
c.sendMsg(alarmMsg)
""",
"""
customer_categories = [value1, value2, value3]
plt.hist(x, bins = 5)
plt.show()
"""
]

vector_database = list()

# actually fetch the embeddings for each of the code snippets
for code in code_corpus:
    vector_database.append(get_embeddings(code))

## Let's query!
nl_query = """
list the last N transactions of a customer in a time window.
"""

nlq_emb = get_embeddings(nl_query)

# get the cosine-similarities for each of the "db entries" and compare
# it against the query
cos_scores = util.cos_sim(nlq_emb, vector_database)[0]
# note that util comes 'sentence-transformers'
# pip install -U sentence-transformers
# https://www.sbert.net/docs/cross_encoder/usage/usage.html
top_results = torch.topk(cos_scores, k=3)

## Print the top results
print("The top results to compare against are:")
print(top_results)

indices = top_results.indices
print(indices)
top_index = indices[0]
print("Query: ",nl_query)
print(f"The most similar was entry {top_index}: {code_corpus[top_index]}")

print("-------------------------------------------------")

# Let's try another query
nl_query = """
Plot a histogram of customer categories
"""

nlq_emb = get_embeddings(nl_query)

# get the cosine-similarities for each of the "db entries" and compare
# it against the query
cos_scores = util.cos_sim(nlq_emb, vector_database)[0]
# note that util comes 'sentence-transformers'
# pip install -U sentence-transformers
# https://www.sbert.net/docs/cross_encoder/usage/usage.html
top_results = torch.topk(cos_scores, k=3)

## Print the top results
print("The top results to compare against are:")
print(top_results)

indices = top_results.indices
print(indices)
top_index = indices[0]
print("Query: ",nl_query)
print(f"The most similar was entry {top_index}: {code_corpus[top_index]}")
