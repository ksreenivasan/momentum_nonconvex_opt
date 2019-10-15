from griewank import *

df = beta_grid_search(n_iter=10000)
df = df.sort_values('f_opt')
print "Top betas!"
print df.head()
df.to_csv("beta_grid_search_results.csv")
