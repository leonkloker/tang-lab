import data

x = [[[1,2,3,4,5],[3,4,5,6,7]], [[1,2,3,4,5]], [[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]]
q = data.get_query_points_marginal(x, n_points=10, n_std=3)
print(q)