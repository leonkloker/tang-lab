import evaluate_model
import data

features = [[[1,2,3],[1.1,2.1,3.1],[1.2,2.2,3.2]], [[4,5,6],[4.3,5.3,6.3]]]

features = data.get_fixed_size_subsample(features, size=3)

print(features)