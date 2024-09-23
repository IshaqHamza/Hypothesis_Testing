# import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import the hypothesis testing functions
from anova import anova_two_way

# For plotting histogram
def plot_histogram(values: np.array, bins: int = 100, xlabel: str = 'x', ylabel: str = 'Frequency', title: str = 'Histogram of x', file_name='Plot1.png') -> None:
    '''Plots a histogram of the given values and saves it as a png file'''
    plt.hist(values, bins=bins)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(file_name)

    return None

# Get the data from the txt file which contains tab separated values
# There are twelve columns for each of the four groups side-by-side, starting from index 1
# Remove rows with missing probe names, expression values or gene symbol
# i.e, remove all rows which are missing any value in the first 50
data = pd.read_csv('data/Raw Data_GeneSpring.txt', sep='\t')
initial_num_rows = data.shape[0]

data = data.dropna(subset=data.columns[0:50])
final_num_rows = data.shape[0]

print(f'Removed {initial_num_rows - final_num_rows} rows with missing values')

# Extract the observations matrix and groups
# Find the groups and their effect vectors for null and alternate hypothesis
observations = np.array(data.iloc[:, 1:4*12+1], dtype=float)
groups = [list(range(12*i, 12*(i + 1))) for i in range(4)]

# Null hypothesis: The effects of gender and smoking are additive
# We fix and order the features as (is_male, is_female, is_smoker, is_non_smoker) and group order as (male non-smoker, male smoker, female non-smoker, female-smoker)
# effect vectors for
#   male non-smoker = (1, 0, 0, 1)
#   male smoker = (1, 0, 1, 0)
#   female non-smoker = (0, 1, 0, 1)
#   female smoker = (0, 1, 1, 0)
effect_vectors_null =[[1, 0, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [0, 1, 1, 0]]

# Alternate hypothesis: The effects of gender and smoking are arbitrary
# We fix and order the features as (male_non_smoker, male_smoker, female_non_smoker, female_smoker)
# effect vectors for
#   male non-smoker = (1, 0, 0, 0)
#   male smoker = (0, 1, 0, 0)
#   female non-smoker = (0, 0, 1, 0)
#   female smoker = (0, 0, 0, 1)
effect_vectors_alternate = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


# Obtain p_values after performing 2-way ANOVA
p_values = anova_two_way(observations, groups, effect_vectors_null, effect_vectors_alternate, rank_null=3, rank_alternate=4)

# Save and show plot of histogram of the p-values
plot_histogram(p_values, bins=100, xlabel='p-values', ylabel='Number of Probes', title='Histogram of p-values across Expressions Levels of Probes', file_name='Plot1.png')
plt.show()

# Add p_values to the data
data['p_value'] = p_values

# There could be multiple probes for the same gene, so it is worth aggregating the p-values for the same gene.

# aggregate by mean and plot
p_values_mean = data.groupby('GeneSymbol')['p_value'].mean()
plot_histogram(p_values_mean, bins=100, xlabel='p-values', ylabel='Number of Genes', title='Histogram of p-values across Genes (aggregated by mean)', file_name='Plot2.png')
plt.show()

# aggregate by minimum and plot
p_values_min = data.groupby('GeneSymbol')['p_value'].min()
plot_histogram(p_values_min, bins=100, xlabel='p-values', ylabel='Number of Genes', title='Histogram of p-values across Genes (aggregated by min)', file_name='Plot3.png')
plt.show()

# aggregate by maximum and plot
p_values_max = data.groupby('GeneSymbol')['p_value'].max()
plot_histogram(p_values_max, bins=100, xlabel='p-values', ylabel='Number of Genes', title='Histogram of p-values across Genes (aggregated by max)', file_name='Plot4.png')
plt.show()