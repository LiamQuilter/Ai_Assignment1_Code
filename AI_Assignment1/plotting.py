import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(style="whitegrid")


csv_file = "gapr1002_results.csv"  
df = pd.read_csv(csv_file)


df.columns = df.columns.str.strip().str.replace(r'\ufeff', '', regex=True)

# Ive done this so i can get the best/average distance at each of these intervals for graphing
generations = [100, 200, 300, 400, 500]
mutation_rates = df['mutation_rate'].unique()  # Extract mutation rates for analysis/ Done this for all of the parameters I was working with

# Reshape Data for Analysis
generation_columns = [
    'fitness_gen_100_best', 'fitness_gen_200_best', 'fitness_gen_300_best',
    'fitness_gen_400_best', 'fitness_gen_500_best'
]

df_melted = df.melt(
    id_vars=['pop_size', 'crossover_rate', 'mutation_rate', 'crossover_type', 'mutation_type', 'best_distance', 'runtime'],
    value_vars=generation_columns,
    var_name='generation', value_name='best_fitness'
)

# Convert 'generation' column into numeric values (100, 200, 300, etc.)
df_melted['generation'] = df_melted['generation'].str.extract('(\d+)').astype(int)

# Convergence of Best Distance Across Mutation Rates
plt.figure(figsize=(10, 5))
for mutation_rate in mutation_rates:
    df_mut = df[df['mutation_rate'] == mutation_rate]
    best_fitness = [df_mut[f'fitness_gen_{gen}_best'].min() for gen in generations]
    plt.plot(generations, best_fitness, marker='o', label=f'Mutation Rate={mutation_rate}')

plt.xlabel('Generations')
plt.ylabel('Best Distance')
plt.title('Impact of Mutation Rate on Best Distance Over Generations')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(x=df['mutation_rate'], y=df['best_distance'])
plt.xlabel('Mutation Rate')
plt.ylabel('Best Distance')
plt.title('Impact of Different Mutation Rates on Best Distance')
plt.show()

# Mutation Rate vs. Average Best Distance per Generation
df_mutation_avg = df_melted.groupby(['mutation_rate', 'generation'])['best_fitness'].mean().reset_index()
plt.figure(figsize=(10, 5))
sns.barplot(data=df_mutation_avg, x='generation', y='best_fitness', hue='mutation_rate')
plt.xlabel('Generations')
plt.ylabel('Average Best Distance')
plt.title('Mutation Rate vs. Average Best Distance per Generation')
plt.legend(title='Mutation Rate')
plt.grid()
plt.show()

# Mutation Type Comparison with boxplot
plt.figure(figsize=(10, 5))
sns.boxplot(x=df['mutation_type'], y=df['best_distance'])
plt.xlabel('Mutation Type')
plt.ylabel('Best Distance')
plt.title('Impact of Different Mutation Types on Best Distance')
plt.show()
