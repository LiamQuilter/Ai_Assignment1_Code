
# AI Assignment 1 - Genetic Algorithm for TSP

## Setup Instructions
Install the required packages using pip: 
```sh
pip install numpy pandas matplotlib
```

## How to Use

1. Prepare your TSP dataset in TSPLIB format and place it in the `datasets` directory.
2. Modify the `main` section in `solver.py` to specify the dataset and parameters for the genetic algorithm.
3. Run the `solver.py` script:
    ```sh
    python solver.py
    ```

## Example Usage

```python
if __name__ == "__main__":
    tsp_data = parse_tsplib("datasets/pr1002.tsp")
    grid_search_results = grid_search_ga_with_csv(
        tsp_data,
        pop_sizes=[100],
        generations_list=[5000],
        crossover_rates=[0.7],
        mutation_rates=[0.3],
        k_values=[3],
        crossover_types=['edg'],
        mutation_types=['reverse'],
        output_file="finalpr_results.csv"
    )
```

This example will run the genetic algorithm on the pr1002 dataset with the specified parameters and save the results to csv specified
