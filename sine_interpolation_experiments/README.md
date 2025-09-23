# Sine Interpolation Experiments

This package provides a modular framework for benchmarking different neural network architectures on the sin(2πkx) interpolation problem using the SSBroyden optimizer.

## Structure

- `problem_utils.py` - Common utilities for the sin(2πkx) problem (data generation, loss functions, oracle)
- `mlp_trainer.py` - MLP model definition and training functions
- `benchmark_mlp.py` - Main benchmarking script for running ablation studies
- `analyze_results.py` - Analysis and visualization of results

## Usage

### Quick Test

Run a quick test to verify everything works:

```bash
cd sine_interpolation_experiments
python benchmark_mlp.py --quick-test
```

### Full Ablation Study

Run a complete ablation study across different architectures:

```bash
python benchmark_mlp.py --widths 32 64 128 256 --depths 1 2 3 4 --ks 1 2 4 8 16 --seeds 0 1 2
```

### Custom Configuration

```bash
python benchmark_mlp.py \
    --widths 64 128 \
    --depths 2 3 \
    --ks 8 16 \
    --steps 25000 \
    --n-train 2048 \
    --seeds 0 1 \
    --lr 1.0 \
    --output-dir my_results
```

### Analyze Results

After running experiments, analyze the results:

```bash
python analyze_results.py results/mlp_ablation_20231215_120000 --output-dir plots
```

## Output Format

Each experiment saves a JSON file with the following structure:

```json
{
  "config": {
    "width": 64,
    "depth": 2,
    "k": 8,
    "steps": 50000,
    "n_train": 4097,
    "n_eval": 10000,
    "seed": 0,
    "lr": 1.0,
    "c1": 1e-4,
    "c2": 0.9,
    "max_ls": 20
  },
  "training": {
    "loss_history": [...],
    "rel_l2_history": [...],
    "final_loss": 1.234e-10,
    "final_rel_l2": 5.678e-6,
    "initial_rel_l2": 1.234
  }
}
```

## Extending for Other Architectures

To test other architectures, create new trainer modules following the pattern in `mlp_trainer.py`:

1. Define your model class
2. Implement a training function that returns the same result format
3. Create a new benchmark script that uses the shared `problem_utils`

The `make_fg_oracle` function can be reused for any architecture that needs the SSBroyden optimizer.

## Dependencies

- JAX
- Flax NNX
- Optax
- tqdm
- numpy
- matplotlib
- seaborn
- pandas
- jaxpi (for SSBroyden optimizer)