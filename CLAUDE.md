# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**macula-nn-nifs** is a high-performance Rust NIF (Native Implemented Functions) library for Macula Neuroevolution. This is an **enterprise-only** package providing 10-15x performance improvements over pure Erlang fallbacks.

## Distribution Model

- **Community Edition**: Users install `macula_tweann` from hex.pm (pure Erlang fallbacks)
- **Enterprise Edition**: Users add `macula_nn_nifs` as a private git dependency to unlock Rust NIFs

The `tweann_nif.erl` module in `macula_tweann` automatically detects and uses these NIFs when available.

## Build Commands

```bash
rebar3 get-deps        # Fetch Erlang dependencies
rebar3 compile         # Compile Erlang wrapper + Rust NIFs
rebar3 eunit           # Run unit tests
rebar3 dialyzer        # Run type analysis
```

## Architecture

### Directory Structure

```
macula-nn-nifs/
├── src/
│   ├── macula_nn_nifs.erl      # Erlang NIF wrapper module
│   └── macula_nn_nifs.app.src  # OTP application resource file
├── native/
│   ├── Cargo.toml              # Rust project configuration
│   └── src/
│       └── lib.rs              # Rust NIF implementations
├── priv/                       # Compiled .so/.dylib/.dll files
└── rebar.config                # Erlang build config with Rust hooks
```

### NIF Categories

1. **Network Compilation and Evaluation**
   - `compile_network/3` - Compile network topology for fast evaluation
   - `evaluate/2` - Forward propagation through compiled network
   - `evaluate_batch/2` - Batch evaluation for multiple inputs
   - `compatibility_distance/5` - NEAT speciation distance
   - `benchmark_evaluate/3` - Performance benchmarking

2. **Signal Aggregation**
   - `dot_product_flat/3` - Fast weighted sum
   - `dot_product_batch/1` - Batch weighted sums
   - `dot_product_preflattened/3` - Pre-optimized dot product
   - `flatten_weights/1` - Weight structure optimization

3. **LTC/CfC (Liquid Time-Constant)**
   - `evaluate_cfc/4` - Closed-form continuous-time evaluation
   - `evaluate_cfc_with_weights/6` - CfC with custom weights
   - `evaluate_ode/5` - ODE-based LTC evaluation
   - `evaluate_ode_with_weights/7` - ODE with custom weights
   - `evaluate_cfc_batch/4` - Batch CfC for time series

4. **Distance and KNN (Novelty Search)**
   - `euclidean_distance/2` - Vector distance
   - `euclidean_distance_batch/2` - Batch distances
   - `knn_novelty/4` - K-nearest neighbor novelty
   - `knn_novelty_batch/3` - Batch novelty computation

5. **Statistics**
   - `fitness_stats/1` - Single-pass statistics
   - `weighted_moving_average/2` - WMA computation
   - `shannon_entropy/1` - Entropy calculation
   - `histogram/4` - Histogram binning

6. **Selection**
   - `build_cumulative_fitness/1` - Roulette wheel setup
   - `roulette_select/3` - Single roulette selection
   - `roulette_select_batch/3` - Batch selection
   - `tournament_select/2` - Tournament selection

7. **Reward and Meta-Controller**
   - `z_score/3` - Z-score normalization
   - `compute_reward_component/2` - Reward signal computation
   - `compute_weighted_reward/1` - Multi-component rewards

8. **Batch Mutation (Evolutionary Genetics)**
   - `mutate_weights/4` - Gaussian weight mutation
   - `mutate_weights_seeded/5` - Reproducible mutation
   - `mutate_weights_batch/1` - Batch mutation with per-genome params
   - `mutate_weights_batch_uniform/4` - Batch with uniform params
   - `random_weights/1` - Generate random weights
   - `random_weights_seeded/2` - Seeded random weights
   - `random_weights_gaussian/3` - Gaussian distributed weights
   - `random_weights_batch/1` - Batch weight generation
   - `weight_distance_l1/2` - L1 (Manhattan) distance
   - `weight_distance_l2/2` - L2 (Euclidean) distance
   - `weight_distance_batch/3` - Batch distance computation

## Integration with macula_tweann

The `tweann_nif` module automatically detects this package:

```erlang
%% In tweann_nif.erl:
detect_impl_module() ->
    case code:which(macula_nn_nifs) of
        non_existing ->
            tweann_nif_fallback;
        _ ->
            case macula_nn_nifs:is_loaded() of
                true -> macula_nn_nifs;
                false -> tweann_nif_fallback
            end
    end.
```

## Usage for Enterprise Customers

Add to your `rebar.config`:

```erlang
{deps, [
    {macula_tweann, "0.6.0"},
    {macula_nn_nifs, {git, "git@github.com:macula-io/macula-nn-nifs.git", {tag, "0.1.0"}}}
]}.
```

The NIFs are automatically detected and used. No code changes required.

## Rust Dependencies

- **rustler** (0.34): Erlang NIF framework for Rust
- **rand** (0.8): Random number generation
- **rand_distr** (0.4): Statistical distributions

## Development Notes

### Building Rust NIFs

The `rebar.config` contains pre/post hooks:
- Pre-compile: Runs `cargo build --release`
- Post-compile: Copies `.so`/`.dylib`/`.dll` to `priv/`

### Testing NIF Loading

```erlang
1> macula_nn_nifs:is_loaded().
true

2> macula_nn_nifs:random_weights(5).
[0.123, -0.456, 0.789, -0.234, 0.567]
```

### Performance Expectations

| Operation | Pure Erlang | Rust NIF | Speedup |
|-----------|-------------|----------|---------|
| Network evaluate | 50μs | 5μs | ~10x |
| Batch mutation | 200μs | 15μs | ~13x |
| KNN novelty | 1ms | 80μs | ~12x |
| Fitness stats | 100μs | 8μs | ~12x |

## EDoc Formatting Rules

**IMPORTANT: EDoc has strict formatting requirements for hex.pm publishing**

- **NO heredoc syntax** - Do not use triple backticks
- **NO HTML tags** - Do not use `<pre>`, `<code>`, etc.
- **NO backticks** - Do not use single backticks for inline code

## License

Apache-2.0 (Enterprise license required for commercial use)
