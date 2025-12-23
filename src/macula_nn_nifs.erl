%% @doc High-performance Rust NIFs for Macula Neuroevolution (Enterprise).
%%
%% This module provides hardware-accelerated implementations of compute-intensive
%% operations for neuroevolution. Enterprise customers can add this as a dependency
%% to get 10-15x performance improvements over the pure Erlang fallbacks.
%%
%% Usage:
%% 1. Add to rebar.config deps:
%%    {macula_nn_nifs, {git, "git@github.com:macula-io/macula-nn-nifs.git", {tag, "0.1.0"}}}
%%
%% 2. The macula_tweann library will automatically detect and use these NIFs.
%%
%% @copyright 2025 Macula.io
%% @license Apache-2.0
-module(macula_nn_nifs).

-export([
    %% NIF loading
    is_loaded/0,

    %% Network compilation and evaluation
    compile_network/3,
    evaluate/2,
    evaluate_batch/2,
    compatibility_distance/5,
    benchmark_evaluate/3,

    %% Signal aggregation
    dot_product_flat/3,
    dot_product_batch/1,
    dot_product_preflattened/3,
    flatten_weights/1,

    %% LTC/CfC functions
    evaluate_cfc/4,
    evaluate_cfc_with_weights/6,
    evaluate_ode/5,
    evaluate_ode_with_weights/7,
    evaluate_cfc_batch/4,

    %% Distance and KNN (Novelty Search)
    euclidean_distance/2,
    euclidean_distance_batch/2,
    knn_novelty/4,
    knn_novelty_batch/3,

    %% Statistics
    fitness_stats/1,
    weighted_moving_average/2,
    shannon_entropy/1,
    histogram/4,

    %% Selection
    build_cumulative_fitness/1,
    roulette_select/3,
    roulette_select_batch/3,
    tournament_select/2,

    %% Reward and Meta-Controller
    z_score/3,
    compute_reward_component/2,
    compute_weighted_reward/1,

    %% Batch Mutation (Evolutionary Genetics)
    mutate_weights/4,
    mutate_weights_seeded/5,
    mutate_weights_batch/1,
    mutate_weights_batch_uniform/4,
    random_weights/1,
    random_weights_seeded/2,
    random_weights_gaussian/3,
    random_weights_batch/1,
    weight_distance_l1/2,
    weight_distance_l2/2,
    weight_distance_batch/3
]).

-on_load(init/0).

%% ============================================================================
%% NIF Loading
%% ============================================================================

-spec init() -> ok | {error, term()}.
init() ->
    PrivDir = case code:priv_dir(?MODULE) of
        {error, bad_name} ->
            %% Module not yet in code path, use relative path
            case code:which(?MODULE) of
                Filename when is_list(Filename) ->
                    filename:join([filename:dirname(Filename), "..", "priv"]);
                _ ->
                    "priv"
            end;
        Dir ->
            Dir
    end,
    SoName = filename:join(PrivDir, "libmacula_nn_nifs"),
    case erlang:load_nif(SoName, 0) of
        ok -> ok;
        {error, {reload, _}} -> ok;  %% Already loaded
        {error, Reason} -> {error, Reason}
    end.

%% @doc Check if NIF is loaded successfully.
-spec is_loaded() -> boolean().
is_loaded() ->
    try
        %% Try a simple NIF call to verify loading
        _ = random_weights(0),
        true
    catch
        error:undef -> false;
        error:nif_not_loaded -> false
    end.

%% ============================================================================
%% Network Compilation and Evaluation
%% ============================================================================

-spec compile_network(list(), non_neg_integer(), [non_neg_integer()]) -> reference().
compile_network(_Nodes, _InputCount, _OutputIndices) ->
    erlang:nif_error(nif_not_loaded).

-spec evaluate(reference(), [float()]) -> [float()].
evaluate(_Network, _Inputs) ->
    erlang:nif_error(nif_not_loaded).

-spec evaluate_batch(reference(), [[float()]]) -> [[float()]].
evaluate_batch(_Network, _InputsList) ->
    erlang:nif_error(nif_not_loaded).

-spec compatibility_distance(list(), list(), float(), float(), float()) -> float().
compatibility_distance(_ConnectionsA, _ConnectionsB, _C1, _C2, _C3) ->
    erlang:nif_error(nif_not_loaded).

-spec benchmark_evaluate(reference(), [float()], pos_integer()) -> float().
benchmark_evaluate(_Network, _Inputs, _Iterations) ->
    erlang:nif_error(nif_not_loaded).

%% ============================================================================
%% Signal Aggregation
%% ============================================================================

-spec dot_product_flat([float()], [float()], float()) -> float().
dot_product_flat(_Signals, _Weights, _Bias) ->
    erlang:nif_error(nif_not_loaded).

-spec dot_product_batch([{[float()], [float()], float()}]) -> [float()].
dot_product_batch(_Batch) ->
    erlang:nif_error(nif_not_loaded).

-spec dot_product_preflattened([float()], [float()], float()) -> float().
dot_product_preflattened(_SignalsFlat, _WeightsFlat, _Bias) ->
    erlang:nif_error(nif_not_loaded).

-spec flatten_weights([{term(), [{float(), float(), float(), list()}]}]) ->
    {[float()], [non_neg_integer()]}.
flatten_weights(_WeightedInputs) ->
    erlang:nif_error(nif_not_loaded).

%% ============================================================================
%% LTC/CfC Functions
%% ============================================================================

-spec evaluate_cfc(float(), float(), float(), float()) -> {float(), float()}.
evaluate_cfc(_Input, _State, _Tau, _Bound) ->
    erlang:nif_error(nif_not_loaded).

-spec evaluate_cfc_with_weights(float(), float(), float(), float(), [float()], [float()]) ->
    {float(), float()}.
evaluate_cfc_with_weights(_Input, _State, _Tau, _Bound, _BackboneWeights, _HeadWeights) ->
    erlang:nif_error(nif_not_loaded).

-spec evaluate_ode(float(), float(), float(), float(), float()) -> {float(), float()}.
evaluate_ode(_Input, _State, _Tau, _Bound, _Dt) ->
    erlang:nif_error(nif_not_loaded).

-spec evaluate_ode_with_weights(float(), float(), float(), float(), float(), [float()], [float()]) ->
    {float(), float()}.
evaluate_ode_with_weights(_Input, _State, _Tau, _Bound, _Dt, _BackboneWeights, _HeadWeights) ->
    erlang:nif_error(nif_not_loaded).

-spec evaluate_cfc_batch([float()], float(), float(), float()) -> [{float(), float()}].
evaluate_cfc_batch(_Inputs, _InitialState, _Tau, _Bound) ->
    erlang:nif_error(nif_not_loaded).

%% ============================================================================
%% Distance and KNN (Novelty Search)
%% ============================================================================

-spec euclidean_distance([float()], [float()]) -> float().
euclidean_distance(_V1, _V2) ->
    erlang:nif_error(nif_not_loaded).

-spec euclidean_distance_batch([float()], [[float()]]) -> [{non_neg_integer(), float()}].
euclidean_distance_batch(_Target, _Others) ->
    erlang:nif_error(nif_not_loaded).

-spec knn_novelty([float()], [[float()]], [[float()]], pos_integer()) -> float().
knn_novelty(_Target, _Population, _Archive, _K) ->
    erlang:nif_error(nif_not_loaded).

-spec knn_novelty_batch([[float()]], [[float()]], pos_integer()) -> [float()].
knn_novelty_batch(_Behaviors, _Archive, _K) ->
    erlang:nif_error(nif_not_loaded).

%% ============================================================================
%% Statistics
%% ============================================================================

-spec fitness_stats([float()]) -> {float(), float(), float(), float(), float(), float()}.
fitness_stats(_Fitnesses) ->
    erlang:nif_error(nif_not_loaded).

-spec weighted_moving_average([float()], float()) -> float().
weighted_moving_average(_Values, _Decay) ->
    erlang:nif_error(nif_not_loaded).

-spec shannon_entropy([float()]) -> float().
shannon_entropy(_Values) ->
    erlang:nif_error(nif_not_loaded).

-spec histogram([float()], pos_integer(), float(), float()) -> [non_neg_integer()].
histogram(_Values, _NumBins, _MinVal, _MaxVal) ->
    erlang:nif_error(nif_not_loaded).

%% ============================================================================
%% Selection
%% ============================================================================

-spec build_cumulative_fitness([float()]) -> {[float()], float()}.
build_cumulative_fitness(_Fitnesses) ->
    erlang:nif_error(nif_not_loaded).

-spec roulette_select([float()], float(), float()) -> non_neg_integer().
roulette_select(_Cumulative, _Total, _RandomVal) ->
    erlang:nif_error(nif_not_loaded).

-spec roulette_select_batch([float()], float(), [float()]) -> [non_neg_integer()].
roulette_select_batch(_Cumulative, _Total, _RandomVals) ->
    erlang:nif_error(nif_not_loaded).

-spec tournament_select([non_neg_integer()], [float()]) -> non_neg_integer().
tournament_select(_Contestants, _Fitnesses) ->
    erlang:nif_error(nif_not_loaded).

%% ============================================================================
%% Reward and Meta-Controller
%% ============================================================================

-spec z_score(float(), float(), float()) -> float().
z_score(_Value, _Mean, _StdDev) ->
    erlang:nif_error(nif_not_loaded).

-spec compute_reward_component([float()], float()) -> {float(), float(), float()}.
compute_reward_component(_History, _Current) ->
    erlang:nif_error(nif_not_loaded).

-spec compute_weighted_reward([{[float()], float(), float()}]) -> float().
compute_weighted_reward(_Components) ->
    erlang:nif_error(nif_not_loaded).

%% ============================================================================
%% Batch Mutation (Evolutionary Genetics)
%% ============================================================================

-spec mutate_weights([float()], float(), float(), float()) -> [float()].
mutate_weights(_Weights, _MutationRate, _PerturbRate, _PerturbStrength) ->
    erlang:nif_error(nif_not_loaded).

-spec mutate_weights_seeded([float()], float(), float(), float(), integer()) -> [float()].
mutate_weights_seeded(_Weights, _MutationRate, _PerturbRate, _PerturbStrength, _Seed) ->
    erlang:nif_error(nif_not_loaded).

-spec mutate_weights_batch([{[float()], float(), float(), float()}]) -> [[float()]].
mutate_weights_batch(_Genomes) ->
    erlang:nif_error(nif_not_loaded).

-spec mutate_weights_batch_uniform([[float()]], float(), float(), float()) -> [[float()]].
mutate_weights_batch_uniform(_Genomes, _MutationRate, _PerturbRate, _PerturbStrength) ->
    erlang:nif_error(nif_not_loaded).

-spec random_weights(non_neg_integer()) -> [float()].
random_weights(_N) ->
    erlang:nif_error(nif_not_loaded).

-spec random_weights_seeded(non_neg_integer(), integer()) -> [float()].
random_weights_seeded(_N, _Seed) ->
    erlang:nif_error(nif_not_loaded).

-spec random_weights_gaussian(non_neg_integer(), float(), float()) -> [float()].
random_weights_gaussian(_N, _Mean, _StdDev) ->
    erlang:nif_error(nif_not_loaded).

-spec random_weights_batch([{non_neg_integer(), float(), float()}]) -> [[float()]].
random_weights_batch(_Specs) ->
    erlang:nif_error(nif_not_loaded).

-spec weight_distance_l1([float()], [float()]) -> float().
weight_distance_l1(_Weights1, _Weights2) ->
    erlang:nif_error(nif_not_loaded).

-spec weight_distance_l2([float()], [float()]) -> float().
weight_distance_l2(_Weights1, _Weights2) ->
    erlang:nif_error(nif_not_loaded).

-spec weight_distance_batch([float()], [[float()]], l1 | l2) -> [float()].
weight_distance_batch(_Target, _Others, _DistanceType) ->
    erlang:nif_error(nif_not_loaded).
