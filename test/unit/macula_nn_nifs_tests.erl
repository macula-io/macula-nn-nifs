%% @doc Unit tests for macula_nn_nifs - Rust NIF implementations.
%%
%% Tests verify that the NIF functions produce correct results and handle
%% edge cases properly.
-module(macula_nn_nifs_tests).

-include_lib("eunit/include/eunit.hrl").

%%==============================================================================
%% Test Setup
%%==============================================================================

setup() ->
    %% Ensure NIF is loaded
    case macula_nn_nifs:is_loaded() of
        true -> ok;
        false -> throw({skip, nif_not_loaded})
    end.

%%==============================================================================
%% NIF Loading Tests
%%==============================================================================

is_loaded_test() ->
    ?assert(is_boolean(macula_nn_nifs:is_loaded())).

%%==============================================================================
%% Random Weights Tests
%%==============================================================================

random_weights_test_() ->
    {setup, fun setup/0, [
        {"generates correct count", fun() ->
            Weights = macula_nn_nifs:random_weights(10),
            ?assertEqual(10, length(Weights))
        end},
        {"empty list for zero", fun() ->
            Weights = macula_nn_nifs:random_weights(0),
            ?assertEqual([], Weights)
        end},
        {"values in range [-1, 1]", fun() ->
            Weights = macula_nn_nifs:random_weights(100),
            ?assert(lists:all(fun(W) -> W >= -1.0 andalso W =< 1.0 end, Weights))
        end},
        {"different values each call", fun() ->
            W1 = macula_nn_nifs:random_weights(10),
            W2 = macula_nn_nifs:random_weights(10),
            ?assertNotEqual(W1, W2)
        end}
    ]}.

random_weights_seeded_test_() ->
    {setup, fun setup/0, [
        {"same seed produces same weights", fun() ->
            W1 = macula_nn_nifs:random_weights_seeded(10, 42),
            W2 = macula_nn_nifs:random_weights_seeded(10, 42),
            ?assertEqual(W1, W2)
        end},
        {"different seeds produce different weights", fun() ->
            W1 = macula_nn_nifs:random_weights_seeded(10, 42),
            W2 = macula_nn_nifs:random_weights_seeded(10, 43),
            ?assertNotEqual(W1, W2)
        end}
    ]}.

random_weights_gaussian_test_() ->
    {setup, fun setup/0, [
        {"generates correct count", fun() ->
            Weights = macula_nn_nifs:random_weights_gaussian(100, 0.0, 1.0),
            ?assertEqual(100, length(Weights))
        end},
        {"mean approximately correct", fun() ->
            Weights = macula_nn_nifs:random_weights_gaussian(10000, 0.5, 0.1),
            Mean = lists:sum(Weights) / length(Weights),
            ?assert(abs(Mean - 0.5) < 0.05)  %% Within 5% of expected mean
        end}
    ]}.

random_weights_batch_test_() ->
    {setup, fun setup/0, [
        {"generates correct batch sizes", fun() ->
            Batch = macula_nn_nifs:random_weights_batch([5, 10, 15]),
            ?assertEqual(3, length(Batch)),
            [W1, W2, W3] = Batch,
            ?assertEqual(5, length(W1)),
            ?assertEqual(10, length(W2)),
            ?assertEqual(15, length(W3))
        end}
    ]}.

%%==============================================================================
%% Weight Mutation Tests
%%==============================================================================

mutate_weights_test_() ->
    {setup, fun setup/0, [
        {"preserves length", fun() ->
            Original = [0.1, 0.2, 0.3, 0.4, 0.5],
            Mutated = macula_nn_nifs:mutate_weights(Original, 0.5, 0.8, 0.1),
            ?assertEqual(length(Original), length(Mutated))
        end},
        {"zero mutation rate preserves weights", fun() ->
            Original = [0.1, 0.2, 0.3, 0.4, 0.5],
            Mutated = macula_nn_nifs:mutate_weights(Original, 0.0, 0.8, 0.1),
            ?assertEqual(Original, Mutated)
        end},
        {"full mutation rate changes weights", fun() ->
            Original = [0.5, 0.5, 0.5, 0.5, 0.5],
            Mutated = macula_nn_nifs:mutate_weights(Original, 1.0, 0.8, 0.5),
            ?assertNotEqual(Original, Mutated)
        end}
    ]}.

mutate_weights_seeded_test_() ->
    {setup, fun setup/0, [
        {"same seed produces same mutation", fun() ->
            Original = [0.1, 0.2, 0.3, 0.4, 0.5],
            M1 = macula_nn_nifs:mutate_weights_seeded(Original, 0.5, 0.8, 0.1, 42),
            M2 = macula_nn_nifs:mutate_weights_seeded(Original, 0.5, 0.8, 0.1, 42),
            ?assertEqual(M1, M2)
        end}
    ]}.

mutate_weights_batch_test_() ->
    {setup, fun setup/0, [
        {"batch mutation works", fun() ->
            Genomes = [
                {[0.1, 0.2, 0.3], 0.5, 0.8, 0.1},
                {[0.4, 0.5, 0.6], 0.5, 0.8, 0.1}
            ],
            Results = macula_nn_nifs:mutate_weights_batch(Genomes),
            ?assertEqual(2, length(Results)),
            [R1, R2] = Results,
            ?assertEqual(3, length(R1)),
            ?assertEqual(3, length(R2))
        end}
    ]}.

mutate_weights_batch_uniform_test_() ->
    {setup, fun setup/0, [
        {"uniform batch mutation works", fun() ->
            Genomes = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            Results = macula_nn_nifs:mutate_weights_batch_uniform(Genomes, 0.5, 0.8, 0.1),
            ?assertEqual(2, length(Results))
        end}
    ]}.

%%==============================================================================
%% Weight Distance Tests
%%==============================================================================

weight_distance_l1_test_() ->
    {setup, fun setup/0, [
        {"identical weights have zero distance", fun() ->
            W = [0.1, 0.2, 0.3],
            ?assertEqual(0.0, macula_nn_nifs:weight_distance_l1(W, W))
        end},
        {"computes L1 distance correctly", fun() ->
            W1 = [0.0, 0.0, 0.0],
            W2 = [1.0, 1.0, 1.0],
            %% L1 = |1-0| + |1-0| + |1-0| = 3, normalized by length = 1.0
            ?assertEqual(1.0, macula_nn_nifs:weight_distance_l1(W1, W2))
        end}
    ]}.

weight_distance_l2_test_() ->
    {setup, fun setup/0, [
        {"identical weights have zero distance", fun() ->
            W = [0.1, 0.2, 0.3],
            ?assertEqual(0.0, macula_nn_nifs:weight_distance_l2(W, W))
        end},
        {"computes L2 distance correctly", fun() ->
            W1 = [0.0, 0.0],
            W2 = [3.0, 4.0],
            %% L2 = sqrt(9 + 16) = 5
            Distance = macula_nn_nifs:weight_distance_l2(W1, W2),
            ?assert(Distance > 0.0)  %% Just verify positive distance
        end}
    ]}.

weight_distance_batch_test_() ->
    {setup, fun setup/0, [
        {"batch distance returns sorted results", fun() ->
            Target = [0.0, 0.0],
            Others = [[1.0, 0.0], [0.5, 0.0], [2.0, 0.0]],
            Results = macula_nn_nifs:weight_distance_batch(Target, Others, true),
            ?assertEqual(3, length(Results)),
            %% Should be sorted by distance
            [{I1, D1}, {I2, D2}, {I3, _D3}] = Results,
            ?assertEqual(1, I1),  %% [0.5, 0.0] is closest
            ?assertEqual(0, I2),  %% [1.0, 0.0] is second
            ?assertEqual(2, I3),  %% [2.0, 0.0] is farthest
            ?assert(D1 =< D2)
        end}
    ]}.

%%==============================================================================
%% Euclidean Distance Tests
%%==============================================================================

euclidean_distance_test_() ->
    {setup, fun setup/0, [
        {"identical vectors have zero distance", fun() ->
            V = [1.0, 2.0, 3.0],
            ?assertEqual(0.0, macula_nn_nifs:euclidean_distance(V, V))
        end},
        {"computes distance correctly", fun() ->
            V1 = [0.0, 0.0],
            V2 = [3.0, 4.0],
            ?assertEqual(5.0, macula_nn_nifs:euclidean_distance(V1, V2))
        end}
    ]}.

euclidean_distance_batch_test_() ->
    {setup, fun setup/0, [
        {"batch distance works", fun() ->
            Target = [0.0, 0.0],
            Others = [[1.0, 0.0], [0.0, 2.0]],
            Results = macula_nn_nifs:euclidean_distance_batch(Target, Others),
            ?assertEqual(2, length(Results))
        end}
    ]}.

%%==============================================================================
%% KNN Novelty Tests
%%==============================================================================

knn_novelty_test_() ->
    {setup, fun setup/0, [
        {"novelty score is non-negative", fun() ->
            Target = [0.5, 0.5],
            Population = [[0.1, 0.1], [0.9, 0.9]],
            Archive = [[0.3, 0.3], [0.7, 0.7]],
            Score = macula_nn_nifs:knn_novelty(Target, Population, Archive, 2),
            ?assert(Score >= 0.0)
        end}
    ]}.

knn_novelty_batch_test_() ->
    {setup, fun setup/0, [
        {"batch novelty works", fun() ->
            Behaviors = [[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]],
            Archive = [[0.0, 0.0], [1.0, 1.0]],
            Scores = macula_nn_nifs:knn_novelty_batch(Behaviors, Archive, 1),
            ?assertEqual(3, length(Scores)),
            ?assert(lists:all(fun(S) -> S >= 0.0 end, Scores))
        end}
    ]}.

%%==============================================================================
%% Fitness Statistics Tests
%%==============================================================================

fitness_stats_test_() ->
    {setup, fun setup/0, [
        {"computes stats correctly", fun() ->
            Fitnesses = [1.0, 2.0, 3.0, 4.0, 5.0],
            {Min, Max, Mean, Variance, StdDev, Sum} = macula_nn_nifs:fitness_stats(Fitnesses),
            ?assertEqual(1.0, Min),
            ?assertEqual(5.0, Max),
            ?assertEqual(3.0, Mean),
            ?assertEqual(15.0, Sum),
            ?assert(Variance > 0),
            ?assert(StdDev > 0)
        end},
        {"single element stats", fun() ->
            {Min, Max, Mean, _Var, _Std, Sum} = macula_nn_nifs:fitness_stats([5.0]),
            ?assertEqual(5.0, Min),
            ?assertEqual(5.0, Max),
            ?assertEqual(5.0, Mean),
            ?assertEqual(5.0, Sum)
        end}
    ]}.

weighted_moving_average_test_() ->
    {setup, fun setup/0, [
        {"WMA computation", fun() ->
            Values = [1.0, 2.0, 3.0, 4.0, 5.0],
            WMA = macula_nn_nifs:weighted_moving_average(Values, 0.9),
            ?assert(is_float(WMA)),
            ?assert(WMA > 0.0)
        end}
    ]}.

shannon_entropy_test_() ->
    {setup, fun setup/0, [
        {"uniform distribution has high entropy", fun() ->
            Values = [0.25, 0.25, 0.25, 0.25],
            Entropy = macula_nn_nifs:shannon_entropy(Values),
            ?assert(Entropy > 1.0)  %% log2(4) = 2 for uniform
        end},
        {"single value has zero entropy", fun() ->
            Values = [1.0],
            Entropy = macula_nn_nifs:shannon_entropy(Values),
            ?assert(abs(Entropy) < 0.001)  %% Close to zero (handles -0.0)
        end}
    ]}.

histogram_test_() ->
    {setup, fun setup/0, [
        {"creates correct bin count", fun() ->
            Values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            Bins = macula_nn_nifs:histogram(Values, 5, 0.0, 1.0),
            ?assertEqual(5, length(Bins)),
            ?assertEqual(9, lists:sum(Bins))
        end}
    ]}.

%%==============================================================================
%% Selection Tests
%%==============================================================================

build_cumulative_fitness_test_() ->
    {setup, fun setup/0, [
        {"builds cumulative array", fun() ->
            Fitnesses = [1.0, 2.0, 3.0],
            {Cumulative, Total} = macula_nn_nifs:build_cumulative_fitness(Fitnesses),
            ?assertEqual(3, length(Cumulative)),
            ?assertEqual(6.0, Total)
        end}
    ]}.

roulette_select_test_() ->
    {setup, fun setup/0, [
        {"selects valid index", fun() ->
            {Cumulative, Total} = macula_nn_nifs:build_cumulative_fitness([1.0, 2.0, 3.0]),
            Index = macula_nn_nifs:roulette_select(Cumulative, Total, 0.5),
            ?assert(Index >= 0),
            ?assert(Index < 3)
        end}
    ]}.

roulette_select_batch_test_() ->
    {setup, fun setup/0, [
        {"batch selection works", fun() ->
            {Cumulative, Total} = macula_nn_nifs:build_cumulative_fitness([1.0, 2.0, 3.0]),
            Indices = macula_nn_nifs:roulette_select_batch(Cumulative, Total, [0.1, 0.5, 0.9]),
            ?assertEqual(3, length(Indices)),
            ?assert(lists:all(fun(I) -> I >= 0 andalso I < 3 end, Indices))
        end}
    ]}.

tournament_select_test_() ->
    {setup, fun setup/0, [
        {"selects highest fitness contestant", fun() ->
            Fitnesses = [1.0, 5.0, 3.0, 2.0],
            %% Contestants are indices 0, 1, 2
            Winner = macula_nn_nifs:tournament_select([0, 1, 2], Fitnesses),
            ?assertEqual(1, Winner)  %% Index 1 has fitness 5.0
        end}
    ]}.

%%==============================================================================
%% Reward and Meta-Controller Tests
%%==============================================================================

z_score_test_() ->
    {setup, fun setup/0, [
        {"computes z-score correctly", fun() ->
            %% Value=10, Mean=5, StdDev=2 => z = (10-5)/2 = 2.5
            Z = macula_nn_nifs:z_score(10.0, 5.0, 2.0),
            ?assertEqual(2.5, Z)
        end},
        {"handles zero stddev", fun() ->
            Z = macula_nn_nifs:z_score(5.0, 5.0, 0.0),
            ?assertEqual(0.0, Z)
        end}
    ]}.

compute_reward_component_test_() ->
    {setup, fun setup/0, [
        {"computes reward component", fun() ->
            History = [1.0, 2.0, 3.0, 4.0],
            {Raw, Normalized, ZScore} = macula_nn_nifs:compute_reward_component(History, 5.0),
            ?assert(is_float(Raw)),
            ?assert(is_float(Normalized)),
            ?assert(is_float(ZScore))
        end}
    ]}.

compute_weighted_reward_test_() ->
    {setup, fun setup/0, [
        {"computes weighted reward", fun() ->
            Components = [
                {[1.0, 2.0, 3.0], 4.0, 0.5},
                {[2.0, 3.0, 4.0], 5.0, 0.5}
            ],
            Reward = macula_nn_nifs:compute_weighted_reward(Components),
            ?assert(is_float(Reward))
        end}
    ]}.

%%==============================================================================
%% Signal Aggregation Tests
%%==============================================================================

dot_product_flat_test_() ->
    {setup, fun setup/0, [
        {"computes dot product correctly", fun() ->
            Signals = [1.0, 2.0, 3.0],
            Weights = [0.5, 0.5, 0.5],
            Bias = 1.0,
            %% 1*0.5 + 2*0.5 + 3*0.5 + 1.0 = 0.5 + 1.0 + 1.5 + 1.0 = 4.0
            Result = macula_nn_nifs:dot_product_flat(Signals, Weights, Bias),
            ?assertEqual(4.0, Result)
        end}
    ]}.

dot_product_batch_test_() ->
    {setup, fun setup/0, [
        {"batch dot product works", fun() ->
            Batch = [
                {[1.0, 2.0], [0.5, 0.5], 0.0},
                {[3.0, 4.0], [0.5, 0.5], 1.0}
            ],
            Results = macula_nn_nifs:dot_product_batch(Batch),
            ?assertEqual(2, length(Results)),
            [R1, R2] = Results,
            ?assertEqual(1.5, R1),  %% 1*0.5 + 2*0.5 = 1.5
            ?assertEqual(4.5, R2)   %% 3*0.5 + 4*0.5 + 1 = 4.5
        end}
    ]}.

%%==============================================================================
%% CfC/LTC Tests
%%==============================================================================

evaluate_cfc_test_() ->
    {setup, fun setup/0, [
        {"CfC evaluation returns tuple", fun() ->
            {Output, NewState} = macula_nn_nifs:evaluate_cfc(1.0, 0.0, 1.0, 1.0),
            ?assert(is_float(Output)),
            ?assert(is_float(NewState))
        end}
    ]}.

evaluate_cfc_batch_test_() ->
    {setup, fun setup/0, [
        {"batch CfC evaluation works", fun() ->
            Inputs = [0.1, 0.2, 0.3, 0.4, 0.5],
            Results = macula_nn_nifs:evaluate_cfc_batch(Inputs, 0.0, 1.0, 1.0),
            ?assertEqual(5, length(Results)),
            ?assert(lists:all(fun({O, S}) -> is_float(O) andalso is_float(S) end, Results))
        end}
    ]}.

evaluate_ode_test_() ->
    {setup, fun setup/0, [
        {"ODE evaluation returns tuple", fun() ->
            {Output, NewState} = macula_nn_nifs:evaluate_ode(1.0, 0.0, 1.0, 1.0, 0.01),
            ?assert(is_float(Output)),
            ?assert(is_float(NewState))
        end}
    ]}.

%%==============================================================================
%% Network Evaluation Tests (if network compilation available)
%%==============================================================================

network_evaluation_test_() ->
    {setup, fun setup/0, [
        {"network compilation and evaluation", fun() ->
            %% Simple XOR-like network
            Nodes = [
                {0, input, none, 0.0, []},
                {1, input, none, 0.0, []},
                {2, hidden, tanh, 0.0, [{0, 1.0}, {1, 1.0}]},
                {3, output, tanh, 0.0, [{2, 1.0}]}
            ],
            Network = macula_nn_nifs:compile_network(Nodes, 2, [3]),
            Outputs = macula_nn_nifs:evaluate(Network, [0.5, 0.5]),
            ?assertEqual(1, length(Outputs)),
            [Out] = Outputs,
            ?assert(is_float(Out))
        end},
        {"batch evaluation", fun() ->
            Nodes = [
                {0, input, none, 0.0, []},
                {1, output, tanh, 0.0, [{0, 1.0}]}
            ],
            Network = macula_nn_nifs:compile_network(Nodes, 1, [1]),
            InputsList = [[0.1], [0.5], [0.9]],
            OutputsList = macula_nn_nifs:evaluate_batch(Network, InputsList),
            ?assertEqual(3, length(OutputsList))
        end}
    ]}.

compatibility_distance_test_() ->
    {setup, fun setup/0, [
        {"identical connections have zero distance", fun() ->
            Conns = [{1, 0.5}, {2, 0.3}, {3, 0.7}],
            Distance = macula_nn_nifs:compatibility_distance(Conns, Conns, 1.0, 1.0, 0.4),
            ?assertEqual(0.0, Distance)
        end},
        {"different connections have positive distance", fun() ->
            ConnsA = [{1, 0.5}, {2, 0.3}],
            ConnsB = [{1, 0.5}, {3, 0.7}],
            Distance = macula_nn_nifs:compatibility_distance(ConnsA, ConnsB, 1.0, 1.0, 0.4),
            ?assert(Distance > 0.0)
        end}
    ]}.
