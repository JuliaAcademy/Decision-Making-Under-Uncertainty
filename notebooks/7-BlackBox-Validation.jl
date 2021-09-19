### A Pluto.jl notebook ###
# v0.16.0

using Markdown
using InteractiveUtils

# ╔═╡ 687f9f3c-3b30-4bbe-af96-60acba88096f
begin
	using PlutoUI

	md"""
	# Black-Box Stress Testing
	##### Julia Academy: _Decision Making Under Uncertainty with POMDPs.jl_

	Showcasing the [`POMDPStressTesting`](https://github.com/sisl/POMDPStressTesting.jl) package that is built on top of the `POMDPs.jl` ecosystem.

	-- Robert Moss (Stanford University) as part of [_Julia Academy_](https://juliaacademy.com/) (Github: [mossr](https://github.com/mossr))
	"""
end

# ╔═╡ 7c1dc177-fdad-4486-8ddd-b1d13c2aebb6
using POMDPStressTesting, Parameters, Distributions

# ╔═╡ 9bda5132-0a28-416d-83aa-94c8d7f0c28e
using Latexify

# ╔═╡ d43b5dac-6398-4c43-aaa6-3f7871ddfeaf
md"""
To find failures in a black-box autonomous system, we can use the [`POMDPStressTesting`](https://github.com/sisl/POMDPStressTesting.jl) package.$^1$

Various solvers—which adhere to the `POMDPs.jl` interface—can be used:
- `MCTSPWSolver` (MCTS with action progressive widening)
- `TRPOSolver` and `PPOSolver` (deep reinforcement learning policy optimization)
- `CEMSolver` (cross-entropy method)
- `RandomSearchSolver`
"""

# ╔═╡ 9d5d5d1b-3cb5-4edb-99c9-3a3c6cc74382
md"""
## Simple Problem: 1D Random Walk
We define a simple problem for _adaptive stress testing_ (AST)$^2$ to find failures. This problem, called Walk1D, samples random walking distances from a standard normal distribution $\mathcal{N}(0,1)$ and defines failures as walking past a certain threshold (which is set to $\pm 10$ in this example). AST will either select the seed which deterministically controls the sampled value from the distribution (i.e. from the transition model) or will directly sample the provided environmental distributions. These action modes are determined by the seed-action or sample-action options. AST will guide the simulation to failure events using a notion of distance to failure, while simultaneously trying to find the set of actions that maximizes the log-likelihood of the samples.
"""

# ╔═╡ f9ada586-0c8e-4a71-821a-08018aa346e3
md"""
#### Black-Box System Under Test
"""

# ╔═╡ 7375c689-be3a-4750-8d66-9b8143ea8476
md"""
Let's contrive an example black-box system (to which we do not actually need to know what's going on internally). Here we create a `BlackBoxSystem` struct and a `run_system` function.
"""

# ╔═╡ 0b818751-4310-47ab-bda3-01a30e3a0b98
md"""
> **The internals of the `BlackBoxSystem` and `run_system` are _unknown_—hence "black box".**
"""

# ╔═╡ 235193c1-7119-4a11-add4-8385e3d21d6b
@with_kw mutable struct BlackBoxSystem
	x::Real = 0 # Current (and initial) x-position
end

# ╔═╡ 5a1b854e-84f1-44e5-9e33-ffad03424352
md"""
All we care about is passing in `inputs` and receiving `outputs` from the system.
"""

# ╔═╡ 287c600e-79ad-4789-971b-9dd49d37d64e
function run_system(system::BlackBoxSystem, input::Real)
	system.x += input # Agent moves based on input
	return system.x   # System output
end

# ╔═╡ 330956a4-25f3-4358-8891-0205e7ef0185
md"""
### Gray-Box Simulator and Environment
The simulator and environment are treated as gray-box because we need access to the state-transition distributions and their associated likelihoods.
"""

# ╔═╡ e256c0da-94db-4b09-b309-edd5d109b066
md"""
##### Parameters
First, we define the parameters of our simulation.
"""

# ╔═╡ cab4c1d3-48f4-4d1a-946a-22c63cde6813
@with_kw mutable struct Walk1DParams
    threshx::Float64 = 10 # ± boundary threshold
    endtime::Int64 = 30   # Simulate end time
end;

# ╔═╡ 7595d65e-9363-4b94-a877-561a7063a63d
md"""
##### Simulation
Next, we define a `GrayBox.Simulation` structure.
"""

# ╔═╡ 117c5000-dc20-4e14-8f22-e0f48a9bebfa
@with_kw mutable struct Walk1DSim <: GrayBox.Simulation
	system::BlackBoxSystem = BlackBoxSystem() # System under test
	output::Real = 0                          # Saved output of the system under test
    params::Walk1DParams = Walk1DParams()     # Parameters
    t::Int64 = 0                              # Current time ±
    distribution::Distribution = Normal(0, 1) # Transition distribution
end;

# ╔═╡ 1a4b8f51-c1c6-4aba-bea3-36f4211c31ae
md"""
#### GrayBox.environment
Then, we define our `GrayBox.Environment` distributions. When using the `ASTSampleAction`, as opposed to `ASTSeedAction`, we need to provide access to the sampleable environment.
"""

# ╔═╡ f7d98d83-ed67-4ec2-b6b2-5477b0614670
GrayBox.environment(sim::Walk1DSim) = GrayBox.Environment(:x => sim.distribution)

# ╔═╡ 13f3f3dc-265a-498e-a3e4-0081d0361ca2
md"""
#### GrayBox.transition!
We override the `transition!` function from the `GrayBox` interface, which takes an environment sample as input. We apply the sample in our simulator, and return the log-likelihood.
"""

# ╔═╡ 546b5aa8-571b-4ab1-8b91-9eaf10cc7a61
function GrayBox.transition!(sim::Walk1DSim, sample::GrayBox.EnvironmentSample)
    sim.t += 1                                 # Keep track of time
    input = sample[:x].value                   # Input to the system under test
	sim.output = run_system(sim.system, input) # Run the system under test
    return logpdf(sample)::Real                # Summation handled by `logpdf()`
end

# ╔═╡ 3ec0a905-3eca-421b-bac0-3664cf41644b
md"""
### Black-Box System
The system under test, in this case a simple single-dimensional moving agent, is always treated as black-box. The following interface functions are overridden to minimally interact with the system, and use outputs from the system to determine failure event indications and distance metrics.
"""

# ╔═╡ a590f1b9-b6d1-45ad-aafa-982af06e6ebd
md"""
#### BlackBox.initialize!
Now we override the `BlackBox` interface, starting with the function that initializes the simulation object. Interface functions ending in `!` may modify the `sim` object in place.
"""

# ╔═╡ e8eabe9f-cea2-44eb-ae98-e8d6cfba4010
function BlackBox.initialize!(sim::Walk1DSim)
    sim.t = 0
	sim.system = BlackBoxSystem()
	sim.output = 0
end

# ╔═╡ 68f3d180-4469-41b9-bb6c-63043af21ead
md"""
#### BlackBox.distance
We define how close we are to a failure event using a non-negative distance metric.
"""

# ╔═╡ 75890d95-b73d-4da2-84f6-6b18409d037d
BlackBox.distance(sim::Walk1DSim) = max(sim.params.threshx - abs(sim.output), 0)

# ╔═╡ 3f4e000d-6c67-4ebd-80dc-835ab74b25c8
md"""
#### BlackBox.isevent
We define an indication that a failure event occurred.
"""

# ╔═╡ 2faa29d0-f42d-44be-a3be-3436d5cef81d
BlackBox.isevent(sim::Walk1DSim) = abs(sim.output) ≥ sim.params.threshx

# ╔═╡ 4af3a2ea-12eb-4afd-894a-bf9f09c38adc
md"""
#### BlackBox.isterminal
Similarly, we define an indication that the simulation is in a terminal state.
"""

# ╔═╡ e2defb9c-a442-4939-83c8-9329e1da182e
function BlackBox.isterminal(sim::Walk1DSim)
	return BlackBox.isevent(sim) || sim.t ≥ sim.params.endtime
end

# ╔═╡ 8ff29459-e714-4a0f-bc03-910f3d24eb3a
md"""
#### BlackBox.evaluate!
Lastly, we use our defined interface to evaluate the system under test. Using the input sample, we return the log-likelihood, distance to an event, and event indication.
"""

# ╔═╡ cca7cd81-e9d2-4494-a710-bbd07dba6731
function BlackBox.evaluate!(sim::Walk1DSim, sample::GrayBox.EnvironmentSample)
    logprob::Real = GrayBox.transition!(sim, sample) # Step simulation
    d::Real       = BlackBox.distance(sim)           # Calculate miss distance
    event::Bool   = BlackBox.isevent(sim)            # Check event indication
    return (logprob::Real, d::Real, event::Bool)
end

# ╔═╡ 5f6fe4db-e7a7-4389-aba9-9ec552faeef8
md"""
### AST Setup and Running
Setting up our simulation, we instantiate our simulation object and pass that to the Markov decision proccess (MDP) object of the adaptive stress testing formulation. We use Monte Carlo tree search (MCTS) with progressive widening on the action space as our solver. Hyperparameters are passed to `MCTSPWSolver`, which is a simple wrapper around the POMDPs.jl implementation of MCTS. Lastly, we solve the MDP to produce a planner. Note we are using the `ASTSampleAction`.
"""

# ╔═╡ 9e1a32a6-32cf-438d-a616-e872fdbf795c
function setup_ast(seed=0)
    # Create gray-box simulation object
    sim::GrayBox.Simulation = Walk1DSim()

    # AST MDP formulation object
    mdp::ASTMDP = ASTMDP{ASTSampleAction}(sim)
    mdp.params.debug = true # record metrics
    mdp.params.top_k = 10   # record top k best trajectories
    mdp.params.seed = seed  # set RNG seed for determinism

    # Hyperparameters for MCTS-PW as the solver
    solver = MCTSPWSolver(n_iterations=1000,        # number of algorithm iterations
                          exploration_constant=1.0, # UCT exploration
                          k_action=1.0,             # action widening
                          alpha_action=0.5,         # action widening
                          depth=sim.params.endtime) # tree depth

    # Get online planner (no work done, yet)
    planner = solve(solver, mdp)

    return planner
end;

# ╔═╡ 917953d6-d7bb-4fe7-b4e5-f4c64b2d2bb2
md"""
#### Searching for Failures
After setup, we search for failures using the planner and output the best action trace.
"""

# ╔═╡ be02f6c7-03fc-40ab-b4d6-86312e88b714
planner = setup_ast();

# ╔═╡ d26959aa-20ba-4943-9a0e-79ed9336d369
action_trace = search!(planner)

# ╔═╡ aaf654fd-3478-4f6d-80af-c16e42881da3
md"""
#### Playback
We can also playback specific trajectories and print intermediate $x$-values.
"""

# ╔═╡ 3679c0ff-142c-49d8-9dc6-3de0f8b96e67
playback_trace = playback(planner, action_trace, sim->sim.output, return_trace=true)

# ╔═╡ 8bc5116a-fe5e-4135-b341-c06900b5ff99
failure_metrics = print_metrics(planner)

# ╔═╡ 934c6cfc-1fce-4225-bf70-8544b1ac82c1
Markdown.parse("""
\$\$\\text{MCTS failure rate} = $(round(failure_metrics.failure_rate, digits=3))\\%\$\$
""")

# ╔═╡ ad1dba31-2b6c-4721-80f7-1841b734526a
md"""
### Other Solvers: Cross-Entropy Method
We can easily take our `ASTMDP` object (`planner.mdp`) and re-solve the MDP using a different solver—in this case the `CEMSolver`.
"""

# ╔═╡ 1d915b62-0281-424a-a317-1e855fbfe046
mdp = setup_ast().mdp;

# ╔═╡ 815033b0-eb9c-4bba-b5c4-fe1d34796fa6
cem_solver = CEMSolver(n_iterations=1000, episode_length=mdp.sim.params.endtime)

# ╔═╡ d0e67875-8f94-402f-adb8-2440f7b9944a
cem_planner = solve(cem_solver, mdp);

# ╔═╡ 39f52e42-eb1e-4817-b842-45a695bdcc37
cem_action_trace = search!(cem_planner);

# ╔═╡ 53403ce4-0af1-4676-b3ab-8f8312590917
cem_playback_trace = playback(cem_planner,
	cem_action_trace, sim->sim.output, return_trace=true)

# ╔═╡ 5616126d-bcd8-4153-a6ff-24c2a2db1f9c
md"Notice the failure rate is about 2× of `MCTSPWSolver`."

# ╔═╡ 43fe1ef2-075f-472d-ab17-ce5d558c7ef2
cem_failure_metrics = print_metrics(cem_planner)

# ╔═╡ a5fedf30-0be4-4221-bcfc-2df4ea7ff400
Markdown.parse("""
\$\$\\text{CEM failure rate} = $(round(cem_failure_metrics.failure_rate, digits=3))\\%\$\$
""")

# ╔═╡ 92f0ab72-dcb6-4360-87f9-fa4569f75a21
md"""
### AST Reward Function
The AST reward function gives a reward of $0$ if an event is found, a reward of negative distance if no event is found at termination, and the log-likelihood during the simulation.
"""

# ╔═╡ 5e7f390e-e1e9-4f1e-8bc3-f1c38c1bbf41
@latexify function R(p,e,d,τ)
    if τ && e
        return 0
    elseif τ && !e
        return -d
    else
        return log(p)
    end
end

# ╔═╡ 70bc9acc-8638-40a7-862d-c1226fd6d53c
md"""
## References
1. Ritchie Lee, Ole J. Mengshoel, Anshu Saksena, Ryan W. Gardner, Daniel Genin, Joshua Silbermann, Michael Owen, and Mykel J. Kochenderfer, "Adaptive Stress Testing: Finding Likely Failure Events with Reinforcement Learning," _Journal of Artificial Intelligence Research_, vol. 69, p. 1165–1201, 2020. [https://arxiv.org/abs/1811.02188](https://arxiv.org/abs/1811.02188)

2. Robert J. Moss, "POMDPStressTesting.jl: Adaptive Stress Testing for Black-Box Systems", _Journal of Open Source Software_, 2021. [https://joss.theoj.org/papers/10.21105/joss.02749](https://joss.theoj.org/papers/10.21105/joss.02749)

"""

# ╔═╡ ecb13e2c-d129-40cc-bf9e-ce8776525811
TableOfContents(title="Black-Box Validation", depth=4)

# ╔═╡ caeaa9b7-afa1-4fb1-8a11-491d32dd0fae
html"""
<script>
var section = 0;
var subsection = 0;
var headers = document.querySelectorAll('h2, h3');
for (var i=0; i < headers.length; i++) {
    var header = headers[i];
    var text = header.innerText;
    var original = header.getAttribute("text-original");
    if (original === null) {
        // Save original header text
        header.setAttribute("text-original", text);
    } else {
        // Replace with original text before adding section number
        text = header.getAttribute("text-original");
    }
    var numbering = "";
    switch (header.tagName) {
        case 'H2':
            section += 1;
            numbering = section + ".";
            subsection = 0;
            break;
        case 'H3':
            subsection += 1;
            numbering = section + "." + subsection;
            break;
    }
    header.innerText = numbering + " " + text;
};
</script>
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
POMDPStressTesting = "6fc570d8-62cd-4d35-b113-bbf3c1b8276a"
Parameters = "d96e819e-fc66-5662-9728-84c9c7592b0a"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
Distributions = "~0.23.11"
Latexify = "~0.15.6"
POMDPStressTesting = "~1.0.0"
Parameters = "~0.12.2"
PlutoUI = "~0.7.9"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "485ee0867925449198280d4af84bdb46a2a404d0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.0.1"

[[AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "f87e559f87a45bece9c9ed97458d3afe98b1ebb9"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.1.0"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[BFloat16s]]
deps = ["LinearAlgebra", "Test"]
git-tree-sha1 = "4af69e205efc343068dc8722b8dfec1ade89254a"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.1.0"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[BeliefUpdaters]]
deps = ["POMDPModelTools", "POMDPs", "Random", "Statistics", "StatsBase"]
git-tree-sha1 = "7d4f9d57116796ae3fc768d195386b0a42b4a58d"
uuid = "8bb6e9a1-7d73-552c-a44a-e5dc5634aac4"
version = "0.2.2"

[[CEnum]]
git-tree-sha1 = "215a9aa4a1f23fbd05b92769fdd62559488d70e9"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.1"

[[CPUTime]]
git-tree-sha1 = "2dcc50ea6a0a1ef6440d6eecd0fe3813e5671f45"
uuid = "a9c8d775-2e2e-55fc-8582-045d282d599e"
version = "1.0.0"

[[CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CompilerSupportLibraries_jll", "DataStructures", "ExprTools", "GPUArrays", "GPUCompiler", "LLVM", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "MacroTools", "Memoize", "NNlib", "Printf", "Random", "Reexport", "Requires", "SparseArrays", "Statistics", "TimerOutputs"]
git-tree-sha1 = "6893a46f357eabd44ce0fc1f9a264120a1a3a732"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "2.6.3"

[[ChainRules]]
deps = ["ChainRulesCore", "Compat", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "0902fc7f416c8f1e3b1e014786bb65d0c2241a9b"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "0.8.24"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "f53ca8d41e4753c41cdafa6ec5f7ce914b34be54"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "0.10.13"

[[CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[CommonRLInterface]]
deps = ["MacroTools"]
git-tree-sha1 = "21de56ebf28c262651e682f7fe614d44623dc087"
uuid = "d842c3ba-07a1-494f-bbec-f5741b0a3e98"
version = "0.3.1"

[[CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "4866e381721b30fac8dda4c8cb1d9db45c8d2994"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.37.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Crayons]]
git-tree-sha1 = "3f71217b538d7aaee0b69ab47d9b7724ca8afa0d"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.0.4"

[[CrossEntropyMethod]]
deps = ["Distributions", "POMDPPolicies", "POMDPSimulators", "POMDPs", "ProgressMeter", "Random"]
git-tree-sha1 = "4c9aceee852f1824b76ed0d75d411070cce17a0e"
uuid = "f7e364ee-db6c-46d3-b598-27dc284d9c4d"
version = "0.1.0"

[[D3Trees]]
deps = ["AbstractTrees", "Base64", "JSON", "Random", "Test"]
git-tree-sha1 = "311af855efa91a595940cd5c0cdb0ff9e8d6b948"
uuid = "e3df1716-f71e-5df9-9e2d-98e193103c45"
version = "0.3.1"

[[DataAPI]]
git-tree-sha1 = "bec2532f8adb82005476c141ec23e921fc20971b"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.8.0"

[[DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "d785f42445b63fc86caa08bb9a9351008be9b765"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.2.2"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[DiffRules]]
deps = ["NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "3ed8fa7178a10d1cd0f1ca524f249ba6937490c0"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.3.0"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "StaticArrays", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "501c11d708917ca09ce357bed163dbaf0f30229f"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.23.12"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[ExprTools]]
git-tree-sha1 = "b7e3d17636b348f005f11040025ae8c6f645fe92"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.6"

[[FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "3c041d2ac0a52a12a27af2782b34900d9c3ee68c"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.11.1"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays"]
git-tree-sha1 = "502b3de6039d5b78c76118423858d981349f3823"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.9.7"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Flux]]
deps = ["AbstractTrees", "Adapt", "CUDA", "CodecZlib", "Colors", "DelimitedFiles", "Functors", "Juno", "LinearAlgebra", "MacroTools", "NNlib", "Pkg", "Printf", "Random", "Reexport", "SHA", "Statistics", "StatsBase", "Test", "ZipFile", "Zygote"]
git-tree-sha1 = "c443bf5a8329573a68364106b2c29bb6938dc6f5"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.11.6"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "NaNMath", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "b5e930ac60b613ef3406da6d4f42c35d8dc51419"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.19"

[[Functors]]
deps = ["MacroTools"]
git-tree-sha1 = "f40adc6422f548176bb4351ebd29e4abf773040a"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.1.0"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GPUArrays]]
deps = ["AbstractFFTs", "Adapt", "LinearAlgebra", "Printf", "Random", "Serialization", "Statistics"]
git-tree-sha1 = "df5b8569904c5c10e84c640984cfff054b18c086"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "6.4.1"

[[GPUCompiler]]
deps = ["DataStructures", "ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "Scratch", "Serialization", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "ef2839b063e158672583b9c09d2cf4876a8d3d55"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "0.10.0"

[[IRTools]]
deps = ["InteractiveUtils", "MacroTools", "Test"]
git-tree-sha1 = "95215cd0076a150ef46ff7928892bc341864c73c"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.3"

[[Inflate]]
git-tree-sha1 = "f5fc07d4e706b84f72d54eedcc1c13d92fb0871c"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[IrrationalConstants]]
git-tree-sha1 = "f76424439413893a832026ca355fe273e93bce94"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.0"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLD2]]
deps = ["DataStructures", "FileIO", "MacroTools", "Mmap", "Pkg", "Printf", "Reexport", "TranscodingStreams", "UUIDs"]
git-tree-sha1 = "192934b3e2a94e897ce177423fd6cf7bdf464bce"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.14"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[Juno]]
deps = ["Base64", "Logging", "Media", "Profile"]
git-tree-sha1 = "07cb43290a840908a771552911a6274bc6c072c7"
uuid = "e5e0dc1b-0480-54bc-9374-aad01c23163d"
version = "0.8.4"

[[LLVM]]
deps = ["CEnum", "Libdl", "Printf", "Unicode"]
git-tree-sha1 = "f57ac3fd2045b50d3db081663837ac5b4096947e"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "3.9.0"

[[LaTeXStrings]]
git-tree-sha1 = "c7f1c695e06c01b95a67f0cd1d34994f3e7db104"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.2.1"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a4b12a1bd2ebade87891ab7e36fdbce582301a92"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.6"

[[LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[LightGraphs]]
deps = ["ArnoldiMethod", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "432428df5f360964040ed60418dd5601ecd240b6"
uuid = "093fc24a-ae57-5d10-9952-331d41423f4d"
version = "1.3.5"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "3d682c07e6dd250ed082f883dc88aee7996bf2cc"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.0"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MCTS]]
deps = ["CPUTime", "Colors", "D3Trees", "POMDPLinter", "POMDPModelTools", "POMDPPolicies", "POMDPSimulators", "POMDPs", "Printf", "ProgressMeter", "Random"]
git-tree-sha1 = "864a39c4136998e421c7be0743b9bcfc770037e5"
uuid = "e12ccd36-dcad-5f33-8774-9175229e7b33"
version = "0.4.7"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "5a5bc6bf062f0f95e62d0fe0a2d99699fed82dd9"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.8"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Media]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "75a54abd10709c01f1b86b84ec225d26e840ed58"
uuid = "e89f7d12-3494-54d1-8411-f7d8b9ae1f27"
version = "0.5.0"

[[Memoize]]
deps = ["MacroTools"]
git-tree-sha1 = "2b1dfcba103de714d31c033b5dacc2e4a12c7caa"
uuid = "c03570c3-d221-55d1-a50c-7939bbd78826"
version = "0.4.4"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NNlib]]
deps = ["Adapt", "ChainRulesCore", "Compat", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "5203a4532ad28c44f82c76634ad621d7c90abcbd"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.7.29"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NamedTupleTools]]
git-tree-sha1 = "63831dcea5e11db1c0925efe5ef5fc01d528c522"
uuid = "d9ec5142-1e00-5aa0-9d6a-321866360f50"
version = "0.13.7"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse", "Test"]
git-tree-sha1 = "95a4038d1011dfdbde7cecd2ad0ac411e53ab1bc"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.10.1"

[[POMDPLinter]]
deps = ["Logging"]
git-tree-sha1 = "cee5817d06f5e1a9054f3e1bbb50cbabae4cd5a5"
uuid = "f3bd98c0-eb40-45e2-9eb1-f2763262d755"
version = "0.1.1"

[[POMDPModelTools]]
deps = ["CommonRLInterface", "Distributions", "LinearAlgebra", "POMDPLinter", "POMDPs", "Random", "SparseArrays", "Statistics", "Tricks", "UnicodePlots"]
git-tree-sha1 = "be6e420779e4a076acac228aa68440ae7ce73331"
uuid = "08074719-1b2a-587c-a292-00f91cc44415"
version = "0.3.7"

[[POMDPPolicies]]
deps = ["BeliefUpdaters", "Distributions", "LinearAlgebra", "POMDPModelTools", "POMDPs", "Parameters", "Random", "SparseArrays", "StatsBase"]
git-tree-sha1 = "2920bc20706b82cf6c5058da51b1bb5d3c391a27"
uuid = "182e52fb-cfd0-5e46-8c26-fd0667c990f4"
version = "0.4.1"

[[POMDPSimulators]]
deps = ["BeliefUpdaters", "DataFrames", "Distributed", "NamedTupleTools", "POMDPLinter", "POMDPModelTools", "POMDPPolicies", "POMDPs", "ProgressMeter", "Random"]
git-tree-sha1 = "1c8a996d3b03023bdeb7589ad87231e73ba93e19"
uuid = "e0d0a172-29c6-5d4e-96d0-f262df5d01fd"
version = "0.3.12"

[[POMDPStressTesting]]
deps = ["CommonRLInterface", "CrossEntropyMethod", "D3Trees", "DataStructures", "Distributed", "Distributions", "Flux", "IterTools", "JLD2", "LinearAlgebra", "MCTS", "Markdown", "POMDPModelTools", "POMDPPolicies", "POMDPSimulators", "POMDPs", "Parameters", "ProgressMeter", "Random", "Requires", "Statistics", "Zygote"]
git-tree-sha1 = "a4c7f430785f57eb14f553e05de066e286e47f31"
uuid = "6fc570d8-62cd-4d35-b113-bbf3c1b8276a"
version = "1.0.0"

[[POMDPs]]
deps = ["Distributions", "LightGraphs", "NamedTupleTools", "POMDPLinter", "Pkg", "Random", "Statistics"]
git-tree-sha1 = "3a8f6cf6a3b7b499ec4294f2eb2b16b9dc8a7513"
uuid = "a93abf59-7444-517b-a68a-c42f96afdd7d"
version = "0.9.3"

[[Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "2276ac65f1e236e0a6ea70baff3f62ad4c625345"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.2"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "438d35d2d95ae2c5e8780b330592b6de8494e779"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.0.3"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlutoUI]]
deps = ["Base64", "Dates", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "Suppressor"]
git-tree-sha1 = "44e225d5837e2a2345e69a1d1e01ac2443ff9fcb"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.9"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a193d6ad9c45ada72c14b731a318bedd3c2f00cf"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.3.0"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "0d1245a357cc61c8cd61934c07447aa569ff22e6"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.1.0"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "afadeba63d90ff223a6a48d2009434ecee2ec9e8"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.1"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["OpenSpecFun_jll"]
git-tree-sha1 = "d8d8b8a9f4119829410ecd706da4cc8594a1e020"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "0.10.3"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "da4cf579416c81994afd6322365d00916c79b8ae"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "0.12.5"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8cbbc098554648c84f79a463c9ff0fd277144b6c"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.10"

[[StatsFuns]]
deps = ["IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "20d1bb720b9b27636280f751746ba4abb465f19d"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.9"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[Suppressor]]
git-tree-sha1 = "a819d77f31f83e5792a76081eee1ea6342ab8787"
uuid = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
version = "0.2.0"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "1162ce4a6c4b7e31e0e6b14486a6986951c73be9"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.5.2"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "209a8326c4f955e2442c07b56029e88bb48299c7"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.12"

[[TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[Tricks]]
git-tree-sha1 = "ae44af2ce751434f5fa52e23f46533b45f0cfd81"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.5"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[UnicodePlots]]
deps = ["Crayons", "Dates", "SparseArrays", "StatsBase"]
git-tree-sha1 = "dc9c7086d41783f14d215ea0ddcca8037a8691e9"
uuid = "b8865327-cd53-5732-bb35-84acbb429228"
version = "1.4.0"

[[ZipFile]]
deps = ["Libdl", "Printf", "Zlib_jll"]
git-tree-sha1 = "c3a5637e27e914a7a445b8d0ad063d701931e9f7"
uuid = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"
version = "0.9.3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "IRTools", "InteractiveUtils", "LinearAlgebra", "MacroTools", "NaNMath", "Random", "Requires", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "8b634fdb4c3c63f2ceaa2559a008da4f405af6b3"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.17"

[[ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "9e7a1e8ca60b742e508a315c17eef5211e7fbfd7"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.1"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─687f9f3c-3b30-4bbe-af96-60acba88096f
# ╠═7c1dc177-fdad-4486-8ddd-b1d13c2aebb6
# ╟─d43b5dac-6398-4c43-aaa6-3f7871ddfeaf
# ╟─9d5d5d1b-3cb5-4edb-99c9-3a3c6cc74382
# ╟─f9ada586-0c8e-4a71-821a-08018aa346e3
# ╟─7375c689-be3a-4750-8d66-9b8143ea8476
# ╟─0b818751-4310-47ab-bda3-01a30e3a0b98
# ╠═235193c1-7119-4a11-add4-8385e3d21d6b
# ╟─5a1b854e-84f1-44e5-9e33-ffad03424352
# ╠═287c600e-79ad-4789-971b-9dd49d37d64e
# ╟─330956a4-25f3-4358-8891-0205e7ef0185
# ╟─e256c0da-94db-4b09-b309-edd5d109b066
# ╠═cab4c1d3-48f4-4d1a-946a-22c63cde6813
# ╟─7595d65e-9363-4b94-a877-561a7063a63d
# ╠═117c5000-dc20-4e14-8f22-e0f48a9bebfa
# ╟─1a4b8f51-c1c6-4aba-bea3-36f4211c31ae
# ╠═f7d98d83-ed67-4ec2-b6b2-5477b0614670
# ╟─13f3f3dc-265a-498e-a3e4-0081d0361ca2
# ╠═546b5aa8-571b-4ab1-8b91-9eaf10cc7a61
# ╟─3ec0a905-3eca-421b-bac0-3664cf41644b
# ╟─a590f1b9-b6d1-45ad-aafa-982af06e6ebd
# ╠═e8eabe9f-cea2-44eb-ae98-e8d6cfba4010
# ╟─68f3d180-4469-41b9-bb6c-63043af21ead
# ╠═75890d95-b73d-4da2-84f6-6b18409d037d
# ╟─3f4e000d-6c67-4ebd-80dc-835ab74b25c8
# ╠═2faa29d0-f42d-44be-a3be-3436d5cef81d
# ╟─4af3a2ea-12eb-4afd-894a-bf9f09c38adc
# ╠═e2defb9c-a442-4939-83c8-9329e1da182e
# ╟─8ff29459-e714-4a0f-bc03-910f3d24eb3a
# ╠═cca7cd81-e9d2-4494-a710-bbd07dba6731
# ╟─5f6fe4db-e7a7-4389-aba9-9ec552faeef8
# ╠═9e1a32a6-32cf-438d-a616-e872fdbf795c
# ╟─917953d6-d7bb-4fe7-b4e5-f4c64b2d2bb2
# ╠═be02f6c7-03fc-40ab-b4d6-86312e88b714
# ╠═d26959aa-20ba-4943-9a0e-79ed9336d369
# ╟─aaf654fd-3478-4f6d-80af-c16e42881da3
# ╠═3679c0ff-142c-49d8-9dc6-3de0f8b96e67
# ╟─934c6cfc-1fce-4225-bf70-8544b1ac82c1
# ╠═8bc5116a-fe5e-4135-b341-c06900b5ff99
# ╟─ad1dba31-2b6c-4721-80f7-1841b734526a
# ╠═1d915b62-0281-424a-a317-1e855fbfe046
# ╠═815033b0-eb9c-4bba-b5c4-fe1d34796fa6
# ╠═d0e67875-8f94-402f-adb8-2440f7b9944a
# ╠═39f52e42-eb1e-4817-b842-45a695bdcc37
# ╠═53403ce4-0af1-4676-b3ab-8f8312590917
# ╟─5616126d-bcd8-4153-a6ff-24c2a2db1f9c
# ╟─a5fedf30-0be4-4221-bcfc-2df4ea7ff400
# ╠═43fe1ef2-075f-472d-ab17-ce5d558c7ef2
# ╟─92f0ab72-dcb6-4360-87f9-fa4569f75a21
# ╠═9bda5132-0a28-416d-83aa-94c8d7f0c28e
# ╠═5e7f390e-e1e9-4f1e-8bc3-f1c38c1bbf41
# ╟─70bc9acc-8638-40a7-862d-c1226fd6d53c
# ╠═ecb13e2c-d129-40cc-bf9e-ce8776525811
# ╟─caeaa9b7-afa1-4fb1-8a11-491d32dd0fae
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
