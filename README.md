Causal Evidence for LSD-induced Hierarchical Network Disintegration

This repository contains the source code for our study investigating the causal hierarchical reorganization of the brain under lysergic acid diethylamide (LSD). Using a personalized surrogate brain modeling framework, we inferred whole-brain effective connectivity (EC) to identify the causal drivers and targets underlying the psychedelic state.

🚀 Analysis Pipeline & Code Structure

The analysis pipeline is divided into four main modules: Surrogate Brain Modeling, Causal Inference, Statistical Analysis & Classification, and Topological Pattern Analysis.

Part 1: Surrogate Brain Modeling

TrainANN_MLP_model_hcp.py
Trains the initial, group-level surrogate brain model (baseline dynamical prior) utilizing the Human Connectome Project (HCP) normative dataset.

individualized_lsd_model_noTrainTest_test.py
Fine-tunes the pre-trained group-level model to generate individualized, state-specific surrogate brain models for each participant under both LSD and placebo conditions.

Part 2: Causal Inference (Effective Connectivity)

EC_infer.py
Infers the whole-brain Effective Connectivity (EC) matrices. This is achieved by systematically applying virtual perturbations to the input signals of each brain region within the individualized surrogate models and quantifying the directed causal responses.

Part 3: Statistical Analysis & Machine Learning

EC_ttest.py
Performs element-wise paired-sample t-tests on the inferred EC matrices to identify statistically significant causal connectivity alterations between the LSD and placebo states.

featureSelectionClassify_region.py
Conducts binary machine learning classification (LSD vs. Placebo). It evaluates the discriminative power of each specific brain region by independently training classifiers on their afferent (inflow/input) and efferent (outflow/output) EC profiles.

Part 4: Topological & Distribution Analysis

analyze_ec_distribution.py
Examines the topological distribution of the absolute EC strengths to verify the presence of a long-tailed distribution. It fits the empirical EC data to four candidate distributions (log-normal, normal, exponential, and inverse Gaussian) and determines the optimal mathematical model.

analyze_connection_patterns.py
Analyzes the spatial organization principles of the inferred connectome. Specifically, it evaluates whether excitatory and inhibitory connections predominantly occur within or between functional resting-state networks (intra-/inter-network) and hemispheres (intra-/inter-hemispheric).
