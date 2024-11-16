# **Awesome NeurIPS 2024 Graph Paper Collection**

This repo contains a comprehensive compilation of **graph and/or GNN** papers that were accepted at the [Thirty-Eighth Annual Conference on Neural Information Processing Systems 2024](https://neurips.cc/). Graph or Geometric machine learning possesses an indispensable role within the domain of machine learning research, providing invaluable insights, methodologies, and solutions to a diverse array of challenges and problems. 

**Short Overview**: We've got around 450-500 papers focusing on Graphs and GNNs in NeurIPS'24, almost 1.75-2x of ICML'24. Seems like diffusion, transformers, agents and knowledge graphs had a good focus in NeurIPS'24.


**Have a look and throw me a review (and, a star ⭐, maybe!)** Thanks!


---



## **All Topics:** 

<details open>
  <summary><b>View Topic list!</b></summary>

- [GNN Theories](#theories)
  - [Weisfeiler Leman](#Weisfeiler-Leman )
  - [Heterophily](#Heterophily)
  - [Hypergraph](#Hypergraph)
  - [Expressivity](#Expressivity)
  - [Generalization](#Generalization)
  - [Equivariant Graph Neural Networks](#Equivariant)
  - [Out-of-Distribution](#OOD)
  - [Diffusion](#Diffusion)
  - [Graph Matching](#GraphMatching)
  - [Flow Matching](#FlowMatching)
  - [Contrastive Learning](#ContrastiveLearning)
  - [Clustering](#Clustering)
  - [Foundational Models](#FM)
  - [Message Passing Neural Networks](#MPNN)
  - [Transformers](#GraphTransformers)
  - [Optimal Transport](#OptimalTransport)
  - [Graph Generation](#ggen)
  - [Unsupervised Learning](#UL)
  - [Meta-learning](#GraphMeta-learning)
  - [Disentanglement](#Disentanglement)
  - [Others](#GNNT-others)
- [GNNs for PDE/ODE/Physics](#PDE)
- [Graph and Large Language Models/Agents](#LLM)
- [Knowledge Graph and Knowledge Graph Embeddings](#KG)
- [GNN Applications](#apps)
- [Spatial and/or Temporal GNNs](#SpatialTemporalGNNs)
- [Explainable AI](#xai)
- [Reinforcement Learning](#rl)
- [Graphs, Molecules and Biology](#molecular)
- [GFlowNets](#GFlowNets)
- [Causal Discovery and Graphs](#Causal)
- [Federated Learning, Privacy, Decentralization](#FL)
- [Scene Graphs](#SceneGraphs)
- [Graphs, GNNs and Efficiency](#Efficiency)
- [Others](#Others)
- [More Possible Works](#more)
</details>


<a name="theories" />

## GNN Theories 

<a name="Weisfeiler-Leman " />

#### Weisfeiler Leman 
- [Weisfeiler and Leman Go Loopy: A New Hierarchy for Graph Representational Learning](https://openreview.net/pdf?id=9O2sVnEHor)

#### Heterophily
- [Unifying Homophily and Heterophily for Spectral Graph Neural Networks via Triple Filter Ensembles](https://openreview.net/pdf?id=uatPOPWzzU)
- [On the Impact of Feature Heterophily on Link Prediction with Graph Neural Networks](https://openreview.net/pdf?id=3LZHatxUa9)

#### Hypergraph
- [Slack-Free Spiking Neural Network Formulation for Hypergraph Minimum Vertex Cover](https://openreview.net/pdf?id=4A5IQEjG8c)
- [Semi-Open 3D Object Retrieval via Hierarchical Equilibrium on Hypergraph](https://openreview.net/pdf?id=A3jHvChR8K)
- [Ada-MSHyper: Adaptive Multi-Scale Hypergraph Transformer for Time Series Forecasting](https://openreview.net/pdf?id=RNbrIQ0se8)
- [Assembly Fuzzy Representation on Hypergraph for Open-Set 3D Object Retrieval](https://openreview.net/pdf?id=xOCAURlVM9)

#### Expressivity
- [On the Expressivity and Sample Complexity of Node-Individualized Graph Neural Networks](https://openreview.net/pdf?id=8APPypS0yN)
- [On the Expressive Power of Tree-Structured Probabilistic Circuits](https://openreview.net/pdf?id=suYAAOI5bd)


#### Generalization
- [Bridging OOD Detection and Generalization: A Graph-Theoretic View](https://openreview.net/pdf?id=qzwAG8qxI1)
- [Compositional PAC-Bayes: Generalization of GNNs with persistence and beyond](https://openreview.net/pdf?id=ZNcJtNN3e8)
- [Improving Generalization of Dynamic Graph Learning via Environment Prompt](https://openreview.net/pdf?id=RJG8ar4wHA)
- [Boosting Sample Efficiency and Generalization in Multi-agent Reinforcement Learning via Equivariance](https://openreview.net/pdf?id=MQIET1VfoV)
- [Topological Generalization Bounds for Discrete-Time Stochastic Optimization Algorithms](https://openreview.net/pdf?id=6U5fCHIWOC)


<a name="Equivariant" />

#### Equivariant Graph Neural Networks
- [A Flexible, Equivariant Framework for Subgraph GNNs via Graph Products and Graph Coarsening](https://openreview.net/pdf?id=9cFyqhjEHC)
- [Equivariant Blurring Diffusion for Hierarchical Molecular Conformer Generation](https://openreview.net/pdf?id=Aj0Zf28l6o)
- [Identifiability Guarantees for Causal Disentanglement from Purely Observational Data](https://openreview.net/pdf?id=M20p6tq9Hq)
- [Are High-Degree Representations Really Unnecessary in Equivariant Graph Neural Networks?](https://openreview.net/pdf?id=M0ncNVuGYN)
- [Equivariant spatio-hemispherical networks for diffusion MRI deconvolution](https://openreview.net/pdf?id=MxWpCherzD)
- [ET-Flow: Equivariant Flow-Matching for Molecular Conformer Generation](https://openreview.net/pdf?id=avsZ9OlR60)
- [Approximately Equivariant Neural Processes](https://openreview.net/pdf?id=dqT9MC5NQl)
- [Equivariant Machine Learning on Graphs with Nonlinear Spectral Filters](https://openreview.net/pdf?id=y8P633E5HQ)
- [Equivariant Neural Diffusion for Molecule Generation](https://openreview.net/pdf?id=40pE5pFhWl)


<a name="OOD" />

#### Out-of-Distribution
- [Reconstruct and Match: Out-of-Distribution Robustness via Topological Homogeneity](https://openreview.net/pdf?id=fkbMlfDBxm)
- [Revisiting Score Propagation in Graph Out-of-Distribution Detection](https://openreview.net/pdf?id=jb5qN3212b)
- [PURE: Prompt Evolution with Graph ODE for Out-of-distribution Fluid Dynamics Modeling](https://openreview.net/pdf?id=z86knmjoUq)

<a name="GraphMatching" />

#### Graph Matching
- [Efficient Graph Matching for Correlated Stochastic Block Models](https://openreview.net/pdf?id=nBhfIcDnRP)
- [Iteratively Refined Early Interaction Alignment for Subgraph Matching based Graph Retrieval](https://openreview.net/pdf?id=udTwwF7tks)


<a name="FlowMatching" />

#### Flow Matching
- [FlowLLM: Flow Matching for Material Generation with Large Language Models as Base Distributions](https://openreview.net/pdf?id=0bFXbEMz8e)
- [Fisher Flow Matching for Generative Modeling over Discrete Data](https://openreview.net/pdf?id=6jOScqwdHU)
- [Variational Flow Matching for Graph Generation](https://openreview.net/pdf?id=UahrHR5HQh)
- [Generalized Protein Pocket Generation with Prior-Informed Flow Matching](https://openreview.net/pdf?id=WyVTj77KEV)


<a name="ContrastiveLearning" />

#### Contrastive Learning
- [Embedding Dimension of Contrastive Learning and $k$-Nearest Neighbors](https://openreview.net/pdf?id=H0qu4moFly)
- [A probability contrastive learning framework for 3D molecular representation learning](https://openreview.net/pdf?id=HYiR6tGQPv)
- [How Molecules Impact Cells: Unlocking Contrastive PhenoMolecular Retrieval](https://openreview.net/pdf?id=LQBlSGeOGm)
- [Inference via Interpolation: Contrastive Representations Provably Enable Planning and Inference](https://openreview.net/pdf?id=PoCs4jq7cV)
- [Exploitation of a Latent Mechanism in Graph Contrastive Learning: Representation Scattering](https://openreview.net/pdf?id=R8SolCx62K)
- [FUG: Feature-Universal Graph Contrastive Pre-training for Graphs with Diverse Node Features](https://openreview.net/pdf?id=VUuOsBrqaw)
- [Unified Graph Augmentations for Generalized Contrastive Learning on Graphs](https://openreview.net/pdf?id=jgkKroLxeC)
- [Leveraging Contrastive Learning for Enhanced Node Representations in Tokenized Graph Transformers](https://openreview.net/pdf?id=u6FuiKzT1K)


### Diffusion
- [Pard: Permutation-Invariant Autoregressive Diffusion for Graph Generation](https://openreview.net/pdf?id=x4Kk4FxLs3)
- [Adapting Diffusion Models for Improved Prompt Compliance and Controllable Image Synthesis](https://openreview.net/pdf?id=sntv8Ac3U2)
- [MIDGArD: Modular Interpretable Diffusion over Graphs for Articulated Designs](https://openreview.net/pdf?id=re2jPCnzkA)
- [Unifying Generation and Prediction on Graphs with Latent Graph Diffusion](https://openreview.net/pdf?id=lvibangnAs)
- [SubgDiff: A Subgraph Diffusion Model to Improve Molecular Representation Learning](https://openreview.net/pdf?id=iSMTo0toDO)
- [DiffusionBlend: Learning 3D Image Prior through Position-aware Diffusion Score Blending for 3D Computed Tomography Reconstruction](https://openreview.net/pdf?id=h3Kv6sdTWO)
- [Diffusion Twigs with Loop Guidance for Conditional Graph Generation](https://openreview.net/pdf?id=fvOCJAAYLx)
- [Graph Diffusion Transformers for Multi-Conditional Molecular Generation](https://openreview.net/pdf?id=cfrDLD1wfO)
- [NaRCan: Natural Refined Canonical Image with Integration of Diffusion Prior for Video Editing](https://openreview.net/pdf?id=bCR2NLm1QW)
- [Differentially Private Graph Diffusion with Applications in Personalized PageRanks](https://openreview.net/pdf?id=aon7bwYBiq)
- [Learning-to-Cache: Accelerating Diffusion Transformer via Layer Caching](https://openreview.net/pdf?id=ZupoMzMNrO)
- [Discrete-state Continuous-time Diffusion for Graph Generation](https://openreview.net/pdf?id=YkSKZEhIYt)
- [Reprogramming Pretrained Target-Specific Diffusion Models for Dual-Target Drug Design](https://openreview.net/pdf?id=Y79L45D5ts)
- [Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion](https://openreview.net/pdf?id=RxkcroC8qP)
- [Geometric Trajectory Diffusion Models](https://openreview.net/pdf?id=OYmms5Mv9H)
- [DiffCut: Catalyzing Zero-Shot Semantic Segmentation with Diffusion Features and Recursive Normalized Cut](https://openreview.net/pdf?id=N0xNf9Qqmc)
- [DiffuBox: Refining 3D Object Detection with Point Diffusion](https://openreview.net/pdf?id=J2wOOtkBx0)
- [Aligning Target-Aware Molecule Diffusion Models with Exact Energy Optimization](https://openreview.net/pdf?id=EWcvxXtzNu)
- [Graph Diffusion Policy Optimization](https://openreview.net/pdf?id=8ohsbxw7q8)
- [Faster Local Solvers for Graph Diffusion Equations](https://openreview.net/pdf?id=3Z0LTDjIM0)

#### Clustering
- [From Dictionary to Tensor: A Scalable Multi-View Subspace Clustering Framework with Triple Information Enhancement](https://openreview.net/pdf?id=p4a1nSvwD7)
- [HC-GAE: The Hierarchical Cluster-based Graph Auto-Encoder for Graph Representation Learning](https://openreview.net/pdf?id=fx6aSBMu6z)
- [Graph Neural Networks Need Cluster-Normalize-Activate Modules](https://openreview.net/pdf?id=faj2EBhdHC)
- [DECRL: A Deep Evolutionary Clustering Jointed Temporal Knowledge Graph Representation Learning Approach](https://openreview.net/pdf?id=V42zfM2GXw)
- [Revisiting Self-Supervised Heterogeneous Graph Learning from Spectral Clustering Perspective](https://openreview.net/pdf?id=I6tRENM5Ya)
- [Clustering then Propagation: Select Better Anchors for Knowledge Graph Embedding](https://openreview.net/pdf?id=BpJ6OTfWw3)
- [Cluster-wise Graph Transformer with Dual-granularity Kernelized Attention](https://openreview.net/pdf?id=3j2nasmKkP)
- [TFGDA: Exploring Topology and Feature Alignment in Semi-supervised Graph Domain Adaptation through Robust Clustering](https://openreview.net/pdf?id=26BdXIY3ik)

<a name="MPNN" />

#### Message Passing Neural Networks
- [How Does Message Passing Improve Collaborative Filtering?](https://openreview.net/pdf?id=c78U5zi4eA)
- [Sequential Signal Mixing Aggregation for Message Passing Graph Neural Networks](https://openreview.net/pdf?id=aRokfUfIQs)
- [Pure Message Passing Can Estimate Common Neighbor for Link Prediction](https://openreview.net/pdf?id=Xa3dVaolKo)
- [Towards Dynamic Message Passing on Graphs](https://openreview.net/pdf?id=4BWlUJF0E9)

<a name="FM" />

#### Foundational Models
- [Cell ontology guided transcriptome foundation model](https://openreview.net/pdf?id=aeYNVtTo7o)
- [A Prompt-Based Knowledge Graph Foundation Model for Universal In-Context Reasoning](https://openreview.net/pdf?id=VQyb9LKmUH)
- [A Foundation Model for Zero-shot Logical Query Reasoning](https://openreview.net/pdf?id=JRSyMBBJi6)
- [MeshXL: Neural Coordinate Field for Generative 3D Foundation Models](https://openreview.net/pdf?id=Gcks157FI3)
- [GFT: Graph Foundation Model with Transferable Tree Vocabulary](https://openreview.net/pdf?id=0MXzbAv8xy)



<a name="GraphTransformers" />

#### Transformers
- [Supra-Laplacian Encoding for Transformer on Dynamic Graphs](https://openreview.net/pdf?id=vP9qAzr2Gw)
- [Interpretable Lightweight Transformer via Unrolling of Learned Graph Smoothness Priors](https://openreview.net/pdf?id=i8LoWBJf7j)
- [Long-range Brain Graph Transformer](https://openreview.net/pdf?id=fjLCqicn64)
- [Graph Convolutions Enrich the Self-Attention in Transformers!](https://openreview.net/pdf?id=ffNrpcBpi6)
- [Molecule Design by Latent Prompt Transformer](https://openreview.net/pdf?id=dg3tI3c2B1)
- [EGSST: Event-based Graph Spatiotemporal Sensitive Transformer for Object Detection](https://openreview.net/pdf?id=cknAewsBhD)
- [Graph Diffusion Transformers for Multi-Conditional Molecular Generation](https://openreview.net/pdf?id=cfrDLD1wfO)
- [CYCLO: Cyclic Graph Transformer Approach to Multi-Object Relationship Modeling in Aerial Videos](https://openreview.net/pdf?id=Zg4zs0l2iH)
- [Knowledge Circuits in Pretrained Transformers](https://openreview.net/pdf?id=YVXzZNxcag)
- [ProTransformer: Robustify Transformers via Plug-and-Play Paradigm](https://openreview.net/pdf?id=UkauUrTbxx)
- [Enhancing Graph Transformers with Hierarchical Distance Structural Encoding](https://openreview.net/pdf?id=U4KldRgoph)
- [Towards Principled Graph Transformers](https://openreview.net/pdf?id=LJCQH6U0pl)
- [Even Sparser Graph Transformers](https://openreview.net/pdf?id=K3k4bWuNnk)
- [Fast Tree-Field Integrators: From Low Displacement Rank to Topological Transformers](https://openreview.net/pdf?id=Eok6HbcSRI)
- [$\\textit{NeuroPath}$: A Neural Pathway Transformer for Joining the Dots of Human Connectomes](https://openreview.net/pdf?id=AvBuK8Ezrg)
- [Understanding Transformer Reasoning Capabilities via Graph Algorithms](https://openreview.net/pdf?id=AfzbDw6DSp)
- [Transformers need glasses! Information over-squashing in language tasks](https://openreview.net/pdf?id=93HCE8vTye)
- [Finding Transformer Circuits With Edge Pruning](https://openreview.net/pdf?id=8oSY3rA9jY)
- [ETO:Efficient Transformer-based Local Feature Matching by Organizing Multiple Homography Hypotheses](https://openreview.net/pdf?id=3xHCaDdYcc)
- [Cluster-wise Graph Transformer with Dual-granularity Kernelized Attention](https://openreview.net/pdf?id=3j2nasmKkP)


<a name="OptimalTransport" />

#### Optimal Transport
- [Any2Graph: Deep End-To-End Supervised Graph Prediction With An Optimal Transport Loss](https://openreview.net/pdf?id=tPgagXpvcV)
- [Low-Rank Optimal Transport through Factor Relaxation with Latent Coupling](https://openreview.net/pdf?id=hGgkdFF2hR)
- [Fairness in Social Influence Maximization via Optimal Transport](https://openreview.net/pdf?id=axW8xvQPkF)


<a name="ggen" />

#### Graph Generation
- [Scene Graph Generation with Role-Playing Large Language Models](https://openreview.net/pdf?id=xpRUi8amtC)
- [Pard: Permutation-Invariant Autoregressive Diffusion for Graph Generation](https://openreview.net/pdf?id=x4Kk4FxLs3)
- [Diffusion Twigs with Loop Guidance for Conditional Graph Generation](https://openreview.net/pdf?id=fvOCJAAYLx)
- [Discrete-state Continuous-time Diffusion for Graph Generation](https://openreview.net/pdf?id=YkSKZEhIYt)
- [FairWire: Fair Graph Generation](https://openreview.net/pdf?id=V0JvwCQlJe)
- [Adaptive Visual Scene Understanding: Incremental Scene Graph Generation](https://openreview.net/pdf?id=6lwKOvL3KN)


<a name="UL" />

#### Unsupervised Learning
- [Unsupervised Homography Estimation on Multimodal Image Pair via Alternating Optimization](https://openreview.net/pdf?id=zkhyrxlwqH)
- [Beyond Redundancy: Information-aware Unsupervised Multiplex Graph Structure Learning](https://openreview.net/pdf?id=xaqPAkJnAS)
- [Graph-based Unsupervised Disentangled Representation Learning via Multimodal Large Language Models](https://openreview.net/pdf?id=a1wf2N967T)



<a name="GraphMeta-learning" />

#### Graph Meta-learning
- [DisenGCD: A Meta Multigraph-assisted Disentangled Graph Learning Framework for  Cognitive Diagnosis](https://openreview.net/pdf?id=lJuQxkDbDo)
- [Long-range Meta-path Search on Large-scale Heterogeneous Graphs](https://openreview.net/pdf?id=hbOWLtJNMK)

#### Disentanglement
- [Scene Graph Disentanglement and Composition for Generalizable Complex Image Generation](https://openreview.net/pdf?id=zGN0YWy2he)



<a name="PDE" />

## GNNs for PDE/ODE/Physics
- [What is my quantum computer good for? Quantum capability learning with physics-aware neural networks](https://openreview.net/pdf?id=4cU9ZvOkBz)
- [Neural P$^3$M: A Long-Range Interaction Modeling Enhancer for Geometric GNNs](https://openreview.net/pdf?id=ncqauwSyl5)


<a name="LLM" />

## Graph and Large Language Models/Agents
- [Online Relational Inference for Evolving Multi-agent Interacting Systems](https://openreview.net/pdf?id=miO8odRzto)
- [Can Graph Learning Improve Planning in LLM-based Agents?](https://openreview.net/pdf?id=bmoS6Ggw4j)
- [UrbanKGent: A Unified Large Language Model Agent Framework for Urban Knowledge Graph Construction](https://openreview.net/pdf?id=Nycj81Z692)
- [Integrating Suboptimal Human Knowledge with Hierarchical Reinforcement Learning for Large-Scale Multiagent Systems](https://openreview.net/pdf?id=NGpMCH5q7Y)
- [Decision-Making Behavior Evaluation Framework for LLMs under Uncertain Context](https://openreview.net/pdf?id=re0ly2Ylcu)
- [GraphVis: Boosting LLMs with Visual Knowledge Graph Integration](https://openreview.net/pdf?id=haVPmN8UGi)
- [Ad Auctions for LLMs via Retrieval Augmented Generation](https://openreview.net/pdf?id=Ujo8V7iXmR)
- [Transcoders find interpretable LLM feature circuits](https://openreview.net/pdf?id=J6zHcScAo0)
- [SG-Nav: Online 3D Scene Graph Prompting for LLM-based Zero-shot Object Navigation](https://openreview.net/pdf?id=HmCmxbCpp2)
- [LLM Dataset Inference: Did you train on my dataset?](https://openreview.net/pdf?id=Fr9d1UMc37)
- [LLMs as Zero-shot Graph Learners: Alignment of GNN Representations with LLM Token Embeddings](https://openreview.net/pdf?id=32g9BWTndc)



<a name="KG" />

## Knowledge Graph and Knowledge Graph Embeddings
- [KG-FIT: Knowledge Graph Fine-Tuning Upon Open-World Knowledge](https://openreview.net/pdf?id=rDoPMODpki)
- [GraphVis: Boosting LLMs with Visual Knowledge Graph Integration](https://openreview.net/pdf?id=haVPmN8UGi)
- [Knowledge Graph Completion by Intermediate Variables Regularization](https://openreview.net/pdf?id=d226uyWYUo)
- [A Prompt-Based Knowledge Graph Foundation Model for Universal In-Context Reasoning](https://openreview.net/pdf?id=VQyb9LKmUH)
- [DECRL: A Deep Evolutionary Clustering Jointed Temporal Knowledge Graph Representation Learning Approach](https://openreview.net/pdf?id=V42zfM2GXw)
- [Text2NKG: Fine-Grained N-ary Relation Extraction for N-ary relational Knowledge Graph Construction](https://openreview.net/pdf?id=V2MBWYXp63)
- [KnowGPT: Knowledge Graph based Prompting for Large Language Models](https://openreview.net/pdf?id=PacBluO5m7)
- [UrbanKGent: A Unified Large Language Model Agent Framework for Urban Knowledge Graph Construction](https://openreview.net/pdf?id=Nycj81Z692)
- [Construction and Application of Materials Knowledge Graph in Multidisciplinary Materials Science via Large Language Model](https://openreview.net/pdf?id=GB5a0RRYuv)
- [Plan-on-Graph: Self-Correcting Adaptive Planning of Large Language Model on Knowledge Graphs](https://openreview.net/pdf?id=CwCUEr6wO5)
- [Clustering then Propagation: Select Better Anchors for Knowledge Graph Embedding](https://openreview.net/pdf?id=BpJ6OTfWw3)





<a name="SpatialTemporalGNNs" />

## Spatial and/or Temporal GNNs
- [Learning from Highly Sparse Spatio-temporal Data](https://openreview.net/pdf?id=rTONicCCJm)
- [EGSST: Event-based Graph Spatiotemporal Sensitive Transformer for Object Detection](https://openreview.net/pdf?id=cknAewsBhD)
- [A Motion-aware Spatio-temporal Graph for Video Salient Object Ranking](https://openreview.net/pdf?id=VUBtAcQN44)
- [DECRL: A Deep Evolutionary Clustering Jointed Temporal Knowledge Graph Representation Learning Approach](https://openreview.net/pdf?id=V42zfM2GXw)
- [State Space Models on Temporal Graphs: A First-Principles Study](https://openreview.net/pdf?id=UaJErAOssN)
- [Improving Temporal Link Prediction via Temporal Walk Matrix Projection](https://openreview.net/pdf?id=Ti3ciyqlS3)
- [Using Time-Aware Graph Neural Networks to Predict Temporal Centralities in Dynamic Graphs](https://openreview.net/pdf?id=6n709MszkP)
- [Temporal Graph Neural Tangent Kernel with Graphon-Guaranteed](https://openreview.net/pdf?id=266nH7kLSV)






<a name="apps" />

## GNN Applications
- [Differentially Private Graph Diffusion with Applications in Personalized PageRanks](https://openreview.net/pdf?id=aon7bwYBiq)
- [A Simple and Adaptive Learning Rate for FTRL in Online Learning with Minimax Regret of $\\Theta(T^{2/3})$ and its Application to Best-of-Both-Worlds](https://openreview.net/pdf?id=XlvUz9F50g)
- [PediatricsGPT: Large Language Models as Chinese Medical Assistants for Pediatric Applications](https://openreview.net/pdf?id=WvoKwq12x5)
- [Construction and Application of Materials Knowledge Graph in Multidisciplinary Materials Science via Large Language Model](https://openreview.net/pdf?id=GB5a0RRYuv)
- [A Textbook Remedy for Domain Shifts: Knowledge Priors for Medical Image Analysis](https://openreview.net/pdf?id=STrpbhrvt3)
- [HEALNet: Multimodal Fusion for Heterogeneous Biomedical Data](https://openreview.net/pdf?id=HUxtJcQpDS)
- [Knowledge-Empowered Dynamic Graph Network for Irregularly Sampled Medical Time Series](https://openreview.net/pdf?id=9hCn01VAdC)


<a name="xai" />

## Explainable AI
- [Enhancing Robustness of Graph Neural Networks on Social Media with Explainable Inverse Reinforcement Learning](https://openreview.net/pdf?id=ziehA15y8k)
- [A hierarchical decomposition for explaining ML performance discrepancies](https://openreview.net/pdf?id=nXXwYsARXB)
- [RegExplainer: Generating Explanations for Graph Neural Networks in Regression Tasks](https://openreview.net/pdf?id=ejWvCpLuwu)
- [MIDGArD: Modular Interpretable Diffusion over Graphs for Articulated Designs](https://openreview.net/pdf?id=re2jPCnzkA)
- [Interpretable Lightweight Transformer via Unrolling of Learned Graph Smoothness Priors](https://openreview.net/pdf?id=i8LoWBJf7j)
- [GraphTrail: Translating GNN Predictions into Human-Interpretable Logical Rules](https://openreview.net/pdf?id=fzlMza6dRZ)
- [Transcoders find interpretable LLM feature circuits](https://openreview.net/pdf?id=J6zHcScAo0)


<a name="rl" />

## Reinforcement Learning
- [Enhancing Robustness of Graph Neural Networks on Social Media with Explainable Inverse Reinforcement Learning](https://openreview.net/pdf?id=ziehA15y8k)
- [FlexPlanner: Flexible 3D Floorplanning via Deep Reinforcement Learning in Hybrid Action Space with Multi-Modality Representation](https://openreview.net/pdf?id=q9RLsvYOB3)
- [Optimizing Automatic Differentiation with Deep Reinforcement Learning](https://openreview.net/pdf?id=hVmi98a0ki)
- [On the Role of Information Structure in Reinforcement Learning for Partially-Observable Sequential Teams and Games](https://openreview.net/pdf?id=QgMC8ftbNd)
- [Integrating Suboptimal Human Knowledge with Hierarchical Reinforcement Learning for Large-Scale Multiagent Systems](https://openreview.net/pdf?id=NGpMCH5q7Y)
- [Federated Natural Policy Gradient and Actor Critic Methods for Multi-task Reinforcement Learning](https://openreview.net/pdf?id=DUFD6vsyF8)
- [Enhancing Chess Reinforcement Learning with Graph Representation](https://openreview.net/pdf?id=97OvPgmjRN)
- [Amortized Active Causal Induction with Deep Reinforcement Learning](https://openreview.net/pdf?id=7AXY27kdNH)
- [Compositional Automata Embeddings for Goal-Conditioned Reinforcement Learning](https://openreview.net/pdf?id=6KDZHgrDhG)




<a name="molecular" />

## Graphs and Molecules
- [TurboHopp: Accelerated Molecule Scaffold Hopping with Consistency Models](https://openreview.net/pdf?id=lBh5kuuY1L)
- [Conditional Synthesis of 3D Molecules with Time Correction Sampler](https://openreview.net/pdf?id=gipFTlvfF1)
- [Molecule Design by Latent Prompt Transformer](https://openreview.net/pdf?id=dg3tI3c2B1)
- [UniIF: Unified Molecule Inverse Folding](https://openreview.net/pdf?id=clqX9cVDKV)
- [QVAE-Mole: The Quantum VAE with Spherical Latent Variable Learning for 3-D Molecule Generation](https://openreview.net/pdf?id=RqvesBxqDo)
- [Aligning Target-Aware Molecule Diffusion Models with Exact Energy Optimization](https://openreview.net/pdf?id=EWcvxXtzNu)
- [Score-based 3D molecule generation with neural fields](https://openreview.net/pdf?id=9lGJrkqJUw)
- [Molecule Generation with Fragment Retrieval Augmentation](https://openreview.net/pdf?id=56Q0qggDlp)
- [FlexSBDD: Structure-Based Drug Design with Flexible Protein Modeling](https://openreview.net/pdf?id=4AB54h21qG)
- [ProtGO: Function-Guided Protein Modeling for Unified Representation Learning](https://openreview.net/pdf?id=0oUutV92YF)
- [Learning Complete Protein Representation by Dynamically Coupling of Sequence and Structure](https://openreview.net/pdf?id=0e5uOaJxo1)



<a name="GFlowNets" />

## **GFlowNets**
- [RGFN: Synthesizable Molecular Generation Using GFlowNets](https://openreview.net/pdf?id=hpvJwmzEHX)
- [Genetic-guided GFlowNets for Sample Efficient Molecular Optimization](https://openreview.net/pdf?id=B4q98aAZwt)



<a name="Causal" />

## Causal Discovery and Graphs
- [Causal Discovery from Event Sequences by Local Cause-Effect Attribution](https://openreview.net/pdf?id=y9zIRxshzj)
- [Hybrid Top-Down Global Causal Discovery with Local Search for Linear and Nonlinear Additive Noise Models](https://openreview.net/pdf?id=xnmm1jThkv)
- [Conditional Generative Models are Sufficient to Sample from Any Causal  Effect Estimand](https://openreview.net/pdf?id=vymkuBMLlh)
- [Partial Structure Discovery is Sufficient for No-regret Learning in Causal Bandits](https://openreview.net/pdf?id=uM3rQ14iex)
- [Disentangled Representation Learning in Non-Markovian Causal Systems](https://openreview.net/pdf?id=uLGyoBn7hm)
- [On Causal Discovery in the Presence of Deterministic Relations](https://openreview.net/pdf?id=pfvcsgFrJ6)
- [Learning the Latent Causal Structure for Modeling Label Noise](https://openreview.net/pdf?id=nJKfNiEBvq)
- [A Simple yet Scalable Granger Causal Structural Learning Approach for Topological Event Sequences](https://openreview.net/pdf?id=mP084aMFsd)
- [Interventional Causal Discovery in a Mixture of DAGs](https://openreview.net/pdf?id=mFrlCI8sov)
- [Identifying General Mechanism Shifts in Linear Causal Representations](https://openreview.net/pdf?id=jWaXhCYTV1)
- [Causal Effect Identification in a Sub-Population with Latent Variables](https://openreview.net/pdf?id=iEsyRsg6t1)
- [Learning Linear Causal Representations from General Environments: Identifiability and Intrinsic Ambiguity](https://openreview.net/pdf?id=dB99jjwx3h)
- [Causal discovery with endogenous context variables](https://openreview.net/pdf?id=cU8d7LeOyx)
- [On the Complexity of Identification in Linear Structural Causal Models](https://openreview.net/pdf?id=bNDwOoxj6W)
- [Learning Mixtures of Unknown Causal Interventions](https://openreview.net/pdf?id=aC9mB1PqYJ)
- [Sample Complexity of Interventional Causal Representation Learning](https://openreview.net/pdf?id=XL9aaXl0u6)
- [Sample Efficient Bayesian Learning of Causal Graphs from Interventions](https://openreview.net/pdf?id=RfSvAom7sS)
- [Linear Causal Bandits: Unknown Graph and Soft Interventions](https://openreview.net/pdf?id=PAu0W5YAKC)
- [Identifying Causal Effects Under Functional Dependencies](https://openreview.net/pdf?id=OIsUWQSvkD)
- [A Local Method for Satisfying Interventional Fairness with Partially Known Causal Graphs](https://openreview.net/pdf?id=NhyDfZXjQX)
- [Consistency of Neural Causal Partial Identification](https://openreview.net/pdf?id=GEbnPxD9EF)
- [On the Parameter Identifiability of Partially Observed Linear Causal Models](https://openreview.net/pdf?id=EQZlEfjrkV)
- [QWO: Speeding Up Permutation-Based Causal Discovery in LiGAMs](https://openreview.net/pdf?id=BptJGaPn9C)
- [Amortized Active Causal Induction with Deep Reinforcement Learning](https://openreview.net/pdf?id=7AXY27kdNH)
- [Complete Graphical Criterion for Sequential Covariate Adjustment in Causal Inference](https://openreview.net/pdf?id=6gIcnPvw2x)
- [CausalStock: Deep End-to-end Causal Discovery for News-driven Multi-stock Movement Prediction](https://openreview.net/pdf?id=5BXXoJh0Vr)



<a name="FL" />

## Federated Learning, Privacy, Decentralization
- [FedNE: Surrogate-Assisted Federated Neighbor Embedding for Dimensionality Reduction](https://openreview.net/pdf?id=zBMKodNgKX)
- [FedGMark: Certifiably Robust Watermarking for Federated Graph Learning](https://openreview.net/pdf?id=xeviQPXTMU)
- [Federated Graph Learning for Cross-Domain Recommendation](https://openreview.net/pdf?id=UBpPOqrBKE)
- [FedSSP: Federated Graph Learning with Spectral Knowledge and Personalized Preference](https://openreview.net/pdf?id=I96GFYalFO)
- [Federated Natural Policy Gradient and Actor Critic Methods for Multi-task Reinforcement Learning](https://openreview.net/pdf?id=DUFD6vsyF8)
- [On provable privacy vulnerabilities of graph representations](https://openreview.net/pdf?id=LSqDcfX3xU)



<a name="SceneGraphs" />

## Scene Graphs
- [Scene Graph Disentanglement and Composition for Generalizable Complex Image Generation](https://openreview.net/pdf?id=zGN0YWy2he)
- [Scene Graph Generation with Role-Playing Large Language Models](https://openreview.net/pdf?id=xpRUi8amtC)
- [SG-Nav: Online 3D Scene Graph Prompting for LLM-based Zero-shot Object Navigation](https://openreview.net/pdf?id=HmCmxbCpp2)
- [Adaptive Visual Scene Understanding: Incremental Scene Graph Generation](https://openreview.net/pdf?id=6lwKOvL3KN)
- [Multiview Scene Graph](https://openreview.net/pdf?id=1ELFGSNBGC)


<a name="Efficiency" />

## Graphs, GNNs and Efficiency
- [Cost-efficient Knowledge-based Question Answering with Large Language Models](https://openreview.net/pdf?id=pje1Y71jad)
- [An Efficient Memory Module for Graph Few-Shot Class-Incremental Learning](https://openreview.net/pdf?id=dqdffX3BS5)
- [Sample Efficient Bayesian Learning of Causal Graphs from Interventions](https://openreview.net/pdf?id=RfSvAom7sS)
- [Efficient Policy Evaluation Across Multiple Different Experimental Datasets](https://openreview.net/pdf?id=PSubtZAitM)
- [Can Graph Neural Networks Expose Training Data Properties? An Efficient Risk Assessment Approach](https://openreview.net/pdf?id=Luxk3z1tSG)
- [Stochastic Taylor Derivative Estimator: Efficient amortization for arbitrary differential operators](https://openreview.net/pdf?id=J2wI2rCG2u)
- [Efficient Streaming Algorithms for Graphlet Sampling](https://openreview.net/pdf?id=EC9Hfi9V3k)
- [Genetic-guided GFlowNets for Sample Efficient Molecular Optimization](https://openreview.net/pdf?id=B4q98aAZwt)
- [Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction](https://openreview.net/pdf?id=859DtlwnAD)
- [Private Edge Density Estimation for Random Graphs: Optimal, Efficient and Robust](https://openreview.net/pdf?id=4NQ24cHnOi)
- [ETO:Efficient Transformer-based Local Feature Matching by Organizing Multiple Homography Hypotheses](https://openreview.net/pdf?id=3xHCaDdYcc)
- [Gaussian Graph Network: Learning Efficient and Generalizable Gaussian Representations from Multi-view Images](https://openreview.net/pdf?id=2dfBpyqh0A)
- [GDeR: Safeguarding Efficiency, Balancing, and Robustness via Prototypical Graph Pruning](https://openreview.net/pdf?id=O97BzlN9Wh)

# Others
- [Graph Edit Distance with General Costs Using Neural Set Divergence](https://openreview.net/pdf?id=u7JRmrGutT)
- [What Matters in Graph Class Incremental Learning? An Information Preservation Perspective](https://openreview.net/pdf?id=tJGX7tpGO8)
- [Generative Modelling of Structurally Constrained Graphs](https://openreview.net/pdf?id=A3hxp0EeNW)
- [Intruding with Words: Towards Understanding Graph Injection Attacks at the Text Level](https://openreview.net/pdf?id=oTzydUKWpq)
- [Boosting Graph Pooling with Persistent Homology](https://openreview.net/pdf?id=WcmqdY2AKu)
- [GaussianCut: Interactive segmentation via graph cut for 3D Gaussian Splatting](https://openreview.net/pdf?id=Ns0LQokxa5)
- [ARC: A Generalist Graph Anomaly Detector with In-Context Learning](https://openreview.net/pdf?id=IdIVfzjPK4)
- [Leveraging Tumor Heterogeneity: Heterogeneous Graph Representation Learning for Cancer Survival Prediction in Whole Slide Images](https://openreview.net/pdf?id=tsIKrvexBd)
- [Road Network Representation Learning with the Third Law of  Geography](https://openreview.net/pdf?id=gPtiGRaVcE)
- [Bayesian Optimization of Functions over Node Subsets in Graphs](https://openreview.net/pdf?id=KxjGi1krBi)
- [IF-Font: Ideographic Description Sequence-Following Font Generation](https://openreview.net/pdf?id=ciwOcmo8CC)
- [Almost Surely Asymptotically Constant Graph Neural Networks](https://openreview.net/pdf?id=Dn68qdfTry)
- [Distributed-Order Fractional Graph Operating Network](https://openreview.net/pdf?id=kEQFjKqiqM)
- [DiGRAF: Diffeomorphic Graph-Adaptive Activation Function](https://openreview.net/pdf?id=ZZoW4Z3le4)
- [Generalizing CNNs to graphs with learnable neighborhood quantization](https://openreview.net/pdf?id=dYIqAZXQNV)
- [InstructG2I: Synthesizing Images from Multimodal Attributed Graphs](https://openreview.net/pdf?id=zWnW4zqkuM)
- [On the Scalability of GNNs for Molecular Graphs](https://openreview.net/pdf?id=klqhrq7fvB)
- [Spiking Graph Neural Network on Riemannian Manifolds](https://openreview.net/pdf?id=VKt0K3iOmO)
- [Rethinking Reconstruction-based Graph-Level Anomaly Detection: Limitations and a Simple Remedy](https://openreview.net/pdf?id=e2INndPINB)
- [DeTikZify: Synthesizing Graphics Programs for Scientific Figures and Sketches with TikZ](https://openreview.net/pdf?id=bcVLFQCOjc)
- [Idiographic Personality Gaussian Process for Psychological Assessment](https://openreview.net/pdf?id=Twqa0GFMGX)
- [What do Graph Neural Networks learn? Insights from Tropical Geometry](https://openreview.net/pdf?id=Oy2x0Xfx0u)
- [Accelerating Non-Maximum Suppression: A Graph Theory Perspective](https://openreview.net/pdf?id=0lau89u4oE)
- [Cryptographic Hardness of Score Estimation](https://openreview.net/pdf?id=URQXbwM0Md)
- [Graph Neural Networks and Arithmetic Circuits](https://openreview.net/pdf?id=0ZeONp33f0)
- [Neural P$^3$M: A Long-Range Interaction Modeling Enhancer for Geometric GNNs](https://openreview.net/pdf?id=ncqauwSyl5)
- [Probabilistic Graph Rewiring via Virtual Nodes](https://openreview.net/pdf?id=LpvSHL9lcK)
- [Fast Graph Sharpness-Aware Minimization for Enhancing and Accelerating Few-Shot Node Classification](https://openreview.net/pdf?id=AF32GbuupC)
- [SpelsNet: Surface Primitive Elements Segmentation by B-Rep Graph Structure Supervision](https://openreview.net/pdf?id=Ad3PzTuqIq)
- [Differentiable Task Graph Learning: Procedural Activity Representation and Online Mistake Detection from Egocentric Videos](https://openreview.net/pdf?id=2HvgvB4aWq)
- [Exploring Consistency in Graph Representations: from Graph Kernels to Graph Neural Networks](https://openreview.net/pdf?id=dg0hO4M11K)
- [Empowering Active Learning for 3D Molecular Graphs with Geometric Graph Isomorphism](https://openreview.net/pdf?id=He2GCHeRML)
- [UniGAD: Unifying Multi-level Graph Anomaly Detection](https://openreview.net/pdf?id=sRILMnkkQd)
- [GraphCroc: Cross-Correlation Autoencoder for Graph Structural Reconstruction](https://openreview.net/pdf?id=zn6s6VQYb0)
- [Visual Data Diagnosis and Debiasing with Concept Graphs](https://openreview.net/pdf?id=XNGsx3WCU9)
- [Deep Graph Mating](https://openreview.net/pdf?id=m4NI2yIwJA)
- [Energy-based Epistemic Uncertainty for Graph Neural Networks](https://openreview.net/pdf?id=6vNPPtWH1Q)
- [Integrating GNN and Neural ODEs for Estimating Non-Reciprocal Two-Body Interactions in Mixed-Species Collective Motion](https://openreview.net/pdf?id=qwl3EiDi9r)
- [Hamba: Single-view 3D Hand Reconstruction with Graph-guided Bi-Scanning Mamba](https://openreview.net/pdf?id=pCJ0l1JVUX)
- [Similarity-Navigated Conformal Prediction for Graph Neural Networks](https://openreview.net/pdf?id=iBZSOh027z)
- [Navigable Graphs for High-Dimensional Nearest Neighbor Search: Constructions and Limits](https://openreview.net/pdf?id=7flSQgZ4RT)
- [Graph Classification via Reference Distribution Learning: Theory and Practice](https://openreview.net/pdf?id=1zVinhehks)
- [Continuous Product Graph Neural Networks](https://openreview.net/pdf?id=XRNN9i1xpi)
- [Uncovering the Redundancy in Graph Self-supervised Learning Models](https://openreview.net/pdf?id=7Ntft3U7jj)
- [Logical characterizations of recurrent graph neural networks with reals and floats](https://openreview.net/pdf?id=atDcnWqG5n)
- [Graph Neural Flows for Unveiling Systemic Interactions Among Irregularly Sampled Time Series](https://openreview.net/pdf?id=tFB5SsabVb)
- [DropEdge not Foolproof: Effective Augmentation Method for Signed Graph Neural Networks](https://openreview.net/pdf?id=CDe2zBPioj)
- [DistrictNet: Decision-aware learning for geographical districting](https://openreview.net/pdf?id=njwYBFau8E)
- [Are Your Models Still Fair? Fairness Attacks on Graph Neural Networks via Node Injections](https://openreview.net/pdf?id=LuqrIkGuru)
- [Probabilistic Weather Forecasting with Hierarchical Graph Neural Networks](https://openreview.net/pdf?id=wTIzpqX121)
- [Unitary Convolutions for Learning on Graphs and Groups](https://openreview.net/pdf?id=lG1VEQJvUH)
- [GRANOLA: Adaptive Normalization for Graph Neural Networks](https://openreview.net/pdf?id=qd8blc0o0F)
- [Mind the Graph When Balancing Data for Fairness or Robustness](https://openreview.net/pdf?id=LQR22jM5l3)
- [Graph Structure Inference with BAM: Neural Dependency Processing via Bilinear Attention](https://openreview.net/pdf?id=3ADBiWNUBb)
- [Tracing Hyperparameter Dependencies for Model Parsing via Learnable Graph Pooling Network](https://openreview.net/pdf?id=Y1edWJH9qB)
- [Analysis of Corrected Graph Convolutions](https://openreview.net/pdf?id=MSsQDWUWpd)
- [GITA: Graph to Visual and Textual Integration for Vision-Language Graph Reasoning](https://openreview.net/pdf?id=SaodQ13jga)
- [Microstructures and Accuracy of Graph Recall by Large Language Models](https://openreview.net/pdf?id=tNhwg9U767)
- [Fair GLASSO: Estimating Fair Graphical Models with Unbiased Statistical Behavior](https://openreview.net/pdf?id=a3cauWMXNV)
- [Continuous Partitioning for Graph-Based Semi-Supervised Learning](https://openreview.net/pdf?id=hCOuip5Ona)
- [On Neural Networks as Infinite Tree-Structured Probabilistic Graphical Models](https://openreview.net/pdf?id=KcmhSrHzJB)
- [What Is Missing For Graph Homophily? Disentangling Graph Homophily For Graph Neural Networks](https://openreview.net/pdf?id=GmdGEF8xxU)
- [Gradient Rewiring for Editable Graph Neural Network Training](https://openreview.net/pdf?id=XY2qrq7cXM)
- [Graph Learning for Numeric Planning](https://openreview.net/pdf?id=Wxc6KvQgLq)
- [Towards Harmless Rawlsian Fairness Regardless of Demographic Prior](https://openreview.net/pdf?id=7U5MwUS3Rw)
- [DFA-GNN: Forward Learning of Graph Neural Networks by Direct Feedback Alignment](https://openreview.net/pdf?id=hKVTwQQu76)
- [Dissecting the Failure of Invariant Learning on Graphs](https://openreview.net/pdf?id=7eFS8aZHAM)
- [Replay-and-Forget-Free Graph Class-Incremental Learning: A Task Profiling and Prompting Approach](https://openreview.net/pdf?id=FXdMgfCDer)
- [Aligning Embeddings and Geometric Random Graphs: Informational Results and Computational Approaches for the Procrustes-Wasserstein Problem](https://openreview.net/pdf?id=4NGlu45uyt)
- [Spatio-Spectral Graph Neural Networks](https://openreview.net/pdf?id=Cb3kcwYBgw)
- [Motion Graph Unleashed: A Novel Approach to Video Prediction](https://openreview.net/pdf?id=4ztP4PujOG)
- [Bridge the Points: Graph-based Few-shot Segment Anything Semantically](https://openreview.net/pdf?id=jYypS5VIPj)
- [Deep Graph Neural Networks via Posteriori-Sampling-based Node-Adaptative Residual Module](https://openreview.net/pdf?id=VywZsAGhp0)
- [RAGraph: A General Retrieval-Augmented Graph Learning Framework](https://openreview.net/pdf?id=Dzk2cRUFMt)
- [Learning on Large Graphs using Intersecting Communities](https://openreview.net/pdf?id=pGR5X4e1gy)
- [DARG: Dynamic Evaluation of Large Language Models via Adaptive Reasoning Graph](https://openreview.net/pdf?id=5IFeCNA7zR)
- [UGC: Universal Graph Coarsening](https://openreview.net/pdf?id=nN6NSd1Qds)
- [Theoretical and Empirical Insights into the Origins of Degree Bias in Graph Neural Networks](https://openreview.net/pdf?id=1mAaewThcz)
- [LLaMo: Large Language Model-based Molecular Graph Assistant](https://openreview.net/pdf?id=WKTNdU155n)
- [Are Graph Neural Networks Optimal Approximation Algorithms?](https://openreview.net/pdf?id=SxRblm9aMs)
- [Spectral Graph Pruning Against Over-Squashing and Over-Smoothing](https://openreview.net/pdf?id=EMkrwJY2de)
- [The Map Equation Goes Neural: Mapping Network Flows with Graph Neural Networks](https://openreview.net/pdf?id=aFWx1N84Fe)
- [Graph Coarsening with Message-Passing Guarantees](https://openreview.net/pdf?id=rIOTceoNc8)
- [CSPG: Crossing Sparse Proximity Graphs for Approximate Nearest Neighbor Search](https://openreview.net/pdf?id=ohvXBIPV7e)
- [Dynamic Rescaling for Training GNNs](https://openreview.net/pdf?id=IfZwSRpqHl)
- [An End-To-End Graph Attention Network Hashing for Cross-Modal Retrieval](https://openreview.net/pdf?id=Q4QUCN2ioc)
- [EGODE: An Event-attended Graph ODE Framework for Modeling Rigid Dynamics](https://openreview.net/pdf?id=js5vZtyoIQ)
- [Robust Offline Active Learning on Graphs](https://openreview.net/pdf?id=MDsl1ifiNS)
- [Active design of two-photon holographic stimulation for identifying neural population dynamics](https://openreview.net/pdf?id=nLQeE8QGGe)
- [Automated Label Unification for Multi-Dataset Semantic Segmentation with GNNs](https://openreview.net/pdf?id=gSGLkCX9sc)
- [G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering](https://openreview.net/pdf?id=MPJ3oXtTZl)
- [The Intelligible and Effective Graph Neural Additive Network](https://openreview.net/pdf?id=SKY1ScUTwA)
- [DeepITE: Designing Variational Graph Autoencoders for Intervention Target Estimation](https://openreview.net/pdf?id=GMsi9966DR)
- [A Structure-Aware Framework for Learning Device Placements on Computation Graphs](https://openreview.net/pdf?id=Kzno1r3Xef)
- [Fairness-Aware Estimation of Graphical Models](https://openreview.net/pdf?id=WvWS8goWyR)
- [HGDL: Heterogeneous Graph Label Distribution Learning](https://openreview.net/pdf?id=OwguhIAh8R)
- [A Topology-aware Graph Coarsening Framework for Continual Graph Learning](https://openreview.net/pdf?id=VpINEEVLX0)
- [Robust Graph Neural Networks via Unbiased Aggregation](https://openreview.net/pdf?id=dz6ex9Ee0Q)
- [Challenges of Generating Structurally Diverse Graphs](https://openreview.net/pdf?id=bbGPoL1NLo)
- [Mixture of Link Predictors on Graphs](https://openreview.net/pdf?id=X3oeoyJlMw)
- [Regression under demographic parity constraints via unlabeled post-processing](https://openreview.net/pdf?id=UtbjD5LGnC)
- [Graph neural networks and non-commuting operators](https://openreview.net/pdf?id=6aJrEC28hR)
- [Graph-based Uncertainty Metrics for Long-form Language Model Generations](https://openreview.net/pdf?id=YgJPQW0lkO)
- [Graph-enhanced Optimizers for Structure-aware Recommendation Embedding Evolution](https://openreview.net/pdf?id=55zLbH7dE1)
- [Learning Plaintext-Ciphertext Cryptographic Problems via ANF-based SAT Instance Representation](https://openreview.net/pdf?id=FzwAQJK4CG)
- [Customized Subgraph Selection and Encoding for Drug-drug Interaction Prediction](https://openreview.net/pdf?id=crlvDzDPgM)
- [GDeR: Safeguarding Efficiency, Balancing, and Robustness via Prototypical Graph Pruning](https://openreview.net/pdf?id=O97BzlN9Wh)
- [FUGAL: Feature-fortified Unrestricted Graph Alignment](https://openreview.net/pdf?id=SdLOs1FR4h)
- [Graphcode: Learning from multiparameter persistent homology using graph neural networks](https://openreview.net/pdf?id=O23XfTnhWR)
- [Stochastic contextual bandits with graph feedback: from independence number to MAS number](https://openreview.net/pdf?id=t8iosEWoyd)
- [Generative Semi-supervised Graph Anomaly Detection](https://openreview.net/pdf?id=zqLAMwVLkt)
- [GraphMorph: Tubular Structure Extraction by Morphing Predicted Graphs](https://openreview.net/pdf?id=hW5QWiCctl)
- [Non-convolutional graph neural networks.](https://openreview.net/pdf?id=JDAQwysFOc)
- [Linear Uncertainty Quantification of Graphical Model Inference](https://openreview.net/pdf?id=XOVks7JHQA)
- [Graph Neural Networks Do Not Always Oversmooth](https://openreview.net/pdf?id=nY7fGtsspU)
- [Schur Nets: exploiting local structure for equivariance in higher order graph neural networks](https://openreview.net/pdf?id=HRnSVflpgt)







<a name="more" />

## More Possible Works
*(Needs Verification Yet)*

- [Markov Equivalence and Consistency in Differentiable Structure Learning](https://openreview.net/pdf?id=TMlGQw7EbC)
- [SpatialRGPT: Grounded Spatial Reasoning in Vision-Language Models](https://openreview.net/pdf?id=JKEIYQUSUc)
- [On the Robustness of Spectral Algorithms for Semirandom Stochastic Block Models](https://openreview.net/pdf?id=kLen1XyW6P)
- [Domain Adaptation for Large-Vocabulary Object Detectors](https://openreview.net/pdf?id=deZpmEfmTo)
- [Tackling Uncertain Correspondences for Multi-Modal Entity Alignment](https://openreview.net/pdf?id=IAse6CAG26)
- [Learning rigid-body simulators over implicit shapes for large-scale scenes and vision](https://openreview.net/pdf?id=QDYts5dYgq)
- [Combining Observational Data and Language for Species Range Estimation](https://openreview.net/pdf?id=IOKLUxB05h)
- [GS-Hider: Hiding Messages into 3D Gaussian Splatting](https://openreview.net/pdf?id=3XLQp2Xx3J)
- [Buffer of Thoughts: Thought-Augmented Reasoning with Large Language Models](https://openreview.net/pdf?id=ANO1i9JPtb)
- [Instance-Optimal Private Density Estimation in the Wasserstein Distance](https://openreview.net/pdf?id=Apq6corvfZ)
- [DDGS-CT: Direction-Disentangled Gaussian Splatting for Realistic Volume Rendering](https://openreview.net/pdf?id=mY0ZnS2s9u)
- [Evaluating the World Model Implicit in a Generative Model](https://openreview.net/pdf?id=aVK4JFpegy)
- [Double-Ended Synthesis Planning with Goal-Constrained Bidirectional Search](https://openreview.net/pdf?id=LJNqVIKSCr)
- [Fair Wasserstein Coresets](https://openreview.net/pdf?id=ylceJ2xIw5)
- [Director3D: Real-world Camera Trajectory and 3D Scene Generation from Text](https://openreview.net/pdf?id=08A6X7FSTs)
- [Delving into the Reversal Curse: How Far Can Large Language Models Generalize?](https://openreview.net/pdf?id=1wxFznQWhp)
- [DreamMesh4D: Video-to-4D Generation with Sparse-Controlled Gaussian-Mesh Hybrid Representation](https://openreview.net/pdf?id=6ZwJSk2kvU)
- [Posture-Informed Muscular Force Learning for Robust Hand Pressure Estimation](https://openreview.net/pdf?id=LtS7pP8rEn)
- [EGonc : Energy-based Open-Set Node Classification with substitute Unknowns](https://openreview.net/pdf?id=3cL2XDyaEB)
- [Practical Shuffle Coding](https://openreview.net/pdf?id=m2DaXpCoIi)
- [Gene-Gene Relationship Modeling Based on Genetic Evidence for Single-Cell RNA-Seq Data Imputation](https://openreview.net/pdf?id=gW0znG5JCG)
- [Divide-and-Conquer Predictive Coding: a structured Bayesian inference algorithm](https://openreview.net/pdf?id=dxwIaCVkWU)
- [Semi-Random Matrix Completion via Flow-Based Adaptive Reweighting](https://openreview.net/pdf?id=XZp1uP0hh2)
- [Divergences between Language Models and Human Brains](https://openreview.net/pdf?id=DpP5F3UfKw)
- [Expected Probabilistic Hierarchies](https://openreview.net/pdf?id=fMdrBucZnj)
- [If You Want to Be Robust, Be Wary of Initialization](https://openreview.net/pdf?id=nxumYwxJPB)
- [Accelerating ERM for data-driven algorithm design using output-sensitive techniques](https://openreview.net/pdf?id=yW3tlSwusb)
- [Towards Flexible Visual Relationship Segmentation](https://openreview.net/pdf?id=kJkp2ECJT7)
- [MKGL: Mastery of a Three-Word Language](https://openreview.net/pdf?id=eqMNwXvOqn)
- [GSDF: 3DGS Meets SDF for Improved Neural Rendering and Reconstruction](https://openreview.net/pdf?id=r6V7EjANUK)
- [Make-it-Real: Unleashing Large Multimodal Model for Painting 3D Objects with Realistic Materials](https://openreview.net/pdf?id=88rbNOtAez)
- [Strategic Littlestone Dimension: Improved Bounds on Online Strategic Classification](https://openreview.net/pdf?id=4Lkzghiep1)
- [Non-Euclidean Mixture Model for Social Network Embedding](https://openreview.net/pdf?id=nuZv2iTlvn)
- [EEG2Video: Towards Decoding Dynamic Visual Perception from EEG Signals](https://openreview.net/pdf?id=RfsfRn9OFd)
- [PageRank Bandits for Link Prediction](https://openreview.net/pdf?id=VSz9na5Jtl)
- [Deep Homomorphism Networks](https://openreview.net/pdf?id=KXUijdMFdG)
- [HardCore Generation: Generating Hard UNSAT Problems for Data Augmentation](https://openreview.net/pdf?id=njvPjG0BfK)
- [Geometry Awakening: Cross-Geometry Learning Exhibits Superiority over Individual Structures](https://openreview.net/pdf?id=347aDObXEa)
- [Sequential Harmful Shift Detection Without Labels](https://openreview.net/pdf?id=jps9KkuSD3)
- [Dynamic 3D Gaussian Fields for Urban Areas](https://openreview.net/pdf?id=xZxXNhndXU)
- [Learning to Solve Quadratic Unconstrained Binary Optimization in a Classification Way](https://openreview.net/pdf?id=p43ObIwJFW)
- [Amortized Eigendecomposition for Neural Networks](https://openreview.net/pdf?id=OYOkkqRLvj)
- [Transferable Boltzmann Generators](https://openreview.net/pdf?id=AYq6GxxrrY)
- [Upping the Game: How 2D U-Net Skip Connections Flip 3D Segmentation](https://openreview.net/pdf?id=QI1ScdeQjp)
- [UniAR: A Unified model for predicting human Attention and Responses on visual content](https://openreview.net/pdf?id=FjssnGuHih)
- [Scaling Continuous Latent Variable Models as Probabilistic Integral Circuits](https://openreview.net/pdf?id=Ke40kfOT2E)
- [Visual Sketchpad: Sketching as a Visual Chain of Thought for Multimodal Language Models](https://openreview.net/pdf?id=GNSMl1P5VR)
- [Deep Equilibrium Algorithmic Reasoning](https://openreview.net/pdf?id=SuLxkxCENa)
- [Semantic Routing via Autoregressive Modeling](https://openreview.net/pdf?id=JvlrUFJMbI)
- [Learning Representations for Hierarchies with Minimal Support](https://openreview.net/pdf?id=HFS800reZK)
- [From an Image to a Scene: Learning to Imagine the World from a Million 360 Videos](https://openreview.net/pdf?id=otxOtsWCMb)
- [DMNet: Self-comparison Driven Model for Subject-independent Seizure Detection](https://openreview.net/pdf?id=mlmTxJwVsb)
- [Enhancing Robustness of Last Layer Two-Stage Fair Model Corrections](https://openreview.net/pdf?id=ChnJ3W4HFG)
- [Hardness of Learning Neural Networks under the Manifold Hypothesis](https://openreview.net/pdf?id=dkkgKzMni7)
- [TopoFR: A Closer Look at Topology Alignment on Face Recognition](https://openreview.net/pdf?id=KVAx5tys2p)
- [IntraMix: Intra-Class Mixup Generation for Accurate Labels and Neighbors](https://openreview.net/pdf?id=0SRJBtTNhX)
- [Neural decoding from stereotactic EEG: accounting for electrode variability across subjects](https://openreview.net/pdf?id=LR1nnsD7H0)
- [UniMTS: Unified Pre-training for Motion Time Series](https://openreview.net/pdf?id=DpByqSbdhI)
- [HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models](https://openreview.net/pdf?id=hkujvAPVsg)
- [eXponential FAmily Dynamical Systems (XFADS): Large-scale nonlinear Gaussian state-space modeling](https://openreview.net/pdf?id=Ln8ogihZ2S)
- [Breaking the curse of dimensionality in structured density estimation](https://openreview.net/pdf?id=dWwin2uGYE)
- [Who Evaluates the Evaluations? Objectively Scoring Text-to-Image Prompt Coherence Metrics with T2IScoreScore (TS2)](https://openreview.net/pdf?id=S4YRCLbUK1)
- [GLinSAT: The General Linear Satisfiability Neural Network Layer By Accelerated Gradient Descent](https://openreview.net/pdf?id=m1PVjNHvtP)
- [The Factorization Curse: Which Tokens You Predict Underlie the Reversal Curse and More](https://openreview.net/pdf?id=f70e6YYFHF)
- [Synergistic Dual Spatial-aware Generation of Image-to-text and Text-to-image](https://openreview.net/pdf?id=YOUh3lgRYI)
- [Improving Robustness of 3D Point Cloud Recognition from a Fourier Perspective](https://openreview.net/pdf?id=4jn7KWPHSD)
- [On Differentially Private U Statistics](https://openreview.net/pdf?id=zApFYcLg6K)
- [Towards Estimating Bounds on the Effect of Policies under Unobserved Confounding](https://openreview.net/pdf?id=u5enPCwaLt)
- [Generative Hierarchical Materials Search](https://openreview.net/pdf?id=PsPR4NOiRC)
- [ChatCam: Empowering Camera Control through Conversational AI](https://openreview.net/pdf?id=IxazPgGF8h)
- [Injecting Undetectable Backdoors in Obfuscated Neural Networks and Language Models](https://openreview.net/pdf?id=KyVBzkConO)
- [Unified Insights: Harnessing Multi-modal Data for Phenotype Imputation via View Decoupling](https://openreview.net/pdf?id=8B3sAX889P)
- [Du-IN: Discrete units-guided mask modeling for decoding speech from Intracranial Neural signals](https://openreview.net/pdf?id=uyLtEFnpQP)
- [Estimating Epistemic and Aleatoric Uncertainty with a Single Model](https://openreview.net/pdf?id=WPxa6OcIdg)
- [NeuralSteiner: Learning Steiner Tree for Overflow-avoiding Global Routing in Chip Design](https://openreview.net/pdf?id=oEKFPSOWpp)
- [Consensus Learning with Deep Sets for Essential Matrix Estimation](https://openreview.net/pdf?id=6sIOBDwr6d)
- [Towards Effective Planning Strategies for Dynamic Opinion Networks](https://openreview.net/pdf?id=LYivxMp5es)
- [Learning Low-Rank Feature for Thorax Disease Classification](https://openreview.net/pdf?id=GkzrVxs9LS)
- [Smoothie: Label Free Language Model Routing](https://openreview.net/pdf?id=pPSWHsgqRp)
- [SpeAr: A Spectral Approach for Zero-Shot Node Classification](https://openreview.net/pdf?id=eU87jJyEK5)
- [A robust inlier identification algorithm for point cloud registration via $\\mathbf{\\ell_0}$-minimization](https://openreview.net/pdf?id=BJrBaLoDRJ)
- [Metric Space Magnitude for Evaluating the Diversity of Latent Representations](https://openreview.net/pdf?id=glgZZAfssH)
- [The Importance of Being Scalable: Improving the Speed and Accuracy of Neural Network Interatomic Potentials Across Chemical Domains](https://openreview.net/pdf?id=Y4mBaZu4vy)
- [Neural Network Reparametrization for Accelerated Optimization in Molecular Simulations](https://openreview.net/pdf?id=FwxOHl0BEl)
- [Transfer Learning for Latent Variable Network Models](https://openreview.net/pdf?id=PK8xOCBQRO)
- [Testing Calibration in Nearly-Linear Time](https://openreview.net/pdf?id=01XV5Za56k)
- [Collaborative Cognitive Diagnosis with Disentangled Representation Learning for Learner Modeling](https://openreview.net/pdf?id=JxlQ2pbyzS)
- [Relational Concept Bottleneck Models](https://openreview.net/pdf?id=G99BSV9pt5)
- [Navigating Chemical Space with Latent Flows](https://openreview.net/pdf?id=aAaV4ZbQ9j)
- [Post-Hoc Reversal: Are We Selecting Models Prematurely?](https://openreview.net/pdf?id=3R7Go6WkDm)
- [Geodesic Optimization for Predictive Shift Adaptation on EEG data](https://openreview.net/pdf?id=qTypwXvNJa)
- [Neural Pfaffians: Solving Many Many-Electron Schrodinger Equations](https://openreview.net/pdf?id=HRkniCWM3E)
- [Differentiable Structure Learning with Partial Orders](https://openreview.net/pdf?id=B2cTLakrhV)
- [Taming the Long Tail in Human Mobility Prediction](https://openreview.net/pdf?id=wT2TIfHKp8)
- [CLIP in Mirror: Disentangling text from visual images through reflection](https://openreview.net/pdf?id=FYm8coxdiR)
- [Multilingual Diversity Improves Vision-Language Representations](https://openreview.net/pdf?id=1WtEqReCyS)
- [Expert-level protocol translation for self-driving labs](https://openreview.net/pdf?id=qXidsICaja)
- [G3: An Effective and Adaptive Framework for Worldwide Geolocalization Using Large Multi-Modality Models](https://openreview.net/pdf?id=21tn63ee15)
- [Learning Discrete Concepts in Latent Hierarchical Models](https://openreview.net/pdf?id=bO5bUxvH6m)
- [MeMo: Meaningful, Modular Controllers via Noise Injection](https://openreview.net/pdf?id=5DJBBACqim)
- [Edit Distance Robust Watermarks via Indexing Pseudorandom Codes](https://openreview.net/pdf?id=FZ45kf5pIA)
- [Wild-GS: Real-Time Novel View Synthesis from Unconstrained Photo Collections](https://openreview.net/pdf?id=Ss7l98DVvD)
- [Generative Forests](https://openreview.net/pdf?id=cRlQHncjwT)
- [bit2bit: 1-bit quanta video reconstruction via self-supervised photon prediction](https://openreview.net/pdf?id=HtlfNbyfOn)
- [Inversion-based Latent Bayesian Optimization](https://openreview.net/pdf?id=TrN5TcWY87)
- [ST$_k$: A Scalable Module for Solving Top-k Problems](https://openreview.net/pdf?id=OdJKB9jSa5)
- [DEL: Discrete Element Learner for Learning 3D Particle Dynamics with Neural Rendering](https://openreview.net/pdf?id=2nvkD0sPOk)
- [Counterfactual Fairness by Combining Factual and Counterfactual Predictions](https://openreview.net/pdf?id=J0Itri0UiN)
- [AUCSeg: AUC-oriented Pixel-level Long-tail Semantic Segmentation](https://openreview.net/pdf?id=ekK26cW5TB)
- [Iterative Methods via Locally Evolving Set Process](https://openreview.net/pdf?id=wT2KhEb97a)
- [Can Models Learn Skill Composition from Examples?](https://openreview.net/pdf?id=1sLdprsbmk)
- [Exponential Quantum Communication Advantage in Distributed Inference and Learning](https://openreview.net/pdf?id=gGR9dJbe3r)
- [DeiSAM: Segment Anything with Deictic Prompting](https://openreview.net/pdf?id=cmSNX47aEH)
- [Group Robust Preference Optimization in Reward-free RLHF](https://openreview.net/pdf?id=PRAsjrmXXK)
- [Harnessing Multiple Correlated Networks for Exact Community Recovery](https://openreview.net/pdf?id=7Fzx3Akdt5)
- [Why the Metric Backbone Preserves Community Structure](https://openreview.net/pdf?id=Kx8I0rP7w2)
- [MambaTree: Tree Topology is All You Need in State Space Model](https://openreview.net/pdf?id=W8rFsaKr4m)
- [On the Optimal Time Complexities in Decentralized Stochastic Asynchronous Optimization](https://openreview.net/pdf?id=IXRa8adMHX)
- [Invariant Tokenization of Crystalline Materials for Language Model Enabled Generation](https://openreview.net/pdf?id=18FGRNd0wZ)
- [Energy-Based Modelling for Discrete and Mixed Data via Heat Equations on Structured Spaces](https://openreview.net/pdf?id=wAqdvcK1Fv)
- [Information Re-Organization Improves Reasoning in Large Language Models](https://openreview.net/pdf?id=SciWuYPNG0)
- [Qualitative Mechanism Independence](https://openreview.net/pdf?id=RE5LSV8QYH)
- [Persistent Homology for High-dimensional Data Based on Spectral Methods](https://openreview.net/pdf?id=ARV1gJSOzV)
- [Fairness without Harm: An Influence-Guided Active Sampling Approach](https://openreview.net/pdf?id=YYJojVBCcd)
- [Extracting Training Data from Molecular Pre-trained Models](https://openreview.net/pdf?id=cV4fcjcwmz)
- [Lambda: Learning Matchable Prior For Entity Alignment with Unlabeled Dangling Cases](https://openreview.net/pdf?id=AWFryOJaGi)
- [LuSh-NeRF: Lighting up and Sharpening NeRFs for Low-light Scenes](https://openreview.net/pdf?id=CcmHlE6N6u)
- [On the Computational Landscape of Replicable Learning](https://openreview.net/pdf?id=1PCsDNG6Jg)
- [What type of inference is planning?](https://openreview.net/pdf?id=TXsRGrzICz)
- [Shape analysis for time series](https://openreview.net/pdf?id=JM0IQSliol)
- [realSEUDO for real-time calcium imaging analysis](https://openreview.net/pdf?id=Ye0O4Nyn21)
- [Normal-GS: 3D Gaussian Splatting with Normal-Involved Rendering](https://openreview.net/pdf?id=kngLs5H6l1)
- [DRACO: A Denoising-Reconstruction Autoencoder for Cryo-EM](https://openreview.net/pdf?id=u1mNGLYN74)
- [Exploring Molecular Pretraining Model at Scale](https://openreview.net/pdf?id=64V40K2fDv)
- [FactorizePhys: Matrix Factorization for Multidimensional Attention in Remote Physiological Sensing](https://openreview.net/pdf?id=qrfp4eeZ47)
- [Entity Alignment with Noisy Annotations from Large Language Models](https://openreview.net/pdf?id=qfCQ54ZTX1)
- [End-to-End Ontology Learning with Large Language Models](https://openreview.net/pdf?id=UqvEHAnCJC)
- [Questioning the Survey Responses of Large Language Models](https://openreview.net/pdf?id=Oo7dlLgqQX)
- [Mixture of neural fields for heterogeneous reconstruction in cryo-EM](https://openreview.net/pdf?id=TuspoNzIdB)
- [TreeVI: Reparameterizable Tree-structured Variational Inference for Instance-level Correlation Capturing](https://openreview.net/pdf?id=YjZ6fQAvT7)
- [SpGesture: Source-Free Domain-adaptive sEMG-based Gesture Recognition with Jaccard Attentive Spiking Neural Network](https://openreview.net/pdf?id=GYqs5Z4joA)
- [Latent Intrinsics Emerge from Training to Relight](https://openreview.net/pdf?id=ltnDg0EzF9)
- [Fractal Patterns May Illuminate the Success of Next-Token Prediction](https://openreview.net/pdf?id=clAFYReaYE)
- [Large language model validity via enhanced conformal prediction methods](https://openreview.net/pdf?id=JD3NYpeQ3R)
- [Effective Rank Analysis and Regularization for Enhanced 3D Gaussian Splatting](https://openreview.net/pdf?id=EwWpAPzcay)



---


**Missing any paper?**
If any paper is absent from the list, please feel free to [mail](mailto:azminetoushik.wasi@gmail.com) or [open an issue](https://github.com/azminewasi/Awesome-Graph-Research-NeurIPS2024/issues/new/choose) or submit a pull request. I'll gladly add that! Also, If I mis-categorized, please knock!

---

## More Collectons:
- [Awesome **NeurIPS'24** ***Molecular ML*** Paper Collection](https://github.com/azminewasi/Awesome-MoML-NeurIPS24)
- [**Awesome ICML 2024 Graph Paper Collection**](https://github.com/azminewasi/Awesome-Graph-Research-ICML2024)
- [**Awesome ICLR 2024 Graph Paper Collection**](https://github.com/azminewasi/Awesome-Graph-Research-ICLR2024)
- [**Awesome-LLMs-ICLR-24**](https://github.com/azminewasi/Awesome-LLMs-ICLR-24/)

---

## ✨ **Credits**
**Azmine Toushik Wasi**

 [![website](https://img.shields.io/badge/-Website-blue?style=flat-square&logo=rss&color=1f1f15)](https://azminewasi.github.io) 
 [![linkedin](https://img.shields.io/badge/LinkedIn-%320beff?style=flat-square&logo=linkedin&color=1f1f18)](https://www.linkedin.com/in/azmine-toushik-wasi/) 
 [![kaggle](https://img.shields.io/badge/Kaggle-%2320beff?style=flat-square&logo=kaggle&color=1f1f1f)](https://www.kaggle.com/azminetoushikwasi) 
 [![google-scholar](https://img.shields.io/badge/Google%20Scholar-%2320beff?style=flat-square&logo=google-scholar&color=1f1f18)](https://scholar.google.com/citations?user=X3gRvogAAAAJ&hl=en) 
 [![facebook](https://img.shields.io/badge/Facebook-%2320beff?style=flat-square&logo=facebook&color=1f1f15)](https://www.facebook.com/cholche.gari.zatrabari/)
