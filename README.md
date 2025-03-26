# horseracing-DGNN
This paper presents a novel approach to predicting horse race outcomes by modeling historical performance data using a dynamic graph neural network (DyGNN). Horses are represented as nodes in a temporal graph, where edges correspond to races, and node states evolve dynamically as horses participate in successive races. Each horse’s embedding is constructed by concatenating three components: (1) a learned state vector updated via an LSTM after each race, (2) features of the horse at the time of the race (e.g., age, breed), and (3) dynamic race-specific conditions (e.g., track surface, distance). These embeddings are processed through convolutional graph neural network (CGNN) layers to model interactions between horses in a race, enabling two key tasks:

1) Rank prediction: Estimating the finishing position of each horse.

2) State propagation: Updating the horse’s hidden state via the LSTM to capture performance trends. The race results are concatenated the the LSTM's input, in order to propagate the race results.

To address computational challenges inherent in dynamic graphs, we propose a snapshot-based training framework where only participating horses and their recent history are loaded for each race. However, since the model’s computational graph grows quadratically (O($n^2$)) with the number of historical races (due to comparisons), we introduce two optimizations:

1) Selective gradient checkpointing: Reducing memory overhead by recomputing intermediate states during backpropagation.

2) Computation graph partitioning: Isolating subgraphs by strategically detaching embeddings from earlier races, thereby truncating backward passes and stabilizing training time.

Although this partitioning limits long-term dependency learning, empirical results demonstrate that the truncated gradients still enable effective state propagation across races. Preliminary experiments shows the training time per epoch stabilized at O(1) after partitioning. While the project is ongoing, this work highlights the potential of DyGNNs for modeling sequential, interaction-heavy systems like competitive sports. 
