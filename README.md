
# Node Classification and Product Recommendation using GraphSAGE

Nurjahan, Nushrat Jahan Ria, Tanzila Tabassum | Spring 2025 | CSC 7700 Web Search

---

## Project Overview

This project applies a Graph Neural Network (GNN) model, **GraphSAGE**, to perform **node classification** on the large-scale **ogbn-products** dataset from the Open Graph Benchmark (OGB).

Key Achievements:
- Trained a scalable **3-layer GraphSAGE model** on a massive Amazon product co-purchase graph (2.4M nodes, 123M edges).
- Achieved **78% test accuracy** in classifying products into categories such as *Books*, *Beauty*, *Home & Kitchen*, etc.
- Built an interactive **Streamlit app** for live node prediction.
- Designed an extension toward **product recommendation** using learned node embeddings.

---

## Dataset: ogbn-products

- Nodes: Amazon products
- Edges: Co-purchasing relationships
- Node Features: 100-dimensional bag-of-words embeddings from product descriptions
- Labels: Product categories (e.g., Books, Beauty, Electronics)

---

## Model Architecture

- **GraphSAGE** with 3 layers
- Hidden dimension: 256
- Activation: ReLU after each hidden layer
- Dropout: 0.5 during training
- Loss: Cross-entropy on labeled nodes
- Neighbor sampling: [15, 10] at each layer for mini-batch training

---

## Training Strategy

- Used **NeighborLoader** for efficient mini-batch training.
- Trained on GPU to handle memory constraints.
- 20 epochs of training.
- Achieved strong generalization on validation and test splits.

---

## Streamlit App

An interactive Streamlit web application was built to:

- Input a **Node ID**.
- Predict and display the **Product Category**.
- Show **Feature Vector (First 10 Features)** of the node.
- Visualize Feature Vector using a small bar chart.

The app is designed for real-time demonstration of node classification results.

---

## Future Extension: Product Recommendation

The project lays the foundation for product recommendation by:

- Extracting **hidden 256-dimensional embeddings** after training.
- Computing **cosine similarity** between node embeddings.
- Recommending **top-5 similar products** based on learned features and graph structure.

This shows how **GraphSAGE embeddings** can be used not only for classification but also for **product recommendation systems**.

---

## Folder Structure

```
├── train.ipynb         # Training code for GraphSAGE
├── testing.ipynb       # Inference and evaluation
├── streamlit_app.py    # Streamlit app for live prediction
├── README.md           # Project documentation
├── GraphSAGE_Presentation_Nurjahan.pptx  # Final presentation slides
├── models/
│   └── graphsage_trained_model.pth   # Trained model weights
├── data/
│   └── products/ogbn_products/     # Dataset and label mapping
```

---

## How to Run

### 1. Train Model
```bash
python train.py
```

### 2. Launch Streamlit App
```bash
streamlit run streamlit_app.py
```

### 3. Extract Embeddings for Recommendation
Modify `inference()` method to output hidden vectors, and run inference.

---

## Key Results

| Metric | Value |
|:-------|:------|
| Test Accuracy | 78% |
| # of Nodes | 2,449,029 |
| # of Edges | 123,718,280 |
| # of Classes | 47 |

---

## Acknowledgements

- Open Graph Benchmark (OGB) Team
- PyTorch Geometric (PyG) Library
- Streamlit Team for rapid app development tools

---

## Author

PhD Student, Louisiana State University  
CSC 7700 Web Search, Spring 2025
