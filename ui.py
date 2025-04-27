import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
from torch_geometric.nn import SAGEConv
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.loader import NeighborLoader

# -------------------------------
# Define GraphSAGE model
# -------------------------------
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3):
        super(GraphSAGE, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.convs[-1](x, edge_index)
        return x

# -------------------------------
# Load model, data, and label mapping
# -------------------------------
@st.cache_resource
def load_model_and_data():
    dataset = PygNodePropPredDataset(name='ogbn-products', root='home/data/products')
    data = dataset[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GraphSAGE(
        in_channels=data.x.size(-1),
        hidden_channels=256,
        out_channels=dataset.num_classes,
        num_layers=3
    ).to(device)

    model.load_state_dict(torch.load('graphsage_trained_model.pth', map_location=device))
    model.eval()

    return model, data, device

@st.cache_data
def load_label_mapping():
    label_map_path = 'home/data/products/ogbn_products/mapping/labelidx2productcategory.csv.gz'
    df = pd.read_csv(label_map_path)
    label_mapping = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))  # safer: by column position
    return label_mapping

# -------------------------------
# Streamlit App Layout
# -------------------------------
st.title("🚀 GraphSAGE Node Classifier on OGBN-Products")

st.write("""
Predict the product category label for a given Node ID using a pre-trained GraphSAGE model!
""")

# Load everything
model, data, device = load_model_and_data()
label_mapping = load_label_mapping()

# User input
node_id = st.number_input(
    label="Enter Node ID (0 to {})".format(data.num_nodes - 1),
    min_value=0,
    max_value=int(data.num_nodes - 1),
    value=0,
    step=1
)

if st.button("Predict Label"):
    with torch.no_grad():
        infer_loader = NeighborLoader(
            data,
            input_nodes=torch.tensor([node_id]),
            num_neighbors=[15, 10],
            batch_size=1,
            shuffle=False
        )

        batch = next(iter(infer_loader))
        batch = batch.to(device)

        out = model(batch.x, batch.edge_index)
        pred_label = out[0].argmax(dim=-1).item()

        # Lookup human-readable category
        category_name = label_mapping.get(pred_label, "Unknown Category")

        st.success(f"🎯 Predicted Label for Node {node_id}: {pred_label} ({category_name})")



# Footer
st.caption("Model: 3-layer GraphSAGE | Dataset: ogbn-products | Test Accuracy: ~78%")
