import torch
import torch.nn.functional as F
from torch_geometric.datasets import WikiCS
from torch_geometric.transforms import NormalizeFeatures, RandomLinkSplit
from torch_geometric.nn import GCNConv, GAE, VGAE
from torch_geometric.utils import negative_sampling, to_networkx
from sklearn.metrics import roc_auc_score, average_precision_score
import networkx as nx

# ======================= 1. Завантаження та розділення =======================
dataset = WikiCS(root='data/WikiCS', transform=NormalizeFeatures(), is_undirected=True)
data = dataset[0]

split = RandomLinkSplit(
    is_undirected=True,
    add_negative_train_samples=False,
    split_labels=True
)
train_data, val_data, test_data = split(data)

# ======================= 2. Вибір моделі =======================
MODE = input("Choose mode (GAE, VGAE, heuristic): ").strip().upper()  # "GAE", "VGAE", "heuristic"

# GAE Encoder
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

# VGAE Encoder
class VariationalEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

# ======================= 3. Підготовка =======================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = train_data.x.to(device)
edge_index = train_data.edge_index.to(device)

if MODE == "GAE":
    model = GAE(GCNEncoder(dataset.num_features, 64)).to(device)
elif MODE == "VGAE":
    model = VGAE(VariationalEncoder(dataset.num_features, 64)).to(device)
else:
    model = None

optimizer = torch.optim.Adam(model.parameters(), lr=0.01) if model else None

# ======================= 4. Навчання =======================
def train():
    model.train()
    optimizer.zero_grad()
    if MODE == "VGAE":
        z = model.encode(x, edge_index)
        loss = model.recon_loss(z, train_data['pos_edge_label_index'].to(device)) + model.kl_loss()
    else:
        z = model.encode(x, edge_index)
        loss = model.recon_loss(z, train_data['pos_edge_label_index'].to(device))
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(pos_edge_index, neg_edge_index):
    model.eval()
    z = model.encode(x, edge_index)
    pos_score = model.decoder(z, pos_edge_index.to(device)).sigmoid()
    neg_score = model.decoder(z, neg_edge_index.to(device)).sigmoid()
    y_true = torch.cat([torch.ones(pos_score.size(0)), torch.zeros(neg_score.size(0))])
    y_score = torch.cat([pos_score, neg_score])
    return roc_auc_score(y_true.cpu(), y_score.cpu()), average_precision_score(y_true.cpu(), y_score.cpu())

# ======================= 5. Візуалізація TSNE =======================
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plot_tsne(z, title):
    z = z.detach().cpu().numpy()
    z_tsne = TSNE(n_components=2).fit_transform(z)
    plt.figure(figsize=(6, 5))
    plt.title(title)
    plt.scatter(z_tsne[:, 0], z_tsne[:, 1], s=10, alpha=0.7)
    plt.show()

# ======================= 6. Запуск =======================
if MODE in ["GAE", "VGAE"]:
    EPOCHS = int(input("Enter number of epochs: "))
    for epoch in range(1, EPOCHS + 1):
        loss = train()
        val_auc, val_ap = test(val_data['pos_edge_label_index'], val_data['neg_edge_label_index'])
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}")

    test_auc, test_ap = test(test_data['pos_edge_label_index'], test_data['neg_edge_label_index'])
    print(f"✅ Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}")
    z = model.encode(x, edge_index)
    plot_tsne(z, title=f"{MODE} Embeddings (TSNE)")

# ======================= 6. Heuristic Methods =======================
if MODE == "heuristic":
    G = to_networkx(data, to_undirected=True)

    print("\n🔎 Common Neighbors (Top 5):")
    cn_scores = []
    limit = 1000  # обмежити кількість пар для прискорення
    for idx, (u, v) in enumerate(nx.non_edges(G)):
        if idx >= limit:
            break
        score = len(list(nx.common_neighbors(G, u, v)))
        cn_scores.append((u, v, score))
    for u, v, score in sorted(cn_scores, key=lambda x: -x[2])[:5]:
        print(f"{u} - {v}: {score}")

    print("\n🔎 Jaccard Coefficient (Top 5):")
    for u, v, p in sorted(nx.jaccard_coefficient(G), key=lambda x: -x[2])[:5]:
        print(f"{u} - {v}: {p:.4f}")

    print("\n🔎 Adamic-Adar Index (Top 5):")
    for u, v, p in sorted(nx.adamic_adar_index(G), key=lambda x: -x[2])[:5]:
        print(f"{u} - {v}: {p:.4f}")
