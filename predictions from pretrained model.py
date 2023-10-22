import torch
import numpy as np
from torch_geometric.utils import dense_to_sparse
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN2, TGCN2
import seaborn as sns
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv, ChebConv


class TGCN2(torch.nn.Module):
    r"""An implementation THAT SUPPORTS BATCHES of the Temporal Graph Convolutional Gated Recurrent Cell.
    For details see this paper: `"T-GCN: A Temporal Graph ConvolutionalNetwork for
    Traffic Prediction." <https://arxiv.org/abs/1811.05320>`_
    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        batch_size (int): Size of the batch.
        improved (bool): Stronger self loops. Default is False.
        cached (bool): Caching the message weights. Default is False.
        add_self_loops (bool): Adding self-loops for smoothing. Default is True.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 batch_size: int,  # this entry is unnecessary, kept only for backward compatibility
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True):
        super(TGCN2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.batch_size = batch_size  # not needed
        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):
        self.conv_z = ChebConv(in_channels=self.in_channels,  out_channels=self.out_channels, K=10, normalization="sym", agrr="add")
        self.linear_z = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_reset_gate_parameters_and_layers(self):
        self.conv_r = ChebConv(in_channels=self.in_channels,  out_channels=self.out_channels, K=10, normalization="sym", agrr="add")
        self.linear_r = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_candidate_state_parameters_and_layers(self):
        self.conv_h = ChebConv(in_channels=self.in_channels,  out_channels=self.out_channels, K=10, normalization="sym", agrr="add")
        self.linear_h = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            # can infer batch_size from X.shape, because X is [B, N, F]
            H = torch.zeros(X.shape[0], X.shape[1], self.out_channels).to(X.device) #(b, 207, 32)
        return H

    def _calculate_update_gate(self, X, edge_index, edge_weight, H):
        Z = torch.cat([self.conv_z(X, edge_index, edge_weight), H], axis=2) # (b, 207, 64)
        Z = self.linear_z(Z) # (b, 207, 32)
        Z = torch.sigmoid(Z)

        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_weight, H):
        R = torch.cat([self.conv_r(X, edge_index, edge_weight), H], axis=2) # (b, 207, 64)
        R = self.linear_r(R) # (b, 207, 32)
        R = torch.sigmoid(R)

        return R

    def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R):
        H_tilde = torch.cat([self.conv_h(X, edge_index, edge_weight), H * R], axis=2) # (b, 207, 64)
        H_tilde = self.linear_h(H_tilde) # (b, 207, 32)
        H_tilde = torch.tanh(H_tilde)

        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde   # # (b, 207, 32)
        return H

    def forward(self,X: torch.FloatTensor, edge_index: torch.LongTensor, edge_weight: torch.FloatTensor = None,
                H: torch.FloatTensor = None ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.
        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.
        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, edge_index, edge_weight, H)
        R = self._calculate_reset_gate(X, edge_index, edge_weight, H)
        H_tilde = self._calculate_candidate_state(X, edge_index, edge_weight, H, R)
        H = self._calculate_hidden_state(Z, H, H_tilde) # (b, 207, 32)
        return H


class A3TGCN2(torch.nn.Module):
    r"""An implementation THAT SUPPORTS BATCHES of the Attention Temporal Graph Convolutional Cell.
    For details see this paper: `"A3T-GCN: Attention Temporal Graph Convolutional
    Network for Traffic Forecasting." <https://arxiv.org/abs/2006.11583>`_
    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        periods (int): Number of time periods.
        improved (bool): Stronger self loops (default :obj:`False`).
        cached (bool): Caching the message weights (default :obj:`False`).
        add_self_loops (bool): Adding self-loops for smoothing (default :obj:`True`).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        periods: int,
        batch_size:int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True):
        super(A3TGCN2, self).__init__()

        self.in_channels = in_channels  # 2
        self.out_channels = out_channels # 32
        self.periods = periods # 12
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.batch_size = batch_size
        self._setup_layers()

    def _setup_layers(self):
        self._base_tgcn = TGCN2(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=self.batch_size,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._attention = torch.nn.Parameter(torch.empty(self.periods, device=device))
        torch.nn.init.uniform_(self._attention)

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None
    ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.
        Arg types:
            * **X** (PyTorch Float Tensor): Node features for T time periods.
            * **edge_index** (PyTorch Long Tensor): Graph edge indices.
            * **edge_weight** (PyTorch Long Tensor, optional)*: Edge weight vector.
            * **H** (PyTorch Float Tensor, optional): Hidden state matrix for all nodes.
        Return types:
            * **H** (PyTorch Float Tensor): Hidden state matrix for all nodes.
        """
        H_accum = 0
        probs = torch.nn.functional.softmax(self._attention, dim=0)
        for period in range(self.periods):

            H_accum = H_accum + probs[period] * self._base_tgcn( X[:, :, :, period], edge_index, edge_weight, H) #([32, 207, 32]

        return H_accum


class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, periods_in, periods_out, batch_size, num_edges):
        super(TemporalGNN, self).__init__()

        # initialize learnable edge weights
        self.edge_weight = torch.nn.Parameter(torch.full((num_edges,), 1 / 8))

        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN2(in_channels=node_features, out_channels=256, periods=periods_in, batch_size=batch_size)

        # Equals single-shot prediction
        self.linear = torch.nn.Linear(256, periods_out)

    def forward(self, x, edge_index):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index, self.edge_weight.relu())
        h = F.relu(h)
        h = self.linear(h)

        return h


class STGNN_model():
    r""""
    This class handles the process of generating predictions from a pretrained spatio-temporal graph neural network
    following the methodology of 'Graph-Based Approach to Human Movement Prediction in Shared Human-Robot Workspaces' by Casper Dik

    Args:
        node_features (int): number of node features used in the pretrained model.
        periods_in (int): number of input time steps of the pretrained model.
        filepath_model (str): the filepath for loading the pretrained PyTorch model.
        adj_reduced (numpy array): a numpy array storing the connectivity information (adjacency matrix) of the
            reduced graph as defined in the thesis. The adjacency matrix can be produced by running the function adj_matrix
            from generate_input_matrices.py, whereafter, the function reduce_graph from reduce_graph.py should be ran to
            get the adjacency matrix of graph without the omitted nodes as specified in the thesis methodology.
        idx (numpy array): a numpy array containing the indices of the omitted nodes. Can be generated using the
            function reduce_graph from reduce_graph.py.
        normalize (boolean): a boolean variable that controls whether not to perform z-score normalization on the input and
            denormalization on the outputs. If the supplied pretrained model is trained on normalized inputs, the
            variable normalize should be set to TRUE.
        means (numpy array, optional): a numpy array containing the means of the normalized node features.
        stdev (numpy array, optional): a numpy array containing the standard deviations of the normalized node features.

    """
    def __init__(self, node_features: int, periods_in: int, periods_out: int, filepath_model: str, adj_reduced: np.ndarray, idx: np.ndarray, normalize: bool = True,
                 means: np.ndarray = None, stdev: np.ndarray = None):
        self.normalize = normalize
        self.periods_in = periods_in
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.A, _ = dense_to_sparse(torch.from_numpy(adj_reduced))
        self.idx = idx
        self.means = means
        self.stdev = stdev

        self.stgnn = TemporalGNN(node_features=node_features, periods_in=self.periods_in, periods_out=periods_out, batch_size=1, num_edges=self.A.shape[1]).to(self.DEVICE)
        self.stgnn.load_state_dict(torch.load(filepath_model, map_location=self.DEVICE))
        self.stgnn.eval()

    def run_model(self, paths_inputs: list):
        self.F = self.create_feature_matrix(paths_inputs)

        self.process_data()
        self.yhat = self.get_outputs()

        return self.yhat

    def create_feature_matrix(self, paths_inputs: list):
        """create feature matrix from inputs"""
        paths_inputs = self.check_time_in(paths_inputs)     # check if

        f0 = self.load_txt_to_array(paths_inputs[0])
        f1 = self.load_txt_to_array(paths_inputs[1])
        F = np.stack((f0, f1))

        for txt in paths_inputs[2:]:
            f = self.load_txt_to_array(txt)
            F = np.vstack((F, f[None, :, :]))

        return F

    def load_txt_to_array(self, txt: str):
        """load a txt file as numpy array"""
        if txt[-4:] == ".txt":          # check if input file is a .txt
            self.delete_hashtag(txt)    # call function to delete the hashtag
            d = np.genfromtxt(txt, delimiter=[1, 20], dtype=[("f0", np.uint8), ("f1", object)]) # load txt as np array
            d = self.load_features(d)   # load the features
        else:
            raise SystemExit("Filetype unsupported. All input files must be .txt")
        return d.astype('uint8')

    def delete_hashtag(self, f: str):
        """remove the # from the input files, otherwise will stop reading after # at later stage"""
        with open(f, "rb") as input_file:
            s = input_file.read()
            input_file.close()
            s = s.replace(b"#", b"")

        with open(f, "wb") as output_file:
            output_file.write(s)

    def load_features(self, d: np.ndarray):
        x = d["f1"].astype("U")

        w = np.where(np.char.find(x, "Wall") > 0, 1, 0)
        c = np.where(np.char.find(x, "coffee") > 0, 1, 0)
        ws = np.where(np.char.find(x, "WS") > 0, 1, 0)

        # first column is human presence, second wall, third coffee, fourth workstation
        d = np.stack((d["f0"], w), axis=1)
        d = np.concatenate((d, c[:, None]), axis=1)
        d = np.concatenate((d, ws[:, None]), axis=1)

        return d

    def check_time_in(self, paths_inputs: list):
        """check if input length is correct. If the length is longer, some observations will be dropped.
        If the length is too short, raise system exit"""

        if len(paths_inputs) > self.periods_in:
            print("The model uses a historical time series of length ", self.periods_in, " but ", len(paths_inputs), " files were supplied.")
            print("First ", self.periods_in-len(paths_inputs), " will be dropped.")
            paths_inputs = paths_inputs[-5:]
        elif len(paths_inputs) < self.periods_in:
            raise SystemExit

        return paths_inputs

    def process_data(self):
        """normalizes, reshape and reduces the grid"""

        # drop nodes
        self.F = np.delete(self.F, self.idx, axis=1)
        # reshape
        self.F = self.F.transpose((1, 2, 0))
        # normalize
        if self.normalize == True:
            self.normalize_zscore()
        # add dimension
        self.F = np.expand_dims(self.F, axis=0)

    def normalize_zscore(self):
        """z-score normalization"""
        if self.means is None:
            raise SystemExit("Normalization is set to True but no means are provided")
        elif self.stdev is None:
            raise SystemExit("Normalization is set to True but no standard deviations are provided")

        self.F = self.F - self.means.reshape(1, -1, 1)
        self.F = self.F / self.stdev.reshape(1, -1, 1)

    def denormalize(self, arr):
        return np.round((arr * self.stdev[0]) + self.means[0], decimals=5)

    def get_outputs(self):
        """runs the model and yields predictions. Denormalizes predictions and adds back omitted nodes"""
        with torch.no_grad():
            yhat = self.stgnn(torch.from_numpy(self.F.astype(np.float32)).to(self.DEVICE), self.A.to(self.DEVICE)).cpu().detach()

        # denormalize
        if self.normalize == True:
            yhat = self.denormalize(yhat)

        # add back omitted nodes
        yhat = np.insert(yhat, self.idx[0] - np.arange(len(self.idx[0])), 0, axis=1)

        return yhat

    def export_as_txt(self, yhat: np.ndarray, path: str = "", regression_output: bool = True, classification_output: bool = False, threshold: float = None):
        """export results to txt files"""
        for i in range(yhat.shape[2]):
            if regression_output:
                np.savetxt(path + "heatmap_reg_t" + str(i+1) + ".txt", yhat[0, :, i])
            if classification_output:
                yhat = np.where(yhat > threshold, 1, 0).astype("uint8")
                np.savetxt(path + "heatmap_class_t" + str(i + 1) + ".txt", yhat[0, :, i])

    def plot_heatmap(self, data, img: str):
        """plots heatmap of an image"""
        data = np.array(np.array_split(data[0], 100))
        data = np.rot90(data)

        ax = sns.heatmap(data, linewidths=0, square=True, cmap='RdYlGn_r', zorder=2, alpha=0.6, cbar=False)
        my_image = mpimg.imread(img)
        ax.imshow(my_image, aspect=ax.get_aspect(), extent=ax.get_xlim() + ax.get_ylim(), zorder=1)

        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        plt.show()


if __name__ == "__main__":
    means_normalization = np.array([0.10162151, 0.03302132, 0.00182007, 0.04836193])
    stdev_normalization = np.array([0.33049121, 0.17869223, 0.04262347, 0.21452985])
    idx_omitted_nodes = np.load("example input files/idx_sim2_100p_5_40.npy")
    A = np.load("example input files/Adj_Matrix_Reduced.npy")
    model_path = "example input files/state_dict.pth"

    model = STGNN_model(node_features=4, periods_in=5, periods_out=40, normalize=True, means=means_normalization,
                        stdev=stdev_normalization, idx=idx_omitted_nodes, adj_reduced=A, filepath_model=model_path)


    txts = ["data/simulation2-100p-100cm/heatmap_08H57m32s.txt",
            "data/simulation2-100p-100cm/heatmap_08H57m33s.txt",
            "data/simulation2-100p-100cm/heatmap_08H57m34s.txt",
            "data/simulation2-100p-100cm/heatmap_08H57m35s.txt",
            "data/simulation2-100p-100cm/heatmap_08H57m36s.txt",
            ]

    yhat = model.run_model(paths_inputs=txts)
    print(yhat.shape)
    print(yhat)

    t = 1
    model.plot_heatmap(yhat[:, :, t].numpy(), img="test input files/simu2.png")
    # model.export_as_txt(yhat, path="data/output/", regression_output=True, classification_output=False, threshold=0.15)

