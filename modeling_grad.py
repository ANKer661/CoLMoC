import torch
import numpy as np
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch import nn, Tensor
from torch.nn.parameter import Parameter
from torch_geometric.data import Dataset, Data
from torch_geometric.typing import Adj
from torch_geometric.utils import scatter, add_self_loops
from torch_geometric.nn import MetaLayer, GraphNorm, LayerNorm, BatchNorm
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import LambdaLR


# Define the dataset
class ConformationDataset(Dataset):
    def __init__(self, datalist, atom_inputs=None, aa_inputs=None, edge_index=None):
        self.datalist = datalist
        self.atom_inputs = atom_inputs
        self.aa_inputs = aa_inputs
        self.edge_index = edge_index
        super().__init__(None)

    def len(self):
        return len(self.datalist)

    def get(self, idx):
        return (self.atom_inputs,
                self.aa_inputs,
                Data(x=self.datalist[idx], edge_index=self.edge_index))


# define the model components
class NodeInit(nn.Module):
    '''
        atom -> embedding
        aa -> embeding
        coords -> layernorm(relu(coords))
        outputs := layernorm(atom + aa + coords)

        why layernorm?
            focus on the feature within a molecule
    '''

    def __init__(
            self,
            num_atom=None,
            num_aa=None,
            coord_inputs_dim=None,
            node_dim=None,
            dropout_prob=None,
            norm_type=None
        ):
        super().__init__()
        self.pre_projection = nn.Sequential(*
                [
                    nn.Sequential(
                        nn.Linear(coord_inputs_dim * (4 ** (i - 1)), 
                                  (4 ** i) * coord_inputs_dim),
                        nn.ReLU()
                    ) for i in range(1, int(np.log2(node_dim / 3) / 2 + 1))
                ]
        )
        self.dense = nn.Linear(
            (4 ** int(np.log2(node_dim / 3) / 2)) * coord_inputs_dim, 
            node_dim
        )
        self.atom_embeddings = nn.Embedding(num_atom, node_dim)
        self.aa_embeddings = nn.Embedding(num_aa, node_dim)
        self.dropout = nn.Dropout(dropout_prob)

        if norm_type == 'layernorm':
            self.Norm_coord = LayerNorm(coord_inputs_dim)
            self.Norm_embedding = LayerNorm(node_dim)
        elif norm_type == 'graphnorm':
            self.Norm_coord = GraphNorm(coord_inputs_dim)
            self.Norm_embedding = GraphNorm(node_dim)
        elif norm_type == 'batchnorm':
            self.Norm_coord = BatchNorm(coord_inputs_dim)
            self.Norm_embedding = BatchNorm(node_dim)
        else:
            raise ValueError('norm_type should be layernorm, graphnorm or batchnorm')

    def forward(
            self,
            x: torch.Tensor,
            atom_ids: torch.Tensor,
            aa_ids: torch.Tensor,
        ) -> torch.Tensor:
        
        batch_size, num_node = atom_ids.shape
        atom_embeddings = self.atom_embeddings(atom_ids)
        aa_embeddings = self.aa_embeddings(aa_ids)

        if isinstance(self.Norm_coord, BatchNorm):
            x = self.Norm_coord(x.view(batch_size, num_node, -1))
            x = self.pre_projection(x)
            x = torch.relu(self.dense(x))
            x = self.Norm_embedding(
                (x + atom_embeddings.view(-1, atom_embeddings.size(-1)) 
                 + aa_embeddings.view(-1, aa_embeddings.size(-1))))
        else:  # layernorm or graphnorm
            batch = torch.arange(batch_size).repeat_interleave(num_node).to(x.device)
            x = self.Norm_coord(x, batch=batch)
            x = self.pre_projection(x)
            x = torch.relu(self.dense(x))
            x = self.Norm_embedding(
                (x + atom_embeddings.view(-1, atom_embeddings.size(-1)) 
                 + aa_embeddings.view(-1, aa_embeddings.size(-1))),
                batch=batch
            )

        return self.dropout(x)


class EdgeInit(nn.Module):
    '''
        obtain Edge feature from nodes
    '''

    def __init__(self, node_dim=None, edge_attr_dim=None, norm_type=None):
        super().__init__()
        self.dense = nn.Linear(node_dim, edge_attr_dim)
        
        if norm_type == 'layernorm':
            self.Norm = LayerNorm(edge_attr_dim)
        elif norm_type == 'graphnorm':
            self.Norm = GraphNorm(edge_attr_dim)
        elif norm_type == 'batchnorm':
            self.Norm = BatchNorm(edge_attr_dim)
        else:
            raise ValueError('norm_type should be' 
                             'layernorm, graphnorm or batchnorm')

    def forward(self, x: torch.Tensor,
                edge_index: Adj, batch: torch.Tensor) -> torch.Tensor:
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # batch: [E] with max entry B - 1.
        row, col = edge_index
        x = F.relu(self.dense(x))
        # src, dest: [E, F_e], where E is the number of edges.
        src = x[row]
        dest = x[col]
        # Average Pooling. Eij = Eji (Pair order invariance)
        if isinstance(self.Norm, BatchNorm):
            edge_attr = self.Norm((src + dest) / 2.)
        else:  # layernorm or graphnorm
            edge_attr = self.Norm((src + dest) / 2., batch=batch)

        return edge_attr


class GlobalInit(nn.Module):
    '''
        obtain Globle feature from edges
    '''

    def __init__(self, edge_attr_dim=None, globle_attr_dim=None, norm_type=None):
        super().__init__()
        self.dense = nn.Linear(edge_attr_dim, globle_attr_dim)
        if norm_type == 'batchnorm':
            self.Norm = BatchNorm(globle_attr_dim)
        elif norm_type == 'layernorm':
            self.Norm = LayerNorm(globle_attr_dim)
        elif norm_type == 'graphnorm':  
            # still use layernorm, bug in graphnorm in the particular case
            self.Norm = LayerNorm(globle_attr_dim)
        else:
            raise ValueError('norm_type should be '
                             'layernorm, graphnorm or batchnorm')

    def forward(self, edge_attr: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # edge_attr: [E, F_e], where E is the number of edges.
        # batch: [E] with max entry B - 1.
        edge_attr = F.relu(self.dense(edge_attr))
        
        if isinstance(self.Norm, BatchNorm):
            return self.Norm(scatter(edge_attr, batch, dim=0, reduce='mean'))
        else:  
            return self.Norm(
                scatter(edge_attr, batch, dim=0, reduce='mean'),
                batch=torch.arange(batch.max().item() + 1).to(edge_attr.device)
            )


class InputsInit(nn.Module):
    '''
        get the initial Node(x), Edge(edge_attr), Global(u) features
        add self-loop to the edge_index
    '''

    def __init__(
            self,
            num_atom=None,
            num_aa=None,
            coord_inputs_dim=None,  # should be 3
            node_dim=None,
            edge_attr_dim=None,
            global_attr_dim=None,
            dropout_prob=None,
            norm_type=None
        ):
        super().__init__()
        self.inputs_layer = NodeInit(
            num_atom=num_atom,
            num_aa=num_aa,
            coord_inputs_dim=coord_inputs_dim,
            node_dim=node_dim,
            dropout_prob=dropout_prob,
            norm_type=norm_type
        )
        self.edge_init = EdgeInit(node_dim, 
                                  edge_attr_dim, norm_type=norm_type)
        self.global_init = GlobalInit(edge_attr_dim, 
                                      global_attr_dim, norm_type=norm_type)

    def get_batch(self, batch_size, num_edge, num_node, device):
        edges_batch = torch.arange(batch_size).repeat_interleave(num_edge)
        nodes_batch = torch.arange(batch_size).repeat_interleave(num_node)
        return torch.concat([edges_batch, nodes_batch]).to(device)

    def forward(
            self,
            x: Tensor, atom_ids: Tensor, aa_ids: Tensor,
            edge_index: Adj,
        ):
        # introduce self-loop, i.e. a edge from node to itself
        num_edge_per_graph = edge_index.shape[1]
        edge_index, _ = add_self_loops(edge_index)
        batch_size, num_node = atom_ids.shape

        batch = self.get_batch(
                batch_size, 
                num_edge_per_graph // batch_size, 
                num_node, device=x.device
        )

        x = self.inputs_layer(x, atom_ids, aa_ids)
        edge_attr = self.edge_init(x, edge_index, batch)
        u = self.global_init(edge_attr, batch)

        return (x, edge_attr, u, edge_index)


class EdgeUpdate(nn.Module):
    def __init__(
            self, 
            node_dim, edge_attr_dim, globle_attr_dim, 
            dropout_prob, norm_type
        ):
        super().__init__()
        self.edge_mlp = nn.Sequential(*[
            nn.Linear(
                2 * node_dim + edge_attr_dim + globle_attr_dim, 
                4 * edge_attr_dim),
            nn.ReLU(),
            nn.Linear(4 * edge_attr_dim, edge_attr_dim)
        ])
        if norm_type == 'layernorm':
            self.Norm = LayerNorm(edge_attr_dim)
        elif norm_type == 'graphnorm':
            self.Norm = GraphNorm(edge_attr_dim)
        elif norm_type == 'batchnorm':
            self.Norm = BatchNorm(edge_attr_dim)
        else:
            raise ValueError('norm_type should be '
                             'layernorm, graphnorm or batchnorm')
        
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, src, dest, edge_attr, u, batch):
        # src, dest: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e], where E is the number of edges.
        # u: [B, F_u], where B is the batch size.
        # batch: [E] with max entry B - 1.

        # Update edge features.
        out = torch.concat([src, dest, edge_attr, u[batch]], dim=1)
        out = self.edge_mlp(out)
        if isinstance(self.Norm, BatchNorm):
            out = self.Norm(out + edge_attr)
        else:  # GraphNorm or LayerNorm
            out = self.Norm(out + edge_attr, batch=batch) 

        return self.dropout(out)


class GlobalUpdate(nn.Module):
    def __init__(
            self, 
            node_dim, edge_attr_dim, globle_attr_dim, 
            dropout_prob, norm_type
        ):
        super().__init__()
        self.global_mlp = nn.Sequential(*[
            nn.Linear(node_dim + edge_attr_dim + globle_attr_dim, 
                      4 * globle_attr_dim),
            nn.ReLU(),
            nn.Linear(4 * globle_attr_dim, globle_attr_dim)
        ])
        self.dropout = nn.Dropout(dropout_prob)
        if norm_type == 'batchnorm':
            self.Norm = BatchNorm(globle_attr_dim)
        elif norm_type == 'layernorm':
            self.Norm = LayerNorm(globle_attr_dim)
        elif norm_type == 'graphnorm':
            # still use layernorm, bug in graphnorm in the particular case
            self.Norm = LayerNorm(globle_attr_dim)
        else:
            raise ValueError('norm_type should be '
                             'layernorm, graphnorm or batchnorm')


    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e], where E is the number of edges.
        # u: [B, F_u], where B is the batch size.
        # batch: [N] with max entry B - 1.

        # Step1: aggregate the edge features globally.
        # Step2: aggregate the node features globally.
        out = torch.concat([
            u,
            scatter(edge_attr, batch[edge_index[0]], dim=0, reduce='mean'),
            scatter(x, batch, dim=0, reduce='mean')
        ], dim=1)

        # Step3: Update global features.
        out = self.global_mlp(out)
        if isinstance(self.Norm, BatchNorm):
            out = self.Norm(out + u)
        else:  # GraphNorm or LayerNorm
            out = self.Norm(
                out + u, batch=torch.arange(u.size(0), device=u.device))
            
        return self.dropout(out)


class NodeUpdate(nn.Module):
    def __init__(
            self, 
            node_dim, edge_attr_dim, global_attr_dim, 
            dropout_prob, norm_type
        ):
        super().__init__()
        self.node_mlp_aggr_edge = nn.Sequential(*[
            nn.Linear(node_dim + edge_attr_dim, 4 * edge_attr_dim),
            nn.ReLU(),
            nn.Linear(4 * edge_attr_dim, edge_attr_dim)
        ])
        self.node_mlp_update_node = nn.Sequential(*[
            nn.Linear(node_dim + edge_attr_dim + global_attr_dim, 
                      4 * node_dim),
            nn.ReLU(),
            nn.Linear(4 * node_dim, node_dim)
        ])
        self.dropout = nn.Dropout(dropout_prob)

        if norm_type == 'layernorm':
            self.Norm = LayerNorm(node_dim)
        elif norm_type == 'graphnorm':
            self.Norm = GraphNorm(node_dim)
        elif norm_type == 'batchnorm':
            self.Norm = BatchNorm(node_dim)
        else:
            raise ValueError('norm_type should be '
                             'layernorm, graphnorm or batchnorm')

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        # row: source, col: destination

        # Step1: aggregate the edge features per node
        # 这里不是直接使用edge_attr聚合
        # 而是使用edge_attr和起始node的特征拼接，再过一个mlp后聚合到目标node
        row, col = edge_index
        out = torch.concat([x[row], edge_attr], dim=1)
        out = self.node_mlp_aggr_edge(out)
        out = scatter(out, col, dim=0, dim_size=x.size(0),
                      reduce='mean')  # [N, F_e]
        
        # Step2: update the node features
        out = torch.concat([x, out, u[batch]], dim=1)
        out = self.node_mlp_update_node(out)
        if isinstance(self.Norm, BatchNorm):
            out = self.Norm(out + x)
        else:  # GraphNorm or LayerNorm
            out = self.Norm(out + x, batch=batch)

        return self.dropout(out)


def get_metalayer_with_model(
        node_dim=None,
        edge_attr_dim=None,
        globle_attr_dim=None,
        dropout_prob=None,
        norm_type=None
):
    '''
        get MetaLayer with EdgeUpdate, GlobalUpdate, NodeUpdate
    '''
    edge_update = EdgeUpdate(node_dim, edge_attr_dim, globle_attr_dim, 
                             dropout_prob, norm_type)
    global_update = GlobalUpdate(node_dim, edge_attr_dim, globle_attr_dim, 
                                 dropout_prob, norm_type)
    node_update = NodeUpdate(node_dim, edge_attr_dim, globle_attr_dim, 
                             dropout_prob, norm_type)
    
    return MetaLayer(edge_update, node_update, global_update)


# Backbone
class GraphNet(nn.Module):
    def __init__(
            self,
            num_atom=None,
            num_aa=None,
            coord_inputs_dim=None,  # should be 3
            node_dim=None,
            edge_attr_dim=None,
            global_attr_dim=None,
            dropout_prob=None,
            num_layers=None,
            norm_type=None
        ):
        super().__init__()
        self.inputs_init = InputsInit(
            num_atom=num_atom,
            num_aa=num_aa,
            coord_inputs_dim=coord_inputs_dim,
            node_dim=node_dim,
            edge_attr_dim=edge_attr_dim,
            global_attr_dim=global_attr_dim,
            dropout_prob=dropout_prob,
            norm_type=norm_type
        )
        self.conv_layers = nn.ModuleList(
            [
                get_metalayer_with_model(
                    node_dim, edge_attr_dim, global_attr_dim, 
                    dropout_prob, norm_type
                ) 
                for _ in range(num_layers)
            ]
        )

    def get_batch(self, batch_size, num_node, device):
        return torch.arange(batch_size).repeat_interleave(num_node).to(device)

    def forward(
            self,
            x: Tensor, atom_ids: Tensor,
            aa_ids: Tensor, edge_index: Adj,
    ):
        # x: [N, F_x], where N is the number of nodes.
        # atom_ids: [B, N], where B is the batch size.
        # aa_ids: [B, N], where B is the batch size.
        # edge_index: [2, E] with max entry N - 1.
        x, edge_attr, u, edge_index = self.inputs_init(
            x, atom_ids, aa_ids, edge_index)
        # batch: [N] with max entry B - 1.
        batch = self.get_batch(atom_ids.size(0), atom_ids.size(1), x.device)

        for conv in self.conv_layers:
            x, edge_attr, u = conv(
                x, edge_index, edge_attr, u, batch)
            
        return (x, edge_attr, u, edge_index)
    

class Projector(nn.Module):
    def __init__(
            self, 
            global_attr_dim, 
            hidden_dim, 
            output_dim, 
            norm_type
        ):
        super().__init__()
        self.dense1 = nn.Linear(global_attr_dim, hidden_dim)
        self.dense2 = nn.Linear( 
            hidden_dim,
            4 ** np.ceil(
                np.log2(hidden_dim / output_dim) / 2 - 1
            ).astype(int) * output_dim,
        )
        self.mlp = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear((4 ** i) * output_dim, 
                            (4 ** (i - 1)) * output_dim),
                    nn.ReLU()
                )
                for i in range(np.ceil(np.log2(hidden_dim / output_dim) / 2 - 1).astype(int), 0, -1)
            ]
        )
        self.mlp[-1].pop(1)  # remove the last relu
        self.mlp[-1].append(nn.Tanh())
        if norm_type == 'batchnorm':
            self.Norm = BatchNorm(hidden_dim)
        elif norm_type == 'layernorm':
            self.Norm = LayerNorm(hidden_dim)
        elif norm_type == 'graphnorm':
            # still use layernorm
            self.Norm = LayerNorm(hidden_dim)
        else:
            raise ValueError('norm_type should be'
                             'batchnorm, layernorm or graphnorm')

    def forward(self, u: Tensor):
        u = F.relu(self.dense1(u))
        # just norm, no add. Because the dim can be different.
        if isinstance(self.Norm, BatchNorm):
            u = self.Norm(u)
        else:  # GraphNorm or LayerNorm
            u = self.Norm(u, batch=torch.arange(u.size(0)).to(u.device))
        u = F.relu(self.dense2(u))

        return self.mlp(u)


# Swav training paradigm
class SwavMoleculeTrain(nn.Module):
    def __init__(
            self,
            num_atom=None,
            num_aa=None,
            coord_inputs_dim=None,  # should be 3
            node_dim=None,
            edge_attr_dim=None,
            global_attr_dim=None,
            dropout_prob=None,
            num_layers=None,
            # temperature=None,
            num_prototypes=None,
            use_projector=False,
            proj_hidden_dim=None,
            proj_output_dim=None,
            norm_type=None
        ):
        super().__init__()
        # self.temperature = temperature
        self.backbone = GraphNet(
            num_atom=num_atom,
            num_aa=num_aa,
            coord_inputs_dim=coord_inputs_dim,
            node_dim=node_dim,
            edge_attr_dim=edge_attr_dim,
            global_attr_dim=global_attr_dim,
            dropout_prob=dropout_prob,
            num_layers=num_layers,
            norm_type=norm_type
        )

        if use_projector:
            self.prototype = nn.Parameter(
                nn.functional.normalize(
                    torch.randn(num_prototypes, proj_output_dim), p=2, dim=1))
            self.projector = Projector(
                global_attr_dim, proj_hidden_dim, proj_output_dim, norm_type)
        else:
            self.prototype = nn.Parameter(
                nn.functional.normalize(
                    torch.randn(num_prototypes, global_attr_dim), p=2, dim=1))
            self.projector = None

    def forward(
            self,
            x_s: Tensor, x_t: Tensor,
            atom_ids: Tensor, aa_ids: Tensor,
            edge_index: Adj,
            sinkhorn_eps: float,
            sinkhorn_iters: int
        ):

        # use global features as the graph representation
        B = atom_ids.size(0)
        _, _, u, _ = self.backbone(
            torch.cat([x_s, x_t], dim=0),
            torch.cat([atom_ids, atom_ids], dim=0),
            torch.cat([aa_ids, aa_ids], dim=0),
            torch.cat([edge_index, edge_index + edge_index.max() + 1], dim=1)
        )

        # project the global features
        if self.projector is not None:
            u = self.projector(u)

        # normalize the global features
        z = u / u.norm(dim=1, keepdim=True, p=2)

        # z: [2B, F_g]
        # prototype: [K, F_g]
        # scores = torch.matmul(z, self.prototype.t())
        scores = torch.einsum('bf, kf -> bk', z, self.prototype)

        scores_s = scores[:B]
        scores_t = scores[B:]

        # compute the assignment matrix Q: [B, K]
        with torch.no_grad():
            q_s = sinkhorn(
                scores_s, eps=sinkhorn_eps, num_iters=sinkhorn_iters)
            q_t = sinkhorn(
                scores_t, eps=sinkhorn_eps, num_iters=sinkhorn_iters)

        """  
            Loss computation. moved to the training loop

        # compute probability
        p_s = nn.functional.softmax(scores_s / self.temperature, dim=-1)
        p_t = nn.functional.softmax(scores_t / self.temperature, dim=-1)

        # swap prediction loss
        # loss = - 0.5 * (q_s * torch.log(p_t)).sum(dim=-1).mean()
        loss = - 0.5 * (
            torch.einsum('bk, bk -> b', q_s, torch.log(p_t))
            + torch.einsum('bk, bk -> b', q_t, torch.log(p_s))
        ).mean()

        """

        return (scores_s, scores_t, q_s, q_t)
    

# use in inference
class SwavMolecule(nn.Module):
    def __init__(
            self,
            num_atom=None,
            num_aa=None,
            coord_inputs_dim=None,  # should be 3
            node_dim=None,
            edge_attr_dim=None,
            global_attr_dim=None,
            dropout_prob=None,
            num_layers=None,
            # temperature=None,
            num_prototypes=None,
            use_projector=False,
            proj_hidden_dim=None,
            proj_output_dim=None,
            norm_type=None
        ):
        super().__init__()
        # self.temperature = temperature
        self.backbone = GraphNet(
            num_atom=num_atom,
            num_aa=num_aa,
            coord_inputs_dim=coord_inputs_dim,
            node_dim=node_dim,
            edge_attr_dim=edge_attr_dim,
            global_attr_dim=global_attr_dim,
            dropout_prob=dropout_prob,
            num_layers=num_layers,
            norm_type=norm_type
        )

        if use_projector:
            self.prototype = nn.Parameter(
                nn.functional.normalize(
                    torch.randn(num_prototypes, proj_output_dim), p=2, dim=1))
            self.projector = Projector(
                global_attr_dim, proj_hidden_dim, proj_output_dim, norm_type)
        else:
            self.prototype = nn.Parameter(
                nn.functional.normalize(
                    torch.randn(num_prototypes, global_attr_dim), p=2, dim=1))
            self.projector = None

    def forward(
            self,
            x: Tensor,
            atom_ids: Tensor, aa_ids: Tensor,
            edge_index: Adj
        ):
        _, _, u, _ = self.backbone(x, atom_ids, aa_ids, edge_index)

        if self.projector is not None:
            u = self.projector(u)

        z = u / u.norm(dim=1, keepdim=True, p=2)
        scores = torch.einsum('bf, kf -> bk', z, self.prototype)

        return {'features': u, 'scores': scores}


def sinkhorn(scores, eps=0.05, num_iters=3):
    # Sinkhorn-Knopp algorithm
    Q = torch.exp(scores / eps)
    Q /= Q.sum()
    B, K = Q.shape
    u = torch.zeros(K, device=Q.device)
    c, r = torch.ones(K, device=Q.device) / K, torch.ones(B, device=Q.device) / B
    for _ in range(num_iters):
        # column normalization
        u = Q.sum(dim=0)                      # [K]
        Q *= (c / u).unsqueeze(0)             # [K] -> [1, K] -> [B, K]
        # row normalization
        Q *= (r / Q.sum(dim=1)).unsqueeze(1)  # [B] -> [B, 1] -> [B, K]

    # row normalization 
    # return the probability of each sample to be assigned to each prototype
    return (Q / Q.sum(dim=1, keepdim=True))  # [B, K]


def get_random_rotation_matrix(batch_size, device=None):
    # generate a batch of random quaternion
    q = torch.rand(batch_size, 4, device=device)
    # normalize the quaternion
    q = q / torch.norm(q, dim=1, keepdim=True)
    # convert quaternion to rotation matrix
    q0, q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([
        1 - 2 * q2**2 - 2 * q3**2, 2 * q1 * q2 - 2 * q0 * q3, 2 * q1 * q3 + 2 * q0 * q2,
        2 * q1 * q2 + 2 * q0 * q3, 1 - 2 * q1**2 - 2 * q3**2, 2 * q2 * q3 - 2 * q0 * q1,
        2 * q1 * q3 - 2 * q0 * q2, 2 * q2 * q3 + 2 * q0 * q1, 1 - 2 * q1**2 - 2 * q2**2
    ], dim=1).reshape(batch_size, 3, 3)

    return R


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(
        model: nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        train_loader: DataLoader = None,
        eval_loader: DataLoader = None,
        log_interval: int = 100,
        num_epochs: int = 100,
        device: str = 'cpu',
        temperature: float = 0.1,
        use_fp16: bool = False,
        sinkhorn_eps: float = 0.05,
        sinkhorn_iters: int = 3,
        freeze_prototypes_nepochs: int = 1
):  
    epoch_losses = []
    epoch_eval_losses = []

    if use_fp16:
        scaler = torch.cuda.amp.GradScaler(init_scale=2**12)

    for epoch in range(num_epochs):
        
        running_losses = 0.0
        # training
        for it, batch in enumerate(train_loader):
            model.train()

            # batch is a tuple of (atom_id, amino_acids_id, DataBatch[x, edge_index])
            atom_id, amino_acids_id, data_batch = batch
            batch_size = atom_id.size(0)
            edge_index = data_batch.edge_index
            x_s = data_batch.x
            
            # data augumentation: random rotation
            x_t = torch.einsum(
                'bij,bnj->bni', 
                get_random_rotation_matrix(
                    atom_id.size(0), 
                    device=atom_id.device),           # (B, 3, 3)
                data_batch.x.view(batch_size, -1, 3)  # (B, N, 3)
            ).reshape(-1, 3)
            
            # backward propagation & optimization
            optimizer.zero_grad()

            if use_fp16:
                with autocast():
                    scores_s, scores_t, q_s, q_t = model(
                        x_s.to(device), x_t.to(device), 
                        atom_id.to(device), amino_acids_id.to(device), 
                        edge_index.to(device),
                        sinkhorn_eps=sinkhorn_eps,
                        sinkhorn_iters=sinkhorn_iters
                    )
                
                    # compute swap prediction loss
                    loss_fn = nn.CrossEntropyLoss()
                    loss = 0.5 * (loss_fn(scores_s / temperature, q_t)
                                + loss_fn(scores_t / temperature, q_s))
                
                scaler.scale(loss).backward()

                # freeze the prototypes to help begining optimization
                if epoch < freeze_prototypes_nepochs:
                    for name, p in model.named_parameters():
                        if "prototype" in name:
                            p.grad = None

                scaler.step(optimizer)
                scaler.update()
            else:
                scores_s, scores_t, q_s, q_t = model(
                    x_s.to(device), x_t.to(device), 
                    atom_id.to(device), amino_acids_id.to(device), 
                    edge_index.to(device),
                    sinkhorn_eps=sinkhorn_eps,
                    sinkhorn_iters=sinkhorn_iters
                )
            
                # compute swap prediction loss
                loss_fn = nn.CrossEntropyLoss()
                loss = 0.5 * (loss_fn(scores_s / temperature, q_t)
                            + loss_fn(scores_t / temperature, q_s))

                loss.backward()

                if epoch < freeze_prototypes_nepochs:
                    for name, p in model.named_parameters():
                        if "prototype" in name:
                            p.grad = None

                optimizer.step()

            # update learning rate
            if scheduler is not None:
                scheduler.step()

            # normalize prototype vectors
            with torch.no_grad():
                w = model.prototype.data.clone()
                w = F.normalize(w, dim=1, p=2)
                model.prototype.data.copy_(w)

            if it % log_interval == 0:
                print(f'Epoch: {epoch + 1}/{num_epochs}, '
                      f'Iter: {it}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Lr: {optimizer.param_groups[0]["lr"]:.6f}')
                print(q_s[0])

            running_losses += loss.item()

        epoch_losses.append(running_losses / it)

        # validation
        running_eval_losses = 0.0
        for it, batch in enumerate(eval_loader):
            model.eval()

            atom_id, amino_acids_id, data_batch = batch
            batch_size = atom_id.size(0)
            edge_index = data_batch.edge_index

            x_s = data_batch.x
            x_t = torch.einsum(
                'bij,bnj->bni', 
                get_random_rotation_matrix(
                    atom_id.size(0), 
                    device=atom_id.device),           # (B, 3, 3)
                data_batch.x.view(batch_size, -1, 3)  # (B, N, 3)
            ).reshape(-1, 3)

            with torch.no_grad():
                scores_s, scores_t, q_s, q_t = model(
                    x_s.to(device), x_t.to(device), 
                    atom_id.to(device), amino_acids_id.to(device), 
                    edge_index.to(device),
                    sinkhorn_eps=sinkhorn_eps,
                    sinkhorn_iters=sinkhorn_iters
                )
            
                loss_fn = nn.CrossEntropyLoss()
                loss = 0.5 * (loss_fn(scores_s / temperature, q_t)
                            + loss_fn(scores_t / temperature, q_s))

            running_eval_losses += loss.item()

        epoch_eval_losses.append(running_eval_losses / it)

        print(f'\nEpoch: {epoch + 1}/{num_epochs}, '
              f'train_loss: {epoch_losses[-1]:.4f}, '
              f'eval_loss: {epoch_eval_losses[-1]:.4f}\n')
        
    return epoch_losses, epoch_eval_losses
        

def cos_lr(step, warmup_steps, total_steps, initial_lr, min_lr=1e-7):
    lr_scale = 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) 
                                 / (total_steps - warmup_steps)))
    return (initial_lr - min_lr) * lr_scale + min_lr


def get_cos_lr_scheduler(
        optimizer, warmup_steps, total_steps, 
        warmup_init_lr, max_lr, min_lr=1e-7
    ):
    """
    Cosine learning rate scheduler
                   linear         cosine
    warmup_init_lr -----> max_lr ------> min_lr
    """
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step:
            cos_lr(
                step, warmup_steps, total_steps,
                max_lr, min_lr
            ) if step > warmup_steps else 
                step / warmup_steps * (max_lr - warmup_init_lr) + warmup_init_lr
    )
    return scheduler


# Copy from nvidia/apex. https://github.com/NVIDIA/apex/blob/master/apex/parallel/LARC.py
class LARC(object):
    """
    :class:`LARC` is a pytorch implementation of both the scaling and clipping variants of LARC,
    in which the ratio between gradient and parameter magnitudes is used to calculate an adaptive 
    local learning rate for each individual parameter. The algorithm is designed to improve
    convergence of large batch training.
     
    See https://arxiv.org/abs/1708.03888 for calculation of the local learning rate.
    In practice it modifies the gradients of parameters as a proxy for modifying the learning rate
    of the parameters. This design allows it to be used as a wrapper around any torch.optim Optimizer.
    ```
    model = ...
    optim = torch.optim.Adam(model.parameters(), lr=...)
    optim = LARC(optim)
    ```
    It can even be used in conjunction with apex.fp16_utils.FP16_optimizer.
    ```
    model = ...
    optim = torch.optim.Adam(model.parameters(), lr=...)
    optim = LARC(optim)
    optim = apex.fp16_utils.FP16_Optimizer(optim)
    ```
    Args:
        optimizer: Pytorch optimizer to wrap and modify learning rate for.
        trust_coefficient: Trust coefficient for calculating the lr. See https://arxiv.org/abs/1708.03888
        clip: Decides between clipping or scaling mode of LARC. If `clip=True` the learning rate is set to `min(optimizer_lr, local_lr)` for each parameter. If `clip=False` the learning rate is set to `local_lr*optimizer_lr`.
        eps: epsilon kludge to help with numerical stability while calculating adaptive_lr
    """

    def __init__(self, optimizer, trust_coefficient=0.02, clip=True, eps=1e-8):
        self.optim = optimizer
        self.trust_coefficient = trust_coefficient
        self.eps = eps
        self.clip = clip

    def __getstate__(self):
        return self.optim.__getstate__()

    def __setstate__(self, state):
        self.optim.__setstate__(state)

    @property
    def state(self):
        return self.optim.state

    def __repr__(self):
        return self.optim.__repr__()

    @property
    def param_groups(self):
        return self.optim.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.optim.param_groups = value
    
    def state_dict(self):
        return self.optim.state_dict()

    def load_state_dict(self, state_dict):
        self.optim.load_state_dict(state_dict)

    def zero_grad(self):
        self.optim.zero_grad()

    def add_param_group(self, param_group):
        self.optim.add_param_group( param_group)

    def step(self):
        with torch.no_grad():
            weight_decays = []
            for group in self.optim.param_groups:
                # absorb weight decay control from optimizer
                weight_decay = group['weight_decay'] if 'weight_decay' in group else 0
                weight_decays.append(weight_decay)
                group['weight_decay'] = 0
                for p in group['params']:
                    if p.grad is None:
                        continue
                    param_norm = torch.norm(p.data)
                    grad_norm = torch.norm(p.grad.data)

                    if param_norm != 0 and grad_norm != 0:
                        # calculate adaptive lr + weight decay
                        adaptive_lr = self.trust_coefficient * (param_norm) / (grad_norm + param_norm * weight_decay + self.eps)

                        # clip learning rate for LARC
                        if self.clip:
                            # calculation of adaptive_lr so that when multiplied by lr it equals `min(adaptive_lr, lr)`
                            adaptive_lr = min(adaptive_lr/group['lr'], 1)

                        p.grad.data += weight_decay * p.data
                        p.grad.data *= adaptive_lr

        self.optim.step()
        # return weight decay control to optimizer
        for i, group in enumerate(self.optim.param_groups):
            group['weight_decay'] = weight_decays[i]

            


        
