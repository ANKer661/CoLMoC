import os
import torch
import numpy as np
import warnings
from modeling_grad import (
    SwavMoleculeTrain,
    ConformationDataset,
    count_parameters,
    train_model,
    get_cos_lr_scheduler,
    LARC
)
from torch_geometric.loader import DataLoader
warnings.filterwarnings('ignore', '.*TypedStorage is deprecated.*')

torch.manual_seed(3)
torch.cuda.manual_seed(3)

# load the data
PATH = r"./data"
atom_inputs = torch.load(f"{PATH}/atom_inputs.pt")
aa_inputs = torch.load(f"{PATH}/aa_inputs.pt")
edge_index = torch.load(f"{PATH}/edge_index.pt")
cartesian_coord = torch.load(f"{PATH}/coords.pt")
cartesian_coord = (cartesian_coord 
                   - cartesian_coord.sum(dim=1, keepdim=True) / cartesian_coord.shape[1])

# shuffle and devide the data into train(90%), eval(10%)
idx = torch.randperm(cartesian_coord.shape[0])
cartesian_coord = cartesian_coord[idx]
cartesian_coord_train = cartesian_coord[:int(len(cartesian_coord)*0.9)]
cartesian_coord_eval = cartesian_coord[int(len(cartesian_coord)*0.9):]

train_dataset = ConformationDataset(
    cartesian_coord_train, atom_inputs, aa_inputs, edge_index)
eval_dataset = ConformationDataset(
    cartesian_coord_eval, atom_inputs, aa_inputs, edge_index)


num_atoms = train_dataset[0][0].unique().size(0)
num_aa = train_dataset[0][1].unique().size(0)
num_prototypes = 16
proj_output_dim = 16
model = SwavMoleculeTrain(
    num_atom=num_atoms,
    num_aa=num_aa,
    coord_inputs_dim=3,
    node_dim=64,
    edge_attr_dim=64,
    global_attr_dim=64,
    dropout_prob=0.0,
    num_layers=12,
    num_prototypes=num_prototypes,
    use_projector=True,
    proj_hidden_dim=256,
    proj_output_dim=proj_output_dim,
    norm_type="graphnorm",
)


print('trainable parameters:', count_parameters(model))


# Parameters & hyperparameters
batch_size = 512
num_epochs = 100
temperature = 0.25
lr = 4.8e-2              # 3e-4, 3*1.414e-4  大的是1.5e-4 or 1e-4

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)
eval_loader = DataLoader(
    eval_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

if torch.cuda.is_available():
    device = 'cuda'
    model = model.cuda()
    print('using cuda')
else:
    device = 'cpu'
    print('using cpu')

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1, momentum=0.9, weight_decay=1e-6)
scheduler = get_cos_lr_scheduler(
    optimizer, warmup_steps=0.1 * num_epochs * len(train_loader), 
    total_steps=num_epochs * len(train_loader),
    warmup_init_lr=0.1 * lr,
    max_lr=lr,
    min_lr=1e-3 * lr
)
optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)

savepath = (f'./c{num_prototypes}'
            f'_d{proj_output_dim}_{temperature}'
            f'_eps0.05_LARS_{lr}')

print(f"savepath: {savepath}")

# train
loss, eval_loss = train_model(
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    train_loader=train_loader,
    eval_loader=eval_loader,
    num_epochs=num_epochs,
    device=device,
    temperature=temperature,
    use_fp16=True,
    sinkhorn_eps=0.05,
)

os.mkdir(savepath)
# save the model
torch.save({'model': model.state_dict()}, savepath + '/swav_mole.pth')

# save the loss
np.save(savepath + '/loss.npy', np.array(loss))
np.save(savepath + '/eval_loss.npy', np.array(eval_loss))




