import copy
from .common import *
from .augmentation import drop_nodes, permute_edges, subgraph, mask_nodes
import torch.nn.functional as F
from .factorized import get_loader_solograph_chunk, get_loader_solograph_word, Solograph_HeteroGNN 

qa_conns = [
    ('q', 'q_a', 'a'),
    ('a', 'a_q', 'q'),
    ('a', 'a_a', 'a'),
    ('q', 'q_q', 'q'),
]

z_conns = [
    ('text', 'text_z', 'z'),
    ('z', 'z_text', 'text'),
    ('audio', 'audio_z', 'z'),
    ('z', 'z_audio', 'audio'),
    ('video', 'video_z', 'z'),
    ('z', 'z_video', 'video'),
    ('z', 'z_z', 'z'),
    ('q', 'q_z', 'z'),
    ('z', 'z_q', 'q'),
    ('a', 'a_z', 'z'),
    ('z', 'z_a', 'a'),

]

non_mod_nodes = ['q', 'a', 'a_idx', 'i_idx', 'z']


class SolographContrastive(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.q_lstm = MyLSTM(768,gc['graph_conv_in_dim'])
        self.a_lstm = MyLSTM(768,gc['graph_conv_in_dim'])

        self.projection_head = nn.Sequential(OrderedDict([
            ('fc0',   nn.Linear(gc['graph_conv_in_dim'], gc['graph_conv_in_dim'])),
            ('drop_1', nn.Dropout(p=gc['drop_1'])),
            ('relu0', nn.ReLU()),
            ('fc1',   nn.Linear(gc['graph_conv_in_dim'], gc['graph_conv_in_dim'])),
            ('drop_2', nn.Dropout(p=gc['drop_2'])),
            ('relu1', nn.ReLU())
        ]))

        self.hetero_gnn_contrastive = Solograph_HeteroGNN(gc['graph_conv_in_dim'], 1, gc['num_gat_layers'])

    def forward(self, batch):
        x_dict = batch.x_dict
        x_dict['q'] = self.q_lstm.step(batch['q']['x'].transpose(1,0))[1][0][0,:,:] # input to q_lstm must be of shape ([25, num_qs, 768])
        x_dict['a'] = self.a_lstm.step(batch['a']['x'].transpose(1,0))[1][0][0,:,:] # input to a_lstm must be of shape ([25, num_qs, 768])
        a_idx, i_idx = x_dict['a_idx'], x_dict['i_idx']
        del x_dict['a_idx'] # should not be used by heterognn
        del x_dict['i_idx']

        if gc['scene_mean']:
            # Data augmentation
            # You can use any single augmentations, or use a combination of them
            edge_index_dict_aug = copy.deepcopy(batch.edge_index_dict)

            # Do not copy Q and A information for augmentations, will cause error
            x_dict_aug = copy.deepcopy({k: v for k, v in x_dict.items() if k in ["text", "video", "audio", "z"]})
            visited_nodes = []
            visited_edges = [] # keep track of visited edge types
            for modality, x in x_dict_aug.items():
                for edge_index_type, edge_index in edge_index_dict_aug.items():
                # Selecting subgraphs according to the chosen modality to augment
                    if (modality in edge_index_type):
                        if gc['drop_nodes']:
                            raise NotImplementedError
                            x_dict_aug[modality], edge_index_dict_aug[edge_index_type] = \
                                drop_nodes(x, edge_index)
                        if gc['permute_edges'] and ("q" not in edge_index_type) and ("a" not in edge_index_type) and (edge_index_type not in visited_edges):
                            x_dict_aug[modality], edge_index_dict_aug[edge_index_type] = \
                                permute_edges(x, edge_index)
                            visited_edges.append(edge_index_type)
                        if gc['subgraph']:
                            raise NotImplementedError
                            x_dict_aug[modality], edge_index_dict_aug[edge_index_type] = \
                                subgraph(x, edge_index)
                        if gc['mask_nodes'] and (modality not in visited_nodes):
                            x_dict_aug[modality] = mask_nodes(x)
                            visited_nodes.append(modality)

            _, _, scene_rep_1 = self.hetero_gnn_contrastive(x_dict, batch.edge_index_dict, batch.batch_dict) # 216, 80; 432, 80; 216, 80
            # Add Q and A information back (we do not augment them)
            x_dict_aug['q'] = copy.copy(x_dict['q'])
            x_dict_aug['a'] = copy.copy(x_dict['a'])
            _, _, scene_rep_2 = self.hetero_gnn_contrastive(x_dict_aug, edge_index_dict_aug, batch.batch_dict) # 216, 80; 432, 80; 216, 80

            # InfoNCE contrastive loss from SimCLR implementation
            # https://github.com/google-research/simclr/blob/master/objective.py
            # Projection head
            scene_rep_1 = self.projection_head(scene_rep_1)
            scene_rep_2 = self.projection_head(scene_rep_2)

            # Actual loss
            LARGE_NUM = 1e9
            batch_size = scene_rep_1.shape[0]
            masks = F.one_hot(torch.arange(batch_size), batch_size).to(scene_rep_1.device)
            labels = F.one_hot(torch.arange(batch_size), batch_size * 2).to(scene_rep_1.device)

            logits_aa = torch.matmul(scene_rep_1, scene_rep_1.T) / gc['temperature']
            logits_aa = logits_aa - masks * LARGE_NUM # stablize training
            logits_bb = torch.matmul(scene_rep_2, scene_rep_2.T) / gc['temperature']
            logits_bb = logits_bb - masks * LARGE_NUM
            logits_ab = torch.matmul(scene_rep_1, scene_rep_2.T) / gc['temperature']
            logits_ba = torch.matmul(scene_rep_2, scene_rep_1.T) / gc['temperature']
            
            loss_a = F.cross_entropy(input=torch.cat([logits_ab, logits_aa], 1),
                    target=torch.argmax(labels, -1), reduction="none")
            loss_b = F.cross_entropy(input=torch.cat([logits_ba, logits_bb], 1),
                    target=torch.argmax(labels, -1), reduction="none")
            loss = loss_a + loss_b
            return torch.mean(loss)
        else:
            raise NotImplementedError
