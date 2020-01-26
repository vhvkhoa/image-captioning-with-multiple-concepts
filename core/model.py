# =========================================================================================
# Implementation of "Show, Attend and Tell: Neural Caption Generator With Visual Attention".
# There are some notations.
# N is batch size.
# L is spacial size of feature vector (196).
# D is dimension of image feature vector (512).
# T is the number of time step which is equal to caption's length-1 (16).
# V is vocabulary size (about 10000).
# M is dimension of word vector which is embedding size (default is 512).
# H is dimension of hidden state (default is 1024).
# =========================================================================================

from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

class CaptionGenerator(nn.Module):
    def __init__(self, feature_dim=[196, 512], num_tags=23, embed_dim=512, hidden_dim=1024,
                  prev2out=True, ctx2out=True, enable_selector=True, dropout=0.5, len_vocab=10000, **kwargs):
        super(CaptionGenerator, self).__init__()
        self.prev2out = prev2out
        self.ctx2out = ctx2out
        self.enable_selector = enable_selector
        self.dropout = dropout
        self.V = len_vocab
        self.L = feature_dim[0] #number of regions
        self.D = feature_dim[1] #size of each region feature
        self.T = num_tags #number of tags
        self.A = kwargs.get('num_actions', 0)# number of actions
        self.M = embed_dim
        self.S = kwargs.get('scene_dim', 0)
        self.AD = kwargs.get('action_dim', 0)
        self.H = hidden_dim

        # Trainable parameters :
        self.lstm_cell = nn.LSTM(self.D + self.M + self.AD + self.S, self.H, dropout=0.5)
        self.hidden_state_init_layer = nn.Linear(self.H, self.H)
        self.cell_state_init_layer = nn.Linear(self.H, self.H)
        self.embedding_lookup = nn.Embedding(self.V, self.M)

        self.feats_proj_layer = nn.Linear(self.D, self.H)
        self.tags_proj_layer = nn.Linear(self.M, self.H)
        self.actions_proj_layer = nn.Linear(self.AD, self.H)
        self.scene_feats_proj_layer = nn.Linear(self.S, self.H)

        self.hidden_to_attention_layer = nn.Linear(self.H, self.H)
        self.features_attention_layer = nn.Linear(self.H, 1)
        self.tags_attention_layer = nn.Linear(self.H, 1)
        self.actions_attention_layer = nn.Linear(self.H, 1)

        self.features_selector_layer = nn.Linear(self.H, 1)
        self.tags_selector_layer = nn.Linear(self.H, 1)
        self.actions_selector_layer = nn.Linear(self.H, 1)
        self.scene_feats_selector_layer = nn.Linear(self.H, 1)

        self.hidden_to_embedding_layer = nn.Linear(self.H, self.M)
        self.features_context_to_embedding_layer = nn.Linear(self.D, self.M)
        self.embedding_to_output_layer = nn.Linear(self.M, self.V)

        # functional layers
        self.features_batch_norm = nn.BatchNorm1d(self.L)
        self.dropout = nn.Dropout(p=dropout)

    def get_initial_lstm(self, feats_proj, tags_proj, actions_proj, scene_feats_proj):
        feats_mean = torch.mean(feats_proj, 1)
        tags_mean = torch.mean(tags_proj, 1)
        actions_mean = torch.mean(actions_proj, 1)

        h = torch.tanh(self.hidden_state_init_layer(feats_mean + tags_mean + actions_mean + scene_feats_proj)).unsqueeze(0)
        c = torch.tanh(self.cell_state_init_layer(feats_mean + tags_mean + actions_mean + scene_feats_proj)).unsqueeze(0)
        return c, h

    def project_features(self, features, project_layer):
        batch, loc, dim = features.size()
        features_flat = features.view(-1, dim)
        features_proj = F.relu(project_layer(features_flat))
        features_proj = features_proj.view(batch, loc, -1)
        return features_proj

    def batch_norm(self, x):
        return self.features_batch_norm(x)

    def word_embedding(self, inputs):
        embed_inputs = self.embedding_lookup(inputs)  # (N, T, M) or (N, M)
        return embed_inputs

    def _attention(self, features, features_proj, hidden_states, attention_layer):
        h_att = F.relu(features_proj + self.hidden_to_attention_layer(hidden_states[-1]).unsqueeze(1))    # (N, L, D)
        loc, dim = features.size()[1:]
        out_att = self.attention_layer(h_att.view(-1, dim)).view(-1, loc)   # (N, L)
        alpha = F.softmax(out_att, dim=-1)
        context = torch.sum(features * alpha.unsqueeze(2), 1)   #(N, D)
        return context, alpha

    def _selector(self, context, hidden_states, selector_layer):
        beta = torch.sigmoid(selector_layer(hidden_states[-1]))    # (N, 1)
        context = context * beta
        return context, beta

    def _decode_lstm(self, x, h, feats_context, tags_context, actions_context, scene_context):
        h = self.dropout(h)
        h_logits = self.hidden_to_embedding_layer(h)

        if self.ctx2out:
            h_logits += self.features_context_to_embedding_layer(feats_context) + tags_context + actions_context + scene_context

        if self.prev2out:
            h_logits += x
        h_logits = torch.tanh(h_logits)

        h_logits = self.dropout(h_logits)
        out_logits = self.embedding_to_output_layer(h_logits)
        return out_logits
    
    def forward(self, features, features_proj, tags_embed, tags_proj, actions_embed, actions_proj, scene_feats_proj, past_captions, hidden_states, cell_states):
        emb_captions = self.word_embedding(inputs=past_captions)

        feats_context, feats_alpha = self._attention(features, features_proj, hidden_states, self.features_attention_layer)
        tags_context, tags_alpha = self._attention(tags_embed, tags_proj, hidden_states, self.tags_attention_layer)
        actions_context, actions_alpha = self._attention(actions_embed, actions_proj, hidden_states, self.actions_attention_layer)

        if self.enable_selector:
            feats_context, feats_beta = self._selector(feats_context, hidden_states, self.features_selector_layer)
            tags_context, tags_beta = self._selector(tags_context, hidden_states, self.tags_selector_layer)
            actions_context, actions_beta = self._selector(actions_context, hidden_states, self.actions_selector_layer)
            scene_context, scenes_beta = self._selector(scene_feats_proj, hidden_states, self.scene_feats_selector_layer)

        next_input = torch.cat((emb_captions, feats_context, tags_context, actions_context, scene_context), 1).unsqueeze(0)

        print(hidden_states.size())
        output, (next_hidden_states, next_cell_states) = self.lstm_cell(next_input, (hidden_states, cell_states))

        logits = self._decode_lstm(emb_captions, output.squeeze(0), feats_context, tags_context, actions_context, scene_context)

        return logits, feats_alpha, tags_alpha, actions_alpha, (next_hidden_states, next_cell_states)
