import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.ops import edge_softmax
from torch.nn.parameter import Parameter
import torch.nn.init as init
import math

class GATConv(nn.Module):
    def __init__(self, args, curvature, in_size, num_heads, aggregator_type='attention', feat_drop=0., attn_drop=0., negative_slope=0.2, residual=False, activation=None,):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self.in_size = in_size
        self.residual = residual
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, in_size)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, in_size)))

        if aggregator_type == 'pool':
            self.fc_pool = nn.Linear(in_size*num_heads, in_size*num_heads)
            nn.init.xavier_normal_(self.fc_pool.weight, gain=1.414)
        self.aggre_type = aggregator_type
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.activation = activation
        self.ball = geoopt.PoincareBall(c=curvature)
        self.args = args

        nn.init.xavier_normal_(self.attn_l, gain=1.414)
        nn.init.xavier_normal_(self.attn_r, gain=1.414)

    def forward(self, graph, feat):
        with graph.local_scope():
            if self.aggre_type == 'attention':
                feat = self.ball.logmap0(feat)
                h_src = h_dst = self.feat_drop(feat).view(-1, self._num_heads, self.in_size)

                el = (h_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
                er = (h_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
                graph.srcdata.update({'ft': h_src, 'el': el})
                graph.dstdata.update({'er': er})
                # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
                graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
                e = self.leaky_relu(graph.edata.pop('e'))
                # compute softmax
                graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
                # message passing
                graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
                rst = graph.dstdata['ft']
                rst = self.ball.expmap0(rst)
                # residual
                if self.residual:
                    rst = rst + h_dst
                # activation
                if self.activation:
                    rst = self.activation(rst)
            elif self.aggre_type == 'mean':
                h_src = self.feat_drop(feat).view(-1, self.in_size*self._num_heads)
                h_src = self.ball.logmap0(h_src)
                graph.srcdata['ft'] = h_src
                graph.update_all(fn.copy_u('ft', 'm'), fn.mean('m', 'ft'))
                rst = graph.dstdata['ft']
                rst = self.ball.expmap0(rst)

            elif self.aggre_type == 'pool':
                h_src = self.feat_drop(feat).view(-1, self.in_size*self._num_heads)
                h_src = self.ball.logmap0(h_src)
                graph.srcdata['ft'] = F.relu(self.fc_pool(h_src))
                graph.update_all(fn.copy_u('ft', 'm'), fn.mean('m', 'ft'))
                rst = graph.dstdata['ft']
                rst = self.ball.expmap0(rst)
            if self.aggre_type == 'attention':
                return rst, e
            else:
                return rst, 0
    def fake_forward_layer(self, graph, feat, attention):
        with graph.local_scope():
            if self.aggre_type == 'attention':
                feat = self.ball.logmap0(feat)
                h_src = h_dst = self.feat_drop(feat).view(-1, self._num_heads, self.in_size)

                el = (h_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
                er = (h_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
                graph.srcdata.update({'ft': h_src, 'el': el})
                graph.dstdata.update({'er': er})
                # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
                graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
                e = self.leaky_relu(graph.edata.pop('e'))
                # 生成反事实的注意力
                # print(e.shape)
                #e = torch.rand(e.shape).to(self.args.device)#attention
                e = torch.rand(e.shape,device=self.args.device)#.to(self.args.device)
                # compute softmax
                graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
                # message passing
                graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
                rst = graph.dstdata['ft']
                rst = self.ball.expmap0(rst)
                # residual
                if self.residual:
                    rst = rst + h_dst
                # activation
                if self.activation:
                    rst = self.activation(rst)
            elif self.aggre_type == 'mean':
                h_src = self.feat_drop(feat).view(-1, self.in_size*self._num_heads)
                h_src = self.ball.logmap0(h_src)
                graph.srcdata['ft'] = h_src
                graph.update_all(fn.copy_u('ft', 'm'), fn.mean('m', 'ft'))
                rst = graph.dstdata['ft']
                rst = self.ball.expmap0(rst)

            elif self.aggre_type == 'pool':
                h_src = self.feat_drop(feat).view(-1, self.in_size*self._num_heads)
                h_src = self.ball.logmap0(h_src)
                graph.srcdata['ft'] = F.relu(self.fc_pool(h_src))
                graph.update_all(fn.copy_u('ft', 'm'), fn.mean('m', 'ft'))
                rst = graph.dstdata['ft']
                rst = self.ball.expmap0(rst)
            return rst

class HetGCNLayer(nn.Module):
    def __init__(self, args, curvature, in_size, aggregator_type='attention', num_heads=8, feat_drop=0., attn_drop=0., negative_slope=0.2, residual=False, activation=None):
        super(HetGCNLayer, self).__init__()
        self.num_heads = num_heads
        self.in_size = in_size
        if aggregator_type == 'pool':
            self.fc_pool = nn.Linear(in_size*num_heads, in_size*num_heads)
            nn.init.xavier_normal_(self.fc_pool.weight, gain=1.414)
        self.aggre_type = aggregator_type

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, self.num_heads, in_size)))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.residual = residual
        self.activation = activation
        self.ball = geoopt.PoincareBall(c=curvature)
        self.args = args

        nn.init.xavier_normal_(self.attn_l, gain=1.414)

    def forward(self, g, feat):
        with g.local_scope():
            if self.aggre_type == 'attention':
                if isinstance(feat, tuple):
                    h_src = self.feat_drop(feat[0]).view(-1, self.num_heads, self.in_size)
                    h_dst = self.feat_drop(feat[1]).view(-1, self.num_heads, self.in_size)
                h_src = self.ball.logmap0(h_src)
                h_dst = self.ball.logmap0(h_dst)
                el = (h_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
                g.srcdata.update({'ft': h_src, 'el': el})
                g.apply_edges(fn.copy_u('el', 'e'))
                e = self.leaky_relu(g.edata.pop('e'))
                g.edata['a'] = self.attn_drop(edge_softmax(g, e))
                g.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
                rst = g.dstdata['ft'].flatten(1)
                rst = self.ball.expmap0(rst)
                if self.residual:
                    rst = rst + h_dst
                if self.activation:
                    rst = self.activation(rst)

            elif self.aggre_type == 'mean':
                h_src = self.feat_drop(feat[0]).view(-1, self.in_size*self.num_heads)
                h_src = self.ball.logmap0(h_src)
                g.srcdata['ft'] = h_src
                g.update_all(fn.copy_u('ft', 'm'), fn.mean('m', 'ft'))
                rst = g.dstdata['ft']
                rst = self.ball.expmap0(rst)

            elif self.aggre_type == 'pool':
                h_src = self.feat_drop(feat[0]).view(-1, self.in_size*self.num_heads)
                h_src = self.ball.logmap0(h_src)
                g.srcdata['ft'] = F.relu(self.fc_pool(h_src))
                g.update_all(fn.copy_u('ft', 'm'), fn.mean('m', 'ft'))
                rst = g.dstdata['ft']
                rst = self.ball.expmap0(rst)
            if self.aggre_type == 'attention':
                return rst, e
            else:
                return rst, 0
    def fake_forward_layer(self, g, feat, attention):
        with g.local_scope():
            if self.aggre_type == 'attention':
                if isinstance(feat, tuple):
                    h_src = self.feat_drop(feat[0]).view(-1, self.num_heads, self.in_size)
                    h_dst = self.feat_drop(feat[1]).view(-1, self.num_heads, self.in_size)
                h_src = self.ball.logmap0(h_src)
                h_dst = self.ball.logmap0(h_dst)
                el = (h_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
                g.srcdata.update({'ft': h_src, 'el': el})
                g.apply_edges(fn.copy_u('el', 'e'))
                e = self.leaky_relu(g.edata.pop('e'))
                # 生成反事实的注意力
                #print(e.shape)
                # e = torch.rand(e.shape).to(self.args.device)  # attention
                e = torch.rand(e.shape,device=self.args.device)
                g.edata['a'] = self.attn_drop(edge_softmax(g, e))
                g.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
                rst = g.dstdata['ft'].flatten(1)
                rst = self.ball.expmap0(rst)
                if self.residual:
                    rst = rst + h_dst
                if self.activation:
                    rst = self.activation(rst)

            elif self.aggre_type == 'mean':
                h_src = self.feat_drop(feat[0]).view(-1, self.in_size*self.num_heads)
                h_src = self.ball.logmap0(h_src)
                g.srcdata['ft'] = h_src
                g.update_all(fn.copy_u('ft', 'm'), fn.mean('m', 'ft'))
                rst = g.dstdata['ft']
                rst = self.ball.expmap0(rst)

            elif self.aggre_type == 'pool':
                h_src = self.feat_drop(feat[0]).view(-1, self.in_size*self.num_heads)
                h_src = self.ball.logmap0(h_src)
                g.srcdata['ft'] = F.relu(self.fc_pool(h_src))
                g.update_all(fn.copy_u('ft', 'm'), fn.mean('m', 'ft'))
                rst = g.dstdata['ft']
                rst = self.ball.expmap0(rst)
            return rst

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)                    # (M, 1)
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1)   # beta就是attention
        return (beta * z).sum(1), beta                       # (N, D * K)
    def fake_forward_layer(self, z, attention):
        return (attention * z).sum(1)




class HMSGLayer(nn.Module):
    def __init__(self, args, curvature, meta_paths, in_size, aggre_type, layer_num_heads, dropout):
        super(HMSGLayer, self).__init__()
        self.nunm_heads = layer_num_heads
        self.semantic_attention_m = SemanticAttention(in_size=in_size * layer_num_heads)
        # self.semantic_attention_a = SemanticAttention(in_size=in_size * layer_num_heads)
        # self.semantic_attention_d = SemanticAttention(in_size=in_size * layer_num_heads)
        self.curvature = curvature
        self.hsmg_layers = nn.ModuleList()
        self.activation_function = HypAct(curvature=curvature, act=F.elu)
        for i in range(len(meta_paths)):
            if meta_paths[i][0][0] == meta_paths[i][-1][-1]:
                self.hsmg_layers.append(GATConv(args, curvature, in_size, num_heads=layer_num_heads, aggregator_type=aggre_type,
                    feat_drop=dropout, attn_drop=dropout, activation=self.activation_function, residual=False))
            else:
                self.hsmg_layers.append(HetGCNLayer(args, curvature, in_size, aggre_type, self.nunm_heads,
                   dropout, dropout, activation=self.activation_function, residual=False))

        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)
        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        attention_list = []
        semantic_embeddings = {'movie':[], 'actor':[], 'director':[]}
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                if len(meta_path) > 1:
                    self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(g, meta_path)  # return a homogeneous or unidirectional bipartite graphs
                elif len(meta_path) == 1:
                    if meta_path in {('am',),}:
                        print('******************am**********************')
                        self._cached_coalesced_graph[meta_path] = dgl.edge_type_subgraph(g, [('actor', 'am', 'movie')])
                    elif meta_path in { ('dm',)}:
                        print('******************dm**********************')
                        self._cached_coalesced_graph[meta_path] = dgl.edge_type_subgraph(g, [('director', 'dm', 'movie')])
                
        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            if new_g.is_homogeneous:
                ntype = new_g.ntypes[0]
                embedding, attention = self.hsmg_layers[i](new_g, h[ntype])
                semantic_embeddings[ntype].append(embedding.flatten(1))
                attention_list.append(attention)
            else:   
                if meta_path in {('am',),}:
                    h_ = (h['actor'], h['movie'])
                    embedding, attention = self.hsmg_layers[i](new_g, h_)
                    semantic_embeddings['movie'].append(embedding)
                    attention_list.append(attention)
                elif meta_path in { ('dm',)}:
                    h_ = (h['director'], h['movie'])
                    embedding, attention = self.hsmg_layers[i](new_g, h_)
                    semantic_embeddings['movie'].append(embedding)
                    attention_list.append(attention)

        embedings = {}
        for ntype in semantic_embeddings.keys():
            if ntype=='movie':
                semantic_embeddings[ntype] = torch.stack(semantic_embeddings[ntype], dim=1) 
                embedings[ntype], attention = self.semantic_attention_m(semantic_embeddings[ntype])
                attention_list.append(attention)
            # elif ntype=='actor' and semantic_embeddings[ntype]:
            #     semantic_embeddings[ntype] = torch.stack(semantic_embeddings[ntype], dim=1)
            #     embedings[ntype] = self.semantic_attention_a(semantic_embeddings[ntype])
            # elif ntype=='director' and semantic_embeddings[ntype]:
            #     semantic_embeddings[ntype] = torch.stack(semantic_embeddings[ntype], dim=1)
            #     embedings[ntype] = self.semantic_attention_d(semantic_embeddings[ntype])
        return embedings, attention_list

    def fake_forward(self, g, h, attention_list):
        semantic_embeddings = {'movie': [], 'actor': [], 'director': []}
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                if len(meta_path) > 1:
                    self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(g,meta_path)  # return a homogeneous or unidirectional bipartite graphs
                elif len(meta_path) == 1:
                    if meta_path in {('am',), }:
                        print('******************am**********************')
                        self._cached_coalesced_graph[meta_path] = dgl.edge_type_subgraph(g, [('actor', 'am', 'movie')])
                    elif meta_path in {('dm',)}:
                        print('******************dm**********************')
                        self._cached_coalesced_graph[meta_path] = dgl.edge_type_subgraph(g,
                                                                                         [('director', 'dm', 'movie')])

        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            if new_g.is_homogeneous:
                ntype = new_g.ntypes[0]
                semantic_embeddings[ntype].append(self.hsmg_layers[i].fake_forward_layer(new_g, h[ntype], attention_list[i]).flatten(1))
            else:
                if meta_path in {('am',), }:
                    h_ = (h['actor'], h['movie'])
                    semantic_embeddings['movie'].append(self.hsmg_layers[i].fake_forward_layer(new_g, h_, attention_list[i]))
                elif meta_path in {('dm',)}:
                    h_ = (h['director'], h['movie'])
                    semantic_embeddings['movie'].append(self.hsmg_layers[i].fake_forward_layer(new_g, h_, attention_list[i]))

        embedings = {}
        for ntype in semantic_embeddings.keys():
            if ntype == 'movie':
                semantic_embeddings[ntype] = torch.stack(semantic_embeddings[ntype], dim=1)
                embedings[ntype], attention = self.semantic_attention_m(semantic_embeddings[ntype])   # 别把这里这个attention忘记了
                # embedings[ntype] = self.semantic_attention_m.fake_forward_layer(semantic_embeddings[ntype], attention_list[-1])
                # embedings[ntype] = self.semantic_attention_m.fake_forward_layer(semantic_embeddings[ntype], attention)
            # elif ntype=='actor' and semantic_embeddings[ntype]:
            #     semantic_embeddings[ntype] = torch.stack(semantic_embeddings[ntype], dim=1)
            #     embedings[ntype] = self.semantic_attention_a(semantic_embeddings[ntype])
            # elif ntype=='director' and semantic_embeddings[ntype]:
            #     semantic_embeddings[ntype] = torch.stack(semantic_embeddings[ntype], dim=1)
            #     embedings[ntype] = self.semantic_attention_d(semantic_embeddings[ntype])
        return embedings


class HMSG(nn.Module):
    def __init__(self, args, manifold, curvature, meta_paths, in_size, hidden_size, out_size, aggre_type, num_heads, dropout):
        super(HMSG, self).__init__()
        """
        self.fc_m = nn.Linear(in_size['movie'], hidden_size*num_heads, bias=True)
        self.fc_a = nn.Linear(in_size['actor'], hidden_size*num_heads, bias=True)
        self.fc_d = nn.Linear(in_size['director'], hidden_size*num_heads, bias=True)
        """
        self.fc_m = HypLinear(manifold=manifold, in_features=in_size['movie'], out_features=hidden_size*num_heads, c=curvature, dropout=dropout, use_bias=True)
        self.fc_a = HypLinear(manifold=manifold, in_features=in_size['actor'], out_features=hidden_size*num_heads, c=curvature, dropout=dropout, use_bias=True)
        self.fc_d = HypLinear(manifold=manifold, in_features=in_size['director'], out_features=hidden_size*num_heads, c=curvature, dropout=dropout, use_bias=True)

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.layers = HMSGLayer(args, curvature, meta_paths, hidden_size, aggre_type, num_heads, dropout)
        self.predict = HypLinear(manifold=manifold, in_features=hidden_size * num_heads, out_features=out_size, c=curvature, dropout=dropout, use_bias=True) #nn.Linear(hidden_size * num_heads, out_size)
        nn.init.xavier_normal_(self.fc_m.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc_a.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc_d.weight, gain=1.414)
        self.history_att = None
        self.args = args

    def forward(self, g, inputs):
        h_trans = {}
        h_trans['movie'] = self.fc_m(inputs['movie']).view(-1, self.num_heads, self.hidden_size)
        h_trans['actor'] = self.fc_a(inputs['actor']).view(-1, self.num_heads, self.hidden_size)
        h_trans['director'] = self.fc_d(inputs['director']).view(-1, self.num_heads, self.hidden_size)
        TDE = 0
        h_trans_, attention_list = self.layers(g, h_trans)
        h_trans_fake = self.layers.fake_forward(g, h_trans, attention_list)
        TDE = self.predict(h_trans_['movie']) - self.predict(h_trans_fake['movie'])
        """
        if self.history_att is None:
            print("历史注意力为空")
            self.history_att = torch.randn(attention.shape).to(self.args.device)#attention
        else:
            h_trans_fake = self.layers.fake_forward(g, h_trans, self.history_att)
            self.history_att = torch.randn(attention.shape).to(self.args.device)#attention
            TDE = self.predict(h_trans_['movie'])-self.predict(h_trans_fake['movie'])
        """
        """
        h_trans_, attention_list = self.layers(g, h_trans)
        # h_trans_, attention_list = self.layers_2(g, h_trans_)
        if self.history_att is None:
            print("历史注意力为空")

        self.history_att = attention_list
        new_attention_list = [torch.ones_like(tensor, dtype=torch.float32) for tensor in attention_list.copy()]
        # else:
        if not new_attention_list:  # 如果 new_attention_list 为空，则重新初始化
            print("new_attention_list为空")
            new_attention_list = [torch.ones_like(tensor, dtype=torch.float32) for tensor in attention_list]

        h_trans_fake = self.layers.fake_forward(g, h_trans, new_attention_list)
        self.history_att = attention_list
        TDE = self.predict(h_trans_['movie']) - self.predict(h_trans_fake['movie'])
        """
        return h_trans_['movie'], self.predict(h_trans_['movie']), TDE



class HypLinear(nn.Module):
    """
    Poincare linear layer.
    """
    def __init__(self, manifold, in_features, out_features, c, dropout=0.6, use_bias=True):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = Parameter(torch.Tensor(out_features), requires_grad=True)
        self.weight = Parameter(torch.Tensor(out_features, in_features), requires_grad=True)
        self.reset_parameters()
        self.ball = geoopt.PoincareBall(c=c)

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        """
        drop_weight = F.dropout(self.weight, p=self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        print(mv)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        """
        drop_weight = F.dropout(self.weight, p=self.dropout, training=self.training)
        mv = self.ball.mobius_matvec(drop_weight, x)
        #print(mv)
        #res = self.ball.proj_(mv, self.c)
        if self.use_bias:
            bias = self.bias.view(1, -1)
            hyp_bias = self.ball.expmap0(bias)
            #hyp_bias = self.ball.proj(hyp_bias, self.c)
            res = self.ball.mobius_add(mv, hyp_bias)
            #res = self.ball.proju().proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )

class HypAct(nn.Module):
    def __init__(self, curvature, act):
        super(HypAct, self).__init__()
        self.curvature = curvature
        self.act = act
        self.ball = geoopt.PoincareBall(c=curvature)
    def forward(self, x):
        x = self.act(self.ball.logmap0(x))
        x = self.ball.expmap0(x)
        return x