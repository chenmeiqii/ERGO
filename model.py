import torch
import torch.nn as nn
from transformers import LongformerModel
import torch.nn.functional as F
import math
from torch.nn.utils.rnn import pad_sequence

class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += (1 - alpha)
            self.alpha[1:] += alpha
        self.gamma = gamma

    def forward(self, preds, labels):
        preds = preds.view(-1, preds.size(-1))
        alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=-1)
        preds_logsoft = F.log_softmax(preds, dim=-1)
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),preds_logsoft)
        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

class Norm(torch.nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        self.alpha = torch.nn.Parameter(torch.ones(self.size))
        self.bias = torch.nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class RGTLayer(torch.nn.Module):

    def __init__(self, in_features, hidden_size, out_features, heads, dropout_att, dropout_emb):
        super(RGTLayer, self).__init__()
        self.nin = in_features
        self.nout = out_features
        self.head = heads
        self.hidden_size = hidden_size
        self.dropout_att = torch.nn.Dropout(dropout_att)
        self.dropout_emb = torch.nn.Dropout(dropout_emb)

        self.norm_1 = Norm(self.nin)
        self.norm_2 = Norm(self.nout)

        self.WQ = torch.nn.Linear(self.nin, heads * self.hidden_size)
        self.WK = torch.nn.Linear(self.nin, heads * self.hidden_size)
        self.WV = torch.nn.Linear(self.nin, heads * self.hidden_size)
        self.WO = torch.nn.Linear(heads * self.hidden_size, self.nout)

    def forward(self, events_pair, mh_self_attention_mask):
        b, ne, d = events_pair.shape
        x = events_pair
        x = self.norm_1(x)
        K = self.WK(x).view(-1, ne, self.head, self.hidden_size)
        Q = self.WQ(x).view(-1, ne, self.head, self.hidden_size)
        V = self.WV(x).view(-1, ne, self.head, self.hidden_size)
        K = K.transpose(1, 2)
        Q = Q.transpose(1, 2)
        V = V.transpose(1, 2)
        K = self.dropout_emb(K)

        tmp = torch.matmul(Q, K.transpose(-1, -2))
        if mh_self_attention_mask.shape[1] > 0:
            mh_self_attention_mask = mh_self_attention_mask.view(b, 1, ne, ne).expand(b, self.head, ne, ne)
        tmp.masked_fill_(mh_self_attention_mask, float("-inf"))
        e = F.softmax(tmp / math.sqrt(self.hidden_size), dim=-1)

        e = self.dropout_att(e)
        x = torch.matmul(e, V)
        x = x.transpose(1, 2).contiguous().view(-1, ne, self.head * self.hidden_size)
        x = self.WO(x)
        x = self.norm_2(x)
        return x, e


class LongformerCausalModel(nn.Module):

    def __init__(self, args):
        super(LongformerCausalModel, self).__init__()
        self.device = args.device
        self.pretrained_model = LongformerModel.from_pretrained(args.model_name_or_path)
        self.RGT_layer = RGTLayer(768 * 2, 768 * 2, 768 * 2, args.num_heads, args.dropout_att, args.dropout_emb)
        self.fc = nn.Linear(768 * 3, args.y_class)
        self.loss_type = args.loss_type
        self.focal_loss = focal_loss(alpha=args.class_weight, gamma=args.gamma, num_classes=args.y_class)
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='mean')

    def forward(self, enc_input_ids, enc_mask_ids, global_attention_mask, t1_pos, t2_pos, mh_self_attention_mask, target):
        enc_output = self.pretrained_model(enc_input_ids, attention_mask=enc_mask_ids, global_attention_mask=global_attention_mask)
        enc_hidden = enc_output[0]
        bs, sequence_len, emb_dim = enc_hidden.shape

        pad_event1_pos = pad_sequence([torch.tensor(pos) for pos in t1_pos]).t().to(self.device).long()  # ne x bs
        pad_event2_pos = pad_sequence([torch.tensor(pos) for pos in t2_pos]).t().to(self.device).long()
        cls = enc_hidden[:, 0, :].view(bs, 1, emb_dim).expand(bs, pad_event1_pos.shape[1], emb_dim)
        event1 = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(enc_hidden, pad_event1_pos)])
        event2 = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(enc_hidden, pad_event2_pos)])

        events_pair = torch.cat([event1, event2], dim=-1)

        events_pair, att = self.RGT_layer(events_pair, mh_self_attention_mask)

        opt = torch.cat((cls, events_pair), dim=-1)
        opt = self.fc(opt)

        opt = torch.cat([j[:len(i)] for i, j in zip(t1_pos, opt)], dim=0)
        target = torch.cat([torch.tensor(t) for t in target], dim=0).to(self.device)

        if self.loss_type == 'focal':
            loss = self.focal_loss(opt, target)
        else:
            loss = self.ce_loss(opt, target)
        return loss, opt