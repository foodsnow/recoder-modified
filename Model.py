from decodeTrans import decodeTransformerBlock
from Embedding import Embedding
from gcnnnormal import GCNNM
from gelu import GELU
from graphTransformer import graphTransformerBlock
from LayerNorm import LayerNorm
from Multihead_Attention import MultiHeadedAttention
from Multihead_Combination import MultiHeadedCombination
from postionEmbedding import PositionalEmbedding
from rightnTransfomer import rightTransformerBlock
from Transfomer import TransformerBlock
from TreeConvGen import TreeConvGen

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_logger import logger


class TreeAttEncoder(nn.Module):
    def __init__(self, args):

        logger.info('starting to initialize TreeAttEncoder')

        super(TreeAttEncoder, self).__init__()

        self.embedding_size = args.embedding_size
        self.nl_len = args.NlLen
        self.word_len = args.WoLen

        self.char_embedding = nn.Embedding(args.Vocsize, self.embedding_size)
        self.token_embedding = Embedding(args.Code_Vocsize, self.embedding_size)

        self.feed_forward_hidden = 4 * self.embedding_size

        self.conv = nn.Conv2d(self.embedding_size, self.embedding_size, (1, self.word_len))
        self.transformerBlocks = nn.ModuleList([TransformerBlock(self.embedding_size, 8, self.feed_forward_hidden, 0.1) for _ in range(3)])
        self.transformerBlocksTree = nn.ModuleList([TransformerBlock(self.embedding_size, 8, self.feed_forward_hidden, 0.1) for _ in range(3)])

    def forward(self, input_code, input_codechar, inputAd):

        logger.info('starting TreeAttEncoder.forward() pass')

        codemask = torch.gt(input_code, 0)
        charEm = self.char_embedding(input_codechar)
        charEm = self.conv(charEm.permute(0, 3, 1, 2))
        charEm = charEm.permute(0, 2, 3, 1).squeeze(dim=-2)
        x = self.token_embedding(input_code.long())
        for trans in self.transformerBlocksTree:
            x = trans.forward(x, codemask, charEm, inputAd, True)
        for trans in self.transformerBlocks:
            x = trans.forward(x, codemask, charEm)
        return x


class NlEncoder(nn.Module):
    def __init__(self, args):

        logger.info('starting to initialize NlEncoder')

        super(NlEncoder, self).__init__()
        self.embedding_size = args.embedding_size
        self.nl_len = args.NlLen
        self.word_len = args.WoLen
        self.char_embedding = nn.Embedding(args.Vocsize, self.embedding_size)
        self.feed_forward_hidden = 4 * self.embedding_size
        self.conv = nn.Conv2d(self.embedding_size,
                              self.embedding_size, (1, self.word_len))
        self.transformerBlocks = nn.ModuleList(
            [graphTransformerBlock(self.embedding_size, 8, self.feed_forward_hidden, 0.1) for _ in range(5)])
        self.pos_embedding = nn.Embedding(5, self.embedding_size)
        '''self.transformerBlocksTree = nn.ModuleList(
            [TransformerBlock(self.embedding_size, 8, self.feed_forward_hidden, 0.1) for _ in range(5)])'''

    def forward(self, nlencoding, nlad, input_nl, inputpos, charEm):

        logger.info('starting NlEncoder.forward() pass')

        nlmask = torch.gt(input_nl, 0)
        posEm = self.pos_embedding(inputpos)
        x = nlencoding
        for trans in self.transformerBlocks:
            x = trans.forward(x, nlmask, posEm, nlad, charEm)
        return x, nlmask


class CopyNet(nn.Module):
    def __init__(self, args):

        logger.info('starting to initialize CopyNet')

        super(CopyNet, self).__init__()
        self.embedding_size = args.embedding_size
        self.LinearSource = nn.Linear(
            self.embedding_size, self.embedding_size, bias=False)
        self.LinearTarget = nn.Linear(
            self.embedding_size, self.embedding_size, bias=False)
        self.LinearRes = nn.Linear(self.embedding_size, 1)
        self.LinearProb = nn.Linear(self.embedding_size, 4)

    def forward(self, source, traget):

        logger.info('starting CopyNet.forward() pass')

        sourceLinear = self.LinearSource(source)
        targetLinear = self.LinearTarget(traget)
        genP = self.LinearRes(F.tanh(sourceLinear.unsqueeze(
            1) + targetLinear.unsqueeze(2))).squeeze(-1)
        prob = F.softmax(self.LinearProb(traget), dim=-1)  # .squeeze(-1))
        return genP, prob


class Decoder(nn.Module):
    def __init__(self, args):

        logger.info('starting to initialize Decoder model instance')

        super(Decoder, self).__init__()

        self.embedding_size = args.embedding_size
        self.word_len = args.WoLen
        self.nl_len = args.NlLen
        self.code_len = args.CodeLen
        self.feed_forward_hidden = 4 * self.embedding_size

        self.conv = nn.Conv2d(self.embedding_size, self.embedding_size, (1, args.WoLen))
        self.path_conv = nn.Conv2d(self.embedding_size, self.embedding_size, (1, 10))
        self.rule_conv = nn.Conv2d(self.embedding_size, self.embedding_size, (1, 2))
        self.depth_conv = nn.Conv2d(self.embedding_size, self.embedding_size, (1, 40))

        self.cnum = args.cnum
        self.resLen = args.rulenum - args.NlLen - self.cnum

        self.encodeTransformerBlock = nn.ModuleList([rightTransformerBlock(self.embedding_size, 8, self.feed_forward_hidden, 0.1) for _ in range(9)])
        self.decodeTransformerBlocksP = nn.ModuleList([decodeTransformerBlock(self.embedding_size, 8, self.feed_forward_hidden, 0.1) for _ in range(2)])

        self.finalLinear = nn.Linear(self.embedding_size, 2048)
        self.resLinear = nn.Linear(2048, self.resLen)

        self.rule_token_embedding = nn.Embedding(args.Code_Vocsize, self.embedding_size)
        self.rule_embedding = nn.Embedding(args.rulenum, self.embedding_size)

        self.encoder = NlEncoder(args)
        self.layernorm = LayerNorm(self.embedding_size)
        self.activate = GELU()
        self.copy = CopyNet(args)
        self.copy2 = CopyNet(args)
        self.copy3 = CopyNet(args)
        self.dropout = nn.Dropout(p=0.1)
        self.depthembedding = nn.Embedding(40, self.embedding_size, padding_idx=0)
        self.char_embedding = nn.Embedding(args.Vocsize, self.embedding_size)

        self.gcnnm = GCNNM(self.embedding_size)
        self.position = PositionalEmbedding(self.embedding_size)

    def getBleu(self, losses, ngram):

        logger.info('starting Decoder.getBleu()')

        bleuloss = F.max_pool1d(losses.unsqueeze(1), ngram, 1).squeeze(1)
        bleuloss = torch.sum(bleuloss, dim=-1)
        return bleuloss

    def forward(self, inputnl, inputnlad, inputrule, inputruleparent, inputrulechild, inputParent, inputParentPath, inputdepth, inputcodechar, tmpf, tmpc, tmpindex, tmpchar, tmpindex2, rulead, antimask, inputRes=None, mode="train"):

        logger.info('starting Decoder.forward() pass')

        selfmask = antimask

        # selfmask = antimask.unsqueeze(0).repeat(inputtype.size(0), 1, 1).unsqueeze(1)
        # .unsqueeze(0).repeat(inputtype.size(0), 1, 1).float()

        admask = torch.eq(inputdepth, 1)
        rulemask = torch.gt(inputrule, 0)
        inputParent = inputParent.float()
        inputnlad = inputnlad.float()

        # encode_token
        charEm = self.char_embedding(tmpchar.long())
        charEm = self.conv(charEm.permute(0, 3, 1, 2))
        charEm = charEm.permute(0, 2, 3, 1).squeeze(dim=-2)
        rule_token_embedding = self.rule_token_embedding(tmpindex2[0])
        rule_token_embedding = rule_token_embedding + charEm[0]

        # encode_nl
        nlencoding = F.embedding(inputnl.long(), rule_token_embedding)
        charEm = self.char_embedding(inputcodechar.long())
        charEm = self.conv(charEm.permute(0, 3, 1, 2))
        charEm = charEm.permute(0, 2, 3, 1).squeeze(dim=-2)
        nlencoding += self.position(inputnl)
        nlencode, nlmask = self.encoder(nlencoding, inputnlad, inputnl, inputdepth, charEm)

        # encode_rule
        # self.rule_token_embedding(tmpc)
        childEm = F.embedding(tmpc, rule_token_embedding)
        childEm = self.conv(childEm.permute(0, 3, 1, 2))
        childEm = childEm.permute(0, 2, 3, 1).squeeze(dim=-2)
        childEm = self.layernorm(childEm)
        # self.rule_token_embedding(tmpf)
        fatherEm = F.embedding(tmpf, rule_token_embedding)
        ruleEmCom = self.rule_conv(torch.stack([fatherEm, childEm], dim=-2).permute(0, 3, 1, 2))
        ruleEmCom = self.layernorm(ruleEmCom.permute(0, 2, 3, 1).squeeze(dim=-2))
        x = self.rule_embedding(tmpindex[0])
        rulenoter = x[:self.cnum]
        ruleter = x[self.cnum:]
        for i in range(9):
            rulenoter = self.gcnnm(rulenoter, rulead[0], ruleEmCom[0]).view(self.cnum, self.embedding_size)
        ruleselect = torch.cat([rulenoter, ruleter], dim=0)
        # self.rule_embedding(inputrule)
        ruleEm = F.embedding(inputrule, ruleselect)
        # self.rule_token_embedding(inputrulechild)
        Ppath = F.embedding(inputrulechild, rule_token_embedding)
        ppathEm = self.path_conv(Ppath.permute(0, 3, 1, 2))
        ppathEm = ppathEm.permute(0, 2, 3, 1).squeeze(dim=-2)
        ppathEm = self.layernorm(ppathEm)
        x = self.dropout(ruleEm + self.position(inputrule))
        for trans in self.encodeTransformerBlock:
            x = trans(x, selfmask, nlencode, nlmask, ppathEm, inputParent, admask)
        decode = x

        # ppath
        # self.rule_token_embedding(inputParentPath)
        Ppath = F.embedding(inputParentPath, rule_token_embedding)
        ppathEm = self.path_conv(Ppath.permute(0, 3, 1, 2))
        ppathEm = ppathEm.permute(0, 2, 3, 1).squeeze(dim=-2)
        ppathEm = self.layernorm(ppathEm)
        x = self.dropout(ppathEm + self.position(inputrule))
        for trans in self.decodeTransformerBlocksP:
            x = trans(x, rulemask, decode, antimask, nlencode, nlmask)
        decode = x
        genP1, _ = self.copy2(rulenoter.unsqueeze(0), decode)
        res1 = F.softmax(genP1, dim=-1)
        genP, prob = self.copy(nlencode, decode)
        genP4, _ = self.copy3(nlencode, decode)
        copymask = nlmask.unsqueeze(1).repeat(1, inputrule.size(1), 1)
        copymask2 = admask.unsqueeze(1).repeat(1, inputrule.size(1), 1)
        genP = genP.masked_fill(copymask == 0, -1e9)
        genP4 = genP4.masked_fill(copymask2 == 0, -1e9)
        # genP = torch.cat([genP1, genP], dim=2)
        res2 = F.softmax(genP, dim=-1)
        res3 = F.softmax(self.resLinear(self.finalLinear(decode)), dim=-1)
        res4 = F.softmax(genP4, dim=-1)
        res1 = res1 * prob[:, :, 0].unsqueeze(-1)
        res2 = res2 * prob[:, :, 1].unsqueeze(-1)
        res3 = res3 * prob[:, :, 2].unsqueeze(-1)
        res4 = res4 * prob[:, :, 3].unsqueeze(-1)
        # genP = F.softmax(genP, dim=-1)

        # x = self.finalLinear(decode)
        # x = self.activate(x)
        # x = self.resLinear(x)
        # resSoftmax = F.softmax(x, dim=-1)

        # resSoftmax = resSoftmax * prob[:,:,0].unsqueeze(-1)
        # genP = genP * prob[:,:,1].unsqueeze(-1)
        # F.softmax(genP, dim=-1)#torch.cat([resSoftmax, genP], -1)

        resSoftmax = torch.cat([res1, res3, res2, res4], dim=-1)

        if mode != "train":
            return resSoftmax

        resmask = torch.gt(inputRes, 0)
        loss = -torch.log(torch.gather(resSoftmax, -1, inputRes.unsqueeze(-1)).squeeze(-1))
        loss = loss.masked_fill(resmask == 0, 0.0)
        resTruelen = torch.sum(resmask, dim=-1).float()
        totalloss = torch.mean(loss, dim=-1) * self.code_len / resTruelen
        # + (self.getBleu(loss, 2) + self.getBleu(loss, 3) + self.getBleu(loss, 4)) / resTruelen
        totalloss = totalloss
        # totalloss = torch.mean(totalloss)
        return totalloss, resSoftmax

