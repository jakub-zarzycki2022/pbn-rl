import argparse
import random
import itertools

from collections import defaultdict
import gymnasium as gym
import gym_PBN
import torch
from gym_PBN.envs.bittner.base import findAttractors

import numpy as np
from sympy import symbols
from sympy.logic import SOPform

from alphaBio import AlphaBio
from bdq_model import BranchingDQN
from bdq_model.utils import AgentConfig
from graph_classifier import GraphClassifier

import math

from gbdq_model import GBDQ
from gbdq_model.utils import ExperienceReplayMemory#, AgentConfig

import seaborn as sns
from matplotlib import pyplot as plt

from gym_PBN.utils.get_attractors_from_cabean import get_attractors

parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int, required=True)
parser.add_argument('--model-path', required=True)
parser.add_argument('--attractors', type=int, default=3)
parser.add_argument('--assa-file', type=str)

args = parser.parse_args()

# model_cls = GraphClassifier
# model_name = "GraphClassifier"

# model_cls = GBDQ
# model_name = "GBDQ"

model_cls = BranchingDQN
model_name = "BranchingDQN"

N = args.n
model_path = args.model_path
# min_attractors = args.attractors

if True:
    env = gym.make("gym-PBN/BittnerMultiGeneral", N=N, min_attractors=args.attractors)

# A systems described by biologist we are cooperating with
# https://docs.google.com/document/d/1ACSjckbhof64rzLWtEUVE0lTKJuVde7dfFSTxNbii3o/edit
if False:
    env = gym.make(f"gym-PBN/PBNEnv",
                   N=17,
                   genes=[
                    "Pax7", "Myf5", "MyoD1", "MyoG", "miR1", "miR206", "FGF8", "SHH",
                    "Pax3", "Mrf4", "Mef2c", "Mef2a", "ID3", "WNT", "WNT3a", "T", "Msg1"
                   ],
                   logic_functions=[
                        [('not miR1 and not MyoG and not miR206', 1.0)],  # pax7
                        [('Pax7 or Pax3 or WNT or SHH', 1.0)],  # myf5
                        [('not ID3 and (FGF8 or Mef2c or Mef2a or Pax7 or SHH or WNT or Pax3)', 1.0)],  # myod1
                        [('MyoG or MyoD1', 1.0)],  # myog
                        [('Myf5', 1.0)],  # mir1
                        [('MyoG or Myf5 or MyoD1 or Mef2c', 1.0)],  # mir206
                        [('FGF8', 1.0)],  # fgf8(in)
                        [('SHH', 1.0)],  # shh(in)
                        [('Pax3', 1.0)],  # pax3(in)
                        [('MyoG or Mef2c or Mef2a', 1.0)],  # mrf4
                        [('Mef2c', 1.0)],  # mef2c(in)
                        [('Mef2a', 1.0)],  # mef2a(in)
                        [('ID3', 1.0)],  # id3(in)
                        [('WNT', 1.0)],  # wnt(in)
                        [('WNT3a', 1.0)],  # wnt3a(in)
                        [('WNT3a', 1.0)],  # t
                        [('WNT3a', 1.0)],  # msg1
                   ])

if False:
    env = gym.make(f"gym-PBN/PBNEnv",
                   N=47,
                   genes=['v_CD4_NKG2D', 'v_CD4_NKG2Dupregulation', 'v_CD8_NKG2D', 'v_DC', 'v_DEF', 'v_FIBROBLAST',
                          'v_GRANZB', 'v_IEC_MICA_B', 'v_IEC_MICA_Bupregulation', 'v_IEC_ULPB1_6', 'v_IFNg', 'v_IL10',
                          'v_IL12', 'v_IL13', 'v_IL15', 'v_IL17', 'v_IL18', 'v_IL1b', 'v_IL2', 'v_IL21', 'v_IL22',
                          'v_IL22upregulation', 'v_IL23', 'v_IL4', 'v_IL6', 'v_LPS', 'v_MACR', 'v_MDP', 'v_MMPs', 'v_NFkB',
                          'v_NK', 'v_NK_NKG2D', 'v_NOD2', 'v_PERFOR', 'v_PGN', 'v_TGFb', 'v_TLR2', 'v_TLR4', 'v_TNFa',
                          'v_Th0', 'v_Th0_M', 'v_Th1', 'v_Th17', 'v_Th17_M', 'v_Th2', 'v_Th2upregulation', 'v_Treg'],
                   logic_functions=[[('((v_PGN and not (v_CD4_NKG2D and ((v_IEC_ULPB1_6 or v_IEC_MICA_B) or v_IL10))) or ((v_MDP and not (v_CD4_NKG2D and ((v_IEC_ULPB1_6 or v_IEC_MICA_B) or v_IL10))) or (((v_CD4_NKG2D and ((v_TNFa or v_IL15) and not v_CD4_NKG2Dupregulation)) and not (v_CD4_NKG2D and ((v_IEC_ULPB1_6 or v_IEC_MICA_B) or v_IL10))) or (v_LPS and not (v_CD4_NKG2D and ((v_IEC_ULPB1_6 or v_IEC_MICA_B) or v_IL10))))))', 1.0)],
                                   [('(v_CD4_NKG2D and (v_TNFa or v_IL15))', 1.0)],
                                   [('((v_MDP and not (v_CD8_NKG2D and (v_IEC_MICA_B or (v_IEC_ULPB1_6 or (v_IL21 and v_IL2))))) or ((v_PGN and not (v_CD8_NKG2D and (v_IEC_MICA_B or (v_IEC_ULPB1_6 or (v_IL21 and v_IL2))))) or (v_LPS and not (v_CD8_NKG2D and (v_IEC_MICA_B or (v_IEC_ULPB1_6 or (v_IL21 and v_IL2)))))))', 1.0)],
                                   [('((v_TLR2 and not (v_DC and v_IL10)) or ((v_TLR4 and not (v_DC and v_IL10)) or (v_NOD2 and not (v_DC and v_IL10))))', 1.0)],
                                   [('(v_IL22 or (v_IL17 or v_NOD2))', 1.0)],
                                   [('((v_IL2 and not (v_FIBROBLAST and (v_IL12 or v_IFNg))) or ((v_MACR and (v_TGFb or (v_IL13 or v_IL4))) and not (v_FIBROBLAST and (v_IL12 or v_IFNg))))', 1.0)],
                                   [('(v_NK_NKG2D or (v_CD8_NKG2D or (v_NK or (v_DC and (not v_PGN or not v_LPS)))))', 1.0)],
                                   [('(((v_IEC_MICA_B and (v_TNFa and not v_IEC_MICA_Bupregulation)) and not v_TGFb) or ((v_LPS and not v_TGFb) or ((v_PGN and not v_TGFb) or (v_MDP and not v_TGFb))))', 1.0)],
                                   [('(v_IEC_MICA_B and v_TNFa)', 1.0)],
                                   [('(v_CD8_NKG2D and (v_PGN or (v_LPS or v_MDP)))', 1.0)],
                                   [('((((v_Th17 and (v_PGN or (v_LPS or v_MDP))) and not (v_IFNg and (v_TGFb or v_IL10))) and not v_Th2) or (((v_Th1 and not (v_IFNg and (v_TGFb or v_IL10))) and not v_Th2) or ((((v_IL18 and (v_IL12 and (v_Th0 or v_MACR))) and not (v_IFNg and (v_TGFb or v_IL10))) and not v_Th2) or ((((v_IL23 and ((v_PGN or (v_LPS or v_MDP)) and v_NK)) and not (v_IFNg and (v_TGFb or v_IL10))) and not v_Th2) or ((((v_NK_NKG2D and (v_IEC_ULPB1_6 or v_IEC_MICA_B)) and not (v_IFNg and (v_TGFb or v_IL10))) and not v_Th2) or (((v_CD8_NKG2D and (v_IEC_ULPB1_6 or v_IEC_MICA_B)) and not (v_IFNg and (v_TGFb or v_IL10))) and not v_Th2))))))', 1.0)],
                                   [('(v_Treg or ((v_DC and v_LPS) or ((v_TLR2 and (v_NFkB and (not v_MACR and not v_IFNg))) or ((v_MACR and (v_LPS and not v_IL4)) or (v_Th2 and not v_IL23)))))', 1.0)],
                                   [('((v_LPS and (v_IFNg and ((v_MACR and v_PGN) or v_DC))) or (v_TLR2 and (v_NFkB and (v_DC or v_MACR))))', 1.0)],
                                   [('v_Th2', 1.0)],
                                   [('((v_FIBROBLAST and (v_PGN or (v_LPS or v_MDP))) or (v_MACR and (v_IFNg or v_LPS)))', 1.0)],
                                   [('(((v_Th17_M and (v_PGN or (v_LPS or v_MDP))) and not (v_IL17 and (v_TGFb or v_IL13))) or (((v_CD4_NKG2D and (v_IEC_ULPB1_6 or v_IEC_MICA_B)) and not (v_IL17 and (v_TGFb or v_IL13))) or (v_Th17 and not (v_IL17 and (v_TGFb or v_IL13)))))', 1.0)],
                                   [('(v_LPS and (v_NFkB and (v_DC or v_MACR)))', 1.0)],
                                   [('(((v_MACR and (v_NFkB and v_LPS)) and not (v_IL10 and v_IL1b)) or ((v_DC and (v_NFkB and v_LPS)) and not (v_IL10 and v_IL1b)))', 1.0)],
                                   [('((v_Th0_M and (v_PGN or (v_LPS or v_MDP))) or (v_Th0 or v_DC))', 1.0)],
                                   [('(((((v_Th0 and v_IL6) and not v_IFNg) and not v_IL4) and not v_TGFb) or v_Th17)', 1.0)],
                                   [('(v_Th17 or ((v_NK and v_IL23) or (((v_Th0 and (v_IL22 and (v_IL21 and not v_IL22upregulation))) and not v_TGFb) or (v_CD4_NKG2D or (v_NK and (v_IL18 and v_IL12))))))', 1.0)],
                                   [('(v_Th0 and (v_IL21 and v_IL22))', 1.0)],
                                   [('((v_MACR and v_IL1b) or v_DC)', 1.0)],
                                   [('v_Th2', 1.0)],
                                   [('((v_DC and (v_PGN or v_LPS)) or ((v_MACR and v_PGN) or ((v_Th17 and v_IL23) or (v_NFkB and (not v_IL10 or not v_IL4)))))', 1.0)],
                                   [('not (v_GRANZB or (v_DEF or v_PERFOR))', 1.0)],
                                   [('((v_NOD2 and not (v_MACR and v_IL10)) or ((v_IFNg and not (v_MACR and v_IL10)) or ((v_IL15 and not (v_MACR and v_IL10)) or ((v_TLR4 and not (v_MACR and v_IL10)) or (v_TLR2 and not (v_MACR and v_IL10))))))', 1.0)],
                                   [('not (v_PERFOR or (v_DEF or v_GRANZB))', 1.0)],
                                   [('((v_FIBROBLAST and (v_TNFa or (v_IL21 or (v_IL17 or v_IL1b)))) or (v_MACR and v_TNFa))', 1.0)],
                                   [('(v_NOD2 or (v_TLR4 or v_TLR2))', 1.0)],
                                   [('(((v_DC and v_IL15) and not (v_NK and v_Treg)) or ((v_IL23 and not (v_NK and v_Treg)) or ((v_IL18 and v_IL10) and not (v_NK and v_Treg))))', 1.0)],
                                   [('((v_MDP and not (v_NK_NKG2D and (v_IEC_ULPB1_6 and (v_TGFb and (v_IEC_MICA_B and (v_IL21 and v_IL12)))))) or ((v_PGN and not (v_NK_NKG2D and (v_IEC_ULPB1_6 and (v_TGFb and (v_IEC_MICA_B and (v_IL21 and v_IL12)))))) or (v_LPS and not (v_NK_NKG2D and (v_IEC_ULPB1_6 and (v_TGFb and (v_IEC_MICA_B and (v_IL21 and v_IL12))))))))', 1.0)],
                                   [('v_MDP', 1.0)],
                                   [('(v_NK_NKG2D or v_NK)', 1.0)],
                                   [('not (v_GRANZB or (v_DEF or v_PERFOR))', 1.0)],
                                   [('(v_Treg or v_MACR)', 1.0)],
                                   [('v_PGN', 1.0)],
                                   [('v_LPS', 1.0)],
                                   [('(((v_MACR and (v_IL2 or (v_PGN or (v_IFNg and v_LPS)))) and not (v_IL10 and (v_TNFa and (v_TLR4 or v_TLR2)))) or (((v_CD8_NKG2D and (v_IEC_ULPB1_6 or v_IEC_MICA_B)) and not (v_IL10 and (v_TNFa and (v_TLR4 or v_TLR2)))) or (((v_FIBROBLAST and v_IFNg) and not (v_IL10 and (v_TNFa and (v_TLR4 or v_TLR2)))) or (((v_NFkB and v_LPS) and not (v_IL10 and (v_TNFa and (v_TLR4 or v_TLR2)))) or (((v_CD4_NKG2D and (v_IEC_ULPB1_6 or v_IEC_MICA_B)) and not (v_IL10 and (v_TNFa and (v_TLR4 or v_TLR2)))) or (((v_NK and ((v_PGN or (v_LPS or v_MDP)) and (v_IL23 or (v_IL15 or v_IL2)))) and not (v_IL10 and (v_TNFa and (v_TLR4 or v_TLR2)))) or ((v_NK_NKG2D and (v_IEC_ULPB1_6 or v_IEC_MICA_B)) and not (v_IL10 and (v_TNFa and (v_TLR4 or v_TLR2))))))))))', 1.0)],
                                   [('(v_MDP or (v_PGN or v_LPS))', 1.0)],
                                   [('(v_Th0_M or (v_Th0 and (v_IL23 or v_IL12)))', 1.0)],
                                   [('((v_Th0 and (v_IL18 or (v_IL12 or v_IFNg))) and not (v_Th1 and (v_IL4 or (v_TGFb or (v_IL10 or (v_Treg or (v_Th2 or (v_IL12 and (v_IL23 or v_IL17)))))))))', 1.0)],
                                   [('((v_Th0 and (v_IL23 or (v_IL6 or v_IL1b))) and not (v_Th17 and (v_IL12 or (v_TGFb or (v_Treg or (v_IFNg or v_IL4))))))', 1.0)],
                                   [('((v_Th0_M and ((v_PGN or (v_LPS or v_MDP)) and ((v_IL6 and v_IL1b) or (v_IL23 or v_IL2)))) or v_Th17_M)', 1.0)],
                                   [('((v_Th0 and (((v_Th2 and v_IL4) and not v_Th2upregulation) or (((v_IL18 and v_IL4) and not v_IL12) or v_IL10))) and not (v_Th2 and (v_TGFb or (v_Treg or v_IFNg))))', 1.0)],
                                   [('(v_Th2 and v_IL4)', 1.0)],
                                   [('((v_Th0 and (v_TGFb or v_TLR2)) and not (v_Treg and (v_IL22 or (v_IL23 or (v_TNFa or (v_IL21 or (v_IL6 or v_Th17)))))))', 1.0)]])

if False:
    env = gym.make(f"gym-PBN/PBNEnv",
                   N=144,
                   genes=["v_ACVR1", "v_AKT1S1", "v_AKT_f", "v_AP1_c", "v_ARHGAP24", "v_ATF2", "v_AXIN1",
                          "v_Antisurvival",
                          "v_BAD", "v_BAX", "v_BCL2", "v_BMPR2", "v_BTRC", "v_CASP3", "v_CASP8", "v_CASP9",
                          "v_CBPp300_c",
                          "v_CCND1", "v_CDC42", "v_CFLAR", "v_CFL_f", "v_CHUK", "v_CK1_f", "v_CREBBP", "v_CSK",
                          "v_CTNNB1",
                          "v_CYCS", "v_DAAM1", "v_DKK_f", "v_DKK_g", "v_DUSP1", "v_DUSP1_g", "v_DUSP6", "v_DVL_f",
                          "v_EGR1",
                          "v_EP300", "v_ERK_f", "v_FOS", "v_FOXO_f", "v_FZD_f", "v_GAB_f", "v_GRAP2", "v_GRB2",
                          "v_GSK3_f",
                          "v_IKBKB", "v_ILK", "v_ILR_f", "v_IRAK1", "v_IRS1", "v_ISGF3_c", "v_ITCH", "v_JAK_f",
                          "v_JNK_f",
                          "v_JUN", "v_KRAS", "v_LEF1", "v_LIF", "v_LIMK1", "v_LIMK2", "v_LRP_f", "v_MAP2K3", "v_MAP2K4",
                          "v_MAP2K7", "v_MAP3K11", "v_MAP3K4", "v_MAP3K5", "v_MAP3K7", "v_MAP3K8", "v_MAPK14",
                          "v_MAPK8IP3",
                          "v_MAPKAPK2", "v_MDM2", "v_MDM2_g", "v_MEK_f", "v_MMP_f", "v_MSK_f", "v_MYC", "v_NFKB_f",
                          "v_NLK",
                          "v_PAK1", "v_PARD6A", "v_PDPK1", "v_PIAS1", "v_PIK3CA", "v_PLCG1", "v_PLK1", "v_PPM1A",
                          "v_PPP1CA", "v_PRKACA", "v_PRKCA", "v_PRKCD", "v_PTEN", "v_PTEN_g", "v_PTPN11", "v_PTPN6",
                          "v_Prosurvival", "v_RAC_f", "v_RAF_f", "v_REL_f", "v_RHEB", "v_RHOA", "v_RND3", "v_ROCK1",
                          "v_RSK_f", "v_RTPK_f", "v_RTPK_g", "v_S6K_f", "v_SFRP1", "v_SFRP1_g", "v_SHC1", "v_SKI",
                          "v_SKP2",
                          "v_SMAD1", "v_SMAD2", "v_SMAD3", "v_SMAD4", "v_SMAD5", "v_SMAD6", "v_SMAD6_g", "v_SMAD7",
                          "v_SMAD7_g", "v_SMURF1", "v_SMURF2", "v_SOCS1", "v_SOCS1_g", "v_SOS1", "v_SRC", "v_SRF",
                          "v_STAT1", "v_STAT2", "v_STAT3", "v_SYK", "v_TAB_f", "v_TCF7_f", "v_TGFB1", "v_TGFBR1",
                          "v_TGFBR2", "v_TIAM1", "v_TP53", "v_TRAF6", "v_TSC_f", "v_VAV1", "v_mTORC1_c", "v_mTORC2_c"],
                   logic_functions=[
                       [("v_BMPR2", 1.0)],
                       [(" not v_AKT_f", 1.0)],
                       [("((v_ILK  or  (v_PDPK1  or  v_mTORC2_c))  and   not v_PPP1CA)", 1.0)],
                       [("(v_SMAD3  or  (v_JUN  or  (v_SMAD4  or  (v_ATF2  or  v_FOS))))", 1.0)],
                       [("v_ROCK1", 1.0)],
                       [("(v_ERK_f  or  (v_MAPK14  or  v_JNK_f))", 1.0)],
                       [("(v_GSK3_f  and   not (v_LRP_f  or  (v_PPM1A  or  v_PPP1CA)))", 1.0)],
                       [("(v_ISGF3_c  or  (v_FOXO_f  or  v_CASP3))", 1.0)],
                       [(" not (v_AKT_f  or  v_RSK_f)", 1.0)],
                       [("v_TP53", 1.0)],
                       [(" not v_BAD", 1.0)],
                       [(" not (v_SMURF1  or  v_SMURF2)", 1.0)],
                       [("((v_GSK3_f  or  (v_CK1_f  or  v_AXIN1))  and   not v_LRP_f)", 1.0)],
                       [("(v_CASP8  or  v_CASP9)", 1.0)],
                       [(" not v_CFLAR", 1.0)],
                       [("(v_CYCS  or  v_PPP1CA)", 1.0)],
                       [("((v_CREBBP  or  v_EP300)  and   not v_TP53)", 1.0)],
                       [("(v_STAT3  or  (v_TCF7_f  or  v_RSK_f))", 1.0)],
                       [("(v_SRC  and   not v_ARHGAP24)", 1.0)],
                       [("(v_AKT_f  and   not v_ITCH)", 1.0)],
                       [("(v_LIMK1  or  v_LIMK2)", 1.0)],
                       [("(v_PRKCA  or  (v_AKT_f  or  v_TRAF6))", 1.0)],
                       [(" not v_LRP_f", 1.0)],
                       [("v_CHUK", 1.0)],
                       [("v_PRKACA", 1.0)],
                       [("(v_CHUK  and   not v_BTRC)", 1.0)],
                       [("(v_BAX  and   not v_BCL2)", 1.0)],
                       [("v_DVL_f", 1.0)],
                       [("v_DKK_g", 1.0)],
                       [("(v_TCF7_f  and   not v_MYC)", 1.0)],
                       [("((v_DUSP1_g  or  (v_MAPK14  or  v_MSK_f))  and   not v_SKP2)", 1.0)],
                       [("(v_ERK_f  or  v_MAPK14)", 1.0)],
                       [("(v_ERK_f  or  v_mTORC1_c)", 1.0)],
                       [("((v_FZD_f  or  v_SMAD1)  and   not v_ITCH)", 1.0)],
                       [(" not v_TCF7_f", 1.0)],
                       [("(v_AKT_f  and   not (v_PRKCD  or  v_SKI))", 1.0)],
                       [("(v_MEK_f  and   not (v_DUSP6  or  v_PPP1CA))", 1.0)],
                       [("(v_ERK_f  or  (v_SRF  or  v_RSK_f))", 1.0)],
                       [(" not (v_NLK  or  (v_CK1_f  or  v_AKT_f))", 1.0)],
                       [(" not v_SFRP1", 1.0)],
                       [("(v_GRB2  and   not v_ERK_f)", 1.0)],
                       [(" not v_MAPK14", 1.0)],
                       [("v_SHC1", 1.0)],
                       [(" not (v_ERK_f  or  (v_MAPK14  or  (v_AKT_f  or  (v_RSK_f  or  (v_S6K_f  or  v_DVL_f)))))",
                         1.0)],
                       [("(v_MAP3K7  and   not (v_PLK1  or  (v_PPM1A  or  v_TP53)))", 1.0)],
                       [("v_PAK1", 1.0)],
                       [("(v_LIF  or  v_AP1_c)", 1.0)],
                       [("(v_ILR_f  and   not v_SOCS1)", 1.0)],
                       [(" not (v_ERK_f  or  (v_S6K_f  or  v_IKBKB))", 1.0)],
                       [("(v_STAT2  or  v_STAT1)", 1.0)],
                       [("v_JNK_f", 1.0)],
                       [("(v_ILR_f  and   not (v_SOCS1  or  v_PTPN6))", 1.0)],
                       [("((v_MAP2K7  or  (v_MAP2K4  or  v_PAK1))  and   not v_DUSP1)", 1.0)],
                       [("(v_JNK_f  and   not v_GSK3_f)", 1.0)],
                       [("(v_PTPN11  or  v_SOS1)", 1.0)],
                       [("v_CTNNB1", 1.0)],
                       [("v_RAF_f", 1.0)],
                       [("(v_ROCK1  or  v_RAC_f)", 1.0)],
                       [("(v_ROCK1  and   not v_PRKCD)", 1.0)],
                       [("((v_ERK_f  or  (v_MAPK14  or  (v_JNK_f  or  v_FZD_f)))  and   not v_DKK_f)", 1.0)],
                       [("(v_MAP3K7  or  v_MAP3K5)", 1.0)],
                       [("(v_MAP3K7  or  (v_MAP3K11  or  (v_MAP3K4  or  v_GRAP2)))", 1.0)],
                       [("(v_MAP3K7  or  (v_MAPK8IP3  or  v_GRAP2))", 1.0)],
                       [("v_RAC_f", 1.0)],
                       [("v_RAC_f", 1.0)],
                       [(" not v_AKT_f", 1.0)],
                       [("v_TAB_f", 1.0)],
                       [("v_IKBKB", 1.0)],
                       [("((v_MAP2K3  or  v_MAP2K4)  and   not v_DUSP1)", 1.0)],
                       [("v_ROCK1", 1.0)],
                       [("v_MAPK14", 1.0)],
                       [("((v_MAPKAPK2  or  (v_AKT_f  or  (v_MDM2_g  or  v_PPP1CA)))  and   not v_S6K_f)", 1.0)],
                       [("(v_NFKB_f  or  v_TP53)", 1.0)],
                       [("((v_RAF_f  or  v_MAP3K8)  and   not v_ERK_f)", 1.0)],
                       [("(v_STAT3  or  v_LEF1)", 1.0)],
                       [("(v_ERK_f  or  v_MAPK14)", 1.0)],
                       [("((v_STAT3  or  (v_TCF7_f  or  v_PLK1))  and   not v_GSK3_f)", 1.0)],
                       [("(v_REL_f  or  (v_MSK_f  or  (v_CHUK  or  v_IKBKB)))", 1.0)],
                       [("v_MAP3K7", 1.0)],
                       [("(v_RAC_f  or  v_CDC42)", 1.0)],
                       [("(v_TGFBR2  or  v_TGFBR1)", 1.0)],
                       [("(v_PIK3CA  and   not v_PTEN)", 1.0)],
                       [("v_MAPKAPK2", 1.0)],
                       [("(v_KRAS  or  (v_GAB_f  or  v_IRS1))", 1.0)],
                       [("v_SYK", 1.0)],
                       [("(v_MAPKAPK2  or  v_PDPK1)", 1.0)],
                       [("v_PTEN", 1.0)],
                       [("(v_SMAD7  and   not v_RTPK_f)", 1.0)],
                       [("(v_NFKB_f  or  v_FOS)", 1.0)],
                       [("v_PLCG1", 1.0)],
                       [("(v_PDPK1  or  v_CASP3)", 1.0)],
                       [("((v_PTEN_g  or  v_ROCK1)  and   not (v_SRC  or  (v_GSK3_f  or  v_CBPp300_c)))", 1.0)],
                       [("v_EGR1", 1.0)],
                       [("v_GAB_f", 1.0)],
                       [("v_SRC", 1.0)],
                       [("(v_CCND1  or  (v_MYC  or  v_RSK_f))", 1.0)],
                       [("((v_VAV1  or  (v_TIAM1  or  (v_mTORC2_c  or  v_DVL_f)))  and   not v_ARHGAP24)", 1.0)],
                       [("(v_KRAS  and   not (v_ERK_f  or  (v_RHEB  or  v_AKT_f)))", 1.0)],
                       [("((v_CBPp300_c  or  (v_MSK_f  or  v_IKBKB))  and   not v_STAT1)", 1.0)],
                       [(" not v_TSC_f", 1.0)],
                       [("(v_DAAM1  and   not (v_SMURF1  or  (v_PARD6A  or  (v_RAC_f  or  v_RND3))))", 1.0)],
                       [("v_ROCK1", 1.0)],
                       [("(v_RHOA  or  v_CASP3)", 1.0)],
                       [("(v_ERK_f  or  v_PDPK1)", 1.0)],
                       [("((v_RTPK_g  or  v_MMP_f)  and   not (v_MAPK14  or  v_MEK_f))", 1.0)],
                       [("v_FOXO_f", 1.0)],
                       [("(v_mTORC1_c  or  v_PDPK1)", 1.0)],
                       [("v_SFRP1_g", 1.0)],
                       [(" not v_MYC", 1.0)],
                       [("((v_RTPK_f  or  (v_SRC  or  (v_ILR_f  or  v_TGFBR1)))  and   not v_PTEN)", 1.0)],
                       [(" not v_AKT_f", 1.0)],
                       [("(v_CCND1  or  (v_ERK_f  or  v_EP300))", 1.0)],
                       [(
                        "(v_ACVR1  and   not (v_SMAD6  or  (v_ERK_f  or  (v_SMURF1  or  (v_GSK3_f  or  (v_PPM1A  or  v_SKI))))))",
                        1.0)],
                       [(
                        "((v_ITCH  or  (v_ERK_f  or  (v_CBPp300_c  or  (v_TGFBR1  or  v_ACVR1))))  and   not (v_PPM1A  or  (v_SMURF2  or  v_SKI)))",
                        1.0)],
                       [(
                        "((v_MAPK14  or  (v_JNK_f  or  (v_TGFBR1  or  v_ACVR1)))  and   not (v_SMAD6  or  (v_ERK_f  or  (v_GSK3_f  or  (v_AKT_f  or  (v_SMAD7  or  (v_PPM1A  or  v_SKI)))))))",
                        1.0)],
                       [(
                        "((v_SMAD3  or  (v_ERK_f  or  (v_PIAS1  or  (v_SMAD5  or  (v_SMAD2  or  v_SMAD1)))))  and   not (v_SMAD6  or  (v_SMURF1  or  (v_SMAD7  or  v_SKI))))",
                        1.0)],
                       [("(v_ACVR1  and   not (v_SMURF2  or  v_SKI))", 1.0)],
                       [("v_SMAD6_g", 1.0)],
                       [("(v_SMAD3  or  (v_SMAD4  or  v_SMAD2))", 1.0)],
                       [("((v_SMAD7_g  or  (v_SMURF1  or  v_EP300))  and   not (v_ITCH  or  (v_AXIN1  or  v_SMURF2)))",
                         1.0)],
                       [("(v_SMAD3  or  (v_SMAD4  or  v_SMAD2))", 1.0)],
                       [("v_SMAD7", 1.0)],
                       [("v_SMAD7", 1.0)],
                       [("v_SOCS1_g", 1.0)],
                       [("v_STAT1", 1.0)],
                       [("((v_PLCG1  or  v_GRB2)  and   not v_ERK_f)", 1.0)],
                       [("(v_RTPK_f  and   not v_CSK)", 1.0)],
                       [("(v_MAPKAPK2  or  (v_CFL_f  or  v_RSK_f))", 1.0)],
                       [("((v_PRKCD  or  (v_SRC  or  (v_JAK_f  or  (v_MAPK14  or  v_IKBKB))))  and   not v_PIAS1)",
                         1.0)],
                       [("v_JAK_f", 1.0)],
                       [(
                        "((v_PRKCD  or  (v_SRC  or  (v_JAK_f  or  (v_ERK_f  or  (v_MAPK14  or  (v_JNK_f  or  (v_mTORC1_c  or  v_IRAK1)))))))  and   not v_PPP1CA)",
                        1.0)],
                       [("v_ILR_f", 1.0)],
                       [("(v_TRAF6  and   not v_MAPK14)", 1.0)],
                       [("(v_CTNNB1  and   not v_NLK)", 1.0)],
                       [("(v_JUN  or  (v_NFKB_f  or  v_FOS))", 1.0)],
                       [("(v_TGFBR2  and   not (v_SMAD6  or  (v_SMURF1  or  (v_SMAD7  or  v_SMURF2))))", 1.0)],
                       [("(v_TGFB1  and   not (v_SMURF1  or  v_SMURF2))", 1.0)],
                       [(" not v_ROCK1", 1.0)],
                       [("((v_PRKCD  or  (v_MAPK14  or  (v_PIAS1  or  v_EP300)))  and   not v_MDM2)", 1.0)],
                       [("(v_TGFBR1  or  v_IRAK1)", 1.0)],
                       [("(v_GSK3_f  and   not (v_ERK_f  or  (v_AKT_f  or  (v_RSK_f  or  v_IKBKB))))", 1.0)],
                       [("v_SYK", 1.0)],
                       [("((v_RHEB  or  v_RSK_f)  and   not v_AKT1S1)", 1.0)],
                       [("((v_PIK3CA  or  v_TSC_f)  and   not v_S6K_f)", 1.0)],
                   ])

# load ispl model
if False:
    with open(args.assa_file, "r") as env_file:
        genes = []
        logic_funcs = defaultdict(list)

        for line in env_file:
            line = line.split()

            if len(line) == 0:
                continue

            # get all vars
            if line[0] == "Vars:":
                while True:
                    line = next(env_file)
                    line = line.split()

                    if line[0] == "end":
                        break

                    if line[0][-1] == ":":
                        genes.append(line[0][:-1])
                    else:
                        genes.append(line[0])

            if line[0] == "Evolution:":
                while True:
                    line = next(env_file)
                    line = line.split()

                    if len(line) == 0:
                        continue

                    if line[0] == "end":
                        break

                    target_gene = line[0].split("=")[0]
                    if line[0].split("=")[1] == "false":
                        continue

                    for i in range(len(line)):
                        sline = line[i].split("=")
                        if sline[-1] == "false":
                            line[i] = f"( not {sline[0]} )"
                        else:
                            line[i] = sline[0]

                    target_fun = " ".join(line[2:])
                    target_fun = target_fun.replace("(", " ( ")
                    target_fun = target_fun.replace(")", " ) ")
                    target_fun = target_fun.replace("|", " or ")
                    target_fun = target_fun.replace("&", " and ")
                    target_fun = target_fun.replace("~", " not ")
                    logic_funcs[target_gene].append((target_fun, 1.0))

    print(list(logic_funcs.keys()))
    print(list(logic_funcs.values()))

    for i in range(len(genes)):
        print(list(logic_funcs.keys())[i], list(logic_funcs.values())[i])

    # Load env
    env = gym.make(f"gym-PBN/PBNEnv",
                   N=args.n,
                   genes=list(logic_funcs.keys()),
                   logic_functions=list(logic_funcs.values()),
                   min_attractors=args.attractors)

# load assa-matlab model
if False:
    def translate(sym, logic_function):
        """
        We need variable names to start with letter.

        """
        if logic_function == 'True':
            return f'{sym[0]} or not {sym[0]}'

        if logic_function == 'False':
            return f'{sym[0]} and not {sym[0]}'

        logic_function = logic_function.replace('~', "not ")
        logic_function = logic_function.replace('|', " or ")
        logic_function = logic_function.replace('&', " and ")
        logic_function = logic_function.replace('(', " ( ")
        logic_function = logic_function.replace(')', " ) ")

        return logic_function


    with open(args.assa_file, "r") as env_file:

        genes = []

        line_no = 0
        # skip two headear lines
        _ = next(env_file)
        _ = next(env_file)

        line_no += 2

        line = next(env_file)
        line_no += 1
        n_genes = int(line)

        line = next(env_file)
        line_no += 1
        number_of_functions = line.split()
        number_of_functions = [int(i) for i in number_of_functions]

        line = next(env_file)
        line_no += 1

        n_predictors = ([int(i) for i in (line.split())])

        truth_tables = defaultdict(list)
        fun_id = 0

        # get truth tables of logic functions
        # some genes have more than one function - we deal with that via bunch of nested for loops
        for node in range(n_genes):
            for _ in range(number_of_functions[node]):
                line = next(env_file)
                line_no += 1
                predictors = n_predictors[fun_id]

                truth_table = np.zeros((predictors + 1, 2 ** predictors))
                truth_table[predictors] = [float(i) for i in line.split()]

                # i'm not sure if this order is the same as the one used in matlab
                for j, state in enumerate(itertools.product([0, 1], repeat=predictors)):
                    for i in range(predictors):
                        truth_table[i][j] = state[i]

                truth_tables[node].append(truth_table)
                fun_id += 1

        predictor_sets = defaultdict(list)
        # get predictor sets of genes
        # some genes have more than one function - we deal with that via bunch of nested for loops
        set_id = 0
        for node in range(n_genes):
            for _ in range(number_of_functions[node]):
                line = next(env_file)
                line_no += 1
                predictor_sets[node].append([f"x{i}" for i in line.split()])

        probas = defaultdict(list)
        for node in range(n_genes):
            line = next(env_file)
            line_no += 1
            probas[node] = [float(i) for i in line.split()]

        line = next(env_file)
        line_no += 1
        perturbation_rate = float(line)

        line = next(env_file)
        line_no += 1

    log_funcs = defaultdict(list)

    for gen in truth_tables:
        lf = []
        for i, truth_table in enumerate(truth_tables[gen]):
            IDs = predictor_sets[gen][i]
            minterms = [list(x)[:-1] for x in truth_table.T if list(x)[-1]]

            if len(IDs) == 1:
                sym = (symbols(",".join(IDs)),)
            else:
                sym = symbols(",".join(IDs))

            fun = str(SOPform(sym, minterms, []))
            fun = translate(sym, fun)
            item = (fun, probas[gen][i])
            log_funcs[gen].append(item)

    genes = [f"x{i}" for i in range(n_genes)]

    # Load env
    # DRUG-SYNERGY-PREDICTION
    # from https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2020.00862/full
    env = gym.make(f"gym-PBN/PBNEnv",
                   N=args.n,
                   genes=genes,
                   logic_functions=log_funcs,
                   min_attractors=args.attractors)

print(type(env.env.env))


env.reset()

DEVICE = 'cpu'

config = AgentConfig()
model = model_cls((N, N), N + 1, config, env)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.EPSILON = 0

action = 0
(state, target), _ = env.reset()

# policy, value = model.predict(state, target);
# policy = policy.numpy()
# action = [np.random.choice(range(N+1), p=policy)]

action = model.predict(state, target)
print(action)
state, *_ = env.step(action)
print(state)

all_attractors = env.all_attractors

lens = []
failed = 0
total = 0

failed_pairs = []

all_attractors = env.all_attractors
print("genereted attractors:")
for a in all_attractors:
    print(a)

# list: (state, target) -> list of lens
lens = [[] for _ in range(args.attractors * args.attractors)]
failed = 0
total = 0

failed_pairs = []

for i in range(1):
    print("testing round ", i)
    id = -1
    for attractor_id, target_id in itertools.product(range(args.attractors), repeat=2):
        # print(f"processing initial_state, target_state = {attractor_id}, {target_id}")
        id += 1
        attractor = all_attractors[attractor_id]
        target = all_attractors[target_id]
        target_state = target[0]
        initial_state = attractor[0]
        total += 1
        actions = []
        state = initial_state
        state = [0 if i == '*' else i for i in list(state)]
        _ = env.reset()
        env.graph.setState(state)
        count = 0

        env.setTarget(target)

        while not env.in_target(state):
            count += 1

            # policy, value = model.predict(state, target_state)
            # policy = policy.numpy()
            # action = [np.random.choice(range(N+1), p=policy)]
            action = model.predict(state, target_state)

            _ = env.step(action)
            state = env.render()
            # action_named = [gen_ids[a-1] for a in action]

            if count > 100:
                print(f"failed to converge for {attractor_id}, {target_id}")
                # print(f"final state was 		     {tuple(state)}")
                print(id)
                failed += 1
                failed_pairs.append((initial_state, target))
                lens[id].append(-1)
                # raise ValueError
                break
        else:
            print(f"for initial state {attractor_id} and target {target_id} got (total of {count} steps)")
            # raise ValueError()
            for a in actions:
                # print(a)
                pass
            if count > 0:
                lens[id].append(count)

    print(f"{failed} failed states out of {total}")

print(lens)
# print(f"the avg is {sum(lens) / len(lens)} with len: {len(lens)}")

data = defaultdict(int)
for i in itertools.chain(*lens):
    if i == -1:
        continue
    data[i] += 1

total = sum(data.values())
last = max(data.keys())

x = list(range(1, last + 1))

y = [math.ceil(data[i]) for i in x]

labels = [i if i % 5 == 0 else '' for i in range(last+1)]
print(labels)
labels[0] = 1

for i in range(30, len(x)):
    if y[i] > 0:
        labels[i] = x[i]
        for j in range(1, 5):
            labels[i - j] = ''

d2 = {'x': x, 'y': y}
plt.figure(figsize=(20, 8))
ax = sns.barplot(data=d2, x='x', y='y', color='blue', label='big')
ax.set_xticklabels(labels)
ax.tick_params(labelsize=40)
plt.savefig(f'bn{N}.pdf', bbox_inches='tight', pad_inches=0.)


# for manual fixes
def avg(l: list):
    s = 0
    count = 0
    for x in l:
        for y in x:
            if y > 0:
                s += y
            count += 1

    return s / count


print(avg(lens))

sns.distplot(lens, bins="doane", kde=False, hist_kws={"align": "left"})
plt.show()
