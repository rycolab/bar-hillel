from rayuela.base.semiring import Boolean, Real, Derivation , MaxPlus , Tropical
from rayuela.base.symbol import Sym, ε

from rayuela.fsa.fsa import FSA
from rayuela.fsa.state import State
from rayuela.fsa.transformer import Transformer as FSATransformer

from rayuela.cfg.nonterminal import NT, S
from rayuela.cfg.cfg import CFG
from rayuela.cfg.misc import * 
from rayuela.base.misc import _random_weight as rw


R=Real

a=Sym("a")
S=NT("S")

cfg=CFG(R)
cfg.add(rw(R), S, S , S )
cfg.add(rw(R), S , a)
cfg.add(rw(R), S , ε)

fsa=FSA(R)
fsa.add_arc(State(0), ε ,State(1), w=rw(R))
fsa.add_arc(State(1), a ,State(2), w=rw(R))
fsa.add_arc(State(2), ε ,State(3), w=rw(R))
fsa.add_arc(State(2), ε ,State(2), w=R(0.3))
fsa.add_arc(State(4), a ,State(1), w=rw(R))
fsa.add_arc(State(1), a ,State(1), w=R(0.4))
fsa.add_arc(State(1), ε ,State(1), w=R(0.4))

fsa.set_I(State(0), w=R(10))
fsa.add_F(State(3), w=R(10))
fsa.set_F(State(1), w=R(10))
fsa.set_I(State(4), w=R(10))

fsa
ftrans=FSATransformer()
fsa_e=ftrans.epsremoval(fsa)
fsa_e

#INTERSECTION WITH E-REMOVED AUTOMATON
ncfg=cfg.intersect_fsa(fsa_e)
print(ncfg.treesum())

ncfg=cfg.intersect_fsa_ε(fsa)
print(ncfg.treesum())