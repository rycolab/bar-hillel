from typing import Tuple, Dict
from collections import defaultdict as dd
from itertools import chain, product
from frozendict import frozendict
from rayuela.base.semiring import Semiring

from rayuela.base.symbol import Sym, ε, φ, dummy
from rayuela.base.partitions import PartitionRefinement
from rayuela.fsa.fsa import FSA
from rayuela.fsa.state import State, MinimizeState, PowerState
from rayuela.fsa.pathsum import Pathsum



class Transformer:
    
    @staticmethod
    def _eps_partition(fsa, partition_symbol: Sym = ε) -> Tuple[FSA, FSA]:
        """Partition FSA into two (one with arcs of the partition symbol and one with all others)

        Args:
            fsa (FSA): The input FSA
            partition_symbol (Sym, optional): The symbol based on which to partition the input FSA

        Returns:
            Tuple[FSA, FSA]: The FSA with non-partition symbol arcs
                             and the FSA with only the partition symbol arcs
        """

        E = fsa.spawn()
        N = fsa.spawn(keep_init=True, keep_final=True)

        for q in fsa.Q:
            E.add_state(q)
            N.add_state(q)

        for i in fsa.Q:
            for a, j, w in fsa.arcs(i):
                if a == partition_symbol:
                    E.add_arc(i, a, j, w)
                else:
                    N.add_arc(i, a, j, w)

        return N, E

    @staticmethod
    def epsremoval(fsa):

        # note that N keeps same initial and final weights
        N, E = Transformer._eps_partition(fsa)
        W = Pathsum(E).lehmann(zero=False)

        for i in fsa.Q:
            for a, j, w in fsa.arcs(i, no_eps=True):
                for k in fsa.Q:
                    N.add_arc(i, a, k, w * W[j, k])

        # additional initial states
        for i, j in product(fsa.Q, repeat=2):
            N.add_I(j, fsa.λ[i] * W[i, j])

        return N

    
    
