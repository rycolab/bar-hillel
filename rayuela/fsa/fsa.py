from typing import Tuple, Generator
import copy
from frozendict import frozendict
from itertools import product

from collections import Counter, defaultdict as dd

from rayuela.base.semiring import Boolean, Semiring, String, ProductSemiring
from rayuela.base.misc import epsilon_filter
from rayuela.base.symbol import Sym, ε, ε_1, ε_2, φ

from rayuela.fsa.state import State, PairState
from rayuela.fsa.pathsum import Pathsum, Strategy

from rayuela.cfg.nonterminal import S


class FSA:
    def __init__(self, R=Boolean):

        # DEFINITION
        # A weighted finite-state automaton is a 5-tuple <R, Σ, Q, δ, λ, ρ> where
        # • R is a semiring;
        # • Σ is an alphabet of symbols;
        # • Q is a finite set of states;
        # • δ is a finite relation Q × Σ × Q × R;
        # • λ is an initial weight function;
        # • ρ is a final weight function.

        # NOTATION CONVENTIONS
        # • single states (elements of Q) are denoted q
        # • multiple states not in sequence are denoted, p, q, r, ...
        # • multiple states in sequence are denoted i, j, k, ...
        # • symbols (elements of Σ) are denoted lowercase a, b, c, ...
        # • single weights (elements of R) are denoted w
        # • multiple weights (elements of R) are denoted u, v, w, ...

        # semiring
        self.R = R

        # alphabet of symbols
        self.Sigma = set([])

        # a finite set of states
        self.Q = set([])

        # transition function : Q × Σ × Q → R
        self.δ = dd(lambda: dd(lambda: dd(lambda: self.R.zero)))

        # initial weight function
        self.λ = R.chart()

        # final weight function
        self.ρ = R.chart()

    def add_state(self, q):
        self.Q.add(q)

    def add_states(self, Q):
        for q in Q:
            self.add_state(q)

    def add_arc(self, i, a, j, w=None):
        if w is None:
            w = self.R.one

        if not isinstance(i, State):
            i = State(i)
        if not isinstance(j, State):
            j = State(j)
        if not isinstance(a, Sym):
            a = Sym(a)
        if not isinstance(w, self.R):
            w = self.R(w)

        self.add_states([i, j])
        self.Sigma.add(a)
        self.δ[i][a][j] += w

    def set_arc(self, i, a, j, w=None):
        if w is None:
            w = self.R.one

        if not isinstance(i, State):
            i = State(i)
        if not isinstance(j, State):
            j = State(j)
        if not isinstance(a, Sym):
            a = Sym(a)
        if not isinstance(w, self.R):
            w = self.R(w)

        self.add_states([i, j])
        self.Sigma.add(a)
        self.δ[i][a][j] = w

    def set_I(self, q, w=None):

        if not isinstance(q, State):
            q = State(q)

        if w is None:
            w = self.R.one
        self.add_state(q)
        self.λ[q] = w

    def set_F(self, q, w=None):

        if not isinstance(q, State):
            q = State(q)

        if w is None:
            w = self.R.one
        self.add_state(q)
        self.ρ[q] = w

    def add_I(self, q, w):
        self.add_state(q)
        self.λ[q] += w

    def add_F(self, q, w):
        self.add_state(q)
        self.ρ[q] += w

    def freeze(self):
        self.Sigma = frozenset(self.Sigma)
        self.Q = frozenset(self.Q)
        self.δ = frozendict(self.δ)
        self.λ = frozendict(self.λ)
        self.ρ = frozendict(self.ρ)

    @property
    def I(self) -> Generator[Tuple[State, Semiring], None, None]:
        for q, w in self.λ.items():
            if w != self.R.zero:
                yield q, w

    @property
    def F(self):
        for q, w in self.ρ.items():
            if w != self.R.zero:
                yield q, w

    def arcs(self, i, no_eps=False, nozero=True):
        for a, T in self.δ[i].items():
            if no_eps and a == ε:
                continue
            for j, w in T.items():
                if w == self.R.zero and nozero:
                    continue
                yield a, j, w

    def accept(self, string):
        """determines whether a string is in the language"""
        assert isinstance(string, str)

        fsa = FSA(R=self.R)
        for i, x in enumerate(list(string)):
            fsa.add_arc(State(i), Sym(x), State(i + 1), self.R.one)

        fsa.set_I(State(0), self.R.one)
        fsa.add_F(State(len(string)), self.R.one)

        tmp = self.intersect(fsa)

        return tmp, Pathsum(tmp).pathsum(Strategy.VITERBI)

    @property
    def num_states(self):
        return len(self.Q)

    @property
    def acyclic(self):
        cyclic, _ = self.dfs()
        return not cyclic

    @property
    def deterministic(self) -> bool:
        for q in self.Q:
            counter = Counter()
            for a, _, _ in self.arcs(q):
                counter[a] += 1
            most_common = counter.most_common(1)
            if len(most_common) > 0 and most_common[0][1] > 1:
                return False
        return True

    @property
    def pushed(self):
        for i in self.Q:
            total = self.ρ[i]
            for a, j, w in self.arcs(i):
                total += w
            if total != self.R.one:
                return False
        return True

    @property
    def epsilon(self):
        for q in self.Q:
            for a, _, _ in self.arcs(q):
                if a == ε:
                    return True
        return False

    def copy(self):
        """deep copies the machine"""
        return copy.deepcopy(self)

    def spawn(self, keep_init=False, keep_final=False):
        """returns a new FSA in the same semiring"""
        F = FSA(R=self.R)

        if keep_init:
            for q, w in self.I:
                F.set_I(q, w)
        if keep_final:
            for q, w in self.F:
                F.set_F(q, w)

        return F

    def unit(self):
        """returns a copy of the current FSA with all the weights set to unity"""
        nfsa = self.spawn()
        one = self.R.one

        for q, _ in self.I:
            nfsa.set_I(q, one)

        for q, _ in self.F:
            nfsa.set_F(q, one)

        for i in self.Q:
            for a, j, _ in self.arcs(i):
                nfsa.add_arc(i, a, j, one)

        return nfsa


    def epsremove(self):
        from rayuela.fsa.transformer import Transformer

        return Transformer.epsremoval(self)

    def to_cfg(self):
        """converts the WFSA to an equivalent WCFG"""

        from rayuela.cfg.nonterminal import NT
        from rayuela.cfg.production import Production
        from rayuela.cfg.cfg import CFG

        cfg = CFG(R=self.R)
        s = State(0)
        NTs = {s: S}

        for i in self.Q:
            if isinstance(i.idx, int):
                NTs[i] = NT(chr(64 + i.idx))
            elif isinstance(i.idx, tuple):
                NTs[i] = NT("".join([chr(64 + elem) for elem in i.idx]))
            elif isinstance(i.idx, str):
                NTs[i] = NT(i.idx)

        for i in self.Q:
            # add production rule for initial states
            if i in self.λ.keys():
                cfg.add(self.λ[i], NTs[s], NTs[i])

            # add production rule for final states
            if i in self.ρ.keys():
                cfg.add(self.ρ[i], NTs[i], ε)

            # add other production rules
            for a, j, w in self.arcs(i):
                cfg.add(w, NTs[i], a, NTs[j])

        return cfg

    def reverse(self):
        """creates a reversed machine"""

        # create the new machine
        R = self.spawn()

        # add the arcs in the reversed machine
        for i in self.Q:
            for a, j, w in self.arcs(i):
                R.add_arc(j, a, i, w)

        # reverse the initial and final states
        for q, w in self.I:
            R.set_F(q, w)
        for q, w in self.F:
            R.set_I(q, w)

        return R

    def undirected(self) -> "FSA":
        """Produces an undirected version of the FSA (where all the transitions)
           run in both directions)>

        Returns:
            FSA: The undirected FSA.
        """

        undirected_fsa = self.copy()
        for q in self.Q:
            for a, t, w in self.arcs(q):
                undirected_fsa.add_arc(t, a, q, w)

        return undirected_fsa

    def accessible(self):
        """computes the set of accessible states"""
        A = set()
        stack = [q for q, w in self.I if w != self.R.zero]
        while stack:
            i = stack.pop()
            for _, j, w in self.arcs(i):
                if j not in A:
                    stack.append(j)
            A.add(i)

        return A

    def coaccessible(self):
        """computes the set of co-accessible states"""
        return self.reverse().accessible()

    def is_parent(self, p: State, q: State) -> bool:
        """Checks whether `p` is a parent of `q` in the FSA.

        Args:
            p (State): The candidate parent
            q (State): The candidate child

        Returns:
            bool: Whether `p` is a parent of `q`
        """
        return q in [t for _, t, _ in self.arcs(p)]

    def connected_by_symbol(self, p: State, q: State, symbol: Sym) -> bool:
        """Checks whereher there is an edge from `p` to `q` with the label `symbol`.

        Args:
            p (State): The candidate parent
            q (State): The candidate child
            symbom (Sym): The arc label to check

        Returns:
            bool: Whereher there is an edge from `p` to `q` with the label `symbol`
        """
        return symbol in self.δ[p] and q in self.δ[p][symbol]

    def has_incoming_arc(self, q: State, symbol: Sym) -> bool:
        """Checks whereher there is an incoming edge into `q` with the label `symbol`.

        Args:
            q (State): The state
            symbom (Sym): The arc label to check

        Returns:
            bool: Whereher there is an incoming edge into `q` with the label `symbol`.
        """
        for p in self.Q:
            for a, t, _ in self.arcs(p):
                if a == symbol and t == q:
                    return True
        return False

    def has_outgoing_arc(self, q: State, symbol: Sym) -> bool:
        """Checks whereher there is an outgoing edge into `q` with the label `symbol`.

        Args:
            q (State): The state
            symbom (Sym): The arc label to check

        Returns:
            bool: Whereher there is an outgoing edge into `q` with the label `symbol`.
        """
        return symbol in self.δ[q]

    def dfs(self):
        """Depth-first search (Cormen et al. 2019; Section 22.3)"""

        in_progress, finished = set([]), {}
        cyclic, counter = False, 0

        def _dfs(p):
            nonlocal in_progress
            nonlocal finished
            nonlocal cyclic
            nonlocal counter

            in_progress.add(p)

            for _, q, _ in self.arcs(p):
                if q in in_progress:
                    cyclic = True
                elif q not in finished:
                    _dfs(q)

            in_progress.remove(p)
            finished[p] = counter
            counter += 1

        for q, _ in self.I:
            _dfs(q)

        return cyclic, finished

    def finish(self, rev=False, acyclic_check=False):
        """
        Returns the nodes in order of their finishing time.
        """

        cyclic, finished = self.dfs()

        if acyclic_check:
            assert self.acyclic

        sort = {}
        for s, n in finished.items():
            sort[n] = s
        if rev:
            for n in sorted(list(sort.keys())):
                yield sort[n]
        else:
            for n in reversed(sorted(list(sort.keys()))):
                yield sort[n]

    def toposort(self, rev=False):
        return self.finish(rev=rev, acyclic_check=True)

    def trim(self):
        from rayuela.fsa.transformer import Transformer

        return Transformer.trim(self)

    def pathsum(self, strategy=Strategy.LEHMANN):
        pathsum = Pathsum(self)
        return pathsum.pathsum(strategy)

    def forward(self, strategy=Strategy.LEHMANN):
        pathsum = Pathsum(self)
        return pathsum.forward(strategy)

    def backward(self, strategy=Strategy.LEHMANN):
        pathsum = Pathsum(self)
        return pathsum.backward(strategy)

    def allpairs(self, strategy=Strategy.LEHMANN):
        pathsum = Pathsum(self)
        return pathsum.allpairs(strategy)

    def booleanize(self):
        fsa = FSA(Boolean)

        for q, w in self.I:
            fsa.add_I(q, fsa.R(w != self.R.zero))

        for q, w in self.F:
            fsa.add_F(q, fsa.R(w != self.R.zero))

        for q in self.Q:
            for a, j, w in self.arcs(q):
                fsa.add_arc(q, a, j, fsa.R(w != self.R.zero))

        return fsa

    # TODO
    def topologically_equivalent(self, fsa):
        """Tests topological equivalence."""

        # We need to enforce both self and fsa are determinized, pushed and minimized
        assert self.deterministic and fsa.deterministic, "The FSA are not deterministic"
        assert self.pushed and fsa.pushed, "The FSA are not pushed"
        assert self.minimized and fsa.minimized, "The FSA are not minimized"

        # Theorem. If G and H are graphs with out-degree at most 1, then
        # the greedy works to determine whether G and H are isomorphic

        # A deterministic machine has exactly one start state

        # Two minimized DFAs are input
        # If number of states is different, return False

        # Our goal it to trying to find a matching the vertices

        stack = [(q1, q2) for (q1, _), (q2, _) in product(self.I, fsa.I)]
        iso = {stack[0][0]: stack[0][1]}

        while stack:
            p, q = stack.pop()
            for a in self.Sigma:
                r, s = self.δ[p][a], fsa.δ[q][a]
                if not iso[r] == s:
                    return False
                iso[r] = s
        return True

    def equivalent(self, fsa):
        """Tests equivalence."""
        from rayuela.fsa.transformer import Transformer

        if self.R is not fsa.R:
            print("Not equivalent due to different semiring")
            return False

        if self.Sigma != fsa.Sigma:
            print("Not equivalent due to different alphabet")
            return False

        fsa0 = Transformer.determinize(
            Transformer.epsremoval(self.single_I().booleanize())
        ).trim()
        fsa1 = Transformer.determinize(
            Transformer.epsremoval(fsa.single_I().booleanize())
        ).trim()

        fsa2 = fsa0.intersect(fsa1.complement())
        fsa3 = fsa1.intersect(fsa0.complement())

        U = fsa2.union(fsa3)

        return U.trim().num_states == 0

    def edge_marginals(self) -> "dd[dd[dd[Semiring]]]":
        """computes the edge marginals μ(q→q')"""
        marginals = dd(lambda: dd(lambda: dd(lambda: self.R.zero)))

        α = Pathsum(self).forward(strategy=Strategy.VITERBI)
        β = Pathsum(self).backward(strategy=Strategy.VITERBI)

        for q in self.Q:
            for a, q_prime, w in self.arcs(q):
                marginals[q][a][q_prime] += α[q] * w * β[q_prime]

        return marginals

    def difference(self, fsa):
        """coputes the difference with a uniweighted DFA"""

        fsa = fsa.complement()

        # fsa will be a boolean FSA, need to make the semirings compatible
        F = FSA(self.R)
        for q, w in fsa.I:
            F.add_I(q, F.R(w.score))
        for q, w in fsa.F:
            F.add_F(q, F.R(w.score))
        for q in fsa.Q:
            for a, j, w in fsa.arcs(q):
                F.add_arc(q, a, j, F.R(w.score))

        return self.intersect(F)

    def union(self, fsa):
        """construct the union of the two FSAs"""

        assert self.R == fsa.R

        U = self.spawn()

        # add arcs, initial and final states from self
        for i in self.Q:
            for a, j, w in self.arcs(i):
                U.add_arc(PairState(State(1), i), a, PairState(State(1), j), w)

        for q, w in self.I:
            U.set_I(PairState(State(1), q), w)

        for q, w in self.F:
            U.set_F(PairState(State(1), q), w)

        # add arcs, initial and final states from argument
        for i in fsa.Q:
            for a, j, w in fsa.arcs(i):
                U.add_arc(PairState(State(2), i), a, PairState(State(2), j), w)

        for q, w in fsa.I:
            U.set_I(PairState(State(2), q), w)

        for q, w in fsa.F:
            U.set_F(PairState(State(2), q), w)

        return U

    def single_I(self):
        """Returns an equivalent FSA with only 1 initial state"""
        if len([q for q, _ in self.I]) == 1:
            return self

        # Find suitable names for the new state
        ixs = [q.idx for q in self.Q]
        start_id = 0, 0
        while f"single_I_{start_id}" in ixs:
            start_id += 1

        F = self.spawn(keep_final=True)

        for i in self.Q:
            for a, j, w in self.arcs(i):
                F.add_arc(i, a, j, w)

        for i, w in self.I:
            F.add_arc(State(f"single_I_{start_id}"), ε, i, w)

        F.set_I(State(f"single_I_{start_id}"), F.R.one)

        return F

    def concatenate(self, fsa):
        """construct the concatenation of the two FSAs"""

        assert self.R == fsa.R

        C = self.spawn()

        # add arcs, initial and final states from self
        for i in self.Q:
            for a, j, w in self.arcs(i):
                C.add_arc(PairState(State(1), i), a, PairState(State(1), j), w)

        for q, w in self.I:
            C.set_I(PairState(State(1), q), w)

        # add arcs, initial and final states from argument
        for i in fsa.Q:
            for a, j, w in fsa.arcs(i):
                C.add_arc(PairState(State(2), i), a, PairState(State(2), j), w)

        for q, w in fsa.F:
            C.set_F(PairState(State(2), q), w)

        # connect the final states from self to initial states from argument
        for (i1, w1), (i2, w2) in product(self.F, fsa.I):
            C.add_arc(PairState(State(1), i1), ε, PairState(State(2), i2), w1 * w2)

        return C

    def kleene_closure(self):

        # Find suitable names for new states
        ixs = [q.idx for q in self.Q]
        start_id, final_id = 0, 0
        while f"kleene_closure_start_{start_id}" in ixs:
            start_id += 1
        while f"kleene_closure_final_{final_id}" in ixs:
            final_id += 1

        K = self.spawn()

        for q in self.Q:
            for a, j, w in self.arcs(q):
                K.set_arc(q, a, j, w)

        i = State(f"kleene_closure_start_{start_id}")
        f = State(f"kleene_closure_final_{final_id}")

        K.add_I(i, K.R.one)
        K.add_F(f, K.R.one)

        for q, w in self.I:
            K.set_arc(i, ε, q, w)

        for q, w in self.F:
            K.set_arc(q, ε, f, w)

        K.set_arc(i, ε, f, K.R.one)

        for (f, wi), (i, wf) in product(self.F, self.I):
            K.set_arc(f, ε, i, wi * wf)

        return K

    def regular_expression(self):
        """Constructs the regular expression corresponding to the FSA"""

        F = self.copy()
        Fs = []

        def concat(a, b):
            if str(a) == str(ε) and str(b) == str(ε):
                return str(a)
            elif str(a) == str(ε):
                return str(b)
            elif str(b) == str(ε):
                return str(a)
            else:
                return str(a) + str(b)

        # TODO: which state idxs to add
        for q, w in self.I:
            F.λ.pop(q, None)
            F.add_arc(State(-1), ε, q, F.R.one)
        for q, w in self.F:
            F.ρ.pop(q, None)
            F.add_arc(q, ε, State(-2), F.R.one)

        F.add_I(State(-1), F.R.one)
        F.add_F(State(-2), F.R.one)

        δinv = {}
        for i in F.Q:
            for a, j, w in F.arcs(i):
                if j not in δinv:
                    δinv[j] = dict()
                if a not in δinv[j]:
                    δinv[j][a] = dict()
                δinv[j][a][i] = w

        new_arcs = set()

        for i in self.Q:

            fwd = dd(list)
            for a, j, w in F.arcs(i):

                # do not include the irrelevant weight of the temporary edges
                if (i, a, j) not in new_arcs and w != F.R.one:
                    fwd[j].append(f"{str(w)}{str(a)}")
                else:
                    fwd[j].append(f"{str(a)}")

            for k, v in fwd.items():
                if len(fwd[k]) > 1:
                    if k == i:
                        fwd[k] = f'({"|".join(v)})*'
                    else:
                        fwd[k] = f'({"|".join(v)})'
                elif len(fwd[k]) > 0:
                    if k == i:
                        fwd[k] = f'({"|".join(v)})*'
                    else:
                        fwd[k] = "|".join(v)

            fwd[i] = fwd[i] if i in fwd else ε

            bwd = dd(list)
            for a in δinv[i]:
                for j, w in δinv[i][a].items():

                    if i == j:
                        continue  # handled above

                    if j not in F.Q:
                        continue  # if the target has been removed

                    # do not include the irrelevant weight of the temporary edges
                    if (j, a, i) not in new_arcs and w != F.R.one:
                        bwd[j].append(f"{str(w)}{str(a)}")
                    else:
                        bwd[j].append(f"{str(a)}")

                    F.δ[j][a].pop(i, None)

            for k, v in bwd.items():
                if len(bwd[k]) > 1:
                    bwd[k] = f'({"|".join(v)})' + ("*" if k == i else "")
                elif len(bwd[k]) > 0:
                    bwd[k] = "|".join(v) + ("*" if k == i else "")

            for b, f in product(bwd, fwd):

                # ignore self-loops
                if f == i:
                    continue

                label = concat(concat(bwd[b], fwd[i]), fwd[f])

                F.add_arc(b, label, f, F.R.one)

                # add the processed arc to the temporary FSA (inverse δ)
                if f not in δinv:
                    δinv[f] = dict()
                if label not in δinv[f]:
                    δinv[f][label] = dict()
                δinv[f][label][b] = F.R.one

                new_arcs.add((b, label, f))

            # remove the processed arcs from the temporary FSA (forward δ)
            F.δ.pop(i, None)
            # remove the processed state from the temporary FSA
            F.Q.remove(i)

            Fs.append(F.copy())

        regex = []
        for i, w in F.I:
            for a, j, w in F.arcs(i):
                if j not in F.Q:
                    continue  # if the target has been removed
                regex.append(str(a))

        # return F, Fs, '|'.join(regex)
        return "|".join(regex)


    def invert(self):
        """computes inverse"""

        zero, one = self.R.zero, self.R.one
        inv = self.spawn(keep_init=True, keep_final=True)

        for i in self.Q:
            for a, j, w in self.arcs(i):
                inv.add_arc(i, a, j, ~w)

        return inv

    def complete(self):
        """constructs a complete FSA"""

        nfsa = self.copy()

        sink = State(self.num_states)
        for a in self.Sigma:
            nfsa.add_arc(sink, a, sink, self.R.one)

        for q in self.Q:
            if q == sink:
                continue
            for a in self.Sigma:
                if a == ε:  # ignore epsilon
                    continue
                if q not in nfsa.δ or not any(sym == a for sym, _, _ in nfsa.arcs(q)):
                    nfsa.add_arc(q, a, sink, self.R.one)

        return nfsa


    def __add__(self, other):
        return self.concatenate(other)

    def __sub__(self, other):
        return self.difference(other)

    def __and__(self, other):
        return self.intersect(other)

    def __or__(self, other):
        return self.union(other)

    def __repr__(self):
        return f"WFSA({self.num_states} states, {self.R})"

    def __str__(self):
        output = []
        for q, w in self.I:
            output.append(f"initial state:\t{q.idx}\t{w}")
        for q, w in self.F:
            output.append(f"final state:\t{q.idx}\t{w}")
        for p in self.Q:
            for a, q, w in self.arcs(p):
                output.append(f"{p}\t----{a}/{w}---->\t{q}")
        return "\n".join(output)

    def __getitem__(self, n):
        return list(self.Q)[n]

    def _repr_html_(self):
        """
        When returned from a Jupyter cell, this will generate the FST visualization
        Based on: https://github.com/matthewfl/openfst-wrapper
        """
        from uuid import uuid4
        import json
        from collections import defaultdict

        ret = []
        if self.num_states == 0:
            return "<code>Empty FST</code>"

        if self.num_states > 64:
            return f"FST too large to draw graphic, use fst.ascii_visualize()<br /><code>FST(num_states={self.num_states})</code>"

        finals = {q for q, _ in self.F}
        initials = {q for q, _ in self.I}

        # print initial
        for q, w in self.I:
            if q in finals:
                label = f"{str(q)} / [{str(w)} / {str(self.ρ[q])}]"
                color = "af8dc3"
            else:
                label = f"{str(q)} / {str(w)}"
                color = "66c2a5"

            ret.append(
                f'g.setNode("{repr(q)}", {{ label: {json.dumps(label)} , shape: "circle" }});\n'
            )
            # f'g.setNode("{repr(q)}", {{ label: {json.dumps(hash(label) // 1e8)} , shape: "circle" }});\n')

            ret.append(f'g.node("{repr(q)}").style = "fill: #{color}"; \n')

        # print normal
        for q in (self.Q - finals) - initials:

            label = str(q)

            ret.append(
                f'g.setNode("{repr(q)}", {{ label: {json.dumps(label)} , shape: "circle" }});\n'
            )
            # f'g.setNode("{repr(q)}", {{ label: {json.dumps(hash(label) // 1e8)} , shape: "circle" }});\n')
            ret.append(f'g.node("{repr(q)}").style = "fill: #8da0cb"; \n')

        # print final
        for q, w in self.F:
            # already added
            if q in initials:
                continue

            if w == self.R.zero:
                continue
            label = f"{str(q)} / {str(w)}"

            ret.append(
                f'g.setNode("{repr(q)}", {{ label: {json.dumps(label)} , shape: "circle" }});\n'
            )
            # f'g.setNode("{repr(q)}", {{ label: {json.dumps(hash(label) // 1e8)} , shape: "circle" }});\n')
            ret.append(f'g.node("{repr(q)}").style = "fill: #fc8d62"; \n')

        for q in self.Q:
            to = defaultdict(list)
            for a, j, w in self.arcs(q):
                if self.R is ProductSemiring and isinstance(w.score[0], String):
                    # the imporant special case of encoding transducers
                    label = f"{str(a)}:{str(w)}"
                else:
                    label = f"{str(a)} / {str(w)}"
                to[j].append(label)

            for dest, values in to.items():
                if len(values) > 6:
                    values = values[0:3] + [". . ."]
                label = "\n".join(values)
                ret.append(
                    f'g.setEdge("{repr(q)}", "{repr(dest)}", {{ arrowhead: "vee", label: {json.dumps(label)} }});\n'
                )

        # if the machine is too big, do not attempt to make the web browser display it
        # otherwise it ends up crashing and stuff...
        if len(ret) > 256:
            return f"FST too large to draw graphic, use fst.ascii_visualize()<br /><code>FST(num_states={self.num_states})</code>"

        ret2 = [
            """
		<script>
		try {
		require.config({
		paths: {
		"d3": "https://cdnjs.cloudflare.com/ajax/libs/d3/4.13.0/d3",
		"dagreD3": "https://cdnjs.cloudflare.com/ajax/libs/dagre-d3/0.6.1/dagre-d3.min"
		}
		});
		} catch {
		  ["https://cdnjs.cloudflare.com/ajax/libs/d3/4.13.0/d3.js",
		   "https://cdnjs.cloudflare.com/ajax/libs/dagre-d3/0.6.1/dagre-d3.min.js"].forEach(function (src) {
			var tag = document.createElement('script');
			tag.src = src;
			document.body.appendChild(tag);
		  })
		}
		try {
		requirejs(['d3', 'dagreD3'], function() {});
		} catch (e) {}
		try {
		require(['d3', 'dagreD3'], function() {});
		} catch (e) {}
		</script>
		<style>
		.node rect,
		.node circle,
		.node ellipse {
		stroke: #333;
		fill: #fff;
		stroke-width: 1px;
		}

		.edgePath path {
		stroke: #333;
		fill: #333;
		stroke-width: 1.5px;
		}
		</style>
		"""
        ]

        obj = "fst_" + uuid4().hex
        ret2.append(
            f'<center><svg width="850" height="600" id="{obj}"><g/></svg></center>'
        )
        ret2.append(
            """
		<script>
		(function render_d3() {
		var d3, dagreD3;
		try { // requirejs is broken on external domains
		  d3 = require('d3');
		  dagreD3 = require('dagreD3');
		} catch (e) {
		  // for google colab
		  if(typeof window.d3 !== "undefined" && typeof window.dagreD3 !== "undefined") {
			d3 = window.d3;
			dagreD3 = window.dagreD3;
		  } else { // not loaded yet, so wait and try again
			setTimeout(render_d3, 50);
			return;
		  }
		}
		//alert("loaded");
		var g = new dagreD3.graphlib.Graph().setGraph({ 'rankdir': 'LR' });
		"""
        )
        ret2.append("".join(ret))

        ret2.append(f'var svg = d3.select("#{obj}"); \n')
        ret2.append(
            f"""
		var inner = svg.select("g");

		// Set up zoom support
		var zoom = d3.zoom().scaleExtent([0.3, 5]).on("zoom", function() {{
		inner.attr("transform", d3.event.transform);
		}});
		svg.call(zoom);

		// Create the renderer
		var render = new dagreD3.render();

		// Run the renderer. This is what draws the final graph.
		render(inner, g);

		// Center the graph
		var initialScale = 0.75;
		svg.call(zoom.transform, d3.zoomIdentity.translate(
		    (svg.attr("width") - g.graph().width * initialScale) / 2, 20).scale(initialScale));

		svg.attr('height', g.graph().height * initialScale + 50);
		}})();

		</script>
		"""
        )

        return "".join(ret2)
