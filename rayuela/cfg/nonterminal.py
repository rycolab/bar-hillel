from rayuela.base.symbol import Sym

class NT:

	def __init__(self, X, label=None, n=None):
		self._X = X
		self._label = label
		self.num = n

	@property
	def X(self):
		return self._X

	@property
	def label(self):
		return self._label

	def number(self):
		return

	def set_label(self, label):
		self._label = label

	def copy(self):
		return NT(self.X)

	def __truediv__(self, Y):
		return Slash(self, Y)

	def __invert__(self):
		return Other(self)

	def __mul__(self, B):
		return Product(self, B)

	def __repr__(self):
		if self.label is not None:
			return f'{self.label}'
		return f'{self.X}'

	def __hash__(self):
		return hash(self.X)

	def __eq__(self, other):
		return isinstance(other, NT) and self.X == other.X

S = NT("S")
bottom = NT("⊥")


class Triplet(NT):
	def __init__(self, p, X, q, label=None):
		super().__init__((p, X, q), label=None)
		self._p, self._X, self._q = p, X, q

	@property
	def p(self):
		return self._p

	@property
	def X(self):
		return self._X

	@property
	def q(self):
		return self._q

	def __hash__(self):
		return hash(self.X)

	def __eq__(self, other):
		return isinstance(other, Triplet) \
			and self.p == other.p \
			and self.X == other.X \
			and self.q == other.q

	def __repr__(self):
		if self.label is not None:
			return f'{self.label}'
		return f"[{self.p}, {self.X}, {self.q}]"


class Delta(NT):

	def __init__(self, X, a, idx):
		assert isinstance(X, NT) or isinstance(a, Sym)
		super().__init__((X, a))
		self._X, self._a, self._idx = X, a, idx

	@property
	def X(self):
		return self._X

	@property
	def a(self):
		return self._a

	@property
	def idx(self):
		return self._idx

	def _downstairs(self):
		if isinstance(self.X , Delta):
			return self.X._downstairs() + str(self.a)
		elif isinstance(self.X, NT):
			return str(self.a)
		raise Exception

	def _upstairs(self):
		if isinstance(self.X , Delta):
			return self.X._upstairs()
		elif isinstance(self.X, NT):
			return str(self.X)
		raise Exception

	def __repr__(self):
		return f"{self._upstairs()}"+"/"+f"{self._downstairs()}"

	def __hash__(self):
		return hash((self.X, self.a))

	def __eq__(self, other):
		return isinstance(other, Delta) \
			and self.idx == other.idx \
			and self.X == other.X \
			and self.a == other.a

class Slash(NT):

	def __init__(self, Y, Z, label=None):
		assert isinstance(Z, NT) or isinstance(Z, Sym)
		super().__init__((Y, Z), label=None)
		self._Y, self._Z = Y, Z

	@property
	def Y(self):
		return self._Y

	@property
	def Z(self):
		return self._Z

	def __repr__(self):
		if self.label is not None:
			return f'{self.label}'
		return f"{self.Y}"+"/"+f"{self.Z}"

	def __hash__(self):
		return hash((self._Y, self._Z))

	def __eq__(self, other):
		return isinstance(other, Slash) \
			and self.Y == other.Y \
			and self.Z == other.Z


class Other(NT):

	def __init__(self, X, label=None):
		super().__init__(X, label=None)

	def __repr__(self):
		if self.label is not None:
			return f'{self.label}'
		return f"~{self.X}"

	def __hash__(self):
		return hash(self.X)

	def __eq__(self, other):
		return isinstance(other, Other) and self.X == other.X


class Nullable(NT):
	def __init__(self, X , label=None):
		super().__init__( X , label=None)
		self._X  =  X

	@property
	def X(self):
		return self._X

	def __hash__(self):
		return hash(self.X)

	def __eq__(self, other):
		return isinstance(other, Nullable) \
			and self.X == other.X \
		
	def __repr__(self):
		if self.label is not None:
			return f'{self.label}'
		return f"{self.X}(ε)"


class Product(NT):

	def __init__(self, A, B, label=None):
		assert isinstance(B, NT) or isinstance(B, Sym)
		super().__init__((A, B), label=None)
		self._A, self._B = A, B

	@property
	def A(self):
		return self._A

	@property
	def B(self):
		return self._B

	def __repr__(self):
		if self.label is not None:
			return f'{self.label}'
		return f"{self.A}"+"*"+f"{self.B}"

	def __hash__(self):
		return hash((self._A, self._B))

	def __eq__(self, other):
		return isinstance(other, Product) \
			and self.A == other.A \
			and self.B == other.B

		
