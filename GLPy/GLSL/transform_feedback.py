from .variable import Variable
from .datatypes import Struct, Array

from numpy import dtype

class FeedbackVarying(Variable):
	def __init__(self, name, datatype):
		super().__init__(name, datatype)

	@property
	def alignment(self):
		if isinstance(self.datatype, Struct):
			return max(m.alignment for m in self)
		else:
			base_type = getattr(self.datatype, 'base', self.datatype)
			return base_type.scalar_type.machine_time.itemsize

	@property
	def dtype(self):
		if isinstance(self.datatype, Array):
			return dtype((self.datatype[0].dtype, len(self.datatype)))
		elif isinstance(self.datatype, Struct):
			dtype([(m.name, m.dtype) for m in self])
		else:
			return self.datatype.machine_type

	def __getitem__(self, idx):
		var = super().__getitem__(idx)
		return FeedbackVarying(var.name, var.datatype)

	def __iter__(self, idx):
		yield from (FeedbackVarying(var.name, var.dtatype) for var in super().__iter__())
