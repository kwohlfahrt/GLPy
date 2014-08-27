Buffer
++++++

General Buffers
===============
.. autoclass:: GLPy.buffers.Empty

.. autoclass:: GLPy.buffers.Buffer
   :members:
   :special-members: __enter__

.. autoclass:: GLPy.buffers.BufferBytes
   :members:
   :special-members: __setitem__, __len__

Vertex Buffers
==============
.. autoclass:: GLPy.vertex.ElementBuffer
   :members:
   :special-members: __setitem__

.. autoclass:: GLPy.vertex.VertexBuffer
   :members:
   :special-members: __setitem__, __len__

.. autoclass:: GLPy.vertex.VertexDataBlock
   :members:
   :special-members: __setitem__

.. autoclass:: GLPy.vertex.VertexDataTrack
   :members:
   :special-members: __setitem__

Uniform Buffers
===============
.. autoclass:: GLPy.uniform.UniformBuffer
   :members: target
   :special-members: __setitem__

.. autoclass:: GLPy.uniform.UniformBlockData
   :members:
   :special-members: __setitem__

.. autoclass:: GLPy.uniform.UniformBlockMemberData
   :members:
   :special-members: __setitem__
