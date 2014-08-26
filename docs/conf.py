import sys, os

sys.path.append(os.path.abspath('..'))

project = 'GLPy'
copyright = '2014, Kai Wohlfahrt'

module_path = os.path.join(os.path.dirname(__file__), '..', 'GLPy')
module_path = os.path.abspath(module_path)

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.doctest', 'sphinx.ext.intersphinx']
source_suffix = '.rst'
master_doc = 'index'
pygments_style = 'sphinx'

#html_theme_options = {'github_fork': 'kwohlfahrt/GLPy'}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
	('index', 'GLPy.tex', 'GLPy Documentation',
	 'Kai Wohlfahrt', 'manual'),
]

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ('index', 'GLPy', 'GLPy Documentation',
     ['Kai Wohlfahrt'], 1)
]

intersphinx_mapping = {'python3': ('http://docs.python.org/3.4/', None),}

rst_epilog = '''
.. |buffer-bind| replace:: :ref:`Binds a buffer <buffer-bind-warning>`
.. |texture-bind| replace:: :ref:`Binds a texture <texture-bind-warning>`
.. _PyOpenGL: http://pyopengl.sourceforge.net/
'''
