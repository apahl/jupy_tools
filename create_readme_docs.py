#!/usr/bin/env python3
"""
Create a simple documentation by extracting the doc strings from the utils module,
then pasting it into the README.
"""

import inspect
import os
from jupy_tools import utils, mol_view

def extract_docstrings(module, exclude=None):
    if exclude is None:
        exclude = set()
    elif isinstance(exclude, str):
        exclude = {exclude}
    elif not isinstance(exclude, set):
        exclude = set(exclude)
    docs = []
    for name, obj in inspect.getmembers(module):
        if name in exclude:
            continue
        if inspect.isfunction(obj) or inspect.isclass(obj):
            docs.append(f"<details>\n<summary>{name}</summary>\n{inspect.getdoc(obj)}\n</details>\n")
    return "\n".join(docs)


if __name__ == "__main__":
    
    # Module `utils`:
    exclude = {
        "Any", "DataFrame", "Mol", "contextmanager", "get_atom_set",
        "glob", "lp", "tqdm"
    }
    doc_content = extract_docstrings(utils, exclude=exclude)
    with open("_doc.md", "w") as readme_file:
        readme_file.write("# `utils` Documentation\n\n")
        readme_file.write(doc_content)
    
    # Module `mol_view`:
    exclude = {
        "Any", "DataFrame", "Mol", "IO", "_apply_link", "chain", 
        "is_interactive_python"
    }
    doc_content = extract_docstrings(mol_view, exclude=exclude)
    # Append to the file created above:
    with open("_doc.md", "a") as readme_file:
        readme_file.write("\n\n# `mol_view` Documentation\n\n")
        readme_file.write(doc_content)
    