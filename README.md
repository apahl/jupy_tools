# jupy_tools

(Again, I am showing everyone how untalented I am in naming things)

A set of convenience tools for my work with JupyterLab.  
They mostly deal with Pandas, RDKit and Cell Painting.  
There are also utilities for displaying HTML tables and grids of molecules.

## Selected Modules
### `utils`
Main list of tools. Please have a look at the included documentation.

### `mol_view`
Utilities for displaying HTML tables and grids of molecules, e.g.:

```Python
from jupy_tools import mol_view as mv
df = pd.DataFrame(
    {
        "Compound_Id": [1, 2, 3],
        "Smiles": ["c1ccccc1CO", "c1ccccc1CN", "c1ccccc1C(=O)N"],
        "Activity": [1.3, 1.5, 2.0]
    }
)
mv.mol_grid(df)
# mv.write_mol_grid(df)  # Saves the grid as one HTML file to disk, 
                         # without external dependencies
                         # (images are embedded and no dependency on external Javascript)
```

![molview](res/mv.png)


## Installation

Recommended way of installation: see [mol_frame](https://github.com/apahl/mol_frame#installation)