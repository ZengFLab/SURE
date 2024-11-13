# SURE: SUccinct REpresentation of cells
 SURE introduces a vector quantization-based probabilistic generative model for calling metacells and use them as landmarks that form a coordinate system for cell ID. Analyzing single-cell omics data in a manner analogous to reference genome-based genomic analysis.

## <code style="color : orangered">UPDATE</code>
An update has been distributed. Users can access to the update here, [SUREv2](https://github.com/ZengFLab/SUREv2). It provides Python classes that users can call SURE in scripts. It also provide the command that users can run SURE in the shell. Additionally, SUREv2 supports the calling of metacells for multi-omics datasets.

## Installation
1. Install [pytorch](https://pytorch.org/get-started/locally/) according to your computational platform
2. Install dependencies:
    - `pip3 install numpy scipy pandas scikit-learn pyro-ppl matplotlib`

    - `pip install datatable`

    - To install SEACells please follow the instructions [here](https://github.com/dpeerlab/SEACells)


## Example 1: Calling metacells for a single-cell dataset

Users can refer to [here](https://github.com/ZengFLab/SURE_example_1) for details.

## Example 2: The hierarchical assembly of large-scale dataset(s)

Users can refer to [here](https://github.com/ZengFLab/SURE_example_2) for details.

## Example 3: Human brain cell atlas

Users can refer to [here](https://github.com/ZengFLab/SURE_example_3) for details.

## Example 4: Metacell calling for scATAC-seq data

Users can refer to [here](https://github.com/ZengFLab/SURE_example_4) for details.


