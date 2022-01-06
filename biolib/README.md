# QM9 SchNet Uncertainty

The app predicts atomization energies of QM9-like molecules with calibrated uncertainty using an ensemble of graph neural network models.
Please [see our paper](https://doi.org/10.1088/2632-2153/ac3eb3) for details.


## Assumptions

The model is trained on the [QM9 dataset](https://doi.org/10.1038/sdata.2014.22) and thus is only expected to produce accurate predictions on molecues with the same constraints as QM9 molecules: small organic molecules made up of CHONF.

This is an early version of the app with very little input validation, so errors might occur.
Please report any issues and provide feedback to jbusk@dtu.dk.


## Input

The input file should be an [ASE database](https://wiki.fysik.dtu.dk/ase/ase/db/db.html) of mulecules defined as `Atoms` objects with atom positions given in Angstrom (Å).


## Output

The ouput is a printed table with the following columns:

* mean: the mean energy prediction.
* var: the predicted total variance (uncertainty) of the energy.
* var_epistemic: the predicted epistemic variance of the energy.
* var_aleatoric: the predicted aleatoric variance of the energy.
* scaling_factor: the scaling factor computed to calibrate the variances.

All energies are given in electronvolt (eV).


## Acknowledgments

The authors acknowledge support from the Novo Nordisk Foundation (SURE, NNF19OC0057822) and the European Union's Horizon 2020 research and innovation program under Grant Agreement No. 957189 (BIG-MAP) and No. 957213 (BATTERY2030PLUS).


## Citation

Please cite the following publication:

    @article{Busk_2021,
    	doi = {10.1088/2632-2153/ac3eb3},
    	url = {https://doi.org/10.1088/2632-2153/ac3eb3},
    	year = 2021,
    	month = {dec},
    	publisher = {{IOP} Publishing},
    	volume = {3},
    	number = {1},
    	pages = {015012},
    	author = {Jonas Busk and Peter Bjørn Jørgensen and Arghya Bhowmik and Mikkel N Schmidt and Ole Winther and Tejs Vegge},
    	title = {Calibrated uncertainty for molecular property prediction using ensembles of message passing neural networks},
    	journal = {Machine Learning: Science and Technology},
    }
