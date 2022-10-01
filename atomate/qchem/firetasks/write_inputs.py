# This module defines firetasks for writing QChem input files

import os

from fireworks import FiretaskBase, explicit_serialize
from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.io.qchem.inputs import QCInput

from atomate.utils.utils import load_class

__author__ = "Brandon Wood"
__copyright__ = "Copyright 2018, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Brandon Wood"
__email__ = "b.wood@berkeley.edu"
__status__ = "Alpha"
__date__ = "5/20/18"
__credits__ = "Sam Blau, Shyam Dwaraknath"


@explicit_serialize
class WriteInputFromIOSet(FiretaskBase):
    """
    Writes QChem Input files from input sets. A dictionary is passed to WriteInputFromIOSet where
    parameters are given as keys in the dictionary.

    required_params:
        qc_input_set (QChemDictSet or str): Either a QChemDictSet object or a string
        name for the QChem input set (e.g., "OptSet"). *** Note that if the molecule is to be inherited through
        fw_spec qc_input_set must be a string name for the QChem input set. ***

    optional_params:
        molecule (Molecule, or list of Molecules): Molecule(s) that will be subjected to an electronic
            structure calculation
        qchem_input_params (dict): When using a string name for QChem input set, use this as a dict
            to specify kwargs for instantiating the input set parameters. This setting is
            ignored if you provide the full object representation of a QChemDictSet. Basic uses
            would be to modify the default inputs of the set, such as dft_rung, basis_set,
            pcm_dielectric, scf_algorithm, or max_scf_cycles. See pymatgen/io/qchem/sets.py for
            default values of all input parameters. For instance, if a user wanted to use a
            more advanced DFT functional, include a pcm with a dielectric of 30, and use a
            larger basis, the user would set qchem_input_params = {"dft_rung": 5,
            "pcm_dielectric": 30, "basis_set": "6-311++g**"}. However, more advanced
            customization of the input is also possible through the overwrite_inputs key which
            allows the user to directly modify the rem, pcm, smd, and solvent dictionaries that
            QChemDictSet passes to inputs.py to print an actual input file. For instance, if a
            user wanted to set the sym_ignore flag in the rem section of the input file to
            true, then they would set qchem_input_params = {"overwrite_inputs": "rem":
            {"sym_ignore": "true"}}. Of course, overwrite_inputs could be used in conjunction
            with more typical modifications, as seen in the test_double_FF_opt workflow test.
        input_file (str): Name of the QChem input file. Defaults to mol.qin
        write_to_dir (str): Path of the directory where the QChem input file will be written,
        the default is to write to the current working directory
    """

    required_params = ["qchem_input_set"]
    optional_params = ["molecule", "qchem_input_params", "input_file", "write_to_dir"]

    def run_task(self, fw_spec):
        input_file = os.path.join(
            self.get("write_to_dir", ""), self.get("input_file", "mol.qin")
        )

        # if a full QChemDictSet object was provided
        if hasattr(self["qchem_input_set"], "write_file"):
            qcin = self["qchem_input_set"]
        # if a molecule is being passed through fw_spec
        elif fw_spec.get("prev_calc_molecule"):
            prev_calc_mol = fw_spec.get("prev_calc_molecule")

            # if a molecule is also passed as an optional parameter
            if self.get("molecule"):
                mol = self.get("molecule")

                # check if mol and prev_calc_mol are isomorphic
                if isinstance(mol, Molecule) and isinstance(prev_calc_mol, Molecule):
                    mol_graph = MoleculeGraph.with_local_env_strategy(mol, OpenBabelNN())
                    prev_mol_graph = MoleculeGraph.with_local_env_strategy(
                        prev_calc_mol, OpenBabelNN()
                    )
                    # If they are isomorphic, aka a previous FW has not changed bonding,
                    # then we will use prev_calc_mol. If bonding has changed, we will use mol.
                    if mol_graph.isomorphic_to(prev_mol_graph):
                        mol = prev_calc_mol
                    elif self["qchem_input_set"] != "OptSet":
                        print(
                            "WARNING: Molecule from spec is not isomorphic to passed molecule!"
                        )
                        mol = prev_calc_mol
                    else:
                        print(
                            "Not using prev_calc_mol as it is not isomorphic to passed molecule!"
                        )
            else:
                mol = prev_calc_mol

            qcin_cls = load_class("pymatgen.io.qchem.sets", self["qchem_input_set"])
            qcin = qcin_cls(mol, **self.get("qchem_input_params", {}))
        # if a molecule is only included as an optional parameter
        elif self.get("molecule"):
            qcin_cls = load_class("pymatgen.io.qchem.sets", self["qchem_input_set"])
            qcin = qcin_cls(self.get("molecule"), **self.get("qchem_input_params", {}))
        # if no molecule is present raise an error
        else:
            raise KeyError(
                "No molecule present, add as an optional param or check fw_spec"
            )
        qcin.write(input_file)


@explicit_serialize
class WriteCustomInput(FiretaskBase):
    """
    Writes QChem Input files from custom input sets. This firetask gives the maximum flexibility when trying
    to define custom input parameters.

    required_params:
        rem (dict):
            A dictionary of all the input parameters for the rem section of QChem input file.
            Ex. rem = {'method': 'rimp2', 'basis': '6-31*G++' ... }

    optional_params:
        opt (dict of lists):
            A dictionary of opt sections, where each opt section is a key and the corresponding
            values are a list of strings. Strings must be formatted as instructed by the QChem manual.
            The different opt sections are: CONSTRAINT, FIXED, DUMMY, and CONNECT
            Ex. opt = {"CONSTRAINT": ["tors 2 3 4 5 25.0", "tors 2 5 7 9 80.0"], "FIXED": ["2 XY"]}
        pcm (dict):
            A dictionary of the PCM section, defining behavior for use of the polarizable continuum model.
            Ex: pcm = {"theory": "cpcm", "hpoints": 194}
        solvent (dict):
            A dictionary defining the solvent parameters used with PCM.
            Ex: solvent = {"dielectric": 78.39, "temperature": 298.15}
        smx (dict):
            A dictionary defining solvent parameters used with the SMD method, a solvent method that adds
            short-range terms to PCM.
            Ex: smx = {"solvent": "water"}
        scan (dict of lists):
            A dictionary of scan variables. Because two constraints of the same type are allowed (for instance, two
            torsions or two bond stretches), each TYPE of variable (stre, bend, tors) should be its own key in the
            dict, rather than each variable. Note that the total number of variable (sum of lengths of all lists)
            CANNOT be
            more than two.
            Ex. scan = {"stre": ["3 6 1.5 1.9 0.1"], "tors": ["1 2 3 4 -180 180 15"]}
        van_der_waals (dict):
            A dictionary of custom van der Waals radii to be used when constructing cavities for the PCM
            model or when computing, e.g. Mulliken charges. They keys are strs whose meaning depends on
            the value of vdw_mode, and the values are the custom radii in angstroms.
        vdw_mode (str): Method of specifying custom van der Waals radii - 'atomic' or 'sequential'.
            In 'atomic' mode (default), dict keys represent the atomic number associated with each
            radius (e.g., 12 = carbon). In 'sequential' mode, dict keys represent the sequential
            position of a single specific atom in the input structure.
        plots (dict):
                A dictionary of all the input parameters for the plots section of the QChem input file.
        nbo (dict):
                A dictionary of all the input parameters for the nbo section of the QChem input file.
        geom_opt (dict):
                A dictionary of input parameters for the geom_opt section of the QChem input file.
                This section is required when using the new libopt3 geometry optimizer.
        cdft (list of lists of dicts):
                A list of lists of dictionaries, where each dictionary represents a charge constraint in the
                cdft section of the QChem input file.

                Each entry in the main list represents one state (allowing for multiconfiguration calculations
                using constrainted density functional theory - configuration interaction (CDFT-CI).
                Each state is relresented by a list, which itself contains some number of constraints (dictionaries).

                Ex:

                1. For a single-state calculation with two constraints:
                 cdft=[[
                    {"value": 1.0, "coefficients": [1.0], "first_atoms": [1], "last_atoms": [2], "types": [None]},
                    {"value": 2.0, "coefficients": [1.0, -1.0], "first_atoms": [1, 17], "last_atoms": [3, 19], "types": ["s"]}
                ]]

                Note that a type of None will default to a charge constraint (which can also be accessed by
                requesting a type of "c" or "charge".

                2. For a multireference calculation:
                cdft=[
                    [
                        {"value": 1.0, "coefficients": [1.0], "first_atoms": [1], "last_atoms": [27], "types": ["c"]},
                        {"value": 0.0, "coefficients": [1.0], "first_atoms": [1], "last_atoms": [27], "types": ["s"]},
                    ],
                    [
                        {"value": 0.0, "coefficients": [1.0], "first_atoms": [1], "last_atoms": [27], "types": ["c"]},
                        {"value": -1.0, "coefficients": [1.0], "first_atoms": [1], "last_atoms": [27], "types": ["s"]},
                    ]
                ]
        almo (list of lists of int 2-tuples):
            A list of lists of int 2-tuples used for calculations of diabatization and state coupling calculations
                relying on the absolutely localized molecular orbitals (ALMO) methodology. Each entry in the main
                list represents a single state (two states are included in an ALMO calculation). Within a single state,
                each 2-tuple represents the charge and spin multiplicity of a single fragment.
            ex: almo=[
                        [
                            (1, 2),
                            (0, 1)
                        ],
                        [
                            (0, 1),
                            (1, 2)
                        ]
                    ]
        input_file (str): Name of the QChem input file. Defaults to mol.qin
        write_to_dir (str): Path of the directory where the QChem input file will be written,
        the default is to write to the current working directory
    """

    required_params = ["rem"]
    # optional_params will need to be modified if more QChem sections are added QCInput
    optional_params = [
        "molecule",
        "opt",
        "pcm",
        "solvent",
        "smx",
        "scan",
        "van_der_waals",
        "vdw_mode",
        "plots",
        "nbo",
        "geom_opt",
        "cdft",
        "almo",
        "input_file",
        "write_to_dir",
    ]

    def run_task(self, fw_spec):
        input_file = os.path.join(
            self.get("write_to_dir", ""), self.get("input_file", "mol.qin")
        )
        # if a molecule is being passed through fw_spec
        if fw_spec.get("prev_calc_molecule"):
            prev_calc_mol = fw_spec.get("prev_calc_molecule")
            # if a molecule is also passed as an optional parameter
            if self.get("molecule"):
                mol = self.get("molecule")

                # check if mol and prev_calc_mol are isomorphic
                if isinstance(mol, Molecule) and isinstance(prev_calc_mol, Molecule):
                    mol_graph = MoleculeGraph.with_local_env_strategy(mol, OpenBabelNN())
                    prev_mol_graph = MoleculeGraph.with_local_env_strategy(
                        prev_calc_mol, OpenBabelNN()
                    )
                    if mol_graph.isomorphic_to(prev_mol_graph):
                        mol = prev_calc_mol
                    else:
                        print(
                            "WARNING: Molecule from spec is not isomorphic to passed molecule!"
                        )
            else:
                mol = prev_calc_mol
        elif self.get("molecule"):
            mol = self.get("molecule")
        else:
            raise KeyError(
                "No molecule present, add as an optional param or check fw_spec"
            )
        # in the current structure there needs to be a statement for every optional QChem section
        # the code below defaults the section to None if the variable is not passed
        opt = self.get("opt", None)
        pcm = self.get("pcm", None)
        solvent = self.get("solvent", None)
        smx = self.get("smx", None)
        scan = self.get("scan", None)
        van_der_waals = self.get("van_der_waals", None)
        vdw_mode = self.get("vdw_mode", "atomic")
        plots = self.get("plots", None)
        nbo = self.get("nbo", None)
        geom_opt = self.get("geom_opt", None)
        cdft = self.get("cdft", None)
        almo = self.get("almo", None)

        qcin = QCInput(
            molecule=mol,
            rem=self["rem"],
            opt=opt,
            pcm=pcm,
            solvent=solvent,
            smx=smx,
            scan=scan,
            van_der_waals=van_der_waals,
            vdw_mode=vdw_mode,
            plots=plots,
            nbo=nbo,
            geom_opt=geom_opt,
            cdft=cdft,
            almo=almo
        )
        qcin.write_file(input_file)


@explicit_serialize
class WriteInput(FiretaskBase):
    """
    Writes QChem input file from QCInput object.

    required_params:
        qc_input (QCInput): QCInput object

    optional_params:
        input_file (str): Name of the QChem input file. Defaults to mol.qin
        write_to_dir (str): Path of the directory where the QChem input file will be written,
        the default is to write to the current working directory

    """

    required_params = ["qc_input"]
    optional_params = ["input_file", "write_to_dir"]

    def run_task(self, fw_spec):
        # if a QCInput object is provided
        input_file = os.path.join(
            self.get("write_to_dir", ""), self.get("input_file", "mol.qin")
        )

        qcin = self["qc_input"]
        qcin.write_file(input_file)
