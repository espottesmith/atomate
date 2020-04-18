# coding: utf-8

# This module defines firetasks for writing pyGSM input files

import os

from atomate.utils.utils import load_class
from fireworks import FiretaskBase, explicit_serialize
from pymatgen.io.qchem.inputs import QCInput
from pymatgen.io.gsm.inputs import QCTemplate, GSMIsomerInput
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.analysis.fragmenter import metal_edge_extender

__author__ = "Evan Spotte-Smith"
__copyright__ = "Copyright 2020, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Evan Spotte-Smith"
__email__ = "espottesmith@gmail.com"
__status__ = "Alpha"
__date__ = "04/17/2020"
__credits__ = "Sam Blau"


@explicit_serialize
class WriteInputFromIOSet(FiretaskBase):
    """
    Writes QChem Input files from input sets. A dictionary is passed to WriteInputFromIOSet where
    parameters are given as keys in the dictionary.

    required_params:
        input_set (QChemDictSet): A QChemDictSet object.

    optional_params:
        molecule (Molecule): Molecule that will be subjected to an electronic structure calculation
        molecule_file (str): Name of the file for the initial molecule geometry. Defaults to
            input.xyz.
        lot_file (str): Name of the pyGSM input file. Defaults to qin
        write_to_dir (str): Path of the directory where the pyGSM input file will be written.
            The default is to write to the current working directory
    """

    required_params = ["input_set"]
    optional_params = [
        "molecule", "molecule_file", "lot_file", "write_to_dir"
    ]

    def run_task(self, fw_spec):
        input_file = os.path.join(self.get("write_to_dir", ""),
                                  self.get("lot_file", "qin"))
        mol_file = os.path.join(self.get("write_to_dir", ""),
                                self.get("molecule_file", "input.xyz"))

        # if a full QChemDictSet object was provided
        qcin = self["qchem_input_set"]
        # if a molecule is being passed through fw_spec

        if fw_spec.get("prev_calc_molecule"):
            prev_calc_mol = fw_spec.get("prev_calc_molecule")
            # if a molecule is also passed as an optional parameter
            if self.get("molecule"):
                mol = self.get("molecule")
                # check if mol and prev_calc_mol are isomorphic
                mol_graph = MoleculeGraph.with_local_env_strategy(mol,
                                                                  OpenBabelNN())
                prev_mol_graph = MoleculeGraph.with_local_env_strategy(prev_calc_mol,
                                                                       OpenBabelNN())
                # If they are isomorphic, aka a previous FW has not changed bonding,
                # then we will use prev_calc_mol. If bonding has changed, we will use mol.
                if mol_graph.isomorphic_to(prev_mol_graph):
                    mol = prev_calc_mol
                else:
                    print("Not using prev_calc_mol as it is not isomorphic to passed molecule!")
            else:
              mol = prev_calc_mol
        elif self.get("molecule"):
            mol = self.get("molecule")
        # if no molecule is present raise an error
        else:
            raise KeyError(
                "No molecule present, add as an optional param or check fw_spec"
            )

        qcin.write(input_file)
        mol.to("xyz", mol_file)


@explicit_serialize
class WriteCustomInput(FiretaskBase):
    """
        Writes QChem Input files from custom input sets. This firetask gives the maximum flexibility when trying
        to define custom input parameters.

        required_params:
            rem (dict): A rem section for a Q-Chem input file.
                Ex: rem = {'method': 'rimp2', 'basis': '6-31*G++' ... }

        optional_params:
            molecule (pymatgen Molecule object): The molecule representing the
                initial molecular geometry.
            pcm (dict): A dictionary of values relating to the polarizable continuum
                model (PCM). Note that, if a pcm dict is provided, then a "solvent"
                dict (described below) should also be provided, but a "smx" dict
                (also described below, and used for SMX methods like SMD) should not
                be provided.
                Ex: pcm = {"theory": "ief-pcm", "radii": "uff"}
            solvent (dict): A dictionary of values related to the solvent used in
                the PCM method. Note that this section should be provided if PCM
                is to be used (see above), but should not be provided for use with
                the SMD or other SMX methods.
                Ex: solvent = {"dielectric": 78.39}
            smx (dict): A dictionary of values for use with the SMX methods (like
                SMD). If this section is provided, then neither the pcm nor the
                solvent sections should be provided.
            molecule_file (str): Name of the file for the initial molecule geometry. Defaults to
                input.xyz.
            lot_file (str): Name of the pyGSM input file. Defaults to qin
            write_to_dir (str): Path of the directory where the QChem input file will be written,
            the default is to write to the current working directory
        """

    required_params = ["rem"]
    # optional_params will need to be modified if more QChem sections are added QCInput
    optional_params = [
        "molecule", "pcm", "solvent", "smx", "molecule_file", "lot_file",
        "write_to_dir"
    ]

    def run_task(self, fw_spec):
        input_file = os.path.join(self.get("write_to_dir", ""),
                                  self.get("lot_file", "mol.qin"))
        mol_file = os.path.join(self.get("write_to_dir", ""),
                                self.get("molecule_file", "input.xyz"))
        # if a molecule is being passed through fw_spec
        if fw_spec.get("prev_calc_molecule"):
            prev_calc_mol = fw_spec.get("prev_calc_molecule")
            # if a molecule is also passed as an optional parameter
            if self.get("molecule"):
                mol = self.get("molecule")
                # check if mol and prev_calc_mol are isomorphic
                mol_graph = MoleculeGraph.with_local_env_strategy(mol,
                                                                  OpenBabelNN())
                prev_mol_graph = MoleculeGraph.with_local_env_strategy(prev_calc_mol,
                                                                       OpenBabelNN())
                if mol_graph.isomorphic_to(prev_mol_graph):
                    mol = prev_calc_mol
                else:
                    print("WARNING: Molecule from spec is not isomorphic to passed molecule!")
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
        pcm = self.get("pcm", None)
        solvent = self.get("solvent", None)
        smx = self.get("smx", None)

        qcin = QCTemplate(
            rem=self["rem"],
            pcm=pcm,
            solvent=solvent,
            smx=smx)
        qcin.write_file(input_file)
        mol.to("xyz", mol_file)


@explicit_serialize
class WriteInput(FiretaskBase):
    """
    Writes QChem input file from QCInput object.

    required_params:
        input (QCTemplate): QCTemplate object
        molecule (pymatgen Molecule object): The molecule representing the
            initial molecular geometry.

    optional_params:
        molecule_file (str): Name of the file for the initial molecule geometry. Defaults to
            input.xyz.
        lot_file (str): Name of the pyGSM input file. Defaults to qin
        write_to_dir (str): Path of the directory where the QChem input file will be written,
            the default is to write to the current working directory

    """
    required_params = ["input", "molecule"]
    optional_params = ["molecule_file", "lot_file", "write_to_dir"]

    def run_task(self, fw_spec):
        # if a QCInput object is provided
        input_file = os.path.join(self.get("write_to_dir", ""),
                                  self.get("lot_file", "mol.qin"))
        mol_file = os.path.join(self.get("write_to_dir", ""),
                                self.get("molecule_file", "input.xyz"))

        qcin = self["input"]
        qcin.write_file(input_file)
        self["molecule"].to("xyz", mol_file)


@explicit_serialize
class WriteIsomer(FiretaskBase):
    """
    Writes a GSM isomer file.

    Note that isomer files are only needed for single-ended techniques.

    required_params:
        isomers (GSMIsomerInput or dict): bonds to be formed/broken and
            angles/out-of-plane bends/dihedrals to be altered. If isomers is a
            dict, it should have at least one of the following as keys:
            bonds_broken
            bonds_formed
            angles
            torsions
            out_of_planes

    optional_params:
        molecule (pymatgen Molecule object)
        isomer_file (str): Name of the pyGSM input file. Defaults to isomers.txt
        write_to_dir (str): Path of the directory where the QChem input file
            will be written, the default is to write to the current working
            directory
    """

    required_params = ["isomers"]
    optional_params = ["molecule", "isomer_file", "write_to_dir"]

    def run_task(self, fw_spec):
        isomer_file = os.path.join(self.get("write_to_dir", ""),
                                   self.get("isomer_file", "isomers.txt"))

        if isinstance(self["isomers"], dict):
            if fw_spec.get("prev_calc_molecule"):
                prev_calc_mol = fw_spec.get("prev_calc_molecule")
                # if a molecule is also passed as an optional parameter
                if self.get("molecule"):
                    mol = self.get("molecule")
                    # check if mol and prev_calc_mol are isomorphic
                    mol_graph = MoleculeGraph.with_local_env_strategy(mol,
                                                                      OpenBabelNN())
                    prev_mol_graph = MoleculeGraph.with_local_env_strategy(prev_calc_mol,
                                                                           OpenBabelNN())
                    if mol_graph.isomorphic_to(prev_mol_graph):
                        mol = prev_calc_mol
                    else:
                        print("WARNING: Molecule from spec is not isomorphic to passed molecule!")
                else:
                  mol = prev_calc_mol
            elif self.get("molecule"):
                mol = self.get("molecule")
            else:
                mol = None

            if mol is None:
                use_graph = False
            else:
                use_graph = True

            isomer_object = GSMIsomerInput(molecule=mol,
                                           bonds_formed=self["isomers"].get("bonds_formed"),
                                           bonds_broken=self["isomers"].get("bonds_broken"),
                                           angles=self["isomers"].get("angles"),
                                           torsions=self["isomers"].get("torsions"),
                                           out_of_planes=self["isomers"].get("out_of_planes"),
                                           use_graph=use_graph)
        elif isinstance(self["isomers"], GSMIsomerInput):
            isomer_object = self["isomers"]
        else:
            raise ValueError("Input 'isomers' should either be a GSMIsomerInput"
                             " object or a dictionary!")

        isomer_object.write_file(isomer_file)