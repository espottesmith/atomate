# coding: utf-8

# Defines standardized Fireworks that can be chained easily to perform various
# sequences of QChem calculations.

from itertools import chain

from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.io.qchem.utils import map_atoms_reaction
from pymatgen.analysis.berny import BernyOptimizer

from fireworks import Firework

from atomate.qchem.firetasks.parse_outputs import QChemToDb
from atomate.qchem.firetasks.run_calc import RunQChemCustodian
from atomate.qchem.firetasks.write_inputs import WriteInputFromIOSet
from atomate.qchem.firetasks.fragmenter import FragmentMolecule

__author__ = "Samuel Blau, Evan Spotte-Smith"
__copyright__ = "Copyright 2018, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Samuel Blau"
__email__ = "samblau1@gmail.com"
__status__ = "Alpha"
__date__ = "5/23/18"
__credits__ = "Brandon Wood, Shyam Dwaraknath"


class SinglePointFW(Firework):
    def __init__(self,
                 molecule=None,
                 name="single point",
                 qchem_cmd=">>qchem_cmd<<",
                 multimode=">>multimode<<",
                 max_cores=">>max_cores<<",
                 qchem_input_params=None,
                 db_file=None,
                 parents=None,
                 **kwargs):
        """

        Args:
            molecule (Molecule): Input molecule.
            name (str): Name for the Firework.
            qchem_cmd (str): Command to run QChem. Supports env_chk.
            multimode (str): Parallelization scheme, either openmp or mpi. Supports env_chk.
            max_cores (int): Maximum number of cores to parallelize over. Supports env_chk.
            qchem_input_params (dict): Specify kwargs for instantiating the input set parameters.
                                       Basic uses would be to modify the default inputs of the set,
                                       such as dft_rung, basis_set, pcm_dielectric, scf_algorithm,
                                       or max_scf_cycles. See pymatgen/io/qchem/sets.py for default
                                       values of all input parameters. For instance, if a user wanted
                                       to use a more advanced DFT functional, include a pcm with a
                                       dielectric of 30, and use a larger basis, the user would set
                                       qchem_input_params = {"dft_rung": 5, "pcm_dielectric": 30,
                                       "basis_set": "6-311++g**"}. However, more advanced customization
                                       of the input is also possible through the overwrite_inputs key
                                       which allows the user to directly modify the rem, pcm, smd, and
                                       solvent dictionaries that QChemDictSet passes to inputs.py to
                                       print an actual input file. For instance, if a user wanted to
                                       set the sym_ignore flag in the rem section of the input file
                                       to true, then they would set qchem_input_params = {"overwrite_inputs":
                                       "rem": {"sym_ignore": "true"}}. Of course, overwrite_inputs
                                       could be used in conjuction with more typical modifications,
                                       as seen in the test_double_FF_opt workflow test.
            db_file (str): Path to file specifying db credentials to place output parsing.
            parents ([Firework]): Parents of this particular Firework.
            **kwargs: Other kwargs that are passed to Firework.__init__.
        """

        qchem_input_params = qchem_input_params or {}
        input_file="mol.qin"
        output_file="mol.qout"
        t = []
        t.append(
            WriteInputFromIOSet(
                molecule=molecule,
                qchem_input_set="SinglePointSet",
                input_file=input_file,
                qchem_input_params=qchem_input_params))
        t.append(
            RunQChemCustodian(
                qchem_cmd=qchem_cmd,
                multimode=multimode,
                input_file=input_file,
                output_file=output_file,
                max_cores=max_cores,
                job_type="normal"))
        t.append(
            QChemToDb(
                db_file=db_file,
                input_file=input_file,
                output_file=output_file,
                additional_fields={"task_label": name}))
        super(SinglePointFW, self).__init__(
            t,
            parents=parents,
            name=name,
            **kwargs)


class OptimizeFW(Firework):
    def __init__(self,
                 molecule=None,
                 name="structure optimization",
                 qchem_cmd=">>qchem_cmd<<",
                 multimode=">>multimode<<",
                 max_cores=">>max_cores<<",
                 qchem_input_params=None,
                 db_file=None,
                 parents=None,
                 **kwargs):
        """
        Optimize the given structure.

        Args:
            molecule (Molecule): Input molecule.
            name (str): Name for the Firework.
            qchem_cmd (str): Command to run QChem. Supports env_chk.
            multimode (str): Parallelization scheme, either openmp or mpi. Defaults to openmp.
            max_cores (int): Maximum number of cores to parallelize over. Supports env_chk.
            qchem_input_params (dict): Specify kwargs for instantiating the input set parameters.
                                       Basic uses would be to modify the default inputs of the set,
                                       such as dft_rung, basis_set, pcm_dielectric, scf_algorithm,
                                       or max_scf_cycles. See pymatgen/io/qchem/sets.py for default
                                       values of all input parameters. For instance, if a user wanted
                                       to use a more advanced DFT functional, include a pcm with a
                                       dielectric of 30, and use a larger basis, the user would set
                                       qchem_input_params = {"dft_rung": 5, "pcm_dielectric": 30,
                                       "basis_set": "6-311++g**"}. However, more advanced customization
                                       of the input is also possible through the overwrite_inputs key
                                       which allows the user to directly modify the rem, pcm, smd, and
                                       solvent dictionaries that QChemDictSet passes to inputs.py to
                                       print an actual input file. For instance, if a user wanted to
                                       set the sym_ignore flag in the rem section of the input file
                                       to true, then they would set qchem_input_params = {"overwrite_inputs":
                                       "rem": {"sym_ignore": "true"}}. Of course, overwrite_inputs
                                       could be used in conjuction with more typical modifications,
                                       as seen in the test_double_FF_opt workflow test.
            db_file (str): Path to file specifying db credentials to place output parsing.
            parents ([Firework]): Parents of this particular Firework.
            **kwargs: Other kwargs that are passed to Firework.__init__.
        """

        qchem_input_params = qchem_input_params or {}
        input_file="mol.qin"
        output_file="mol.qout"
        t = []
        t.append(
            WriteInputFromIOSet(
                molecule=molecule,
                qchem_input_set="OptSet",
                input_file=input_file,
                qchem_input_params=qchem_input_params))
        t.append(
            RunQChemCustodian(
                qchem_cmd=qchem_cmd,
                multimode=multimode,
                input_file=input_file,
                output_file=output_file,
                max_cores=max_cores,
                job_type="normal"))
        t.append(
            QChemToDb(
                db_file=db_file,
                input_file=input_file,
                output_file=output_file,
                additional_fields={"task_label": name}))
        super(OptimizeFW, self).__init__(
            t,
            parents=parents,
            name=name,
            **kwargs)


class FrequencyFW(Firework):
    def __init__(self,
                 molecule=None,
                 name="frequency calculation",
                 qchem_cmd=">>qchem_cmd<<",
                 multimode=">>multimode<<",
                 max_cores=">>max_cores<<",
                 qchem_input_params=None,
                 db_file=None,
                 parents=None,
                 **kwargs):
        """
        Optimize the given structure.

        Args:
            molecule (Molecule): Input molecule.
            name (str): Name for the Firework.
            qchem_cmd (str): Command to run QChem. Supports env_chk.
            multimode (str): Parallelization scheme, either openmp or mpi. Defaults to openmp.
            max_cores (int): Maximum number of cores to parallelize over. Supports env_chk.
            qchem_input_params (dict): Specify kwargs for instantiating the input set parameters.
                                       Basic uses would be to modify the default inputs of the set,
                                       such as dft_rung, basis_set, pcm_dielectric, scf_algorithm,
                                       or max_scf_cycles. See pymatgen/io/qchem/sets.py for default
                                       values of all input parameters. For instance, if a user wanted
                                       to use a more advanced DFT functional, include a pcm with a
                                       dielectric of 30, and use a larger basis, the user would set
                                       qchem_input_params = {"dft_rung": 5, "pcm_dielectric": 30,
                                       "basis_set": "6-311++g**"}. However, more advanced customization
                                       of the input is also possible through the overwrite_inputs key
                                       which allows the user to directly modify the rem, pcm, smd, and
                                       solvent dictionaries that QChemDictSet passes to inputs.py to
                                       print an actual input file. For instance, if a user wanted to
                                       set the sym_ignore flag in the rem section of the input file
                                       to true, then they would set qchem_input_params = {"overwrite_inputs":
                                       "rem": {"sym_ignore": "true"}}. Of course, overwrite_inputs
                                       could be used in conjuction with more typical modifications,
                                       as seen in the test_double_FF_opt workflow test.
            db_file (str): Path to file specifying db credentials to place output parsing.
            parents ([Firework]): Parents of this particular Firework.
            **kwargs: Other kwargs that are passed to Firework.__init__.
        """

        qchem_input_params = qchem_input_params or {}
        input_file="mol.qin"
        output_file="mol.qout"
        t = []
        t.append(
            WriteInputFromIOSet(
                molecule=molecule,
                qchem_input_set="FreqSet",
                input_file=input_file,
                qchem_input_params=qchem_input_params))
        t.append(
            RunQChemCustodian(
                qchem_cmd=qchem_cmd,
                multimode=multimode,
                input_file=input_file,
                output_file=output_file,
                max_cores=max_cores,
                job_type="normal"))
        t.append(
            QChemToDb(
                db_file=db_file,
                input_file=input_file,
                output_file=output_file,
                additional_fields={"task_label": name}))
        super(FrequencyFW, self).__init__(
            t,
            parents=parents,
            name=name,
            **kwargs)


class TransitionStateFW(Firework):
    def __init__(self,
                 molecule=None,
                 name="transition state structure optimization",
                 qchem_cmd=">>qchem_cmd<<",
                 multimode=">>multimode<<",
                 max_cores=">>max_cores<<",
                 qchem_input_params=None,
                 db_file=None,
                 parents=None,
                 **kwargs):
        """
        Optimize the given molecule to a saddle point of the potential energy surface (transition
        state).

        Args:
            molecule (Molecule): Input molecule.
            name (str): Name for the Firework.
            qchem_cmd (str): Command to run QChem. Supports env_chk.
            multimode (str): Parallelization scheme, either openmp or mpi. Defaults to openmp.
            max_cores (int): Maximum number of cores to parallelize over. Supports env_chk.
            qchem_input_params (dict): Specify kwargs for instantiating the input set parameters.
                                       Basic uses would be to modify the default inputs of the set,
                                       such as dft_rung, basis_set, pcm_dielectric, scf_algorithm,
                                       or max_scf_cycles. See pymatgen/io/qchem/sets.py for default
                                       values of all input parameters. For instance, if a user wanted
                                       to use a more advanced DFT functional, include a pcm with a
                                       dielectric of 30, and use a larger basis, the user would set
                                       qchem_input_params = {"dft_rung": 5, "pcm_dielectric": 30,
                                       "basis_set": "6-311++g**"}. However, more advanced customization
                                       of the input is also possible through the overwrite_inputs key
                                       which allows the user to directly modify the rem, pcm, smd, and
                                       solvent dictionaries that QChemDictSet passes to inputs.py to
                                       print an actual input file. For instance, if a user wanted to
                                       set the sym_ignore flag in the rem section of the input file
                                       to true, then they would set qchem_input_params = {"overwrite_inputs":
                                       "rem": {"sym_ignore": "true"}}. Of course, overwrite_inputs
                                       could be used in conjuction with more typical modifications,
                                       as seen in the test_double_FF_opt workflow test.
            db_file (str): Path to file specifying db credentials to place output parsing.
            parents ([Firework]): Parents of this particular Firework.
            **kwargs: Other kwargs that are passed to Firework.__init__.
        """

        qchem_input_params = qchem_input_params or {}
        input_file = "mol.qin"
        output_file = "mol.qout"
        t = list()
        t.append(
            WriteInputFromIOSet(
                molecule=molecule,
                qchem_input_set="TransitionStateSet",
                input_file=input_file,
                qchem_input_params=qchem_input_params))
        t.append(
            RunQChemCustodian(
                qchem_cmd=qchem_cmd,
                multimode=multimode,
                input_file=input_file,
                output_file=output_file,
                max_cores=max_cores,
                job_type="normal"))
        t.append(
            QChemToDb(
                db_file=db_file,
                input_file=input_file,
                output_file=output_file,
                additional_fields={"task_label": name}))
        super(TransitionStateFW, self).__init__(
            t,
            parents=parents,
            name=name,
            **kwargs)


class FreezingStringFW(Firework):
    def __init__(self,
                 reactants,
                 products,
                 name="freezing string method calculation",
                 qchem_cmd=">>qchem_cmd<<",
                 multimode=">>multimode<<",
                 max_cores=">>max_cores<<",
                 qchem_input_params=None,
                 db_file=None,
                 parents=None,
                 map_atoms=True,
                 additions_allowed=0,
                 **kwargs):
        """
        Identify a guess geometry for a reaction transition state using the freezing string method.

        Args:
            molecule (Molecule): Input molecule.
            name (str): Name for the Firework.
            qchem_cmd (str): Command to run QChem. Supports env_chk.
            multimode (str): Parallelization scheme, either openmp or mpi. Defaults to openmp.
            max_cores (int): Maximum number of cores to parallelize over. Supports env_chk.
            qchem_input_params (dict): Specify kwargs for instantiating the input set parameters.
                                       Basic uses would be to modify the default inputs of the set,
                                       such as dft_rung, basis_set, pcm_dielectric, scf_algorithm,
                                       or max_scf_cycles. See pymatgen/io/qchem/sets.py for default
                                       values of all input parameters. For instance, if a user wanted
                                       to use a more advanced DFT functional, include a pcm with a
                                       dielectric of 30, and use a larger basis, the user would set
                                       qchem_input_params = {"dft_rung": 5, "pcm_dielectric": 30,
                                       "basis_set": "6-311++g**"}. However, more advanced customization
                                       of the input is also possible through the overwrite_inputs key
                                       which allows the user to directly modify the rem, pcm, smd, and
                                       solvent dictionaries that QChemDictSet passes to inputs.py to
                                       print an actual input file. For instance, if a user wanted to
                                       set the sym_ignore flag in the rem section of the input file
                                       to true, then they would set qchem_input_params = {"overwrite_inputs":
                                       "rem": {"sym_ignore": "true"}}. Of course, overwrite_inputs
                                       could be used in conjuction with more typical modifications,
                                       as seen in the test_double_FF_opt workflow test.
            db_file (str): Path to file specifying db credentials to place output parsing.
            parents ([Firework]): Parents of this particular Firework.
            map_atoms (Bool): Should an attempt be made to map reactant atoms to product atoms?
            additions_allowed (int): If mapping is to be done, can fictitious bonds be added to ensure
                                     that a subgraph isomorphism is found?
            **kwargs: Other kwargs that are passed to Firework.__init__.
        """

        qchem_input_params = qchem_input_params or {}
        input_file = "mol.qin"
        output_file = "mol.qout"

        if map_atoms:
            rct_mgs = [MoleculeGraph.with_local_env_strategy(r, OpenBabelNN(),
                                                             extend_structure=False, reorder=False)
                       for r in reactants]
            pro_mgs = [MoleculeGraph.with_local_env_strategy(p, OpenBabelNN(),
                                                             extend_structure=False, reorder=False)
                       for p in products]
            # TODO: make this a FireTask
            if len(products) == 1:
                mapping = map_atoms_reaction(rct_mgs, pro_mgs[0],
                                             num_additions_allowed=additions_allowed)
                if mapping is None:
                    raise ValueError("Reactant atoms cannot be mapped to product molecules using existing methods. "
                                     "Please map atoms by hand and set map_atoms=False to try again.")

                species = [None for _ in range(len(products[0]))]
                coords = [None for _ in range(len(products[0]))]
                for e, site in enumerate(products[0]):
                    species[mapping[e]] = site.species
                    coords[mapping[e]] = site.coords
                product = Molecule(species, coords, charge=products[0].charge,
                                   spin_multiplicity=products[0].spin_multiplicity)
                molecule = {"reactants": reactants, "products": [product]}
            elif len(reactants) == 1:
                mapping = map_atoms_reaction(pro_mgs, rct_mgs[0],
                                             num_additions_allowed=additions_allowed)
                # print(mapping)
                species = [None for _ in range(len(reactants[0]))]
                coords = [None for _ in range(len(reactants[0]))]
                for e, site in enumerate(reactants[0]):
                    species[mapping[e]] = site.species
                    coords[mapping[e]] = site.coords
                reactant = Molecule(species, coords, charge=reactants[0].charge,
                                    spin_multiplicity=reactants[0].spin_multiplicity)
                molecule = {"reactants": [reactant], "products": products}
            else:
                raise ValueError("Cannot map atoms with more than one product and more than one "
                                 "reactant.")
        else:
            molecule = {"reactants": reactants, "products": products}

        t = list()
        t.append(
            WriteInputFromIOSet(
                molecule=molecule,
                qchem_input_set="FreezingStringSet",
                input_file=input_file,
                qchem_input_params=qchem_input_params))
        t.append(
            RunQChemCustodian(
                qchem_cmd=qchem_cmd,
                multimode=multimode,
                input_file=input_file,
                output_file=output_file,
                max_cores=max_cores,
                job_type="normal",
                gzipped_output=False))
        t.append(
            QChemToDb(
                db_file=db_file,
                input_file=input_file,
                output_file=output_file,
                additional_fields={"task_label": name},
                extra_files=["Vfile.txt", "stringfile.txt", "perp_grad_file.txt"]))
        super(FreezingStringFW, self).__init__(
            t,
            parents=parents,
            name=name,
            **kwargs)


class GrowingStringFW(Firework):
    def __init__(self,
                 reactants,
                 products,
                 name="growing string method calculation",
                 qchem_cmd=">>qchem_cmd<<",
                 multimode=">>multimode<<",
                 max_cores=">>max_cores<<",
                 qchem_input_params=None,
                 db_file=None,
                 parents=None,
                 map_atoms=True,
                 additions_allowed=0,
                 **kwargs):
        """
        Identify a guess geometry for a reaction transition state using the growing string method.

        Args:
            molecule (Molecule): Input molecule.
            name (str): Name for the Firework.
            qchem_cmd (str): Command to run QChem. Supports env_chk.
            multimode (str): Parallelization scheme, either openmp or mpi. Defaults to openmp.
            max_cores (int): Maximum number of cores to parallelize over. Supports env_chk.
            qchem_input_params (dict): Specify kwargs for instantiating the input set parameters.
                                       Basic uses would be to modify the default inputs of the set,
                                       such as dft_rung, basis_set, pcm_dielectric, scf_algorithm,
                                       or max_scf_cycles. See pymatgen/io/qchem/sets.py for default
                                       values of all input parameters. For instance, if a user wanted
                                       to use a more advanced DFT functional, include a pcm with a
                                       dielectric of 30, and use a larger basis, the user would set
                                       qchem_input_params = {"dft_rung": 5, "pcm_dielectric": 30,
                                       "basis_set": "6-311++g**"}. However, more advanced customization
                                       of the input is also possible through the overwrite_inputs key
                                       which allows the user to directly modify the rem, pcm, smd, and
                                       solvent dictionaries that QChemDictSet passes to inputs.py to
                                       print an actual input file. For instance, if a user wanted to
                                       set the sym_ignore flag in the rem section of the input file
                                       to true, then they would set qchem_input_params = {"overwrite_inputs":
                                       "rem": {"sym_ignore": "true"}}. Of course, overwrite_inputs
                                       could be used in conjuction with more typical modifications,
                                       as seen in the test_double_FF_opt workflow test.
            db_file (str): Path to file specifying db credentials to place output parsing.
            parents ([Firework]): Parents of this particular Firework.
            map_atoms (Bool): Should an attempt be made to map reactant atoms to product atoms?
            additions_allowed (int): If mapping is to be done, can fictitious bonds be added to ensure
                                     that a subgraph isomorphism is found?
            **kwargs: Other kwargs that are passed to Firework.__init__.
        """

        qchem_input_params = qchem_input_params or {}
        input_file = "mol.qin"
        output_file = "mol.qout"

        if map_atoms:
            rct_mgs = [MoleculeGraph.with_local_env_strategy(r, OpenBabelNN(),
                                                             extend_structure=False, reorder=False)
                       for r in reactants]
            pro_mgs = [MoleculeGraph.with_local_env_strategy(p, OpenBabelNN(),
                                                             extend_structure=False, reorder=False)
                       for p in products]
            # TODO: make this a FireTask
            if len(products) == 1:
                mapping = map_atoms_reaction(rct_mgs, pro_mgs[0],
                                             num_additions_allowed=additions_allowed)
                if mapping is None:
                    raise ValueError("Reactant atoms cannot be mapped to product molecules using existing methods. "
                                     "Please map atoms by hand and set map_atoms=False to try again.")

                species = [None for _ in range(len(products[0]))]
                coords = [None for _ in range(len(products[0]))]
                for e, site in enumerate(products[0]):
                    species[mapping[e]] = site.species
                    coords[mapping[e]] = site.coords
                product = Molecule(species, coords, charge=products[0].charge,
                                   spin_multiplicity=products[0].spin_multiplicity)
                molecule = {"reactants": reactants, "products": [product]}
            elif len(reactants) == 1:
                mapping = map_atoms_reaction(pro_mgs, rct_mgs[0],
                                             num_additions_allowed=additions_allowed)

                species = [None for _ in range(len(reactants[0]))]
                coords = [None for _ in range(len(reactants[0]))]
                for e, site in enumerate(reactants[0]):
                    species[mapping[e]] = site.species
                    coords[mapping[e]] = site.coords
                reactant = Molecule(species, coords, charge=reactants[0].charge,
                                    spin_multiplicity=reactants[0].spin_multiplicity)
                molecule = {"reactants": [reactant], "products": products}
            else:
                raise ValueError("Cannot map atoms with more than one product and more than one "
                                 "reactant.")
        else:
            molecule = {"reactants": reactants, "products": products}

        t = list()
        t.append(
            WriteInputFromIOSet(
                molecule=molecule,
                qchem_input_set="GrowingStringSet",
                input_file=input_file,
                qchem_input_params=qchem_input_params))
        t.append(
            RunQChemCustodian(
                qchem_cmd=qchem_cmd,
                multimode=multimode,
                input_file=input_file,
                output_file=output_file,
                max_cores=max_cores,
                job_type="normal",
                gzipped_output=False))
        t.append(
            QChemToDb(
                db_file=db_file,
                input_file=input_file,
                output_file=output_file,
                additional_fields={"task_label": name},
                extra_files=["Vfile.txt", "perp_grad_file.txt"]))
        super(GrowingStringFW, self).__init__(
            t,
            parents=parents,
            name=name,
            **kwargs)


class PESScanFW(Firework):
    def __init__(self,
                 molecule=None,
                 name="potential energy surface scan",
                 qchem_cmd=">>qchem_cmd<<",
                 multimode=">>multimode<<",
                 max_cores=">>max_cores<<",
                 qchem_input_params=None,
                 scan_variables=None,
                 db_file=None,
                 parents=None,
                 **kwargs):
        """
        Perform a potential energy surface scan by varying bond lengths, angles,
        and/or dihedral angles in a molecule.

        Args:
           molecule (Molecule): Input molecule.
            name (str): Name for the Firework.
            qchem_cmd (str): Command to run QChem. Supports env_chk.
            multimode (str): Parallelization scheme, either openmp or mpi. Supports env_chk.
            max_cores (int): Maximum number of cores to parallelize over. Supports env_chk.
            qchem_input_params (dict): Specify kwargs for instantiating the input set parameters.
                                       Basic uses would be to modify the default inputs of the set,
                                       such as dft_rung, basis_set, pcm_dielectric, scf_algorithm,
                                       or max_scf_cycles. See pymatgen/io/qchem/sets.py for default
                                       values of all input parameters. For instance, if a user wanted
                                       to use a more advanced DFT functional, include a pcm with a
                                       dielectric of 30, and use a larger basis, the user would set
                                       qchem_input_params = {"dft_rung": 5, "pcm_dielectric": 30,
                                       "basis_set": "6-311++g**"}. However, more advanced customization
                                       of the input is also possible through the overwrite_inputs key
                                       which allows the user to directly modify the rem, pcm, smd, and
                                       solvent dictionaries that QChemDictSet passes to inputs.py to
                                       print an actual input file.
            scan_variables (dict): dict {str: list}, where the key is the type of variable ("stre"
                                   for bond length, "bend" for angle, "tors" for dihedral angle),
                                   and the list contains all of the variable set information
            db_file (str): Path to file specifying db credentials to place output parsing.
            parents ([Firework]): Parents of this particular Firework.
            **kwargs: Other kwargs that are passed to Firework.__init__.
        """

        if scan_variables is None:
            raise ValueError("Some variable input must be given! Provide some "
                             "bond, angle, or dihedral angle information.")

        qchem_input_params = qchem_input_params or dict()
        qchem_input_params["scan_variables"] = scan_variables
        input_file = "mol.qin"
        output_file = "mol.qout"
        t = list()

        t.append(
            WriteInputFromIOSet(
                molecule=molecule,
                qchem_input_set="PESScanSet",
                input_file=input_file,
                qchem_input_params=qchem_input_params))
        t.append(
            RunQChemCustodian(
                qchem_cmd=qchem_cmd,
                multimode=multimode,
                input_file=input_file,
                output_file=output_file,
                max_cores=max_cores,
                job_type="normal"))
        t.append(
            QChemToDb(
                db_file=db_file,
                input_file=input_file,
                output_file=output_file,
                additional_fields={"task_label": name}))
        super(PESScanFW, self).__init__(
            t,
            parents=parents,
            name=name,
            **kwargs)


class FrequencyFlatteningOptimizeFW(Firework):
    def __init__(self,
                 molecule=None,
                 name="frequency flattening structure optimization",
                 qchem_cmd=">>qchem_cmd<<",
                 multimode=">>multimode<<",
                 max_cores=">>max_cores<<",
                 qchem_input_params=None,
                 max_iterations=10,
                 max_molecule_perturb_scale=0.3,
                 linked=False,
                 db_file=None,
                 parents=None,
                 **kwargs):
        """
        Iteratively optimize the given structure and flatten imaginary frequencies to ensure that
        the resulting structure is a true minima and not a saddle point.

        Args:
            molecule (Molecule): Input molecule.
            name (str): Name for the Firework.
            qchem_cmd (str): Command to run QChem. Supports env_chk.
            multimode (str): Parallelization scheme, either openmp or mpi. Supports env_chk.
            max_cores (int): Maximum number of cores to parallelize over. Supports env_chk.
            qchem_input_params (dict): Specify kwargs for instantiating the input set parameters.
                                       Basic uses would be to modify the default inputs of the set,
                                       such as dft_rung, basis_set, pcm_dielectric, scf_algorithm,
                                       or max_scf_cycles. See pymatgen/io/qchem/sets.py for default
                                       values of all input parameters. For instance, if a user wanted
                                       to use a more advanced DFT functional, include a pcm with a
                                       dielectric of 30, and use a larger basis, the user would set
                                       qchem_input_params = {"dft_rung": 5, "pcm_dielectric": 30,
                                       "basis_set": "6-311++g**"}. However, more advanced customization
                                       of the input is also possible through the overwrite_inputs key
                                       which allows the user to directly modify the rem, pcm, smd, and
                                       solvent dictionaries that QChemDictSet passes to inputs.py to
                                       print an actual input file. For instance, if a user wanted to
                                       set the sym_ignore flag in the rem section of the input file
                                       to true, then they would set qchem_input_params = {"overwrite_inputs":
                                       "rem": {"sym_ignore": "true"}}. Of course, overwrite_inputs
                                       could be used in conjuction with more typical modifications,
                                       as seen in the test_double_FF_opt workflow test.
            max_iterations (int): Number of perturbation -> optimization -> frequency
                                  iterations to perform. Defaults to 10.
            max_molecule_perturb_scale (float): The maximum scaled perturbation that can be
                                                applied to the molecule. Defaults to 0.3.
            db_file (str): Path to file specifying db credentials to place output parsing.
            parents ([Firework]): Parents of this particular Firework.
            **kwargs: Other kwargs that are passed to Firework.__init__.
        """

        qchem_input_params = qchem_input_params or {}
        input_file = "mol.qin"
        output_file = "mol.qout"

        t = list()
        t.append(
            WriteInputFromIOSet(
                molecule=molecule,
                qchem_input_set="OptSet",
                input_file=input_file,
                qchem_input_params=qchem_input_params))
        t.append(
            RunQChemCustodian(
                qchem_cmd=qchem_cmd,
                multimode=multimode,
                input_file=input_file,
                output_file=output_file,
                max_cores=max_cores,
                job_type="opt_with_frequency_flattener",
                max_iterations=max_iterations,
                max_molecule_perturb_scale=max_molecule_perturb_scale,
                linked=linked))
        t.append(
            QChemToDb(
                db_file=db_file,
                input_file=input_file,
                output_file=output_file,
                additional_fields={
                    "task_label": name,
                    "special_run_type": "frequency_flattener",
                    "linked": linked
                }))
        super(FrequencyFlatteningOptimizeFW, self).__init__(
            t,
            parents=parents,
            name=name,
            **kwargs)


class FrequencyFlatteningTransitionStateFW(Firework):
    def __init__(self,
                 molecule=None,
                 name="frequency flattening transition state optimization",
                 qchem_cmd=">>qchem_cmd<<",
                 multimode=">>multimode<<",
                 max_cores=">>max_cores<<",
                 qchem_input_params=None,
                 max_iterations=10,
                 max_molecule_perturb_scale=0.3,
                 linked=False,
                 db_file=None,
                 parents=None,
                 **kwargs):
        """
        Iteratively optimize the transition state structure and flatten imaginary frequencies to
        ensure that the resulting structure is a true transition state.

        Args:
            molecule (Molecule): Input molecule.
            name (str): Name for the Firework.
            qchem_cmd (str): Command to run QChem. Supports env_chk.
            multimode (str): Parallelization scheme, either openmp or mpi. Supports env_chk.
            max_cores (int): Maximum number of cores to parallelize over. Supports env_chk.
            qchem_input_params (dict): Specify kwargs for instantiating the input set parameters.
                                       Basic uses would be to modify the default inputs of the set,
                                       such as dft_rung, basis_set, pcm_dielectric, scf_algorithm,
                                       or max_scf_cycles. See pymatgen/io/qchem/sets.py for default
                                       values of all input parameters. For instance, if a user wanted
                                       to use a more advanced DFT functional, include a pcm with a
                                       dielectric of 30, and use a larger basis, the user would set
                                       qchem_input_params = {"dft_rung": 5, "pcm_dielectric": 30,
                                       "basis_set": "6-311++g**"}. However, more advanced customization
                                       of the input is also possible through the overwrite_inputs key
                                       which allows the user to directly modify the rem, pcm, smd, and
                                       solvent dictionaries that QChemDictSet passes to inputs.py to
                                       print an actual input file. For instance, if a user wanted to
                                       set the sym_ignore flag in the rem section of the input file
                                       to true, then they would set qchem_input_params = {"overwrite_inputs":
                                       "rem": {"sym_ignore": "true"}}. Of course, overwrite_inputs
                                       could be used in conjuction with more typical modifications,
                                       as seen in the test_double_FF_opt workflow test.
            max_iterations (int): Number of perturbation -> optimization -> frequency
                                  iterations to perform. Defaults to 10.
            max_molecule_perturb_scale (float): The maximum scaled perturbation that can be
                                                applied to the molecule. Defaults to 0.3.
            db_file (str): Path to file specifying db credentials to place output parsing.
            parents ([Firework]): Parents of this particular Firework.
            **kwargs: Other kwargs that are passed to Firework.__init__.
        """

        qchem_input_params = qchem_input_params or {}
        input_file = "mol.qin"
        output_file = "mol.qout"
        runs = list(chain.from_iterable([["ts_" + str(ii), "freq_" + str(ii)]
                                         for ii in range(10)]))

        t = list()
        t.append(
            WriteInputFromIOSet(
                molecule=molecule,
                qchem_input_set="TransitionStateSet",
                input_file=input_file,
                qchem_input_params=qchem_input_params))
        t.append(
            RunQChemCustodian(
                qchem_cmd=qchem_cmd,
                multimode=multimode,
                input_file=input_file,
                output_file=output_file,
                max_cores=max_cores,
                job_type="opt_with_frequency_flattener",
                max_iterations=max_iterations,
                max_molecule_perturb_scale=max_molecule_perturb_scale,
                transition_state=True,
                linked=linked))
        t.append(
            QChemToDb(
                db_file=db_file,
                input_file=input_file,
                output_file=output_file,
                runs=runs,
                additional_fields={
                    "task_label": name,
                    "special_run_type": "ts_frequency_flattener",
                    "linked": linked
                }))

        super(FrequencyFlatteningTransitionStateFW, self).__init__(
            t,
            parents=parents,
            name=name,
            **kwargs)


class BernyOptimizeFW(Firework):
    def __init__(self,
                 molecule=None,
                 name="structure optimization with Berny optimizer",
                 qchem_cmd=">>qchem_cmd<<",
                 multimode=">>multimode<<",
                 max_cores=">>max_cores<<",
                 qchem_input_params=None,
                 transition_state=False,
                 optimizer_params=None,
                 max_iterations=10,
                 db_file=None,
                 parents=None,
                 **kwargs):
        """
        Optimize a molecule with energy and gradient calculations from Q-Chem and
        optimization steps determined by a Berny optimizer.

        Args:
                molecule (Molecule): Input molecule.
                name (str): Name for the Firework.
                qchem_cmd (str): Command to run QChem. Supports env_chk.
                multimode (str): Parallelization scheme, either openmp or mpi. Supports env_chk.
                max_cores (int): Maximum number of cores to parallelize over. Supports env_chk.
                transition_state (bool): If True (default False), optimize for a transition state,
                                         rather than a stable molecule. This changes the
                                         optimization algorithm.
                qchem_input_params (dict): Specify kwargs for instantiating the input set parameters.
                                           Basic uses would be to modify the default inputs of the set,
                                           such as dft_rung, basis_set, pcm_dielectric, scf_algorithm,
                                           or max_scf_cycles. See pymatgen/io/qchem/sets.py for default
                                           values of all input parameters. For instance, if a user wanted
                                           to use a more advanced DFT functional, include a pcm with a
                                           dielectric of 30, and use a larger basis, the user would set
                                           qchem_input_params = {"dft_rung": 5, "pcm_dielectric": 30,
                                           "basis_set": "6-311++g**"}. However, more advanced customization
                                           of the input is also possible through the overwrite_inputs key
                                           which allows the user to directly modify the rem, pcm, smd, and
                                           solvent dictionaries that QChemDictSet passes to inputs.py to
                                           print an actual input file. For instance, if a user wanted to
                                           set the sym_ignore flag in the rem section of the input file
                                           to true, then they would set qchem_input_params = {"overwrite_inputs":
                                           "rem": {"sym_ignore": "true"}}. Of course, overwrite_inputs
                                           could be used in conjuction with more typical modifications,
                                           as seen in the test_double_FF_opt workflow test.
                optimizer_params (dict): Specify kwargs for instantiating the optimizer parameters,
                                         including the logging method, verbosity, convergence parameters,
                                         and initial trust radius.
                max_iterations (int): Number of perturbation -> optimization -> frequency
                                      iterations to perform. Defaults to 10.
                db_file (str): Path to file specifying db credentials to place output parsing.
                parents ([Firework]): Parents of this particular Firework.
                **kwargs: Other kwargs that are passed to Firework.__init__.
        """

        qchem_input_params = qchem_input_params or {}
        qchem_input_params["geom_opt_max_cycles"] = 1
        optimizer_params = optimizer_params or {}
        optimizer_params["transition_state"] = transition_state
        if "max_steps" not in optimizer_params:
            optimizer_params["max_steps"] = 250

        input_file = "mol.qin"
        output_file = "mol.qout"
        runs = list()
        for ii in range(max_iterations):
            for jj in range(optimizer_params["max_steps"]):
                runs.append("opt_{}_{}".format(ii, jj))
            runs.append("freq_{}".format(ii))

        t = list()
        t.append(
            WriteInputFromIOSet(
                molecule=molecule,
                qchem_input_set="OptSet",
                input_file=input_file,
                qchem_input_params=qchem_input_params))
        t.append(
            RunQChemCustodian(
                qchem_cmd=qchem_cmd,
                multimode=multimode,
                input_file=input_file,
                output_file=output_file,
                max_cores=max_cores,
                job_type="berny_opt_with_frequency_flattener",
                max_iterations=max_iterations,
                transition_state=transition_state,
                handler_group="no_opt",
                optimizer_params=optimizer_params))
        t.append(
            QChemToDb(
                db_file=db_file,
                input_file=input_file,
                output_file=output_file,
                runs=runs,
                additional_fields={
                    "task_label": name,
                    "special_run_type": "berny_optimization",
                    "linked": True
                }))

        super(BernyOptimizeFW, self).__init__(
            t,
            parents=parents,
            name=name,
            **kwargs)


class FragmentFW(Firework):
    def __init__(self,
                 molecule=None,
                 depth=1,
                 open_rings=True,
                 additional_charges=None,
                 do_triplets=True,
                 linked=False,
                 name="fragment and optimize",
                 qchem_input_params=None,
                 db_file=None,
                 check_db=True,
                 parents=None,
                 **kwargs):
        """
        Fragment the given structure and optimize all unique fragments

        Args:
            molecule (Molecule): Input molecule.
            depth (int): Fragmentation depth. Defaults to 1. See fragmenter firetask for more details.
            open_rings (bool): Whether or not to open any rings encountered during fragmentation.
                               Defaults to True. See fragmenter firetask for more details.
            additional_charges (list): List of additional charges besides the defaults. See fragmenter
                                       firetask for more details.
            do_triplets (bool): Whether to simulate triplets as well as singlets for molecules with an
                                even number of electrons. Defaults to True.
            name (str): Name for the Firework.
            qchem_input_params (dict): Specify kwargs for instantiating the input set parameters.
                                       Basic uses would be to modify the default inputs of the set,
                                       such as dft_rung, basis_set, pcm_dielectric, scf_algorithm,
                                       or max_scf_cycles. See pymatgen/io/qchem/sets.py for default
                                       values of all input parameters. For instance, if a user wanted
                                       to use a more advanced DFT functional, include a pcm with a
                                       dielectric of 30, and use a larger basis, the user would set
                                       qchem_input_params = {"dft_rung": 5, "pcm_dielectric": 30,
                                       "basis_set": "6-311++g**"}. However, more advanced customization
                                       of the input is also possible through the overwrite_inputs key
                                       which allows the user to directly modify the rem, pcm, smd, and
                                       solvent dictionaries that QChemDictSet passes to inputs.py to
                                       print an actual input file. For instance, if a user wanted to
                                       set the sym_ignore flag in the rem section of the input file
                                       to true, then they would set qchem_input_params = {"overwrite_inputs":
                                       "rem": {"sym_ignore": "true"}}. Of course, overwrite_inputs
                                       could be used in conjuction with more typical modifications,
                                       as seen in the test_double_FF_opt workflow test.
            db_file (str): Path to file specifying db credentials to place output parsing.
            check_db (bool): Whether or not to check the database for equivalent structures
                             before adding new fragment fireworks. Defaults to True.
            parents ([Firework]): Parents of this particular Firework.
            **kwargs: Other kwargs that are passed to Firework.__init__.
        """

        qchem_input_params = qchem_input_params or {}
        additional_charges = additional_charges or []
        t = []
        t.append(
            FragmentMolecule(
                molecule=molecule,
                depth=depth,
                open_rings=open_rings,
                additional_charges=additional_charges,
                do_triplets=do_triplets,
                linked=linked,
                qchem_input_params=qchem_input_params,
                db_file=db_file,
                check_db=check_db))
        super(FragmentFW, self).__init__(
            t,
            parents=parents,
            name=name,
            **kwargs)
