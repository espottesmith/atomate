import networkx as nx
import networkx.algorithms.isomorphism as iso

from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.io.babel import BabelMolAdaptor
from pymatgen.analysis.local_env import OpenBabelNN

from fireworks import Workflow
from atomate.qchem.fireworks.core import FrequencyFlatteningOptimizeFW
from atomate.utils.utils import get_logger

__author__ = "Evan Spotte-Smith"
__copyright__ = "Copyright 2019, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Evan Spotte-Smith"
__email__ = "espottesmith@gmail.com"
__status__ = "Alpha"
__date__ = "October 2019"
__credits__ = "Sam Blau"

logger = get_logger(__name__)


def get_wf_substitution_opt_molecule(molecule, index,
                                     func_group, bond_order=1, pre_optimize=False,
                                     linked=False,
                                     qchem_input_params=None,
                                     name="functional_group_substitution",
                                     db_file=">>db_file<<",
                                     **kwargs):
        """
        Modify a molecule through a functional group substitution, and then
        create a workflow with the modified molecule.

        Args:
            molecule (Molecule): Molecule to be modified
            index (int): Index (in the reactant molecule) where the functional
                group is to be substituted.
            func_group (Molecule or str): Either a string representing a functional group (from
                pymatgen.structure.core.FunctionalGroups), or a Molecule with a
                dummy atom X.
            bond_order (int): Order of the bond between the functional group and
                the base molecule. Default 1, for single bond.
            pre_optimize (bool): If True (default False), use OpenBabel to
                perform an initial optimization and conformer search
            linked (bool): If True (default False), connect each calculation in the
                frequency flattening optimization calculation such that the
                scratch files from one calculation can be fed into the next
                calculation.
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
            name (str): Name of the Workflow
            db_file (str): path to file containing the database credentials.
            kwargs (keyword arguments): additional kwargs to be passed to Workflow

        Returns:
            Workflow
        """

        # Set up - strategy to extract bond orders
        # Node match for isomorphism check
        strat = OpenBabelNN()

        # Set up molecule graphs, including node attributes
        mg = MoleculeGraph.with_local_env_strategy(molecule,
                                                   strat,
                                                   reorder=False,
                                                   extend_structure=False)
        mg.set_node_attributes()

        mg.substitute_group(index, func_group, OpenBabelNN,
                            bond_order=bond_order,
                            extend_structure=False)

        if pre_optimize:
            obmol = BabelMolAdaptor(mg.molecule)
            obmol.rotor_conformer(25, 2500, False)
            obmol.localopt()
            mol = obmol.pymatgen_mol
        else:
            mol = mg.molecule

        fws = list()

        fws.append(FrequencyFlatteningOptimizeFW(molecule=mol,
                                                 name=name,
                                                 qchem_cmd=">>qchem_cmd<<",
                                                 max_cores=">>max_cores<<",
                                                 qchem_input_params=qchem_input_params,
                                                 linked=linked,
                                                 db_file=db_file))

        wfname = "{}:{}".format(molecule.composition.reduced_formula, name)

        return Workflow(fws, name=wfname, **kwargs)
