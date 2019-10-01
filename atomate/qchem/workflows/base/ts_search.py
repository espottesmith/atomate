# coding: utf-8

from __future__ import absolute_import, division, print_function, \
    unicode_literals

# This module defines a workflow for identifying a guess transition-state geometry using the
# freezing string method and then optimizing the transition state using a frequency-flattening
# method

from pymatgen.analysis.reaction_calculator import Reaction

from fireworks import Workflow
from atomate.qchem.fireworks.core import (FrequencyFlatteningTransitionStateFW,
                                          FreezingStringFW,
                                          GrowingStringFW)
from atomate.utils.utils import get_logger

__author__ = "Evan Spotte-Smith"
__copyright__ = "Copyright 2019, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Evan Spotte-Smith"
__email__ = "espottesmith@gmail.com"
__status__ = "Alpha"
__date__ = "September 2019"
__credits__ = "Sam Blau"

logger = get_logger(__name__)


def get_wf_ts_search(reactants,
                     products,
                     qchem_cmd=">>qchem_cmd<<",
                     max_cores=">>max_cores<<",
                     multimode=">>multimode<<",
                     qchem_input_params=None,
                     linked=False,
                     method="fsm",
                     name="ts_search",
                     db_file=">>db_file<<",
                     **kwargs):
    """

    Firework 1 : map atoms in reactants to corresponding atoms in products,
                 write QChem input for an freezing-string method (FSM) calculation,
                 run fsm QCJob,
                 parse directory and insert into db,
                 pass ts_guess to fw_spec and on to fw2,

    Firework 2 : write QChem input for a frequency-flattening ts calculation starting with the
                 ts guess from fw1
                 run FF_opt QCJob,
                 parse directory and insert into db

    Args:
        reactants (list): list of pymatgen Molecules representing the (optimized) reactant
            molecules.
        products (list): list of pymatgen Molecules representing the (optimized) product molecules.
        qchem_cmd (str): Command to run QChem.
        max_cores (int): Maximum number of cores to parallelize over.
            Defaults to 32.
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
        linked (bool): if True (default False), pass scratch files from calculation to calculation
            and have outputs from one calculation inform the next.
        method (string): Determines what string method to use for the initial transition state guess.
            Default is "fsm", which will use the Freezing-String Method. "gsm" (for Growing-String
            Method) is also a valid option.
        name (string): name for the Workflow
        db_file (str): path to file containing the database credentials.
        kwargs (keyword arguments): additional kwargs to be passed to Workflow

    Returns:
        Workflow
    """

    # Guess the transition state structure
    if method.lower() == "fsm":
        fw1 = FreezingStringFW(
            reactants=reactants,
            products=products,
            name="ts_search_fsm",
            qchem_cmd=qchem_cmd,
            max_cores=max_cores,
            multimode=multimode,
            qchem_input_params=qchem_input_params,
            db_file=db_file)
    elif method.lower() == "gsm":
        fw1 = GrowingStringFW(
            reactants=reactants,
            products=products,
            name="ts_search_gsm",
            qchem_cmd=qchem_cmd,
            max_cores=max_cores,
            multimode=multimode,
            qchem_input_params=qchem_input_params,
            db_file=db_file)

    fw2 = FrequencyFlatteningTransitionStateFW(
        name="ff_ts_optimization",
        qchem_cmd=qchem_cmd,
        max_cores=max_cores,
        multimode=multimode,
        qchem_input_params=qchem_input_params,
        linked=linked,
        db_file=db_file,
        parents=fw1)
    fws = [fw1, fw2]

    reaction = Reaction([r.composition for r in reactants], [p.composition for p in products])

    wfname = "{}:{}".format(str(reaction), name)

    return Workflow(fws, name=wfname, **kwargs)
