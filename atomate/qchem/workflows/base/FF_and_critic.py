# coding: utf-8

from fireworks import Workflow
from atomate.qchem.fireworks.core import (OptimizeFW,
                                          FrequencyFlatteningOptimizeFW,
                                          FrequencyFlatteningTransitionStateFW,
                                          CubeAndCritic2FW)
from atomate.utils.utils import get_logger

__author__ = "Samuel Blau, Evan Spotte-Smith"

logger = get_logger(__name__)


def get_wf_FFopt_and_critic(molecule,
                            suffix,
                            qchem_input_params=None,
                            db_file=">>db_file<<",
                            **kwargs):
    """
    """

    # FFopt
    fw1 = FrequencyFlatteningOptimizeFW(
         molecule=molecule,
         name="{}:{}".format(molecule.composition.alphabetical_formula, "FFopt_" + suffix),
         qchem_cmd=">>qchem_cmd<<",
         max_cores=">>max_cores<<",
         qchem_input_params=qchem_input_params,
         linked=True,
         db_file=db_file
    )

    # Critic
    fw2 = CubeAndCritic2FW(
         name="{}:{}".format(molecule.composition.alphabetical_formula, "CC2_" + suffix),
         qchem_cmd=">>qchem_cmd<<",
         max_cores=">>max_cores<<",
         qchem_input_params=qchem_input_params,
         db_file=db_file,
         parents=fw1)
    fws = [fw1, fw2]

    wfname = "{}:{}".format(molecule.composition.alphabetical_formula, "FFopt_CC2_WF_" + suffix)

    return Workflow(fws, name=wfname, **kwargs)


def get_wf_preopt_FFopt_critic(molecule,
                               suffix,
                               qchem_input_params=None,
                               db_file=">>db_file<<",
                               **kwargs):
    """
    """

    # Pre-optimization
    fw1 = OptimizeFW(
        molecule=molecule,
        name="{}:{}".format(molecule.composition.alphabetical_formula, "preopt_" + suffix),
        qchem_cmd=">>qchem_cmd<<",
        max_cores=">>max_cores<<",
        qchem_input_params=qchem_input_params,
        db_file=db_file
    )

    # FFopt
    fw2 = FrequencyFlatteningOptimizeFW(
         name="{}:{}".format(molecule.composition.alphabetical_formula, "FFopt_" + suffix),
         qchem_cmd=">>qchem_cmd<<",
         max_cores=">>max_cores<<",
         qchem_input_params=qchem_input_params,
         linked=True,
         db_file=db_file,
         parents=fw1
    )

    # Critic
    fw3 = CubeAndCritic2FW(
         name="{}:{}".format(molecule.composition.alphabetical_formula, "CC2_" + suffix),
         qchem_cmd=">>qchem_cmd<<",
         max_cores=">>max_cores<<",
         qchem_input_params=qchem_input_params,
         db_file=db_file,
         parents=fw2
    )
    fws = [fw1, fw2, fw3]

    wfname = "{}:{}".format(molecule.composition.alphabetical_formula, "preopt_FFopt_CC2_WF_" + suffix)

    return Workflow(fws, name=wfname, **kwargs)


def get_wf_FFTSopt_and_critic(molecule,
                              suffix,
                              qchem_input_params=None,
                              db_file=">>db_file<<",
                              **kwargs):
    """
    """

    # FFopt
    fw1 = FrequencyFlatteningTransitionStateFW(
        molecule=molecule,
        name="{}:{}".format(molecule.composition.alphabetical_formula, "FFTSopt_" + suffix),
        qchem_cmd=">>qchem_cmd<<",
        max_cores=">>max_cores<<",
        qchem_input_params=qchem_input_params,
        linked=True,
        freq_before_opt=True,
        db_file=db_file
    )

    # Critic
    fw2 = CubeAndCritic2FW(
         name="{}:{}".format(molecule.composition.alphabetical_formula, "CC2_" + suffix),
         qchem_cmd=">>qchem_cmd<<",
         max_cores=">>max_cores<<",
         qchem_input_params=qchem_input_params,
         db_file=db_file,
         parents=fw1)
    fws = [fw1, fw2]

    wfname = "{}:{}".format(molecule.composition.alphabetical_formula, "FFTSopt_CC2_WF_" + suffix)

    return Workflow(fws, name=wfname, **kwargs)