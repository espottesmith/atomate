# coding: utf-8

from __future__ import division, print_function, unicode_literals, absolute_import

import os
import unittest

from fireworks import FWorker
from fireworks.core.rocket_launcher import rapidfire
from atomate.utils.testing import AtomateTest
from pymatgen.core import Molecule
from pymatgen.io.qchem.inputs import QCInput
from atomate.qchem.powerups import use_fake_qchem
from atomate.qchem.workflows.base.ts_search import get_wf_ts_search

__author__ = "Evan Spotte-Smith"
__copyright__ = "Copyright 2019, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Evan Spotte-Smith"
__email__ = "espottesmith@gmail.com"
__status__ = "Alpha"
__date__ = "September 2019"
__credits__ = "Sam Blau"


module_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
db_dir = os.path.join(module_dir, "..", "..", "..", "common", "test_files")


class TestTSSearch(AtomateTest):
    def test_ts_search_fsm(self):
        # location of test files
        ts_search_files = os.path.join(module_dir, "..", "..",
                                       "test_files", "fsm_ts_freq")
        # define starting molecule and workflow object
        initial_qcin = QCInput.from_file(
            os.path.join(ts_search_files, "block", "launcher_first",
                         "mol.qin"))
        initial_mol = initial_qcin.molecule

        params = {"dft_rung": 3,
                  "basis_set": "6-31g*",
                  "overwrite_inputs": {"rem": {"xc_grid": 3}}}

        real_wf = get_wf_ts_search(
            reactants=initial_mol["reactants"],
            products=initial_mol["products"],
            qchem_input_params=params,
            linked=True,
            multimode="openmp",
            method="fsm")
        # use powerup to replace run with fake run
        ref_dirs = {
            "ts_search_fsm":
            os.path.join(ts_search_files, "block", "launcher_first"),
            "ts_search_ff_ts":
            os.path.join(ts_search_files, "block", "launcher_second")
        }
        fake_wf = use_fake_qchem(real_wf, ref_dirs)
        self.lp.add_wf(fake_wf)
        rapidfire(
            self.lp,
            fworker=FWorker(env={"max_cores": 32, "db_file": os.path.join(db_dir, "db.json"),
                                 "qchem_cmd": ""}))

        wf_test = self.lp.get_wf_by_fw_id(1)
        self.assertTrue(
            all([s == "COMPLETED" for s in wf_test.fw_states.values()]))

        fsm = self.get_task_collection().find_one({
            "task_label":
            "ts_search_fsm"
        })

        fsm_ts_guess_mol = Molecule.from_dict(fsm["output"]["ts_guess"])

        ff_ts = self.get_task_collection().find_one({
            "task_label":
            "ts_search_ff_ts"
        })

        ff_ts_initial_mol = Molecule.from_dict(
            ff_ts["input"]["initial_molecule"])

        self.assertEqual(fsm_ts_guess_mol, ff_ts_initial_mol)


if __name__ == "__main__":
    unittest.main()
