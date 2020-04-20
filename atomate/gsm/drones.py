# coding: utf-8

import os
import datetime
from fnmatch import fnmatch
from collections import OrderedDict
import json
import glob
import traceback
from itertools import chain
import copy

from monty.io import zopen
from monty.json import jsanitize

from pymatgen.apps.borg.hive import AbstractDrone
from pymatgen.io.babel import BabelMolAdaptor
from pymatgen.io.qchem import QCInput
from pymatgen.io.qchem import QCOutput
from pymatgen.io.gsm.inputs import (QCTemplate, GSMIsomerInput,
                                    parse_multi_xyz)
from pymatgen.io.gsm.outputs import (GSMOutput,
                                     GSMOptimizedStringParser,
                                     GSMInternalCoordinateDataParser)
from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.analysis.fragmenter import metal_edge_extender
from pymatgen.symmetry.analyzer import PointGroupAnalyzer

from atomate.utils.utils import get_logger
from atomate import __version__ as atomate_version

__author__ = "Evan Spotte-Smith"
__copyright__ = "Copyright 2020, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Evan Spotte-Smith"
__email__ = "espottesmith@gmail.com"
__status__ = "Alpha"
__date__ = "02/18/20"
__credits__ = "Sam Blau, Brandon Wood, Shyam Dwaraknath, Xiaohui Qu, Kiran Mathew, Shyue Ping Ong, Anubhav Jain"

logger = get_logger(__name__)


class GSMDrone(AbstractDrone):
    """
    A drone to parse calculations from pyGSM and insert an organized, searchable entry into the database.
    """

    # note: the version is inserted into the task doc
    __version__ = atomate_version

    # Schema def of important keys and sub-keys; used in validation
    schema = {
        "root": {
            "dir_name", "input", "output", "smiles", "walltime", "cputime",
            "formula_pretty", "formula_anonymous", "chemsys", "pointgroup",
            "formula_alphabetical"
        },
        "input": {"initial_reactants", "initial_products", "mode", "num_nodes",
                  "ends_fixed"},
        "output": {"string_nodes", "ts_guess", "ts_energy", "absolute_ts_energy"}
    }

    def __init__(self, additional_fields=None):
        """
        Initialize a GSM drone to parse pyGSM calculations
        Args:
            additional_fields (dict): dictionary of additional fields to add to output document
        """

        self.additional_fields = additional_fields or dict()

    def assimilate(self, path, molecule_file="input.xyz", template_file="qin",
                   output_file="gsm.out", isomers_file=None):
        """
        Parses qchem input and output files and insert the result into the db.

        Args:
            path (str): Path to the directory containing output file.
            molecule_file (str): Name of the input molecule geometry file.
                Default is "input.xyz".
            template_file (str): Name of the input QChem template file.
                Default is "qin".
            output_file (str): Name of the pyGSM output file. Default is
                "gsm.out".
            isomers_file (str): For single-ended calculations, this is the
                name of the isomers file that defines what coordinates to
                vary. Default is None; however, note that this should be
                provided for single-ended calculations.

        Returns:
            d (dict): a task dictionary
        """
        logger.info("Getting task doc for base dir :{}".format(path))

        all_files = os.listdir(path)
        if "scratch" in all_files:
            all_scratch_files = os.listdir(os.path.join(path, "scratch"))

        # Populate important input and output files
        mol_file = None
        temp_file = None
        out_file = None
        iso_file = None
        ic_file = None
        opt_file = None
        for file in all_files:
            if isomers_file is not None:
                if file == isomers_file and iso_file is None:
                    iso_file = os.path.join(path, file)
                    continue

            if file == molecule_file and mol_file is None:
                mol_file = os.path.join(path, file)
            elif file == template_file and temp_file is None:
                temp_file = os.path.join(path, file)
            elif file == output_file and out_file is None:
                out_file = os.path.join(path, file)
            elif "IC_data" in file and ic_file is None:
                ic_file = os.path.join(path, file)
            elif "opt_converged" in file and opt_file is None:
                opt_file = os.path.join(path, file)

        # Only check scratch directory if we're missing files
        any_needed_none = any([e is None for e in [mol_file, temp_file,
                                                   out_file, ic_file,
                                                   opt_file]])
        iso_needed_none = isomers_file is not None and iso_file is None

        if any_needed_none or iso_needed_none:
            for file in all_scratch_files:
                if isomers_file is not None:
                    if file == isomers_file and iso_file is None:
                        iso_file = os.path.join(path, "scratch", file)
                        continue

                if file == molecule_file and mol_file is None:
                    mol_file = os.path.join(path, "scratch", file)
                elif file == template_file and temp_file is None:
                    temp_file = os.path.join(path, "scratch", file)
                elif file == output_file and out_file is None:
                    out_file = os.path.join(path, "scratch", file)
                elif "IC_data" in file and ic_file is None:
                    ic_file = os.path.join(path, "scratch", file)
                elif "opt_converged" in file and opt_file is None:
                    opt_file = os.path.join(path, "scratch", file)

        any_needed_none = any([e is None for e in [mol_file, temp_file,
                                                   out_file, ic_file,
                                                   opt_file]])
        iso_needed_none = isomers_file is not None and iso_file is None

        if not(any_needed_none or iso_needed_none):
            d = self.generate_doc(path=path, molecule_file=mol_file,
                                  template_file=temp_file, output_file=out_file,
                                  isomers_file=iso_file, internal_coordinate_file=ic_file,
                                  optimized_geom_file=opt_file)
            self.post_process(d)
        else:
            raise ValueError("Either input or output not found!")
        self.validate_doc(d)
        return jsanitize(d, strict=True, allow_bson=True)

    def generate_doc(self, path, molecule_file, template_file, output_file,
                     isomers_file, internal_coordinate_file,
                     optimized_geom_file):

        try:
            fullpath = os.path.abspath(path)
            d = dict()

            d["schema"] = {
                "code": "atomate",
                "version": GSMDrone.__version__
            }

            d["dir_name"] = fullpath

            # TODO: Consider error handlers
            # Include an "orig" section to the doc

            # Parse all relevant files
            initial_mol = parse_multi_xyz(molecule_file)
            temp_file = QCTemplate.from_file(template_file)
            iso_file = GSMIsomerInput.from_file(isomers_file)
            out_file = GSMOutput(output_file)
            ic_file = GSMInternalCoordinateDataParser(internal_coordinate_file)
            opt_file = GSMOptimizedStringParser()

            #TODO: YOU ARE HERE

            d["structure_change"] = []
            d["warnings"] = {}
            for entry in d["calcs_reversed"]:
                if "structure_change" in entry and "structure_change" not in d["warnings"]:
                    if entry["structure_change"] != "no_change":
                        d["warnings"]["structure_change"] = True
                if "structure_change" in entry:
                    d["structure_change"].append(entry["structure_change"])
                for key in entry["warnings"]:
                    if key not in d["warnings"]:
                        d["warnings"][key] = True

            d_calc_init = d["calcs_reversed"][-1]
            d_calc_final = d["calcs_reversed"][0]

            d["input"] = {
                "initial_molecule": d_calc_init["initial_molecule"],
                "job_type": d_calc_init["input"]["rem"]["job_type"]
            }
            d["output"] = {
                "initial_molecule": d_calc_final["initial_molecule"],
                "job_type": d_calc_final["input"]["rem"]["job_type"],
                "mulliken": d_calc_final["Mulliken"][-1]
            }
            if "RESP" in d_calc_final:
                d["output"]["resp"] = d_calc_final["RESP"][-1]
            elif "ESP" in d_calc_final:
                d["output"]["esp"] = d_calc_final["ESP"][-1]

            if d["output"]["job_type"] in ["opt", "optimization", "ts"]:
                if "molecule_from_optimized_geometry" in d_calc_final:
                    d["output"]["optimized_molecule"] = d_calc_final[
                        "molecule_from_optimized_geometry"]
                    d["output"]["final_energy"] = d_calc_final["final_energy"]
                else:
                    d["output"]["final_energy"] = "unstable"
                if d_calc_final["opt_constraint"]:
                    d["output"]["constraint"] = [
                        d_calc_final["opt_constraint"][0],
                        float(d_calc_final["opt_constraint"][6])
                    ]
            elif d["output"]["job_type"] in ["freq", "frequency"]:
                d["output"]["frequencies"] = d_calc_final["frequencies"]
                d["output"]["enthalpy"] = d_calc_final["total_enthalpy"]
                d["output"]["entropy"] = d_calc_final["total_entropy"]
                if d["input"]["job_type"] in ["opt", "optimization", "ts"]:
                    d["output"]["optimized_molecule"] = d_calc_final[
                        "initial_molecule"]
                    d["output"]["final_energy"] = d["calcs_reversed"][1][
                        "final_energy"]
            elif d["output"]["job_type"] in ["fsm", "gsm"]:
                d["input"]["initial_reactant_molecule"] = d_calc_final["string_initial_reactant_molecules"]
                d["input"]["initial_product_molecule"] = d_calc_final["string_initial_product_molecules"]
                d["input"]["initial_reactant_geometry"] = d_calc_final["string_initial_reactant_geometry"]
                d["input"]["initial_product_geometry"] = d_calc_final["string_initial_product_geometry"]
                d["output"]["num_images"] = d_calc_final["string_num_images"]
                d["output"]["string_relative_energies"] = d_calc_final["string_relative_energies"]
                d["output"]["string_geometries"] = d_calc_final["string_geometries"]
                d["output"]["string_molecules"] = d_calc_final["string_molecules"]
                d["output"]["string_gradient_magnitudes"] = d_calc_final["string_gradient_magnitudes"]
                d["output"]["ts_guess"] = d_calc_final["string_ts_guess"]

                if d["output"]["job_type"] == "fsm":
                    d["output"]["string_energies"] = d_calc_final["string_energies"]
                    d["output"]["string_absolute_distances"] = d_calc_final["string_absolute_distances"]
                    d["output"]["string_proportional_distances"] = d_calc_final["string_proportional_distances"]
                    d["output"]["max_energy"] = d_calc_final["string_max_energy"]
                else:
                    d["output"]["string_relative_energies_iterations"] = d_calc_final["string_relative_energies_iterations"]
                    d["output"]["string_gradient_magnitudes_iterations"] = d_calc_final["string_gradient_magnitudes_iterations"]
                    d["output"]["string_total_gradient_magnitude"] = d_calc_final["string_total_gradient_magnitude"]
                    d["output"]["string_total_gradient_magnitude_iterations"] = d_calc_final["string_total_gradient_magnitude_iterations"]
                    d["output"]["string_max_relative_energy"] = d_calc_final["string_max_relative_energy"]

            if "final_energy" not in d["output"]:
                if d_calc_final["final_energy"] != None:
                    d["output"]["final_energy"] = d_calc_final["final_energy"]
                else:
                    d["output"]["final_energy"] = d_calc_final["SCF"][-1][-1][0]

            if d_calc_final["completion"]:
                total_cputime = 0.0
                total_walltime = 0.0
                for calc in d["calcs_reversed"]:
                    if "walltime" in calc and "cputime" in calc:
                        if calc["walltime"] is not None:
                            total_walltime += calc["walltime"]
                        if calc["cputime"] is not None:
                            total_cputime += calc["cputime"]
                d["walltime"] = total_walltime
                d["cputime"] = total_cputime
            else:
                d["walltime"] = None
                d["cputime"] = None

            comp = d["output"]["initial_molecule"].composition
            d["formula_pretty"] = comp.reduced_formula
            d["formula_anonymous"] = comp.anonymized_formula
            d["formula_alphabetical"] = comp.alphabetical_formula
            d["chemsys"] = "-".join(sorted(set(d_calc_final["species"])))
            if d_calc_final["point_group"] != None:
                d["pointgroup"] = d_calc_final["point_group"]
            else:
                try:
                    d["pointgroup"] = PointGroupAnalyzer(d["output"]["initial_molecule"]).sch_symbol
                except ValueError:
                    d["pointgroup"] = "PGA_error"

            bb = BabelMolAdaptor(d["output"]["initial_molecule"])
            pbmol = bb.pybel_mol
            smiles = pbmol.write(str("smi")).split()[0]
            d["smiles"] = smiles

            d["state"] = "successful" if d_calc_final["completion"] else "unsuccessful"

            if "special_run_type" in d:
                if d["special_run_type"] == "frequency_flattener":
                    if d["state"] == "successful":
                        orig_num_neg_freq = sum(1 for freq in d["calcs_reversed"][-2]["frequencies"] if freq < 0)
                        orig_energy = d_calc_init["final_energy"]
                        final_num_neg_freq = sum(1 for freq in d_calc_final["frequencies"] if freq < 0)
                        final_energy = d["calcs_reversed"][1]["final_energy"]
                        d["num_frequencies_flattened"] = orig_num_neg_freq - final_num_neg_freq
                        if final_num_neg_freq > 0: # If a negative frequency remains,
                            # and it's too large to ignore,
                            if final_num_neg_freq > 1 or abs(d["output"]["frequencies"][0]) >= 15.0:
                                d["state"] = "unsuccessful" # then the flattening was unsuccessful
                        if final_energy > orig_energy:
                            d["warnings"]["energy_increased"] = True

                elif d["special_run_type"] == "berny_optimization":
                    logfiles = [f for f in os.listdir(dir_name)
                                if f.startswith("berny.log")]
                    berny_traj = list()
                    for log in logfiles:
                        parsed = BernyLogParser(os.path.join(dir_name, log)).data
                        doc = dict()

                        doc["internals"] = parsed["internals"]
                        doc["initial_energy"] = parsed["energy_trajectory"][0]
                        doc["final_energy"] = parsed["final_energy"]
                        doc["trust"] = parsed["trust"]
                        doc["step_walltimes"] = parsed["opt_step_times"]
                        doc["walltime"] = parsed["opt_walltime"]
                        if d["walltime"] is not None:
                            d["walltime"] += doc["walltime"]

                        berny_traj.append(doc)
                    d["berny_trajectory"] = berny_traj

                if d["special_run_type"] in ["frequency_flattener", "berny_optimization"]:
                    opt_traj = list()
                    for entry in d["calcs_reversed"]:
                        if entry["input"]["rem"]["job_type"] in ["opt", "optimization", "ts"]:
                            doc = {"initial": {}, "final": {}}
                            doc["initial"]["molecule"] = entry["initial_molecule"]
                            doc["final"]["molecule"] = entry["molecule_from_last_geometry"]
                            doc["initial"]["total_energy"] = entry["energy_trajectory"][0]
                            doc["final"]["total_energy"] = entry["energy_trajectory"][-1]
                            doc["initial"]["scf_energy"] = entry["SCF"][0][-1][0]
                            doc["final"]["scf_energy"] = entry["SCF"][-1][-1][0]
                            doc["structure_change"] = entry["structure_change"]
                            opt_traj.append(doc)
                    opt_traj.reverse()
                    opt_trajectory = {"trajectory": opt_traj, "structure_change": [[ii, entry["structure_change"]] for ii,entry in enumerate(opt_traj)], "energy_increase": []}
                    for ii, entry in enumerate(opt_traj):
                        if entry["final"]["total_energy"] > entry["initial"]["total_energy"]:
                            opt_trajectory["energy_increase"].append([ii, entry["final"]["total_energy"]-entry["initial"]["total_energy"]])
                        if ii != 0:
                            if entry["final"]["total_energy"] > opt_traj[ii-1]["final"]["total_energy"]:
                                opt_trajectory["energy_increase"].append([ii-1, ii, entry["final"]["total_energy"]-opt_traj[ii-1]["final"]["total_energy"]])
                            struct_change = check_for_structure_changes(opt_traj[ii-1]["final"]["molecule"], entry["final"]["molecule"])
                            if struct_change != entry["structure_change"]:
                                opt_trajectory["structure_change"].append([ii-1, ii, struct_change])
                                d["warnings"]["between_iteration_structure_change"] = True
                    if "linked" in d:
                        if d["linked"] == True:
                            opt_trajectory["discontinuity"] = {"structure": [], "scf_energy": [], "total_energy": []}
                            for ii, entry in enumerate(opt_traj):
                                if ii != 0:
                                    if entry["initial"]["molecule"] != opt_traj[ii-1]["final"]["molecule"]:
                                        opt_trajectory["discontinuity"]["structure"].append([ii-1,ii])
                                        d["warnings"]["linked_structure_discontinuity"] = True
                                    if entry["initial"]["total_energy"] != opt_traj[ii-1]["final"]["total_energy"]:
                                        opt_trajectory["discontinuity"]["total_energy"].append([ii-1,ii])
                                    if entry["initial"]["scf_energy"] != opt_traj[ii-1]["final"]["scf_energy"]:
                                        opt_trajectory["discontinuity"]["scf_energy"].append([ii-1,ii])
                    d["opt_trajectory"] = opt_trajectory

            d["last_updated"] = datetime.datetime.utcnow()
            return d

        except Exception:
            logger.error(traceback.format_exc())
            logger.error("Error in " + os.path.abspath(dir_name) + ".\n" +
                         traceback.format_exc())
            raise

    @staticmethod
    def process_qchemrun(dir_name, taskname, input_file, output_file):
        """
        Process a QChem calculation, aka an input/output pair.
        """
        qchem_input_file = os.path.join(dir_name, input_file)
        qchem_output_file = os.path.join(dir_name, output_file)
        d = QCOutput(qchem_output_file).data
        temp_input = QCInput.from_file(qchem_input_file)
        d["input"] = {}
        d["input"]["molecule"] = temp_input.molecule
        d["input"]["rem"] = temp_input.rem
        d["input"]["opt"] = temp_input.opt
        d["input"]["pcm"] = temp_input.pcm
        d["input"]["solvent"] = temp_input.solvent
        d["input"]["smx"] = temp_input.smx
        d["task"] = {"type": taskname, "name": taskname}
        return d

    @staticmethod
    def post_process(dir_name, d):
        """
        Post-processing for various files other than the QChem input and output files.
        """
        logger.info("Post-processing dir:{}".format(dir_name))
        fullpath = os.path.abspath(dir_name)
        filenames = glob.glob(os.path.join(fullpath, "custodian.json*"))
        if len(filenames) >= 1:
            with zopen(filenames[0], "rt") as f:
                d["custodian"] = json.load(f)
        filenames = glob.glob(os.path.join(fullpath, "solvent_data*"))
        if len(filenames) >= 1:
            with zopen(filenames[0], "rt") as f:
                d["custom_smd"] = f.readlines()[0]
        filenames = glob.glob(os.path.join(fullpath, "processed_critic2.json*"))
        if len(filenames) >= 1:
            with zopen(filenames[0], "rt") as f:
                d["critic2"] = {}
                d["critic2"]["processed"] = json.load(f)
            filenames = glob.glob(os.path.join(fullpath, "CP.json*"))
            if len(filenames) >= 1:
                with zopen(filenames[0], "rt") as f:
                    d["critic2"]["CP"] = json.load(f)
            filenames = glob.glob(os.path.join(fullpath, "YT.json*"))
            if len(filenames) >= 1:
                with zopen(filenames[0], "rt") as f:
                    d["critic2"]["YT"] = json.load(f)
            filenames = glob.glob(os.path.join(fullpath, "bonding.json*"))
            if len(filenames) >= 1:
                with zopen(filenames[0], "rt") as f:
                    d["critic2"]["bonding"] = json.load(f)

    def validate_doc(self, d):
        """
        Sanity check, aka make sure all the important keys are set. Note that a failure
        to pass validation is unfortunately unlikely to be noticed by a user.
        """
        for k, v in self.schema.items():
            diff = v.difference(set(d.get(k, d).keys()))
            if diff:
                logger.warning("The keys {0} in {1} not set".format(diff, k))

    @staticmethod
    def get_valid_paths(self, path):
        return [path]
