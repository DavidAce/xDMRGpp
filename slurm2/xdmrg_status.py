from __future__ import annotations

import traceback
from pathlib import Path

import h5py


# Inspect one xDMRG HDF5 file and return a structured `(state, detail)` pair.
def get_h5_status(filename: Path) -> tuple[str, str | None]:
    if not filename.is_file():
        return "MISSING", None

    expected_dset_paths = [
        "/common/finished_all",
        "/xDMRG/model/hamiltonian",
        "/xDMRG/state_emid/measurements",
        "/xDMRG/state_emid/status",
        "/xDMRG/state_emid/subsystem_entanglement_entropies",
    ]
    expected_link_attrs = {
        "initial_pattern": "/xDMRG/state_emid",
        "initial_state": "/xDMRG/state_emid",
    }

    try:
        with h5py.File(filename, "r") as h5file:
            if h5file["/common/finished_all"][()] == 0:
                return "FAILED", "simulation has not finished"

            # First verify that the mandatory datasets and attrs exist.
            expected_dsets = [h5file.get(path) for path in expected_dset_paths]
            expected_attrs = [h5file.get(link).attrs.get(attr) for attr, link in expected_link_attrs.items() if link in h5file]
            missing_dsets = [path for dset, path in zip(expected_dsets, expected_dset_paths) if dset is None]
            missing_attrs = [path for link, path in zip(expected_attrs, expected_link_attrs) if link is None]
            if missing_dsets:
                return "FAILED", f"missing datasets:{missing_dsets}"
            if missing_attrs:
                return "FAILED", f"missing attributes:{missing_attrs}"

            enum_event = h5py.check_enum_dtype(h5file["xDMRG/state_emid/status"].dtype["event"])
            enum_algo_stop = h5py.check_enum_dtype(h5file["xDMRG/state_emid/status"].dtype["algo_stop"])
            if algorithm_stop := h5file["/xDMRG/state_emid"].attrs.get("algorithm_stop"):
                # Newer files store an explicit algorithm_stop attribute.
                if algorithm_stop != "SUCCESS":
                    for evar, (int_event, prec_limit, _) in zip(
                            h5file["xDMRG/state_emid/measurements"]["energy_variance"][::-1],
                            h5file["xDMRG/state_emid/status"]["event", "energy_variance_prec_limit", "algo_stop"][::-1]):
                        if int_event != enum_event["FINISHED"]:
                            continue
                        if evar < 100 * prec_limit:
                            return "FINISHED", None
                        return "FAILED", f"algorithm_stop:{algorithm_stop}:variance:{evar:.5e}[limit:{prec_limit:.5e}]"
            else:
                # Older files only expose the stop reason through the status table.
                for evar, (int_event, prec_limit, int_algo_stop) in zip(
                        h5file["xDMRG/state_emid/measurements"]["energy_variance"][::-1],
                        h5file["xDMRG/state_emid/status"]["event", "energy_variance_prec_limit", "algo_stop"][::-1]):
                    if int_event != enum_event["FINISHED"]:
                        continue
                    if evar <= prec_limit:
                        return "FINISHED", None
                    if int_algo_stop == enum_algo_stop["SUCCESS"] and evar > 10 * prec_limit:
                        return "FAILED", f"variance is too high:{evar:.5e}[limit:{prec_limit:.5e}]"
                    if int_algo_stop != enum_algo_stop["SUCCESS"]:
                        algo_stop = next(name for name, value in enum_algo_stop.items() if value == int_algo_stop)
                        return "FAILED", f"algorithm_stop:{algo_stop}"

            return "FINISHED", None
    except Exception as exc:  # pragma: no cover - defensive path
        return "FAILED", f"{exc} | {traceback.format_exc(limit=1).strip()}"
