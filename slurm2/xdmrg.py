from __future__ import annotations

import traceback
from pathlib import Path


# Split the legacy `STATE|detail` format into structured pieces.
def split_state_detail(raw_state: str) -> tuple[str, str | None]:
    state, _, detail = raw_state.partition("|")
    return state, detail or None


# Inspect one xDMRG HDF5 file and return the legacy string status form.
def get_h5_status(filename: Path, model_type: str) -> str:
    import h5py

    if not filename.is_file():
        return "MISSING"

    if model_type != "ising_majorana":
        raise AssertionError(f"Expected model_type==ising_majorana. Got: {model_type}")

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
                return "FAILED|simulation has not finished"
            # First verify that the mandatory datasets and attrs exist.
            expected_dsets = [h5file.get(path) for path in expected_dset_paths]
            expected_attrs = [h5file.get(link).attrs.get(attr) for attr, link in expected_link_attrs.items() if link in h5file]
            missing_dsets = [path for dset, path in zip(expected_dsets, expected_dset_paths) if dset is None]
            missing_attrs = [path for link, path in zip(expected_attrs, expected_link_attrs) if link is None]
            if missing_dsets:
                return f"FAILED|missing datasets:{missing_dsets}"
            if missing_attrs:
                return f"FAILED|missing attributes:{missing_attrs}"

            enum_event = h5py.check_enum_dtype(h5file["xDMRG/state_emid/status"].dtype["event"])
            enum_algo_stop = h5py.check_enum_dtype(h5file["xDMRG/state_emid/status"].dtype["algo_stop"])

            if algorithm_stop := h5file["/xDMRG/state_emid"].attrs.get("algorithm_stop"):
                # Newer files store an explicit algorithm_stop attribute.
                if algorithm_stop != "SUCCESS":
                    for evar, (int_event, prec_limit, _int_algo_stop) in zip(
                        h5file["xDMRG/state_emid/measurements"]["energy_variance"][::-1],
                        h5file["xDMRG/state_emid/status"]["event", "energy_variance_prec_limit", "algo_stop"][::-1],
                    ):
                        if int_event != enum_event["FINISHED"]:
                            continue
                        if evar < 100 * prec_limit:
                            return "FINISHED"
                        return f"FAILED|algorithm_stop:{algorithm_stop}:variance:{evar:.5e}[limit:{prec_limit:.5e}]"
            else:
                # Older files only expose the stop reason through the status table.
                for evar, (int_event, prec_limit, int_algo_stop) in zip(
                    h5file["xDMRG/state_emid/measurements"]["energy_variance"][::-1],
                    h5file["xDMRG/state_emid/status"]["event", "energy_variance_prec_limit", "algo_stop"][::-1],
                ):
                    if int_event != enum_event["FINISHED"]:
                        continue
                    if evar <= prec_limit:
                        return "FINISHED"
                    if int_algo_stop == enum_algo_stop["SUCCESS"] and evar > 10 * prec_limit:
                        return f"FAILED|variance is too high:{evar:.5e}[limit:{prec_limit:.5e}]"
                    if int_algo_stop != enum_algo_stop["SUCCESS"]:
                        str_algo_stop = list(enum_algo_stop.keys())[list(enum_algo_stop.values()).index(int_algo_stop)]
                        return f"FAILED|algorithm_stop:{str_algo_stop}"

            return "FINISHED"
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(traceback.format_exc())
        return f"FAILED|{exc}"
