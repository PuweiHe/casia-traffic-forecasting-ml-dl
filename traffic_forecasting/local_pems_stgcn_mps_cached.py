"""Opt-in cached-data runner: requires a separate protocol and output directory."""
from traffic_forecasting import local_pems_stgcn_mps as runner
from traffic_forecasting import multihorizon_study as study
from traffic_forecasting.cached_windows import CachedMaskedWindows


def main():
    # Run only in a new root: freeze() rejects this adapter in an existing study.
    runner.MaskedWindows = CachedMaskedWindows
    study.CODE_FILES.extend(('traffic_forecasting/cached_windows.py',
                             'traffic_forecasting/local_pems_stgcn_mps_cached.py'))
    runner.main()


if __name__ == '__main__':
    main()
