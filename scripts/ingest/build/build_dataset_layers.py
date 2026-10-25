"""Build or validate raw -> normalized -> runtime dataset layers."""

from __future__ import annotations

import argparse

from core.dataset_assets import build_asset_layers, validate_asset_layers
from core.dataset_layers import build_layers, build_selected_layers, validate_layers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true",
        help="validate the generated catalog and files without rebuilding",
    )
    parser.add_argument(
        "--skip-hashes", action="store_true",
        help="skip full-file hashes during --check",
    )
    parser.add_argument(
        "--require-raw", action="store_true",
        help="also require and verify every source-native raw archive",
    )
    parser.add_argument(
        "--table", action="append", default=[],
        help="rebuild only this governed table against the existing catalog; repeatable",
    )
    args = parser.parse_args()
    if args.check and args.table:
        parser.error("--check cannot be combined with --table")
    if not args.check:
        if args.table:
            build_selected_layers(args.table)
            print(f"rebuilt {len(set(args.table))} governed tables")
        else:
            catalog = build_layers()
            catalog = build_asset_layers()
            print(
                f"built {catalog['table_count']} governed tables and indexed "
                f"{catalog['asset_count']} assets across {catalog['source_count']} sources"
            )
    errors = validate_layers(verify_hashes=not args.skip_hashes)
    errors.extend(validate_asset_layers(require_raw=args.require_raw))
    if errors:
        raise SystemExit("\n".join(errors))
    print("dataset layers valid")


if __name__ == "__main__":
    main()
