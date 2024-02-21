import argparse
from pathlib import Path

# import pyfdhi.disp.MossRoss2011 as mr11
# import pyfdhi.disp.KuehnEtAl2023 as kea23
# import pyfdhi.disp.YoungsEtAl2003 as yea03
# import pyfdhi.disp.PetersenEtAl2011 as pea11
# import pyfdhi.disp.WellsCoppersmith1994 as wc94

from pyfdhi.disp.moss_ross_2011 import MossRoss2011 as MR11
from pyfdhi.disp.kuehn_et_al_2023 import KuehnEtAl2023 as Kea23

# models = {"mr11": MR11, "kea23": kea23, "yea03": yea03, "pea11": pea11, "wc94": wc94}
models = {"mr11": MR11, "kea23": Kea23}


# @click.group()
# def cli():
#     pass
#
#
# @cli.command()
# @click.argument("model", type=click.Choice([k for k, m in models.items() if hasattr(m, "run_ad")]))
# @click.option("-m", "--mag", type=click.FLOAT, required=True, multiple=True)
# @click.option(
#     "-s",
#     "--style",
#     type=click.Choice(("strike-slip", "reverse", "normal")),
#     required=True,
#     multiple=True,
# )
# def disp_avg(model, mag, style):
def disp_avg(args):
    kwds = vars(args)
    print(kwds)
    model = kwds.pop("model")

    results = models[model].calc_disp_avg(**kwds)
    print(results)

    # Prompt to save results to CSV
    save_option = input("Do you want to save the results to a CSV (yes/no)? ").strip().lower()

    if save_option in ["y", "yes"]:
        file_path = input("Enter filepath to save results: ").strip()
        if file_path:
            # Create the directory if it doesn't exist
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            results.to_csv(file_path, index=False)
            print(f"Results saved to {file_path}")
        else:
            print("Invalid file path. Results not saved.")
    else:
        print("Results not saved.")


# @cli.command()
# def disp_profile():
#     pass
#
#
# @cli.command()
# def disp_model():
#     pass
#
#
# @cli.command()
# def disp_prob_exc():
#     pass


parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(required=True)
parser_da = subparsers.add_parser("disp_avg", help="Average displacement")
parser_da.set_defaults(func=disp_avg)

parser_da.add_argument(
    "model",
    type=str,
    choices={k for k, m in models.items() if hasattr(m, "_calc_disp_avg")},
    help="Model.",
)


parser_da.add_argument(
    "-m",
    "--magnitude",
    required=True,
    nargs="+",
    type=float,
    help="Earthquake moment magnitude.",
)
parser_da.add_argument(
    "-l",
    "--location",
    required=True,
    nargs="+",
    type=float,
    help="Normalized location along rupture length, range [0, 1.0].",
)
parser_da.add_argument(
    "-p",
    "--percentile",
    required=True,
    nargs="+",
    type=float,
    help=" Aleatory quantile value. Use -1 for mean.",
)
parser_da.add_argument(
    "-shape",
    "--submodel",
    default="elliptical",
    nargs="+",
    type=str.lower,
    choices=("elliptical", "quadratic", "bilinear"),
    help="PEA11 shape model name (case-insensitive). Default is 'elliptical'.",
)
parser_da.add_argument(
    "-s",
    "--style",
    default="strike-slip",
    nargs="+",
    type=str.lower,
    help="Style of faulting (case-insensitive). Default is 'strike-slip'; other styles not recommended.",
)

# FIXME: bilinear model debugger issue
parser_da.add_argument(
    "--debug",
    dest="debug_bilinear_model",
    action="store_true",
    help="Return bilinear results that are erroneous for debugging purposes.",
    default=False,
)


def cli():
    args = parser.parse_args()
    args.func(args)
