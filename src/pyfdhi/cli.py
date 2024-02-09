from pathlib import Path

import click

import pyfdhi.disp.MossRoss2011 as mr11
import pyfdhi.disp.KuehnEtAl2023 as kea23
import pyfdhi.disp.YoungsEtAl2003 as yea03
import pyfdhi.disp.PetersenEtAl2011 as pea11
import pyfdhi.disp.WellsCoppersmith1994 as wc94

models = {"mr11": mr11, "kea23": kea23, "yea03": yea03, "pea11": pea11, "wc94": wc94}


@click.group()
def cli():
    pass


@cli.command()
@click.argument("model", type=click.Choice([k for k, m in models.items() if hasattr(m, "run_ad")]))
@click.option("-m", "--mag", type=click.FLOAT, required=True, multiple=True)
@click.option(
    "-s",
    "--style",
    type=click.Choice(("strike-slip", "reverse", "normal")),
    required=True,
    multiple=True,
)
def disp_avg(model, mag, style):
    results = models[model].run_ad(magnitude=mag, style=style)
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


@cli.command()
def disp_profile():
    pass


@cli.command()
def disp_model():
    pass


@cli.command()
def disp_prob_exc():
    pass
