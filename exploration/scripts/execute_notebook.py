"""Exécuter ce notebook séquentiel avec IPython, sans serveur ni sockets Jupyter."""
from pathlib import Path
import argparse
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT / '.matplotlib')
os.environ['IPYTHONDIR'] = str(ROOT / '.ipython')
os.chdir(ROOT)

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('notebook', nargs='?', default='notebooks/01_analyse_et_comparaison.ipynb')
    args = parser.parse_args()
    path = (ROOT / args.notebook).resolve()
    notebook = nbformat.read(path, as_version=4)
    shell = InteractiveShell.instance()
    import matplotlib
    from matplotlib_inline.backend_inline import configure_inline_support
    matplotlib.use('module://matplotlib_inline.backend_inline')
    configure_inline_support(shell, 'inline')
    count = 0
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        count += 1
        started = time.perf_counter()
        print(f'Cellule {count} : démarrage', flush=True)
        with capture_output() as captured:
            result = shell.run_cell(cell.source, store_history=True)
        outputs = []
        for name, value in [('stdout', captured.stdout), ('stderr', captured.stderr)]:
            if value:
                outputs.append(nbformat.v4.new_output('stream', name=name, text=value))
        for rich in captured.outputs:
            outputs.append(nbformat.v4.new_output('display_data', data=rich.data, metadata=rich.metadata))
        cell.execution_count = count
        cell.outputs = outputs
        nbformat.write(notebook, path)
        error = result.error_before_exec or result.error_in_exec
        if error:
            raise RuntimeError(f'Échec de la cellule {count}: {error}') from error
        print(f'Cellule {count} : OK ({time.perf_counter() - started:.1f} s)', flush=True)
    nbformat.validate(notebook)
    print(f'Notebook exécuté et validé : {count} cellules de code', flush=True)


if __name__ == '__main__':
    main()
