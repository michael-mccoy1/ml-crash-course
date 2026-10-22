"""Convert notebooks_src/*.py (percent format) -> notebooks/*.ipynb"""
import glob, os, re
import nbformat as nbf

here = os.path.dirname(os.path.abspath(__file__))
for src in sorted(glob.glob(os.path.join(here, "notebooks_src", "*.py"))):
    text = open(src, encoding="utf-8").read()
    chunks = re.split(r"^# %%(?: \[markdown\])?[ \t]*\n", text, flags=re.M)[1:]
    kinds = re.findall(r"^# %%( \[markdown\])?[ \t]*$", text, flags=re.M)
    nb = nbf.v4.new_notebook()
    for kind, chunk in zip(kinds, chunks):
        chunk = chunk.strip("\n")
        if kind:
            md = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in chunk.splitlines())
            nb.cells.append(nbf.v4.new_markdown_cell(md))
        else:
            nb.cells.append(nbf.v4.new_code_cell(chunk))
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    out = os.path.join(here, "notebooks", os.path.basename(src).replace(".py", ".ipynb"))
    nbf.write(nb, out); print("wrote", out, f"({len(nb.cells)} cells)")
