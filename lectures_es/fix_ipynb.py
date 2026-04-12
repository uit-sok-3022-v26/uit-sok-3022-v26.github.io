import sys
import os

path = os.path.dirname(os.path.abspath(__file__))

def fix_file(fname):
    # Read file

    base = os.path.splitext(path+'/'+fname)[0]
    fname = base + ".ipynb"
    with open(fname, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace attachment:
    content = content.replace("attachment:", "")

    # Remove repeated \r\n before <span
    content = content.replace('\\n",\n        "<span', "<span")
    content = content.replace('>\n$$', ">\\n\n$$")
    # Write result
    with open(fname, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        fix_file('derivatives')
        print("Usage: python fix_quarto_ipynb.py <filename>")
        sys.exit(1)

    fix_file(sys.argv[1])