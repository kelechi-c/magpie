#!/usr/bin/env bash
# on the colab vm: re-freeze the lettering item (gate.WIDE[15]) with the kernels-off port, gate the current build on
# the sixteen images at both sizes, then read the poster's lettering back.
set -uo pipefail
cd /content/magpie && export HF_HOME=/hf PYTHONPATH=src:tools
mkdir -p /out/bottles && cp --parents /out/baseline/turbo/*/15.png /out/baseline/turbo/*/edit16_*/15.* /out/bottles/ 2>/dev/null
python tools/refreeze_item.py 15
python tools/gate.py --variant turbo --tag poster --cases edit16_1024,edit16_2048
python tools/lettering_item.py poster
echo finished > /out/poster.done
