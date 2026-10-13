# modal_bench's judge and trace run in place on a machine that is not modal (the colab vm):
#   python tools/colab_extras.py judge <gate tag> <case>   the vlm's opinion on every item -> /out/judge-<tag>.json
#   python tools/colab_extras.py trace <tag>               one profiled edit_1024 request -> /out/trace/<tag>/
import pathlib as pl
import sys
import types

import modal_bench

modal_bench.out_volume = modal_bench.hf_volume = types.SimpleNamespace(commit=lambda: None)


def main(mode: str, tag: str, case: str = "edit_1024") -> None:
    if mode == "judge":
        items = sorted(int(path.stem) for path in (pl.Path("/out/gate") / tag / "turbo" / case).glob("*.png"))
        pl.Path(f"/out/judge-{tag}.json").write_text(modal_bench.judge_flagged.local("turbo", tag, {case: items}))
    else:
        pl.Path(f"/out/trace/{tag}").mkdir(parents=True, exist_ok=True)
        pl.Path(f"/out/trace/{tag}/report.json").write_text(modal_bench.trace.local("turbo", [case], tag))


if __name__ == "__main__":
    main(*sys.argv[1:])
