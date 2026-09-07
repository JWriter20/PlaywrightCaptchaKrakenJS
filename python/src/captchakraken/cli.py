"""
CaptchaKraken CLI (v2).

Modes:
  captchakraken image.png [model_name] [api_provider] [api_key]
        Solve a captcha from ONE still image. `api_provider` is kept for v1
        compat; only `captchaKrakenApi` is supported and is the default.

  captchakraken solve-animated --frames-dir DIR [--fps N] [--model M] [--api-key K]
        Solve an ANIMATED challenge from a recorded burst. DIR holds one
        zero-padded PNG per frame; the burst is sliced into keyframes (written to
        DIR/keyframes/) and sent to the model as one multi-image request. Prints
        {"actions": [...], "keyframes": [...], "keyframe_mode": ...}. Every action
        carries `await_keyframe` — the still the driver must see on screen before
        acting — and `frame`, its number. Poll for that state with `match-region`.

  captchakraken match-region ref.png live.png cx cy [tolerance]
        Does `live` look like `ref` around the 0-1 point (cx, cy)? The wait gate
        behind an animated click -> {"match": bool, "diff": float}. Also a `serve`
        cmd (`match-region`), which is what a driver polling every ~120ms wants.

  captchakraken check-movement   img1.png img2.png [threshold]
  captchakraken check-movement-batch threshold img1 img2 [img3 ...]
        Frame-diff helpers used by the Playwright lib's settle monitor.

  captchakraken find-grid       image.png
  captchakraken detect-selected image.png
  captchakraken get-numbered-grid image.png
  captchakraken find-checkbox   image.png
        OpenCV tool calls.

  captchakraken find-move       image.png
        Detect every hCaptcha "Move" draggable pill -> {"indicators": [[x,y,w,h]]}.
  captchakraken find-movable    image.png
        Detect each Move pill AND the movable card/object below it
        -> {"items": [{"indicator": [...], "content": [...]}]}.

  captchakraken grid-cell-states imgA.png imgB.png
        Batched per-poll grid-cell state across two consecutive frames:
        {"empty": [...], "changing": [...], "loaded": [...], "selected": [...]}
        (1-indexed), or {"grid": null} if no grid is painted yet. This is the
        hot path the Playwright lib polls while waiting for reCAPTCHA tiles to
        settle — one subprocess per poll, not one per cell.

  captchakraken is-empty-cell    image.png cell_number
  captchakraken is-cell-selected image.png cell_number
  captchakraken is-cell-changing imgA.png imgB.png cell_number
  captchakraken wait-for-cell-loaded cell_number img1.png img2.png [...]
        Single-cell state helpers (1-indexed cell_number), mainly for debug.

  captchakraken server start | stop | status | run
        Manage the local vLLM server. `start` launches it in the background and
        waits until healthy; `run` runs it in the foreground; `stop` terminates
        it; `status` reports the endpoint + model config. You normally never run
        these — a local server auto-starts on the first solve (disable with
        CAPTCHA_KRAKEN_AUTOSTART=0). Point VLLM_BASE_URL at your own server to
        skip local management entirely.

  captchakraken fetch [--weights-only|--engine-only] [--no-restart] [--dry-run]
        Update in one step: pull the latest CaptchaKraken model from the HF org
        (https://huggingface.co/CaptchaKraken) AND upgrade the vLLM serving
        stack, then restart a running local server so it takes effect. Use this
        to get new model revisions + engine fixes without re-running setup.sh.
"""

import argparse
import json
import os
import subprocess
import sys
from typing import Optional

from .errors import CaptchaKrakenAPIError
from .solver import CaptchaSolver, UnsupportedCaptchaError
from .timing import timed


def _handle_movement_commands() -> bool:
    if len(sys.argv) <= 1:
        return False

    cmd = sys.argv[1]

    if cmd == "check-movement":
        if len(sys.argv) < 4:
            print(
                json.dumps({"error": "Usage: captchakraken check-movement img1.png img2.png [threshold]"}),
                file=sys.stderr,
            )
            sys.exit(1)

        from .image_processor import ImageProcessor

        img1 = sys.argv[2]
        img2 = sys.argv[3]
        threshold = 0.005
        if len(sys.argv) > 4:
            try:
                threshold = float(sys.argv[4])
            except ValueError:
                pass
        has_movement = ImageProcessor.detect_movement(img1, img2, threshold)
        print(json.dumps({"has_movement": has_movement}))
        return True

    if cmd == "check-movement-batch":
        if len(sys.argv) < 5:
            print(
                json.dumps({"error": "Usage: captchakraken check-movement-batch threshold img1 img2 [img3 ...]"}),
                file=sys.stderr,
            )
            sys.exit(1)

        import cv2

        try:
            threshold = float(sys.argv[2])
        except ValueError:
            threshold = 0.003
        paths = sys.argv[3:]

        imgs = [cv2.imread(p) for p in paths]
        valid = [(p, im) for p, im in zip(paths, imgs) if im is not None]
        if len(valid) < 2:
            print(json.dumps({"has_movement": False, "max_ratio": 0.0, "valid_samples": len(valid)}))
            return True

        max_ratio = 0.0
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                a, b = valid[i][1], valid[j][1]
                if a.shape != b.shape:
                    max_ratio = 1.0
                    break
                diff = cv2.absdiff(a, b)
                gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                _, thr = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
                ratio = cv2.countNonZero(thr) / (thr.shape[0] * thr.shape[1])
                if ratio > max_ratio:
                    max_ratio = ratio
            if max_ratio >= 1.0:
                break

        print(
            json.dumps(
                {
                    "has_movement": max_ratio > threshold,
                    "max_ratio": max_ratio,
                    "valid_samples": len(valid),
                }
            )
        )
        return True

    if cmd == "is-cell-changing":
        # captchakraken is-cell-changing imgA.png imgB.png cell_number
        if len(sys.argv) < 5:
            print(
                json.dumps({"error": "Usage: captchakraken is-cell-changing imgA.png imgB.png cell_number"}),
                file=sys.stderr,
            )
            sys.exit(1)

        from .tool_calls.find_grid import find_grid, is_cell_opacity_changing

        img_a, img_b = sys.argv[2], sys.argv[3]
        try:
            cell_number = int(sys.argv[4])
        except ValueError:
            print(json.dumps({"error": "cell_number must be an integer (1-indexed)"}), file=sys.stderr)
            sys.exit(1)

        grid_boxes = find_grid(img_b)
        if not grid_boxes:
            print(json.dumps({"error": "No grid detected"}), file=sys.stderr)
            sys.exit(1)
        print(json.dumps({"is_changing": is_cell_opacity_changing(img_a, img_b, grid_boxes, cell_number)}))
        return True

    if cmd == "wait-for-cell-loaded":
        # captchakraken wait-for-cell-loaded cell_number img1.png img2.png [img3 ...]
        if len(sys.argv) < 4:
            print(
                json.dumps({"error": "Usage: captchakraken wait-for-cell-loaded cell_number img1.png img2.png [...]"}),
                file=sys.stderr,
            )
            sys.exit(1)

        from .tool_calls.find_grid import find_grid, wait_for_cell_loaded

        try:
            cell_number = int(sys.argv[2])
        except ValueError:
            print(json.dumps({"error": "cell_number must be an integer (1-indexed)"}), file=sys.stderr)
            sys.exit(1)
        frame_paths = sys.argv[3:]

        grid_boxes = find_grid(frame_paths[-1])
        if not grid_boxes:
            print(json.dumps({"error": "No grid detected"}), file=sys.stderr)
            sys.exit(1)
        print(json.dumps({"is_loaded": wait_for_cell_loaded(frame_paths, grid_boxes, cell_number)}))
        return True

    return False


def _handle_cell_commands() -> bool:
    """Per-cell state helpers that take a single image plus a 1-indexed cell
    number: is-empty-cell, is-cell-selected. (is-cell-changing and
    wait-for-cell-loaded take multiple images and live in
    _handle_movement_commands.)"""
    if len(sys.argv) <= 1:
        return False
    cmd = sys.argv[1]
    if cmd not in {"is-empty-cell", "is-cell-selected"}:
        return False

    if len(sys.argv) < 4:
        print(
            json.dumps({"error": f"Usage: captchakraken {cmd} image.png cell_number"}),
            file=sys.stderr,
        )
        sys.exit(1)

    image_path = sys.argv[2]
    if not os.path.exists(image_path):
        print(json.dumps({"error": f"Image not found: {image_path}"}), file=sys.stderr)
        sys.exit(1)
    try:
        cell_number = int(sys.argv[3])
    except ValueError:
        print(json.dumps({"error": "cell_number must be an integer (1-indexed)"}), file=sys.stderr)
        sys.exit(1)

    try:
        from .tool_calls.find_grid import find_grid, is_empty_cell, is_cell_selected

        grid_boxes = find_grid(image_path)
        if not grid_boxes:
            print(json.dumps({"error": "No grid detected"}), file=sys.stderr)
            sys.exit(1)
        if cmd == "is-empty-cell":
            result = {"is_empty": is_empty_cell(image_path, grid_boxes, cell_number)}
        else:
            result = {"is_selected": is_cell_selected(image_path, grid_boxes, cell_number)}
        print(json.dumps(result))
        return True
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


def _compute_grid_cell_states(img_a, img_b, grid_boxes):
    """Pure core for grid-cell-states / -fixed and the persistent worker. Given
    two frame paths and the grid boxes to use, returns the
    {empty, changing, loaded, selected} dict (1-indexed). Single source of truth
    so every entry point (one-shot CLI, -fixed, serve) produces identical
    output."""
    from .tool_calls.find_grid import (
        is_empty_cell,
        is_cell_opacity_changing,
        detect_selected_cells,
    )

    empty, changing, loaded = [], [], []
    for c in range(1, len(grid_boxes) + 1):
        e = is_empty_cell(img_b, grid_boxes, c)
        ch = is_cell_opacity_changing(img_a, img_b, grid_boxes, c)
        if e:
            empty.append(c)
        if ch:
            changing.append(c)
        if not e and not ch:
            loaded.append(c)
    selected, _ = detect_selected_cells(img_b, grid_boxes)
    return {"empty": empty, "changing": changing, "loaded": loaded, "selected": selected}


def _handle_grid_cell_states() -> bool:
    """Batched per-poll grid state across TWO consecutive frames. One subprocess
    per poll (find_grid once, then loop all cells) — never one spawn per cell.

      captchakraken grid-cell-states imgA.png imgB.png

    Returns {"empty": [...], "changing": [...], "loaded": [...],
    "selected": [...]} (1-indexed). If no grid is detected it returns
    {"grid": null} with exit 0 so the JS poller treats it as "keep polling"
    rather than a hard error."""
    if len(sys.argv) <= 1 or sys.argv[1] != "grid-cell-states":
        return False

    if len(sys.argv) < 4:
        print(
            json.dumps({"error": "Usage: captchakraken grid-cell-states imgA.png imgB.png"}),
            file=sys.stderr,
        )
        sys.exit(1)

    img_a, img_b = sys.argv[2], sys.argv[3]
    for p in (img_a, img_b):
        if not os.path.exists(p):
            print(json.dumps({"error": f"Image not found: {p}"}), file=sys.stderr)
            sys.exit(1)

    try:
        from .tool_calls.find_grid import find_grid

        # Detect the grid on the latest frame; bboxes are reused for both frames.
        grid_boxes = find_grid(img_b)
        if not grid_boxes:
            # Not "an error" — the grid simply hasn't painted yet. Let JS poll on.
            print(json.dumps({"grid": None}))
            return True

        print(json.dumps(_compute_grid_cell_states(img_a, img_b, grid_boxes)))
        return True
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


def _handle_grid_cell_states_fixed() -> bool:
    """Like grid-cell-states, but the GRID BOXES ARE SUPPLIED EXPLICITLY instead
    of re-detected per frame:

      captchakraken grid-cell-states-fixed imgA.png imgB.png '<json grid_boxes>'

    The dynamic reCAPTCHA refresh blanks tiles to near-white, which makes
    find_grid fail on that frame (no separator lines) and grid-cell-states then
    returns {"grid": null}. The JS driver caches the grid from the first solid
    frame and passes it here so per-cell empty/changing/selected stays correct
    even while tiles are blank/fading. grid_boxes is a JSON array of
    [x1,y1,x2,y2] pixel tuples in screenshot space (the same shape find-grid
    emits). Returns {"empty","changing","loaded","selected"} (1-indexed)."""
    if len(sys.argv) <= 1 or sys.argv[1] != "grid-cell-states-fixed":
        return False

    if len(sys.argv) < 5:
        print(
            json.dumps({"error": "Usage: captchakraken grid-cell-states-fixed imgA.png imgB.png '<json grid_boxes>'"}),
            file=sys.stderr,
        )
        sys.exit(1)

    img_a, img_b, boxes_json = sys.argv[2], sys.argv[3], sys.argv[4]
    for p in (img_a, img_b):
        if not os.path.exists(p):
            print(json.dumps({"error": f"Image not found: {p}"}), file=sys.stderr)
            sys.exit(1)

    try:
        raw = json.loads(boxes_json)
        grid_boxes = [tuple(int(v) for v in box) for box in raw]
        if not grid_boxes:
            print(json.dumps({"error": "empty grid_boxes"}), file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": f"bad grid_boxes JSON: {e}"}), file=sys.stderr)
        sys.exit(1)

    try:
        print(json.dumps(_compute_grid_cell_states(img_a, img_b, grid_boxes)))
        return True
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


def _handle_serve() -> bool:
    """Persistent worker mode — the big latency win for the reCAPTCHA poll loop.

      captchakraken serve

    Instead of spawning a fresh `captchakraken` (≈0.4s of interpreter +
    cv2/numpy import) for every poll, the JS lib starts ONE long-lived process
    and streams requests over stdin, one JSON object per line, each answered with
    one JSON line on stdout. cv2/numpy are imported once at startup.

    Request line:  {"id": <n>, "cmd": "<name>", ...args}
    Response line: {"id": <n>, "ok": true, "result": <value>}  on success
                   {"id": <n>, "ok": false, "error": "<msg>"}  on failure

    Supported cmds (results are byte-identical to the one-shot subcommands):
      find-grid               {image}                       -> grid_boxes | null
      grid-cell-states        {a, b}                        -> states | {grid: null}
      grid-cell-states-fixed  {a, b, grid_boxes}            -> states
      check-movement          {a, b, threshold?}            -> {has_movement: bool}
      match-region            {ref, live, cx, cy, ...}      -> {match, diff, ...}
      track-piece             {before, after, exclude?}     -> {bbox, moved}

    Unknown cmd / malformed line -> an {ok:false} response (the process keeps
    running). EOF on stdin ends the loop cleanly. All heavy detection delegates
    to the exact same functions the one-shot handlers use."""
    if len(sys.argv) <= 1 or sys.argv[1] != "serve":
        return False

    # Import once, up front — this is the whole point of the worker.
    from .tool_calls.find_grid import find_grid

    def handle(req):
        cmd = req.get("cmd")
        if cmd == "find-grid":
            return find_grid(req["image"])
        if cmd == "grid-cell-states":
            grid_boxes = find_grid(req["b"])
            if not grid_boxes:
                return {"grid": None}
            return _compute_grid_cell_states(req["a"], req["b"], grid_boxes)
        if cmd == "grid-cell-states-fixed":
            grid_boxes = [tuple(int(v) for v in box) for box in req["grid_boxes"]]
            if not grid_boxes:
                raise ValueError("empty grid_boxes")
            return _compute_grid_cell_states(req["a"], req["b"], grid_boxes)
        if cmd == "check-movement":
            # Freshness check for the JS solver: did the captcha frame change
            # (tiles faded in / refreshed) between the screenshot we sent the
            # model and now? Same primitive the one-shot `check-movement` uses.
            from .image_processor import ImageProcessor

            threshold = float(req.get("threshold", 0.005))
            moved = ImageProcessor.detect_movement(req["a"], req["b"], threshold)
            return {"has_movement": bool(moved)}
        if cmd == "match-region":
            # Animated wait gate: is the widget back in the state the model
            # answered about? Polled every ~120ms while a click is held, so it
            # belongs on the worker rather than paying a process spawn per poll.
            return _match_region(req["ref"], req["live"],
                                 float(req["cx"]), float(req["cy"]),
                                 req.get("tolerance"))
        if cmd == "track-piece":
            # Slider closed loop: where has the piece got to? Called several
            # times per drag WITH THE MOUSE HELD DOWN, so a process spawn per
            # reading would stretch the drag into something no human hand does.
            return _track_piece(req["before"], req["after"], req.get("exclude"))
        raise ValueError(f"unknown cmd: {cmd!r}")

    # Signal readiness so the JS side knows imports are done before it polls.
    sys.stdout.write(json.dumps({"ready": True}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        rid = None
        try:
            req = json.loads(line)
            rid = req.get("id")
            result = handle(req)
            sys.stdout.write(json.dumps({"id": rid, "ok": True, "result": result}) + "\n")
        except Exception as e:  # noqa: BLE001 — worker must never die on one bad req
            sys.stdout.write(json.dumps({"id": rid, "ok": False, "error": str(e)}) + "\n")
        sys.stdout.flush()
    return True


def _frames_in(frames_dir: str) -> list:
    """The recorded burst's frames in capture order.

    Sorted by NAME, which is why the recorder must zero-pad: `frame_9.png` sorts
    after `frame_10.png`, and a shuffled burst destroys the temporal structure the
    cycle detector reads. `_handle_solve_animated` rejects an unsorted-looking set
    rather than silently slicing noise.
    """
    import glob

    return sorted(
        p for ext in ("png", "jpg", "jpeg")
        for p in glob.glob(os.path.join(frames_dir, f"*.{ext}"))
    )


def _handle_solve_animated() -> bool:
    """`captchakraken solve-animated --frames-dir DIR [--fps N] [...]`

    The animated counterpart of the default solve command, for the TypeScript
    driver: TS owns the browser, so IT records the burst (one zero-padded PNG per
    frame into a directory it owns) and this slices the burst into keyframes, sends
    them to the model as one multi-image request, and prints the actions.

    Each action carries `await_keyframe`, an absolute path to the still the driver
    must see on screen before it acts. Those files are written into
    `DIR/keyframes/`, i.e. inside the caller's own directory — the CLI never leaves
    temp files behind for the caller to guess at, and the caller cleans up when it
    cleans up the burst. Poll for the state with `match-region`.
    """
    if len(sys.argv) <= 1 or sys.argv[1] != "solve-animated":
        return False

    parser = argparse.ArgumentParser(prog="captchakraken solve-animated")
    parser.add_argument("--frames-dir", required=True,
                        help="Directory of the recorded burst (zero-padded PNGs).")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="Frame rate the burst was recorded at (default 10, the "
                             "rate the training data was collected at).")
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--expert", default=None,
                        choices=["pixel", "grid", "video", "text"])
    args = parser.parse_args(sys.argv[2:])

    frames = _frames_in(args.frames_dir)
    if not frames:
        print(json.dumps({"error": f"no frames in {args.frames_dir}"}), file=sys.stderr)
        sys.exit(1)

    try:
        import cv2

        from .keyframes import extract_keyframes, write_keyframes

        with timed("cli.total"):
            imgs = [im for im in (cv2.imread(p) for p in frames) if im is not None]
            if not imgs:
                print(json.dumps({"error": "no frame decoded"}), file=sys.stderr)
                sys.exit(1)
            kfset = extract_keyframes(imgs, fps=float(args.fps))
            kf_dir = os.path.join(args.frames_dir, "keyframes")
            paths = [str(os.path.abspath(p))
                     for p in write_keyframes(kfset, kf_dir, stem="challenge")]

            solver = CaptchaSolver(model=args.model, api_key=args.api_key,
                                   expert=args.expert)
            result = solver.solve_keyframes(paths)

        action_data = [a.model_dump() for a in result]
        print(json.dumps({
            "actions": action_data,
            "token_usage": solver.planner.token_usage,
            # Reported so the driver can log what it actually sent, and so a
            # surprising slicing (6 frames collapsing to 1) is visible in the
            # solve log rather than only inferable from the answer.
            "keyframes": paths,
            "keyframe_mode": kfset.mode,
            # What the board LOOKS like, as distinct from how it was
            # sliced. The driver gates its clicks on this; see
            # KeyframeSet.steady_screens.
            "steady_screens": kfset.steady_screens,
            "source_frames": len(imgs),
        }))
    except UnsupportedCaptchaError as e:
        print(json.dumps({"error": str(e), "unsupported": True}), file=sys.stderr)
        sys.exit(2)
    except CaptchaKrakenAPIError as e:
        print(json.dumps(e.to_payload()), file=sys.stderr)
        sys.exit(3)
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)
    return True


def _match_region(ref: str, live: str, cx: float, cy: float,
                  tolerance: Optional[float] = None) -> dict:
    """Does `live` look like `ref` around the 0–1 point (cx, cy)?

    The wait-for-state gate behind an animated click. Compares only the
    neighbourhood of the action point, with the same box and metric the training
    label's frame was chosen with (`keyframes.region_box` / `region_diff_ratio`) —
    local rather than whole-frame because everything else in these puzzles is also
    moving, so a whole-frame match would need every unrelated sprite to align too
    and would essentially never open.
    """
    import cv2

    from .keyframes import MATCH_REGION_TOLERANCE, region_box, region_diff_ratio

    tol = MATCH_REGION_TOLERANCE if tolerance is None else float(tolerance)
    a, b = cv2.imread(ref), cv2.imread(live)
    if a is None or b is None:
        return {"match": False, "diff": 1.0, "error": "unreadable image"}
    box = region_box(a.shape[1::-1], (cx, cy))
    d = region_diff_ratio(a, b, box)
    return {"match": bool(d <= tol), "diff": float(d), "tolerance": tol}


def _track_piece(before: str, after: str, exclude=None) -> dict:
    """Where did the puzzle piece get to? See tool_calls/track_piece.py."""
    from .tool_calls.track_piece import changed_bbox

    bbox = changed_bbox(before, after, exclude)
    return {"bbox": bbox, "moved": bbox is not None}


def _handle_track_piece() -> bool:
    """`captchakraken track-piece before.png after.png [exclude_json]`

    `exclude_json` is `[x1, y1, x2, y2]` in pixels — the slider handle, which is
    moving too and would otherwise be measured instead of the piece.

    Also available over the persistent worker as cmd `track-piece`, which is
    what the driver should use: this runs mid-drag, several times, with the
    mouse button held.
    """
    if len(sys.argv) <= 1 or sys.argv[1] != "track-piece":
        return False
    if len(sys.argv) < 4:
        print(json.dumps({
            "error": "Usage: captchakraken track-piece before.png after.png [exclude_json]"
        }), file=sys.stderr)
        sys.exit(1)
    exclude = json.loads(sys.argv[4]) if len(sys.argv) > 4 else None
    print(json.dumps(_track_piece(sys.argv[2], sys.argv[3], exclude)))
    return True


def _handle_match_region() -> bool:
    """`captchakraken match-region ref.png live.png cx cy [tolerance]`

    cx/cy are 0–1 fractions of the reference image (the centre of the action).
    Also available over the persistent worker as cmd `match-region`, which is what
    the driver should use — this is polled every ~120 ms while waiting.
    """
    if len(sys.argv) <= 1 or sys.argv[1] != "match-region":
        return False
    if len(sys.argv) < 6:
        print(json.dumps({
            "error": "Usage: captchakraken match-region ref.png live.png cx cy [tolerance]"
        }), file=sys.stderr)
        sys.exit(1)
    tol = float(sys.argv[6]) if len(sys.argv) > 6 else None
    print(json.dumps(_match_region(sys.argv[2], sys.argv[3],
                                   float(sys.argv[4]), float(sys.argv[5]), tol)))
    return True


def _handle_report_outcome() -> bool:
    """`captchakraken report-outcome <session-id> solved|failed`

    Tell the hosted API whether the widget accepted this solve. The one fact the
    server cannot see for itself: a wrong answer arrives there as a 200 with
    well-formed JSON in it, and only the driver watched the widget.

    THIS EXISTS FOR THE TypeScript PORT. The Python driver calls
    `planner.report_outcome` in-process at the end of `PageSolver.solve`; the TS
    driver spawns the CLI, which is the same subprocess boundary every other
    model call already crosses. One implementation of the endpoint, the
    credential and the extra headers, rather than a second copy in TypeScript
    that agrees with this one until one of them is edited — the drift
    `model-name.ts` already exists to prevent, on the same seam.

    ALWAYS EXITS 0. The caller is a `finally` block in a driver whose solve has
    already finished, so there is no outcome here worth failing over, and a
    non-zero exit would make a self-hosted user's logs report an error on every
    solve. What happened is on stdout for anyone who wants it.
    """
    if len(sys.argv) <= 1 or sys.argv[1] != "report-outcome":
        return False
    if len(sys.argv) < 4 or sys.argv[3] not in ("solved", "failed"):
        print(json.dumps({
            "error": "Usage: captchakraken report-outcome <session-id> solved|failed"
        }), file=sys.stderr)
        sys.exit(1)
    delivered = False
    try:
        from .planner import ActionPlanner
        delivered = ActionPlanner().report_outcome(sys.argv[2], sys.argv[3] == "solved")
    except Exception as exc:  # noqa: BLE001 — see the docstring
        print(json.dumps({"reported": False, "error": str(exc)}))
        return True
    print(json.dumps({"reported": bool(delivered)}))
    return True


def _handle_tool_commands() -> bool:
    if len(sys.argv) <= 1:
        return False
    cmd = sys.argv[1]
    if cmd not in {"find-grid", "find-checkbox", "detect-selected", "get-numbered-grid"}:
        return False

    if len(sys.argv) < 3:
        print(json.dumps({"error": f"Usage: captchakraken {cmd} image.png"}), file=sys.stderr)
        sys.exit(1)

    image_path = sys.argv[2]
    if not os.path.exists(image_path):
        print(json.dumps({"error": f"Image not found: {image_path}"}), file=sys.stderr)
        sys.exit(1)

    try:
        if cmd == "find-grid":
            from .tool_calls.find_grid import find_grid

            result = find_grid(image_path)
        elif cmd == "detect-selected":
            from .tool_calls.find_grid import detect_selected_cells, find_grid

            grid_boxes = find_grid(image_path)
            if not grid_boxes:
                result = {"error": "No grid detected"}
            else:
                selected, loading = detect_selected_cells(image_path, grid_boxes)
                result = {"selected": selected, "loading": loading}
        elif cmd == "get-numbered-grid":
            from .tool_calls.find_grid import find_grid, get_numbered_grid_overlay

            grid_boxes = find_grid(image_path)
            if not grid_boxes:
                result = {"error": "No grid detected"}
            else:
                overlay_path = get_numbered_grid_overlay(image_path, grid_boxes)
                result = {"overlay_image": overlay_path}
        else:
            from .tool_calls.find_checkbox import find_checkbox

            result = find_checkbox(image_path)

        print(json.dumps(result))
        return True
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


def _handle_move_commands() -> bool:
    """hCaptcha drag-puzzle "Move" pill tools (pure OpenCV):

      captchakraken find-move       image.png
      captchakraken find-movable    image.png
    """
    if len(sys.argv) <= 1:
        return False
    cmd = sys.argv[1]
    if cmd not in {"find-move", "find-movable"}:
        return False

    if len(sys.argv) < 3:
        print(json.dumps({"error": f"Usage: captchakraken {cmd} image.png"}), file=sys.stderr)
        sys.exit(1)

    image_path = sys.argv[2]
    if not os.path.exists(image_path):
        print(json.dumps({"error": f"Image not found: {image_path}"}), file=sys.stderr)
        sys.exit(1)

    try:
        import cv2

        from .tool_calls.move_indicator import (
            find_movable_content,
            find_move_indicators,
        )

        im = cv2.imread(image_path)
        if im is None:
            print(json.dumps({"error": f"Could not read image: {image_path}"}), file=sys.stderr)
            sys.exit(1)

        indicators = find_move_indicators(im)

        if cmd == "find-move":
            result = {"indicators": indicators}
        else:  # find-movable
            items = []
            for ind in indicators:
                items.append({"indicator": ind, "content": find_movable_content(im, ind)})
            result = {"items": items}

        print(json.dumps(result))
        return True
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


def _handle_server_commands() -> bool:
    """Local vLLM lifecycle: `captchakraken server <start|stop|status|run>`.

    Distinct from the CV-worker `serve` subcommand (that one is the OpenCV poll
    worker the Playwright lib drives). This manages the inference server.
    """
    if len(sys.argv) <= 1 or sys.argv[1] != "server":
        return False

    from . import server_manager

    action = sys.argv[2] if len(sys.argv) > 2 else "status"
    try:
        if action == "start":
            print(json.dumps(server_manager.start(background=True)))
        elif action == "run":
            server_manager.run_foreground()  # never returns
        elif action == "stop":
            print(json.dumps(server_manager.stop()))
        elif action in ("status", ""):
            print(json.dumps(server_manager.status()))
        else:
            print(
                json.dumps({"error": f"unknown server action {action!r} "
                                     "(use start|stop|status|run)"}),
                file=sys.stderr,
            )
            sys.exit(2)
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)
    return True


def _handle_fetch() -> bool:
    """`captchakraken fetch` — pull the latest published weights from the HF org
    and refresh the vLLM serving stack, restarting a running local server so the
    update takes effect. See updater.fetch for the full behavior."""
    if len(sys.argv) <= 1 or sys.argv[1] not in {"fetch", "update"}:
        return False

    flags = set(sys.argv[2:])
    unknown = flags - {"--weights-only", "--engine-only", "--no-restart", "--dry-run"}
    if unknown:
        print(
            json.dumps({"error": f"unknown fetch flag(s): {sorted(unknown)} "
                                 "(use --weights-only|--engine-only|--no-restart|--dry-run)"}),
            file=sys.stderr,
        )
        sys.exit(2)
    if "--weights-only" in flags and "--engine-only" in flags:
        print(json.dumps({"error": "--weights-only and --engine-only are mutually exclusive"}),
              file=sys.stderr)
        sys.exit(2)

    from . import updater

    weights = "--engine-only" not in flags
    engine = "--weights-only" not in flags
    try:
        result = updater.fetch(
            weights=weights,
            engine=engine,
            restart="--no-restart" not in flags,
            dry_run="--dry-run" in flags,
        )
        print(json.dumps(result))
        return True
    except subprocess.CalledProcessError as e:
        print(json.dumps({"error": f"fetch step failed (exit {e.returncode}): "
                                   f"{' '.join(e.cmd) if isinstance(e.cmd, list) else e.cmd}"}),
              file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


def main():
    if _handle_server_commands():
        return
    if _handle_fetch():
        return
    if _handle_movement_commands():
        return
    if _handle_move_commands():
        return
    if _handle_serve():
        return
    if _handle_grid_cell_states():
        return
    if _handle_grid_cell_states_fixed():
        return
    if _handle_cell_commands():
        return
    if _handle_tool_commands():
        return
    if _handle_solve_animated():
        return
    if _handle_match_region():
        return
    if _handle_track_piece():
        return
    if _handle_report_outcome():
        return

    parser = argparse.ArgumentParser(description="CaptchaKraken v2 (vLLM)")
    parser.add_argument("image_path", help="Path to the captcha image or video")
    parser.add_argument(
        "model",
        nargs="?",
        default=None,
        help="LoRA name registered with vLLM (default: 'captcha').",
    )
    parser.add_argument(
        "api_provider",
        nargs="?",
        default="captchaKrakenApi",
        choices=["captchaKrakenApi"],
        help="Kept for v1 argv compatibility; only captchaKrakenApi is supported in v2.",
    )
    parser.add_argument(
        "api_key",
        nargs="?",
        default=None,
        help="Bearer token (or set VLLM_API_KEY / CAPTCHA_KRAKEN_API_KEY).",
    )
    parser.add_argument(
        "--puzzle-source",
        default="unknown",
        choices=["hcaptcha", "recaptcha", "unknown"],
        help="Vendor hint from the Playwright wrapper. Constrains which grid "
        "shapes a detection may be solved as: hCaptcha ships only a 3x3, so a "
        "16-cell lattice on one is a false positive (find_grid latches onto the "
        "header/footer bands of click puzzles). An unrecognised or absent hint "
        "allows every shape, which is what GeeTest and Prosopo boards report.",
    )
    parser.add_argument(
        "--retry-mode",
        default=None,
        choices=["missed-tiles"],
        help="Hint that the previous selection was rejected by the captcha vendor "
        "with an under-selection error (e.g. reCAPTCHA's 'Please select all matching "
        "images'). Switches the grid prompt to a more aggressive variant that "
        "instructs the LoRA to look at the full grid for tiles it missed.",
    )
    parser.add_argument(
        "--expert",
        default=None,
        choices=["pixel", "grid", "video", "text"],
        help="Force one expert of a ROUTED model (Abyss) instead of letting the "
        "prompt family choose. Routing is automatic and this is the override a "
        "benchmark needs: serve one arm, grade the types it owns, serve the "
        "next. Refused against a model that serves a single adapter, because "
        "quietly measuring the generalist and reporting it as the expert is a "
        "number nobody can catch. Also settable as CAPTCHA_EXPERT.",
    )
    parser.add_argument(
        "--text-mode",
        action="store_true",
        help="The widget has a text box, so the answer is a string to type rather "
        "than a place to click. Sends the distorted-text prompt and skips grid "
        "detection. Set by the Playwright wrapper from the DOM — the picture "
        "alone does not distinguish a BotDetect code from an hCaptcha "
        "'click the matching letter'.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(json.dumps({"error": f"Image not found: {args.image_path}"}), file=sys.stderr)
        sys.exit(1)

    try:
        with timed("cli.total"):
            solver = CaptchaSolver(model=args.model, api_key=args.api_key,
                                   expert=args.expert)
            result = solver.solve(
                args.image_path,
                puzzle_source=args.puzzle_source,
                retry_mode=args.retry_mode,
                text_mode=args.text_mode,
            )

        if isinstance(result, list):
            action_data = [a.model_dump() for a in result]
        elif hasattr(result, "model_dump"):
            action_data = result.model_dump()
        else:
            action_data = result

        print(json.dumps({"actions": action_data, "token_usage": solver.planner.token_usage}))
    except UnsupportedCaptchaError as e:
        # Expected outcome, not a crash: this LoRA only handles grids and
        # checkboxes. Emit a clean error with no traceback.
        print(json.dumps({"error": str(e), "unsupported": True}), file=sys.stderr)
        sys.exit(2)
    except CaptchaKrakenAPIError as e:
        # The hosted API refused us and told us why. That is an answer, not a
        # crash, so no traceback: a stack trace here buries the one line the
        # user needs ("out of credits, top up at …") under frames from a module
        # they have never opened.
        #
        # Exit 3, distinct from 1 (genuine failure) and 2 (unsupported puzzle),
        # and the payload carries the machine-readable fields so the JS driver
        # can rebuild the error rather than regex it out of prose.
        print(json.dumps(e.to_payload()), file=sys.stderr)
        sys.exit(3)
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
