"""An isolated browser canvas that executes mouse events against rendered GUIs.

The policy interface is ``observe()`` plus the fixed instruction. All hit boxes,
enabled flags, counters and inspection methods belong only to the evaluator.
No OS/native mouse input and no existing browser session are used.
"""

from __future__ import annotations

import base64
import copy
import io
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .dataset import _resolve_image

ENV_VERSION = "controlled_save_canvas_v1"
DEFAULT_EDGE = Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")

_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<style>html,body{margin:0;padding:0;overflow:hidden;background:#fff;}canvas{display:block;margin:0;padding:0;touch-action:none;}</style>
</head><body><canvas id="screen"></canvas><script>
window.__setup = async (payload) => {
  const canvas = document.getElementById('screen');
  canvas.width = payload.width; canvas.height = payload.height;
  canvas.style.width = payload.width + 'px'; canvas.style.height = payload.height + 'px';
  const ctx = canvas.getContext('2d', {alpha:false, colorSpace:'srgb'});
  ctx.imageSmoothingEnabled = false;
  const image = new Image();
  image.src = payload.image;
  await image.decode();
  const state = {save_count:0, other_action_count:0, other_attempt_count:0,
                 attempts:[], feedback_text:null, first_hit:null, mapped_target:null,
                 target_hit:false, rendered_feedback:null};
  const render = () => {
    ctx.drawImage(image, 0, 0);
    if (state.feedback_text !== null) {
      const x = Math.max(24, payload.width - 308), y = 70;
      ctx.fillStyle = state.feedback_text === '保存成功' ? '#18734c' : '#315f9a';
      ctx.fillRect(x, y, 278, 52);
      ctx.strokeStyle = '#eff7ff'; ctx.lineWidth = 1; ctx.strokeRect(x+0.5,y+0.5,277,51);
      ctx.fillStyle = '#ffffff'; ctx.font = '23px "Microsoft YaHei", "Noto Sans SC", sans-serif';
      ctx.textBaseline = 'top'; ctx.fillText(state.feedback_text, x + 15, y + 4);
      ctx.font = '14px "Microsoft YaHei", "Noto Sans SC", sans-serif';
      ctx.fillText('操作次数 ' + (state.save_count + state.other_action_count), x + 15, y + 33);
    }
    state.rendered_feedback = state.feedback_text;
  };
  canvas.onclick = (event) => {
    const x = event.clientX, y = event.clientY;
    const control = payload.controls.find(c => x >= c.bbox_px[0] && x < c.bbox_px[2]
                                           && y >= c.bbox_px[1] && y < c.bbox_px[3]) || null;
    const target = control !== null && control.text === payload.target_text;
    let outcome = 'blank';
    if (control !== null) {
      if (!target) state.other_attempt_count += 1;
      if (!control.enabled) outcome = target ? 'disabled_target' : 'disabled_other';
      else if (target) { state.save_count += 1; outcome='saved'; state.feedback_text='保存成功'; }
      else { state.other_action_count += 1; outcome='other_action'; state.feedback_text='已执行：' + control.text; }
    }
    const attempt = {index:state.attempts.length, x, y, trusted:event.isTrusted,
                     mapped_target:control, target_hit:target, outcome};
    state.attempts.push(attempt);
    if (state.attempts.length === 1) state.first_hit = control;
    state.mapped_target = control; state.target_hit = target;
    render();
  };
  render();
  window.__state = state;
  await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
};
</script></body></html>"""


class ControlledSaveEnvironment:
    """Context-managed, non-persistent, independently launched headless Edge.

    Coordinates are original-image pixels. ``click`` uses page.mouse.click;
    ``inspect`` and ``verify`` are evaluator-only and must never reach a model.
    """

    def __init__(self, executable_path: str | Path | None = None, headless: bool = True) -> None:
        self.executable_path = Path(executable_path).resolve() if executable_path is not None else DEFAULT_EDGE
        self.headless = headless
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._initial: Image.Image | None = None
        self._width = self._height = 0
        self._pixel_identity_verified = False

    def __enter__(self) -> "ControlledSaveEnvironment":
        if not self.executable_path.is_file():
            raise FileNotFoundError(f"Browser executable is unavailable: {self.executable_path}")
        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()
        try:
            self._browser = self._playwright.chromium.launch(
                executable_path=str(self.executable_path), headless=self.headless,
                args=["--force-color-profile=srgb", "--disable-gpu"], timeout=30000,
            )
            self._context = self._browser.new_context(device_scale_factor=1, forced_colors="none", offline=True)
            self._page = self._context.new_page()
            self._page.set_default_timeout(10000)
        except BaseException:
            self.close()
            raise
        return self

    def close(self) -> None:
        try:
            if self._context is not None:
                self._context.close()
        finally:
            self._context = self._page = None
            try:
                if self._browser is not None:
                    self._browser.close()
            finally:
                self._browser = None
                if self._playwright is not None:
                    self._playwright.stop()
                self._playwright = None
                self._initial = None

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _ready(self) -> None:
        if self._page is None:
            raise RuntimeError("Use ControlledSaveEnvironment as a context manager")

    def reset(self, sample: dict[str, Any], manifest_dir: str | Path) -> None:
        """Load the PNG verbatim, initialize counters and assert pixel identity."""
        self._ready()
        root = Path(manifest_dir).resolve()
        image_path = _resolve_image(sample.get("image"), root)
        with Image.open(image_path) as opened:
            initial = opened.convert("RGB")
        width, height = initial.size
        controls = copy.deepcopy(sample.get("metadata", {}).get("controls"))
        if not isinstance(controls, list):
            raise ValueError("Environment samples require metadata.controls")
        for index, control in enumerate(controls):
            if not isinstance(control, dict) or not isinstance(control.get("text"), str) or not isinstance(control.get("enabled"), bool):
                raise ValueError(f"Malformed control {index}")
            box = control.get("bbox")
            if (not isinstance(box, (list, tuple)) or len(box) != 4 or
                any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in box) or
                not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1)):
                raise ValueError(f"Control {index} requires a normalized xyxy bbox")
            control["bbox_px"] = [round(box[0] * width), round(box[1] * height),
                                  round(box[2] * width), round(box[3] * height)]
            control["control_index"] = index
        self._width, self._height, self._initial = width, height, initial
        self._pixel_identity_verified = False
        self._page.set_viewport_size({"width": width, "height": height})
        self._page.set_content(_PAGE, wait_until="load")
        data = io.BytesIO()
        initial.save(data, format="PNG")
        self._page.evaluate("payload => window.__setup(payload)", {
            "width": width, "height": height,
            "image": "data:image/png;base64," + base64.b64encode(data.getvalue()).decode("ascii"),
            "controls": controls, "target_text": sample.get("target_text", "保存"),
        })
        observed = self.observe()
        if not np.array_equal(np.asarray(observed), np.asarray(initial)):
            raise RuntimeError("Initial browser screenshot differs from source pixels; do not run the policy")
        self._pixel_identity_verified = True

    def observe(self) -> Image.Image:
        """Policy-visible observation: a real viewport screenshot, RGB pixels only."""
        self._ready()
        if self._initial is None:
            raise RuntimeError("Call reset before observe")
        data = self._page.screenshot(type="png", full_page=False, scale="css", animations="disabled",
                                     clip={"x": 0, "y": 0, "width": self._width, "height": self._height})
        with Image.open(io.BytesIO(data)) as image:
            return image.convert("RGB")

    def inspect(self) -> dict[str, Any]:
        """Evaluator-only counters, trusted click attempts and mapped controls."""
        self._ready()
        if self._initial is None:
            raise RuntimeError("Call reset before inspect")
        return {"environment_version": ENV_VERSION, **self._page.evaluate("() => JSON.parse(JSON.stringify(window.__state))")}

    def verify(self) -> dict[str, Any]:
        """Check UI bookkeeping and actual changed pixels against the reset PNG."""
        state = self.inspect()
        observed = self.observe()
        changed = np.any(np.asarray(observed) != np.asarray(self._initial), axis=2)
        changed_count = int(changed.sum())
        expected_change = state["save_count"] + state["other_action_count"] > 0
        feedback_ok = state["feedback_text"] == state["rendered_feedback"]
        return {"pixel_changed": changed_count > 0, "changed_pixel_count": changed_count,
                "reset_pixel_identity_verified": self._pixel_identity_verified,
                "ui_state_consistent": feedback_ok and bool(changed_count) == expected_change,
                "feedback_text": state["rendered_feedback"],
                "save_succeeded": state["save_count"] > 0 and state["rendered_feedback"] == "保存成功" and changed_count > 0}

    def click(self, x: float, y: float) -> dict[str, Any]:
        """Dispatch a genuine page mouse event; never invoke a JS action directly."""
        self._ready()
        if self._initial is None:
            raise RuntimeError("Call reset before click")
        if (isinstance(x, bool) or isinstance(y, bool) or not isinstance(x, (int, float)) or not isinstance(y, (int, float)) or
            not math.isfinite(x) or not math.isfinite(y) or not (0 <= x < self._width and 0 <= y < self._height)):
            raise ValueError("Click coordinates must be finite original-image pixels inside the viewport")
        before = len(self.inspect()["attempts"])
        self._page.mouse.click(float(x), float(y))
        self._page.wait_for_function("n => window.__state.attempts.length > n", arg=before)
        state = self.inspect()
        return {**state, "last_attempt": state["attempts"][-1], "verification": self.verify()}
