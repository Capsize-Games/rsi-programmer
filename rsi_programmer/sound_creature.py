"""Canvas-drawn companion creature for the sound learning window."""
from __future__ import annotations

import math
import tkinter as tk


class SoundCreature:
    """Draw and animate a small creature that reflects the current state."""

    def __init__(self, canvas: tk.Canvas) -> None:
        self.canvas = canvas
        self.pose = "idle"
        self.frame = 0
        self._draw()

    def set_pose(self, pose: str) -> None:
        """Set the animation pose, falling back to idle for unknown values."""
        if pose in {"idle", "attentive", "wake", "sleep", "dance", "timer"}:
            self.pose = pose

    def animate(self) -> None:
        """Advance the animation and redraw the creature."""
        self.frame += 1
        self._draw()

    def _draw(self) -> None:
        c = self.canvas
        c.delete("all")
        width = max(c.winfo_width(), 280)
        height = max(c.winfo_height(), 180)
        cx, cy = width / 2, height / 2 + 10
        phase = self.frame / 8
        sleeping = self.pose == "sleep"
        dancing = self.pose == "dance"
        awake = self.pose == "wake"
        bounce = math.sin(phase * 2) * (7 if dancing else 2)
        cy += bounce
        body = "#55d6c2" if not sleeping else "#629d9e"
        c.create_oval(cx - 78, cy + 70, cx + 78, cy + 88,
                      fill="#101c27", outline="")
        # Antennae sway independently to make the listening state readable.
        sway = math.sin(phase) * (10 if self.pose in {"attentive", "wake"}
                                  else 3)
        c.create_line(cx - 28, cy - 64, cx - 37 + sway, cy - 109,
                      fill="#74eee0", width=4, capstyle=tk.ROUND)
        c.create_oval(cx - 46 + sway, cy - 118, cx - 28 + sway,
                      cy - 100, fill="#ffbe63", outline="")
        c.create_line(cx + 28, cy - 64, cx + 38 - sway, cy - 110,
                      fill="#74eee0", width=4, capstyle=tk.ROUND)
        c.create_oval(cx + 29 - sway, cy - 119, cx + 47 - sway,
                      cy - 101, fill="#ffbe63", outline="")
        c.create_oval(cx - 69, cy - 73, cx + 69, cy + 76,
                      fill=body, outline="#a1fff0", width=2)
        arm_y = cy + (18 if dancing else 23)
        arm_spread = 114 if dancing or awake else 94
        c.create_line(cx - 53, arm_y, cx - arm_spread, arm_y - 14,
                      fill="#74eee0", width=12, capstyle=tk.ROUND)
        c.create_line(cx + 53, arm_y, cx + arm_spread, arm_y - 14,
                      fill="#74eee0", width=12, capstyle=tk.ROUND)
        blink = self.frame % 37 in (0, 1) and not sleeping
        eye_y = cy - 18
        if sleeping or blink:
            for ex in (cx - 27, cx + 27):
                c.create_arc(ex - 10, eye_y - 3, ex + 10, eye_y + 10,
                             start=0, extent=180, style=tk.ARC,
                             outline="#17313a", width=3)
        else:
            eye_scale = 1.35 if awake else 1.0
            for ex in (cx - 27, cx + 27):
                c.create_oval(ex - 7 * eye_scale, eye_y - 8 * eye_scale,
                              ex + 7 * eye_scale, eye_y + 8 * eye_scale,
                              fill="#17313a", outline="")
                c.create_oval(ex - 2, eye_y - 5, ex + 1, eye_y - 2,
                              fill="#ffffff", outline="")
        if sleeping:
            mouth_y = cy + 12
            c.create_arc(cx - 13, mouth_y - 3, cx + 13, mouth_y + 13,
                         start=190, extent=160, style=tk.ARC,
                         outline="#264b50", width=2)
        else:
            c.create_arc(cx - 13, cy - 2, cx + 13, cy + 15,
                         start=200, extent=140, style=tk.ARC,
                         outline="#264b50", width=2)
        if dancing:
            c.create_text(cx + 98, cy - 65, text="♪", fill="#ffbe63",
                          font=("TkDefaultFont", 22, "bold"))
            c.create_text(cx - 98, cy - 47, text="✦", fill="#ffbe63",
                          font=("TkDefaultFont", 15, "bold"))
        elif self.pose == "timer":
            c.create_text(cx + 101, cy - 55, text="◷", fill="#ffbe63",
                          font=("TkDefaultFont", 24, "bold"))
