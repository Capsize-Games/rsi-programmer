"""Native Tk interface for teaching and testing sound actions."""
from __future__ import annotations

from collections.abc import Callable
import tkinter as tk
from tkinter import ttk
from typing import Any

from rsi_programmer.sound_creature import SoundCreature


class SoundWindow:
    """Build the sound companion interface and expose controller updates."""

    ACTIONS = ("Wake", "Sleep", "Dance", "Start timer")
    BG = "#101820"
    PANEL = "#182630"
    TEXT = "#e8f0f3"
    MUTED = "#9db0b8"
    CYAN = "#55d6c2"
    AMBER = "#ffbe63"

    def __init__(self, root: tk.Tk,
                 callbacks: dict[str, Callable[..., Any]]) -> None:
        self.root = root
        self.callbacks = callbacks
        self._sounds: list[dict[str, Any]] = []
        self._selected_label = ""
        self._busy = False
        self._connected = False
        self._listening = False
        self._test_count = 0
        self._test_matches = 0
        self._can_correct = False
        self._memory_revision: Any = 0
        self._tests_by_label: dict[str, list[int]] = {}
        self._comparison = {
            "snn": {"correct": 0, "total": 0},
            "matcher": {"correct": 0, "total": 0},
        }
        self._build_style()
        self._build_window()
        self._refresh_controls()
        self._animate()

    def _build_style(self) -> None:
        self.root.configure(bg=self.BG)
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background=self.BG)
        style.configure("Panel.TFrame", background=self.PANEL)
        style.configure("TLabel", background=self.BG, foreground=self.TEXT,
                        font=("TkDefaultFont", 10))
        style.configure("Panel.TLabel", background=self.PANEL,
                        foreground=self.TEXT)
        style.configure("Muted.TLabel", background=self.PANEL,
                        foreground=self.MUTED)
        style.configure("Title.TLabel", font=("TkDefaultFont", 21, "bold"))
        style.configure("Section.TLabel", background=self.PANEL,
                        foreground=self.TEXT,
                        font=("TkDefaultFont", 12, "bold"))
        style.configure("Accent.TButton", background=self.CYAN,
                        foreground="#102027", padding=(12, 8),
                        font=("TkDefaultFont", 10, "bold"))
        style.map("Accent.TButton",
                  background=[("disabled", "#263740"),
                              ("active", "#7ce9db")],
                  foreground=[("disabled", "#82969e"),
                              ("!disabled", "#102027")])
        style.configure("TButton", padding=(9, 6))
        style.map("TButton", background=[("disabled", "#263740"),
                                          ("active", "#29434b")],
                  foreground=[("disabled", "#82969e"),
                              ("!disabled", self.TEXT)])
        style.configure("TEntry", fieldbackground="#0f1921",
                        foreground=self.TEXT, insertcolor=self.TEXT)
        style.configure("TCombobox", fieldbackground="#0f1921",
                        foreground=self.TEXT, arrowcolor=self.CYAN)
        style.map("TCombobox",
                  fieldbackground=[("disabled", self.PANEL),
                                   ("readonly", "#0f1921")],
                  foreground=[("disabled", self.MUTED),
                              ("readonly", self.TEXT)],
                  selectbackground=[("readonly", "#0f1921")],
                  selectforeground=[("readonly", self.TEXT)])
        style.configure("Treeview", background="#111c25",
                        fieldbackground="#111c25", foreground=self.TEXT,
                        rowheight=27, borderwidth=0)
        style.configure("Treeview.Heading", background=self.PANEL,
                        foreground=self.MUTED, relief="flat")
        style.map("Treeview", background=[("selected", "#24534f")])

    def _build_window(self) -> None:
        self.root.title("Sound Companion")
        self.root.geometry("1000x800")
        self.root.minsize(760, 800)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        header = ttk.Frame(self.root, padding=(24, 18, 24, 10))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Sound Companion",
                  style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="Teach sounds, then check what it remembers.",
                  foreground=self.MUTED).grid(row=1, column=0, sticky="w",
                                              pady=(4, 0))
        self.backend_var = tk.StringVar(
            value="Example matcher controls the creature")
        ttk.Label(header, textvariable=self.backend_var,
                  foreground=self.CYAN).grid(row=2, column=0, sticky="w",
                                             pady=(4, 0))
        self.status_var = tk.StringVar(value="Microphone is off")
        ttk.Label(header, textvariable=self.status_var,
                  foreground=self.CYAN, wraplength=300).grid(
                      row=0, column=1, rowspan=2, sticky="e",
                      padx=(20, 0))
        body = ttk.Frame(self.root, padding=(24, 10, 24, 22))
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=3, minsize=340)
        body.columnconfigure(1, weight=2, minsize=300)
        body.rowconfigure(0, weight=1)
        left = ttk.Frame(body)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=3)
        left.rowconfigure(1, weight=2)
        self._build_creature_panel(left)
        self._build_learning_panel(left)
        self._build_right_panel(body)

    def _panel(self, parent: ttk.Frame, row: int, title: str,
               padding: int = 16) -> ttk.Frame:
        panel = ttk.Frame(parent, style="Panel.TFrame", padding=padding)
        panel.grid(row=row, column=0, sticky="nsew", pady=(0, 12))
        panel.columnconfigure(0, weight=1)
        ttk.Label(panel, text=title, style="Section.TLabel").grid(
            row=0, column=0, sticky="w")
        return panel

    def _build_creature_panel(self, parent: ttk.Frame) -> None:
        panel = self._panel(parent, 0, "Companion")
        panel.rowconfigure(1, weight=1)
        self.creature_canvas = tk.Canvas(panel, bg=self.PANEL,
                                         highlightthickness=0, height=225)
        self.creature_canvas.grid(row=1, column=0, sticky="nsew", pady=4)
        self.creature = SoundCreature(self.creature_canvas)
        footer = ttk.Frame(panel, style="Panel.TFrame")
        footer.grid(row=2, column=0, sticky="ew")
        footer.columnconfigure(1, weight=1)
        self.listening_var = tk.StringVar(value="Paused")
        ttk.Label(footer, text="Listening", style="Muted.TLabel").grid(
            row=0, column=0, sticky="w")
        ttk.Label(footer, textvariable=self.listening_var,
                  style="Panel.TLabel").grid(row=0, column=1, sticky="w",
                                             padx=(12, 0))
        self.toggle_button = ttk.Button(footer, text="Enable microphone",
                                        style="Accent.TButton",
                                        command=self._toggle_listening)
        self.toggle_button.grid(row=0, column=2, sticky="e")

    def _build_learning_panel(self, parent: ttk.Frame) -> None:
        panel = self._panel(parent, 1, "Teach a sound")
        panel.columnconfigure(0, weight=1)
        ttk.Label(panel, text="Sound name", style="Muted.TLabel").grid(
            row=1, column=0, sticky="w", pady=(10, 4))
        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(panel, textvariable=self.name_var)
        self.name_entry.grid(row=2, column=0, sticky="ew")
        ttk.Label(panel, text="When heard, do", style="Muted.TLabel").grid(
            row=3, column=0, sticky="w", pady=(10, 4))
        self.action_var = tk.StringVar(value=self.ACTIONS[0])
        self.action_box = ttk.Combobox(panel, textvariable=self.action_var,
                                       values=self.ACTIONS, state="readonly")
        self.action_box.grid(row=4, column=0, sticky="ew")
        self.teach_button = ttk.Button(panel, text="Record example",
                                       style="Accent.TButton",
                                       command=self._teach)
        self.teach_button.grid(row=5, column=0, sticky="ew", pady=(12, 0))
        ttk.Label(panel, text="Record 3 examples, then test a new one.",
                  style="Muted.TLabel").grid(row=6, column=0, sticky="w",
                                             pady=(8, 0))

    def _build_right_panel(self, parent: ttk.Frame) -> None:
        right = ttk.Frame(parent)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=0)
        self._build_memory_panel(right)
        self._build_test_panel(right)

    def _build_memory_panel(self, parent: ttk.Frame) -> None:
        panel = self._panel(parent, 0, "Sound memory", padding=12)
        panel.rowconfigure(1, weight=1)
        panel.columnconfigure(0, weight=1)
        columns = ("sound", "action", "examples")
        tree_frame = ttk.Frame(panel, style="Panel.TFrame")
        tree_frame.grid(row=1, column=0, sticky="nsew", pady=(8, 5))
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)
        self.memory_tree = ttk.Treeview(
            tree_frame, columns=columns, show="headings",
                                    selectmode="browse", height=4,
                                    yscrollcommand=self._set_memory_scroll)
        self.memory_tree.heading("sound", text="Sound")
        self.memory_tree.heading("action", text="Action")
        self.memory_tree.heading("examples", text="Count")
        self.memory_tree.column("sound", width=100, minwidth=65,
                                anchor="w", stretch=True)
        self.memory_tree.column("action", width=85, minwidth=60,
                                anchor="w", stretch=True)
        self.memory_tree.column("examples", width=60, minwidth=54,
                                anchor="center", stretch=False)
        self.memory_tree.grid(row=0, column=0, sticky="nsew")
        self.memory_scrollbar = ttk.Scrollbar(
            tree_frame, orient="vertical", command=self.memory_tree.yview)
        self.memory_scrollbar.grid(row=0, column=1, sticky="ns")
        self.memory_tree.bind("<<TreeviewSelect>>", self._selection_changed)
        self.memory_count_var = tk.StringVar(value="0 sounds · revision 0")
        ttk.Label(panel, textvariable=self.memory_count_var,
                  style="Muted.TLabel").grid(row=2, column=0, sticky="w")
        actions = ttk.Frame(panel, style="Panel.TFrame")
        actions.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        actions.columnconfigure((0, 1), weight=1)
        self.correct_button = ttk.Button(actions, text="Correct last sound",
                                         command=self._correct)
        self.correct_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.delete_button = ttk.Button(actions, text="Forget sound",
                                        command=self._delete)
        self.delete_button.grid(row=0, column=1, sticky="ew", padx=(5, 0))

    def _build_test_panel(self, parent: ttk.Frame) -> None:
        panel = self._panel(parent, 1, "Test recognition", padding=12)
        ttk.Label(panel, text="Expected sound", style="Muted.TLabel").grid(
            row=1, column=0, sticky="w", pady=(7, 3))
        self.expected_var = tk.StringVar()
        self.expected_box = ttk.Combobox(panel, textvariable=self.expected_var,
                                         state="readonly")
        self.expected_box.grid(row=2, column=0, columnspan=2, sticky="ew")
        self.test_button = ttk.Button(panel, text="Record test sound",
                                      command=self._test)
        self.test_button.grid(row=3, column=0, columnspan=2,
                              sticky="ew", pady=(6, 6))
        self.latest_var = tk.StringVar(value="No sound recorded yet")
        ttk.Label(panel, textvariable=self.latest_var, style="Panel.TLabel",
                  wraplength=300).grid(row=4, column=0, columnspan=2,
                                       sticky="w")
        self.level_var = tk.StringVar(value="Input level: —")
        ttk.Label(panel, textvariable=self.level_var,
                  style="Muted.TLabel").grid(row=5, column=0, sticky="w",
                                             pady=(5, 0))
        self.timer_var = tk.StringVar(value="Timer: —")
        ttk.Label(panel, textvariable=self.timer_var,
                  style="Panel.TLabel").grid(row=5, column=1, sticky="e",
                                              pady=(5, 0))
        self.feedback_var = tk.StringVar(value="")
        ttk.Label(panel, textvariable=self.feedback_var,
                  style="Muted.TLabel", wraplength=300).grid(
                      row=6, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self.test_history_var = tk.StringVar(
            value="Session: 0 tests · 0 correct")
        ttk.Label(panel, textvariable=self.test_history_var,
                  style="Muted.TLabel").grid(row=7, column=0, columnspan=2,
                                              sticky="w", pady=(4, 0))
        self.comparison_var = tk.StringVar(
            value="Probes · SNN 0/0 · Example matcher 0/0")
        ttk.Label(panel, textvariable=self.comparison_var,
                  style="Panel.TLabel", wraplength=300).grid(
                      row=8, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.learning_var = tk.StringVar(value="SNN parameters: unavailable")
        ttk.Label(panel, textvariable=self.learning_var,
                  style="Muted.TLabel", wraplength=330).grid(
                      row=9, column=0, columnspan=2, sticky="w", pady=(3, 0))

        def wrap_labels(event: tk.Event) -> None:
            for widget in panel.winfo_children():
                if not isinstance(widget, ttk.Label):
                    continue
                length = widget.cget("wraplength")
                if str(length) and widget.winfo_pixels(length):
                    widget.configure(wraplength=max(120, event.width - 24))

        panel.bind("<Configure>", wrap_labels)

    def _set_memory_scroll(self, first: str, last: str) -> None:
        """Keep the memory list's scrollbar synchronized with its rows."""
        if hasattr(self, "memory_scrollbar"):
            self.memory_scrollbar.set(first, last)

    def _callback(self, name: str, *args: Any) -> None:
        callback = self.callbacks.get(name)
        if callback is not None:
            callback(*args)

    def _toggle_listening(self) -> None:
        if self._busy:
            return
        if not self._connected:
            self._callback("connect")
        else:
            self._callback("toggle_listening")

    def _teach(self) -> None:
        label = self.name_var.get().strip()
        if label and self._connected and not self._busy:
            self._callback("teach", label, self.action_var.get())

    def _test(self) -> None:
        label = self.expected_var.get().strip()
        if label and self._connected and not self._busy:
            self.feedback_var.set("Test capture is separate from listening.")
            self._callback("test", label)

    def _selection_changed(self, _event: tk.Event[Any]) -> None:
        selected = self.memory_tree.selection()
        self._selected_label = selected[0] if selected else ""
        self._refresh_controls()

    def _correct(self) -> None:
        if self._selected_label and self._can_correct and not self._busy:
            self._callback("correction", self._selected_label)

    def _delete(self) -> None:
        if self._selected_label and not self._busy:
            self._callback("delete", self._selected_label)

    def _refresh_controls(self) -> None:
        state = "disabled" if self._busy else "normal"
        self.toggle_button.configure(
            state=state,
            text=("Pause listening" if self._listening else
                  "Start listening" if self._connected else
                  "Enable microphone"))
        capture_state = state if self._connected else "disabled"
        self.teach_button.configure(state=capture_state)
        self.test_button.configure(state=capture_state)
        self.correct_button.configure(
            state=(state if self._selected_label and self._can_correct
                   else "disabled"))
        self.delete_button.configure(
            state=state if self._selected_label else "disabled")

    def _animate(self) -> None:
        self.creature.animate()
        self.root.after(130, self._animate)

    def set_status(self, text: str) -> None:
        """Show a controller-provided microphone or operation status."""
        self.status_var.set(text)

    def set_level(self, db: float) -> None:
        """Show measured input level."""
        self.level_var.set(f"Input level: {db:.1f} dB")

    def set_connected(self, connected: bool) -> None:
        """Update microphone availability after the controller connects."""
        self._connected = connected
        if not connected:
            self._listening = False
        self._refresh_controls()

    def set_can_correct(self, can_correct: bool) -> None:
        """Enable correction when the controller retains a recent clip."""
        self._can_correct = can_correct
        self._refresh_controls()

    def set_listening(self, listening: bool) -> None:
        """Update the continuous recognition state shown in the window."""
        self._listening = listening and self._connected
        self.listening_var.set("Listening" if self._listening else "Paused")
        if self._listening:
            self.creature.set_pose("attentive")
        self._refresh_controls()

    def set_busy(self, busy: bool) -> None:
        """Disable capture controls while a controller operation is active."""
        self._busy = busy
        self._refresh_controls()

    def set_memory(self, summary: dict[str, Any]) -> None:
        """Replace the displayed sound memory with a controller snapshot."""
        backend = summary.get("backend")
        learning = summary.get("learning", {})
        if summary.get("load_error"):
            backend_text = "SNN unavailable; saved memory needs repair"
        elif backend == "trained_snn" and isinstance(learning, dict):
            backend_text = (
                "SNN controls the creature" if learning.get("trained")
                else "SNN ready; teach a sound")
        elif backend == "trained_snn":
            backend_text = "SNN ready; teach a sound"
        else:
            backend_text = "Example matcher controls the creature"
        self.backend_var.set(backend_text)
        comparison = summary.get("comparison")
        if isinstance(comparison, dict):
            for key in ("snn", "matcher"):
                counts = comparison.get(key)
                if isinstance(counts, dict):
                    self._comparison[key] = {
                        "correct": int(counts.get("correct", 0)),
                        "total": int(counts.get("total", 0)),
                    }
            self._refresh_comparison()
        if isinstance(learning, dict):
            fingerprint = (learning.get("model_hash")
                           or learning.get("parameter_hash"))
            if summary.get("load_error"):
                detail = "SNN parameters unavailable"
            elif not learning.get("trained"):
                detail = "SNN parameters: not trained"
            elif fingerprint:
                detail = f"SNN parameters {str(fingerprint)[:10]}"
            else:
                detail = "SNN parameters: unavailable"
            metrics = []
            if learning.get("updates") is not None:
                metrics.append(f"{learning['updates']} updates total")
            if metrics:
                detail += " · " + " · ".join(metrics)
            self.learning_var.set(detail)
        raw_sounds = summary.get("sounds", summary.get("labels", []))
        self._sounds = []
        for sound in raw_sounds:
            item = dict(sound)
            item["label"] = item.get("label", item.get("name", ""))
            self._sounds.append(item)
        for item in self.memory_tree.get_children():
            self.memory_tree.delete(item)
        for sound in self._sounds:
            label = str(sound.get("label", ""))
            action = str(sound.get("action", ""))
            count = sound.get("examples", sound.get("example_count", 0))
            self.memory_tree.insert("", "end", iid=label,
                                    values=(label, action, str(count)))
        revision = summary.get("revision", 0)
        self._memory_revision = revision
        self.memory_count_var.set(
            f"{len(self._sounds)} sounds · revision {revision} · "
            "saved locally")
        labels = [str(sound.get("label", "")) for sound in self._sounds]
        self.expected_box.configure(values=labels)
        if self.expected_var.get() not in labels:
            self.expected_var.set(labels[0] if labels else "")
        if self._selected_label not in labels:
            self._selected_label = ""
        if not self._sounds:
            if summary.get("load_error"):
                self.feedback_var.set("")
            else:
                self.feedback_var.set(
                    "Teach a sound to begin building memory.")
        elif self.feedback_var.get() == (
                "Teach a sound to begin building memory."):
            self.feedback_var.set("")
        self._refresh_controls()

    def set_recognition(self, result: dict[str, Any]) -> None:
        """Display a live recognition event and its action, if any."""
        raw_label = result.get("label")
        label = str(raw_label) if raw_label else "Unknown"
        action = str(result.get("action", ""))
        message = str(result.get("message", result.get("reason", "")))
        self.latest_var.set(
            f"Heard: {label}" + (f" · {action}" if action else ""))
        self.feedback_var.set(message)
        self._refresh_controls()

    def set_test_result(self, result: dict[str, Any]) -> None:
        """Report an expected-versus-recognized test result plainly."""
        expected = str(result.get("expected", ""))
        raw_recognized = result.get("recognized")
        recognized = (str(raw_recognized) if raw_recognized else "Unknown")
        correct = bool(result.get("correct", recognized == expected))
        message = str(result.get("message", ""))
        self._test_count += 1
        label_counts = self._tests_by_label.setdefault(expected, [0, 0])
        label_counts[0] += 1
        if correct:
            self._test_matches += 1
            label_counts[1] += 1
        snn_counts = self._comparison["snn"]
        snn_counts["total"] += 1
        snn_counts["correct"] += int(correct)
        baseline_correct = result.get("baseline_correct")
        if baseline_correct is not None:
            matcher_counts = self._comparison["matcher"]
            matcher_counts["total"] += 1
            matcher_counts["correct"] += int(bool(baseline_correct))
        self._refresh_comparison()
        self.test_history_var.set(
            f"Session: {self._test_count} tests · "
            f"{self._test_matches} correct")
        verdict = "Correct" if correct else "Incorrect"
        revision = result.get("memory_revision", self._memory_revision)
        self.latest_var.set(
            f"Test · heard {recognized} (expected {expected}) · "
            f"rev {revision}")
        self.feedback_var.set(
            message or f"{verdict} · {expected} {label_counts[1]}/"
            f"{label_counts[0]}")

    def _refresh_comparison(self) -> None:
        """Show cumulative held-out correctness for both recognizers."""
        snn = self._comparison["snn"]
        matcher = self._comparison["matcher"]
        self.comparison_var.set(
            f"Probes · SNN {snn['correct']}/{snn['total']} · "
            f"Example matcher {matcher['correct']}/{matcher['total']}")

    def perform_action(self, action: str) -> None:
        """Reflect an action dispatched by the controller."""
        self.creature.set_pose(self._pose_for_action(action))

    def set_timer(self, seconds: int | None) -> None:
        """Show the timer countdown value reported by the controller."""
        self.timer_var.set("Timer: —" if seconds is None else
                           f"Timer: {max(0, seconds)} s")

    @staticmethod
    def _pose_for_action(action: str) -> str:
        normalized = action.casefold()
        if normalized in {"wake", "waking"}:
            return "wake"
        if normalized == "sleep":
            return "sleep"
        if normalized == "dance":
            return "dance"
        if normalized in {"start timer", "start_timer", "timer"}:
            return "timer"
        return "attentive"
