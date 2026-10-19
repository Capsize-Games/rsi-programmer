# Sound Companion validation

The native microphone teaching demo supports Wake, Sleep, Dance and a
one-minute timer. Its backend stores labeled feature examples and uses
nearest-example matching. It does not use an SNN or an LLM.

## Completed local checks

- 53 affected checks passed in 1.635 seconds, including real Tk controls
  on an isolated Xvfb display, fake microphone transitions, independent
  unknown-sound rejection, correction routing and persistence.
- Three shortcut checks passed separately in 0.060 seconds. The escaping
  round trip uses the system GLib parser. `desktop-file-validate` accepted
  the installed application-menu entry.
- A finite 1.6-second check of the physical default microphone measured
  -24.223 dBFS RMS and no clipped samples. This establishes capture, not
  recognition accuracy. PCM was discarded.
- The rendered 1000 by 800 native window was inspected for clipped controls
  and unreadable states. Its screenshot has the microphone off.

All audio/model/test checks ran serially under the shared builder lock.
No full suite or remote CI was awaited. The 56-check combined attempt had
an installer-test parser failure; the 53 app checks passed in that attempt.
The installer test was corrected and its three checks passed separately.
The Python research environment remains unchanged; the desktop uses a
separate environment with system Tk because the standalone Tk build
aborted when creating entry widgets on this machine.

## Generated learn/add/reload check

The development check teaches three generated sound families with three
examples each. It queries one different seeded example per family, adds
one example of a fourth family, and queries all four before and after
reloading memory from disk.

| Stage | Correct | Wrong label | Rejected |
| --- | ---: | ---: | ---: |
| Three taught families | 3/3 | 0 | 0 |
| After adding fourth family | 4/4 | 0 | 0 |
| After memory reload | 4/4 | 0 | 0 |

The original three queries are identical across stages. Their stored
example fingerprints remain unchanged when the fourth family is added.
Silence and generated noise are rejected. Test queries are never added to
training memory and never invoke actions.

These queries informed development, so this is a regression demonstration,
not an independent accuracy benchmark. It does not establish recognition
of a person's sounds, open-world rejection, an SNN advantage or chip energy.
The app starts with an empty personal memory; its test panel is for fresh
microphone examples after the owner teaches sounds.

## Failures retained

The initial generated check recognized 1/3 families and then 3/4 after
adding the fourth. It accidentally changed the original query seeds
between those two stages, so it cannot establish retention. Its report is
retained unchanged. The corrected same-query check after timing alignment
recognized 1/3 and 2/4; no wrong labels occurred.

Adding similar positive examples had narrowed acceptance below the
single-example default. The final matcher preserves that existing minimum
tolerance and uses alignment limited to one temporal bin. A separate
energy-weighting attempt caused two unknown/rhythm rejection failures and
was removed. The rejection tests were kept intact.

The first shortcut test incorrectly used Python `shlex` to model desktop
escaping; it was replaced by the actual system GLib parser. Earlier Tk
aborts and the missing Spikeforge environment configuration are also
preserved in the evidence directory.

Receipts, source hashes, compressed logs, generated aggregate reports and
screenshots are in
[`artifacts/sound-creature-validation-20261018/`](
../artifacts/sound-creature-validation-20261018/).
The final generated report is `synthetic-coverage-corrected-report.json`;
the final app receipt is `desktop-coverage-corrected.json`, and the final
shortcut receipt is `installer-corrected.json`. No raw microphone audio or
personal sound memory is committed.
