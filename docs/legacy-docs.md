# Legacy documents

Before this site existed, BlueSky's documentation consisted of a mix of the
GitHub wiki and a folder of PDFs, Word documents, and a spreadsheet. Those
files haven't been deleted — they've been moved to `docs/legacy/` in the
repository, since some of them (design notes, validation reports) contain
detail that hasn't yet been migrated into this site.

They're excluded from the built HTML on this site (they're large binary
files, and mostly superseded by the pages linked from each entry below), but
remain browsable directly on GitHub:

| File | Superseded by / see instead |
|------|------------------------------|
| [`BLUESKY-COMMAND-TABLE.TXT`](https://github.com/jangroter/bluesky/blob/read-the-docs/docs/legacy/BLUESKY-COMMAND-TABLE.TXT), `.pdf`, `.xlsx` | [Stack command reference](reference/commands/index.md) (auto-generated from the code, always current) |
| [`BlueSky_Command_Reference.doc`](https://github.com/jangroter/bluesky/blob/read-the-docs/docs/legacy/BlueSky_Command_Reference.doc), [`BlueSky-Command-Reference-in-3-pages.pdf`](https://github.com/jangroter/bluesky/blob/read-the-docs/docs/legacy/BlueSky-Command-Reference-in-3-pages.pdf) | [Stack command reference](reference/commands/index.md) |
| [`Routes and FMS Guidance in BlueSky.pdf`/`.docx`](https://github.com/jangroter/bluesky/blob/read-the-docs/docs/legacy/Routes%20and%20FMS%20Guidance%20in%20BlueSky.pdf) | [Autopilot, FMS and routes](concepts/autopilot-fms.md) |
| [`Aircraft Performance in BlueSky_manual.pdf`](https://github.com/jangroter/bluesky/blob/read-the-docs/docs/legacy/Aircraft%20Performance%20in%20BlueSky_manual.pdf) | [Aircraft performance models](concepts/performance-models.md) |
| [`ASAS-CD&R-info.pdf`](https://github.com/jangroter/bluesky/blob/read-the-docs/docs/legacy/ASAS-CD%26R-info.pdf) | [Conflict detection and resolution](concepts/conflict-detection-resolution.md) |
| [`Guidance-Lay-out-BlueSky.pdf`](https://github.com/jangroter/bluesky/blob/read-the-docs/docs/legacy/Guidance-Lay-out-BlueSky.pdf), [`Software Documents/`](https://github.com/jangroter/bluesky/tree/read-the-docs/docs/legacy/Software%20Documents) | [Architecture overview](concepts/architecture.md), [The traffic model](concepts/traffic-model.md) |
| [`BlueSky Paper.pdf`](https://github.com/jangroter/bluesky/blob/read-the-docs/docs/legacy/BlueSky%20Paper.pdf) | [Citing BlueSky](citation.md) |
| [`python_demo.ipynb`](https://github.com/jangroter/bluesky/blob/read-the-docs/docs/legacy/python_demo.ipynb) | [Using BlueSky as a Python library](api/index.md) |
| [`NLR TMX Reference Manual`](https://github.com/jangroter/bluesky/blob/read-the-docs/docs/legacy/NLR%20TMX%20Reference%20Manual%20version%202016-11-03.pdf), [`XP-APT1000-Spec.pdf`](https://github.com/jangroter/bluesky/blob/read-the-docs/docs/legacy/XP-APT1000-Spec.pdf), [`BlueSky-QTGL_Supported_Systems.txt`](https://github.com/jangroter/bluesky/blob/read-the-docs/docs/legacy/BlueSky-QTGL_Supported_Systems.txt) | Reference material without a direct modern equivalent yet |
| [`workshop_programme.png`](https://github.com/jangroter/bluesky/blob/read-the-docs/docs/legacy/workshop_programme.png) | Historical workshop material |

```{note}
These links point at the `jangroter/bluesky` fork's `read-the-docs` branch.
If this documentation is merged upstream, update them to point at
`TUDelft-CNS-ATM/bluesky` instead.
```
