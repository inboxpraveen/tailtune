# 05. Data ingestion

**What this document decides:**

- One function, `tailtune.data.read`, sits behind the Python API, the CLI and notebooks. It runs eight stages: `sniff → read → normalize → detect → map → render → validate → freeze` (D-21).
- v0.1 reads everything Unsloth documents for structured training data, plus `.xlsx .xlsm .xls .xlsb .ods` as real tables. Documents, remote storage, streaming and media come in later tiers. A few Unsloth features are deliberately left out, and they are named in section 2.
- Spreadsheets are read with python-calamine for values and openpyxl for structure. Every cell becomes text, by fixed rules. Nothing is guessed silently: every change made to the user's file is reported.
- Format detection uses ordered schema rules, then alias lists, and gives a confidence score. When the mapping is ambiguous and nobody is there to confirm it, the run stops with exit code 2.
- A `mapping.yaml` is always written, can be reused, and is never required.
- Rendering uses the model's own chat template, with no second BOS. A template hash is recorded, and [10](10-export-and-deployment.md) verifies it at export and at load.
- Validation runs by default. Double BOS, missing EOS and all-masked labels are blocking, with exit code 2.

Evidence labels are defined in the [plan index](README.md#evidence-labels). All code in this document is an untested design proposal. All example output is illustrative.

Terms used throughout:

- **BOS and EOS:** the special tokens that mark the beginning and end of a sequence.
- **Chat template:** the Jinja program shipped with a tokenizer that turns a list of messages into one string with the model's role markers.
- **Label masking:** setting a token's label to −100 so it does not count in the loss. Usually everything except the assistant's reply is masked.
- **Arrow and Parquet:** a columnar in-memory format, and its on-disk counterpart.
- **Magic bytes:** the fixed bytes at the start of a file that identify its real format.

## 1. The brief, corrected

The owner asked for training to start from "direct Excels or HF dataset or whatever they have in Unsloth, same way". The research found that Unsloth's Excel support is weaker than commonly assumed.

| Common assumption | What the research found |
|---|---|
| Unsloth trains directly from Excel | Its docs send "CSV or Excel" users to a Google Drive Colab that could not be read, so its method is unknown [S](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama.md) |
| Unsloth Studio's training upload takes xlsx | It accepts `.csv .jsonl .json .parquet` only [S](https://github.com/unslothai/unsloth/blob/5f1d55556fd82f889abfe17a446c3b3a97a2f42b/studio/frontend/src/features/training/lib/native-dataset-drop.ts). The backend adds `.tsv` and `.arrow` [S](https://github.com/unslothai/unsloth/blob/5f1d55556fd82f889abfe17a446c3b3a97a2f42b/studio/backend/hub/services/datasets/local.py) |
| Studio reads xlsx as a table | xlsx appears only among unstructured documents for Data Recipes [S](https://github.com/unslothai/unsloth/blob/5f1d55556fd82f889abfe17a446c3b3a97a2f42b/studio/backend/routes/data_recipe/seed.py). Each sheet is flattened to `a \| b \| c` text and chunked [S](https://github.com/unslothai/unsloth/blob/5f1d55556fd82f889abfe17a446c3b3a97a2f42b/studio/backend/core/rag/office_formats.py). That destroys the column-to-role mapping that training needs |
| Unsloth ships an Excel library | No Excel library is declared in its dependencies [S](https://github.com/unslothai/unsloth/blob/5f1d55556fd82f889abfe17a446c3b3a97a2f42b/pyproject.toml) |

Competitors are no better. Oumi reads xlsx but concatenates all sheets [S](https://github.com/oumi-ai/oumi/blob/4c0aa76443dda98537a0886f82ed6f12b5c32440/src/oumi/utils/io_utils.py). Hugging Face `datasets` has no spreadsheet module [S](https://github.com/huggingface/datasets/blob/a1269ca9dbb6e0858a9c375e8982329a3fcc84af/src/datasets/packaged_modules/__init__.py). LlamaFactory, TRL and Axolotl have no reader. Training Hub's spreadsheet example converts xlsx to Markdown for continued pretraining, which is not table-to-Q&A [S](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/1eb3d9b2034362883d7ecf9d1e69dab99c315a61/examples/scripts/osft_cpt_spreadsheet_example.py).

**Demand is unproven.** GitHub shows 7 issue hits for "excel" in Unsloth, few of them about training [S](https://github.com/unslothai/unsloth/issues?q=is%3Aissue+excel). Nobody has surveyed this.

**Why it is still worth doing** [J]:

- It is cheap. The libraries exist, and the hard part is a set of rules this document already specifies.
- The naive path is actively harmful. Several of its failures are silent (section 6), so tailtune would be worse than a hand-exported CSV without these rules. The differentiator is not "reads Excel" but "tells you exactly what it did to your Excel".
- The same pipeline serves every other input. The spreadsheet work is mostly the read and normalize stages.
- The lead is temporary. Studio already parses xlsx and has heuristic column mapping, so Unsloth could add a table path quickly. Ship it early, and pair it with the retention report, which is harder to copy ([01](01-product-strategy.md)).

## 2. Input surface

### By release tier

Tier assignments after v0.1 are a judgement [J].

| Release | Inputs | Notes |
|---|---|---|
| v0.1 (Gate 2) | Hub ID with `config`, `split` and `revision` | Revision resolved to a commit SHA and recorded |
| v0.1 | Local `.csv .tsv .json .jsonl .parquet .arrow` | Detected by content, not extension |
| v0.1 | `.xlsx .xlsm .xls .xlsb .ods` as tables | Section 6 |
| v0.1 | `.txt .md` as raw text, one document per file | For continued pretraining, which ships at launch (D-16). No extra dependency |
| v0.1 | A directory of any of the above, or a list of paths | Combined only when their mapped schemas match |
| v0.1 | In memory: `datasets.Dataset`, `DatasetDict`, pandas or polars `DataFrame`, `list[dict]` | DataFrames go through the same cell-to-text rules |
| v0.2 (Gate 3) | `hf:// s3:// gs://` and other fsspec URLs | fsspec, s3fs and gcsfs are BSD [S](https://pypi.org/pypi/fsspec/json). `az://` follows if asked for |
| v0.2 | Streaming JSONL and Parquet | Section 14 |
| v0.3 (Gate 4) | `.pdf .docx .html` through the `tailtune[docs]` extra | MarkItDown by default, Docling if installed. Raw text for continued pretraining only |
| v0.3 | ModelScope, as an optional extra | Matches Unsloth's legacy CLI |
| After v1.0 | Vision and audio columns (path, URL or bytes) | Needs multimodal training, which is out of scope at launch |
| Not planned | Synthetic data generation | Connect to Data Designer (Apache-2.0) [S](https://pypi.org/pypi/data-designer/json) instead |

### Coverage against Unsloth

The verdict: v0.1 matches every structured format and every chat schema Unsloth documents or detects. Documents, remote sources and media arrive later. Four items are deliberately not covered. So "covers everything" holds for training data in tables and JSON, not for every file type Studio accepts.

| Unsloth feature | Evidence | tailtune |
|---|---|---|
| Hub dataset | [Datasets Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide.md) | v0.1 |
| ModelScope (`UNSLOTH_USE_MODELSCOPE`) | [unsloth-cli.py](https://github.com/unslothai/unsloth/blob/5f1d55556fd82f889abfe17a446c3b3a97a2f42b/unsloth-cli.py) | v0.3, optional |
| CSV, JSON, JSONL, Parquet, TSV, Arrow | [local.py](https://github.com/unslothai/unsloth/blob/5f1d55556fd82f889abfe17a446c3b3a97a2f42b/studio/backend/hub/services/datasets/local.py) | v0.1 |
| Raw text corpus; `RawTextDataLoader` for text, Markdown, JSON lines and CSV text columns | [raw_text.py](https://github.com/unslothai/unsloth/blob/5f1d55556fd82f889abfe17a446c3b3a97a2f42b/unsloth/dataprep/raw_text.py) | v0.1 |
| Alpaca, ShareGPT, ChatML `messages`, preference pairs | [Datasets Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide.md) | v0.1 detection. Preference training waits for DPO (D-16) |
| Prompt and completion pairs, detected in Studio | [format_detection.py](https://github.com/unslothai/unsloth/blob/5f1d55556fd82f889abfe17a446c3b3a97a2f42b/studio/backend/utils/datasets/format_detection.py) | v0.1 |
| `format_type`: auto, alpaca, chatml, sharegpt, conversational, raw | [config.py](https://github.com/unslothai/unsloth/blob/5f1d55556fd82f889abfe17a446c3b3a97a2f42b/unsloth_cli/config.py) | v0.1, as `format=` |
| Multi-column merge with `{col}` and `[[...]]` (`to_sharegpt`) | [chat_templates.py](https://github.com/unslothai/unsloth/blob/5f1d55556fd82f889abfe17a446c3b3a97a2f42b/unsloth/chat_templates.py) | v0.1, section 9 |
| Role aliases (`standardize_data_formats`) | [dataset_utils.py](https://github.com/unslothai/unsloth-zoo/blob/18dd2236add5a80dd79ccc8390dc9f49df37e5e5/unsloth_zoo/dataset_utils.py) | v0.1, reimplemented (LGPL source) |
| Heuristic column mapping | [format_detection.py](https://github.com/unslothai/unsloth/blob/5f1d55556fd82f889abfe17a446c3b3a97a2f42b/studio/backend/utils/datasets/format_detection.py) | v0.1, from this spec (AGPL source) |
| Multiple datasets | [Datasets Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide.md) | v0.1 for matching schemas. Weighted mixing later |
| Excel | Colab (unread) and Studio as a document | v0.1 as a real table |
| PDF, DOCX, HTML | [seed.py](https://github.com/unslothai/unsloth/blob/5f1d55556fd82f889abfe17a446c3b3a97a2f42b/studio/backend/routes/data_recipe/seed.py) | v0.3 through `[docs]` |
| PPTX, EPUB, ODT, EML, MSG | Same | Only if the chosen converter handles them [U] |
| Vision messages; audio rows | [Datasets Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide.md), [TTS guide](https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning.md) | After v1.0 |
| `conversation_extension`, which merges random single-turn rows into one conversation | [Datasets Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide.md) | **Not planned** [J]: it invents conversations that never happened |
| Synthetic data: `SyntheticDataKit`, Data Recipes | [synthetic.py](https://github.com/unslothai/unsloth/blob/5f1d55556fd82f889abfe17a446c3b3a97a2f42b/unsloth/dataprep/synthetic.py) | **Not planned**: D-03. Connect to Data Designer |
| Ebooks: MOBI, FB2 (`load_ebook_dataset`, AGPL) | [ebooks.py](https://github.com/unslothai/unsloth/blob/5f1d55556fd82f889abfe17a446c3b3a97a2f42b/unsloth/dataprep/ebooks.py) | **Not planned** |
| LLM-assisted mapping | [llm_assist.py](https://github.com/unslothai/unsloth/blob/5f1d55556fd82f889abfe17a446c3b3a97a2f42b/studio/backend/utils/datasets/llm_assist.py) | **Not at launch.** Later, optional and local only [J], because spreadsheets often hold private business data |

One Unsloth trap is designed out. Its legacy CLI sends any `.json` file to a raw-text loader that keeps only the first string field, so a `messages` file yields no text [S](https://github.com/unslothai/unsloth/blob/5f1d55556fd82f889abfe17a446c3b3a97a2f42b/unsloth-cli.py). In tailtune, every surface calls the same function.

## 3. The pipeline at a glance

| Stage | Input | Output | Libraries (licence) | On failure |
|---|---|---|---|---|
| sniff | Path, URL, Hub ID or object | `SourceSpec`: kind, real format, encoding, SHA-256, revision | stdlib, charset-normalizer (MIT) | Error naming the supported formats. Never a Hub lookup for a local path |
| read | `SourceSpec` | `RawTable` per sheet or file: cells, plus structure metadata | python-calamine (MIT), openpyxl (MIT), pyarrow (Apache-2.0), `datasets` (Apache-2.0) | Error with the file, sheet and cell |
| normalize | `RawTable` | `Table`: string columns, `source_row`, events | stdlib | Rows are dropped with a reason, never silently |
| detect | `Table` | `Detection`: format, roles, confidence, reasons, alternatives | stdlib | Ambiguous → ask, or exit 2 when non-interactive |
| map | `Table`, `Detection`, user options | `Mapping` and canonical `messages` per example | stdlib | Unknown column → error. Empty required field → row dropped |
| render | Examples and a tokenizer | Text, `input_ids` and loss spans per example | Transformers v5 (Apache-2.0) | Template errors → blocking check |
| validate | Everything above | `DataReport` | stdlib | Blocking → exit 2 |
| freeze | Kept examples and report | `data.parquet`, `mapping.yaml`, `data_report.json`, hashes | pyarrow | Atomic write. On error, nothing partial is left |

Python surface, consistent with the [plan index](README.md#shared-vocabulary). Final names and the CLI spelling belong to [04](04-api-and-ux.md).

```python
# Design proposal, untested.
def read(
    source: str | os.PathLike | list[str | os.PathLike] | "Dataset" | "DatasetDict"
            | "pandas.DataFrame" | "polars.DataFrame" | list[dict],
    *,
    sheet: str | list[str] | None = None,     # sheet names; None = auto-select
    header_row: int | None = None,            # 1-based sheet row; None = detect
    hub_config: str | None = None,            # Hub config (named as in 04)
    split: str | None = None,                 # Hub split
    revision: str | None = None,              # Hub branch, tag or SHA
    format: str = "auto",                     # auto|messages|sharegpt|alpaca|prompt_completion|preference|text
    system: str | None = None,                # templates, e.g. "{question}"
    user: str | None = None,
    assistant: str | None = None,
    mapping: str | os.PathLike | None = None, # reuse a mapping.yaml
    model: str | None = None,                 # tokenizer for render and token checks
    max_seq_length: int | None = None,
    on_overflow: str = "drop",                # drop|truncate|raise
    eval_fraction: float | None = None,
    seed: int = 0,
    interactive: bool | None = None,          # None = only if attached to a terminal
) -> "tailtune.Dataset": ...                  # named as in 04; not datasets.Dataset
```

Without `model`, the first five stages run, plus every check that needs no tokenizer. `report()` then shows "token checks: pending". `tt.train` completes render and the token checks using the tokenizer alone, before any weights load.

## 4. Sniff

Algorithm:

1. **In-memory objects** are recognised by type and need no sniffing.
2. **Local paths.** If the path exists, read the first 8 KB and decide by magic bytes. Extension is a tie-breaker only.
3. **Missing local paths.** A string with a path separator or a file extension that does not exist is "file not found". Never try the Hub. The research measured that `load_dataset("multi.xlsx")` reports a misleading `FileNotFoundError` and `load_dataset("xlsx", ...)` tries the Hub [M].
4. **Hub IDs** match `owner/name`. Resolve the revision to a commit SHA and record both.
5. **URLs** with a scheme go to fsspec, from v0.2.

| Bytes at offset 0 | Format | How to confirm |
|---|---|---|
| `PK\x03\x04` (zip) | xlsx, xlsm, xlsb, ods, docx | Zip members: `xl/workbook.xml` means xlsx; plus `xl/vbaProject.bin` means macros (xlsm); `xl/workbook.bin` means xlsb; a `mimetype` member reading `application/vnd.oasis.opendocument.spreadsheet` means ods |
| `D0 CF 11 E0 A1 B1 1A E1` (OLE2 container) | xls, or an encrypted xlsx | Stream names. An OLE2 file named `.xlsx` is treated as password-protected [U]: confirm with a fixture saved by Excel |
| `PAR1` at both ends | Parquet | pyarrow footer read |
| Arrow IPC file or stream | Arrow | Try both pyarrow readers |
| `%PDF-` | PDF | v0.3 only |
| Text | JSON, JSONL, CSV, TSV, TXT, MD | See below |

For text:

- **Encoding.** Check for a BOM (UTF-8, UTF-16 LE and BE). Otherwise try strict UTF-8 on the first 1 MB, then charset-normalizer (MIT) [S](https://pypi.org/pypi/charset-normalizer/json). Report the encoding and confidence. `encoding=` overrides. The research measured that a cp1252 CSV fails in `load_dataset` and that a UTF-8 BOM is handled [M].
- **Shape.** The first non-blank character `[` means JSON. One complete JSON object per line on the first two lines means JSONL. Otherwise sniff a delimiter among `,` `\t` `;` `|`. If that fails, and the extension is `.txt` or `.md`, the file is raw text.
- **A mismatch is reported**, for example "faq.csv is actually an xlsx workbook". Magic bytes win.

Every source gets a SHA-256 of its bytes, streamed. Directories list files recursively in sorted order, skip hidden files and Excel lock files (`~$*`), and report each file.

## 5. Read

Every value leaves this stage as text, or as "empty". Only a truly empty cell or field is empty.

| Source | Reader | Rule |
|---|---|---|
| Spreadsheets | python-calamine 0.9.1 (MIT) for values [S](https://pypi.org/pypi/python-calamine/json); openpyxl 3.1.5 (MIT) for structure [S](https://pypi.org/pypi/openpyxl/json) | Section 6 |
| CSV, TSV | pyarrow CSV reader, every column forced to string, no null markers | `NA`, `null`, `None` and `N/A` stay text. A fixture proves it |
| JSON, JSONL | Python `json`, line by line, in 10,000-row chunks | Nested `messages` kept as lists. Scalars converted to text by the rules below. A type change mid-file is not an error |
| Parquet, Arrow | pyarrow | Scalar columns converted to text. List and struct columns kept for `messages` |
| Hub | `datasets.load_dataset(id, name=config, split=split, revision=sha)` | Never run dataset loading scripts (section 15) |
| `Dataset` | As is | Scalar columns converted to text |
| DataFrame | Per cell | Never `Dataset.from_pandas` on object columns: the research measured `ArrowInvalid` on a mixed-type column [M] |
| `.txt`, `.md` | Read whole file | One row, `text` column, per file |

Why not `load_dataset` for local CSV: its CSV builder reads in 10,000-row chunks [S](https://github.com/huggingface/datasets/blob/a1269ca9dbb6e0858a9c375e8982329a3fcc84af/src/datasets/packaged_modules/csv/csv.py), and the research measured a type change after row 12,000 failing late with `DatasetGenerationError` [M].

**Cell-to-text rules.** These apply to spreadsheet cells and to scalar values from any other source.

| Value type | Text | Example |
|---|---|---|
| String | Unchanged, except `\r\n` and `\r` become `\n` | `00123` → `00123` |
| Whole-number float, magnitude below 2^53 | No decimal point | `1.0` → `1` |
| Other float | Shortest decimal that round-trips. Positional notation for magnitudes from 1e-6 to below 1e16, scientific otherwise [J] | `2.5` → `2.5` |
| Integer | Decimal | `42` → `42` |
| Boolean | `TRUE` or `FALSE`, as Excel shows it [J] | |
| Date | ISO 8601 | `2026-01-02` |
| Date-time | ISO 8601, no time zone, seconds kept | `2026-01-02T13:45:00` |
| Time | `HH:MM:SS` | |
| Duration | `H:MM:SS` [J] | |
| Excel error such as `#N/A` | The error literal, counted as a warning | How calamine surfaces errors is untested [U] |
| Empty, or empty string | Empty | |

Number formats such as percentages and currency are **not** applied, because calamine returns stored values (section 6). Columns whose number format differs from General are reported, for example "column `rate` uses format `0.00%`: the stored value 0.125 was used, while Excel shows 12.50%".

## 6. Spreadsheets

### How calamine and openpyxl work together

| Need | Source | Available for |
|---|---|---|
| Cell values | calamine `iter_rows` | xlsx, xlsm, xls, xlsb, ods |
| Sheet names and visibility | calamine `sheets_metadata` [M] | xlsx confirmed; others untested [U] |
| Merged ranges | calamine `merged_cell_ranges` [M] | xlsx confirmed; others untested [U] |
| Formula text, to find uncached formulas | openpyxl, read-only mode, `data_only=False` [M] | xlsx, xlsm |
| Number formats | openpyxl, read-only mode [M] | xlsx, xlsm |
| Hidden rows and columns | openpyxl, full mode (`row_dimensions`, `column_dimensions`) [M] | xlsx, xlsm, under the size limit in section 14 |

openpyxl's read-only mode gives "(near) constant memory" but "relies on applications and libraries that created the file providing correct information" [S](https://openpyxl.readthedocs.io/en/stable/optimized.html). Its read-only worksheets have no merged ranges and no row dimensions [M]. For formats openpyxl cannot read, the report says which structure checks did not run. pandas plans to move its default Excel engine from openpyxl to calamine [S](https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html).

### What has and has not been tested

**Failures reproduced in the research** [M]. Scope: pandas 3.1.1, openpyxl 3.1.5, python-calamine 0.9.1, `datasets` 5.1.0, Windows 11, a single run on synthetic files written by a script. The required behaviour follows the report's table.

| Observed failure in the naive path | Required behaviour |
|---|---|
| `pd.read_excel` on a 3-sheet workbook returned only the first sheet, 2 of 4 data rows, with no warning. Concatenating all sheets, as Oumi does, merged a Notes sheet's `author, date` columns into training columns | List every sheet with rows and columns. Auto-select only when exactly one looks like data. Combine only on matching headers |
| Merged cells, both engines: only the first row kept the value, and the others became `NaN`. A merged banner row became a data row | Report merges, fill vertical merges down, detect the header row, allow `header_row` |
| A mixed-type column raised `ArrowInvalid: Could not convert 'forty-two' with type str` | Read every cell as text. Never pass inferred object columns to Arrow |
| `00123` became `123`; an integer column with a blank became `[1.0, nan, 3.0]`; the strings `NA`, `None`, `null` and `N/A` became missing | Preserve what the user sees. Only truly empty cells are empty |
| A formula with no cached value read as `NaN` in both engines | Compare with an openpyxl formula pass. Warn with a cell count |
| `load_dataset` gave misleading errors on xlsx; a mid-file type change failed late; a cp1252 CSV failed | Clear messages naming the supported formats; encoding detection; report what was chosen |
| Reading 100,000 rows (1.7 MB): openpyxl 12.2 s, calamine 1.1 s | Use calamine for values |

**Probed for this plan** [M]. Scope: python-calamine 0.9.1, openpyxl 3.1.5, Python 3.13.9, Windows 11, one workbook written by openpyxl, a single run.

| Probe | Result | Design consequence |
|---|---|---|
| Sheet visibility | `sheets_metadata` reported Visible, Hidden and VeryHidden correctly | Use calamine for sheet listing |
| Merged ranges | `merged_cell_ranges` returned zero-based ranges, such as `((3,0),(5,0))` | No openpyxl full load is needed for merges |
| Hidden row | calamine returned the hidden row as ordinary data. openpyxl full mode flagged it | Hidden rows need the openpyxl pass |
| Integer cell `1` | Returned as float `1.0` | Cell-to-text rule for whole floats |
| Cell with format `0.00%` | Stored value returned, format ignored | Report format mismatches |
| Midnight date-time | Returned as a `date` | ISO output handles both |
| Text `00123`, `NA`, `null` | Returned as strings, unchanged | Reading calamine directly, not through pandas, preserves them |
| Empty cell | Returned as `''` | Treat `''` as empty |
| Formula `=1+1` with no cached value | calamine `''`. openpyxl read-only showed `=1+1` | Detect uncached formulas by comparing the two |
| openpyxl read-only mode | Gave formula text and number formats, but no merged ranges and no row dimensions | Split the structure pass as in the table above |

**Not tested** [U]: files saved by real Excel or LibreOffice (these normally hold cached formula values, so the uncached-formula result applies only to script-written files); `.xls`, `.xlsb` and `.ods`; encrypted workbooks; XML entity attacks; memory use on large files; non-English headers; hidden columns; the 1900 and 1904 date systems; how calamine reports error cells.

### Sheet selection

1. List every sheet: name, visibility, used rows and columns, and whether it "looks like data".
2. A sheet looks like data when it is visible, has a detected header row, at least 2 non-empty columns and at least 3 data rows, and either detection finds a user and assistant pair with confidence of at least 0.6, or the sheet holds every column named in an explicit template [J].
3. Exactly one data-like sheet: select it, and list the others as ignored with a reason (hidden, not data, empty).
4. No data-like sheet: blocking, with the sheet list.
5. Several: ask when interactive. When non-interactive, exit 2 with the candidates and the `sheet=` option.
6. Several sheets may be combined, by `sheet=["Q1", "Q2"]`, only if their normalized headers match as sets. A union of mismatched columns is an error. Each row records its `source_sheet`.
7. Hidden sheets are never auto-selected. Naming one explicitly works, with a warning.

### Header row detection

Applies to spreadsheets, and to CSV files whose first row fails the test.

1. Look at the first 20 non-empty, non-hidden rows.
2. A row qualifies as the header when:
   - it has at least 2 non-empty cells, or 1 for a one-column table;
   - every non-empty cell is text that does not parse as a number or a date;
   - the values are distinct after normalization;
   - it is not a full-width merged banner;
   - in the next 3 rows, at least half of its non-empty columns hold values.
3. Pick the first row that qualifies. Rows above it are the preamble. They are dropped and reported with their first text, for example "2 title rows skipped".
4. When two consecutive rows qualify and the upper one contains horizontal merges, treat it as a group header. Use the lower row, and warn `multi_row_header` [J].
5. When no row qualifies, columns are named `column_A`, `column_B` and so on. The mapping must then be explicit.
6. `header_row=N` (the 1-based sheet row) overrides all of this.

Header text is trimmed and inner whitespace collapsed. Duplicates get `_2`, `_3` suffixes. Matching is case-insensitive. The original text is kept for display.

### Merged cells

| Merge shape | Behaviour |
|---|---|
| Vertical, inside the data region | Fill the value down to every row in the range. Counted |
| Spans every column of the table | Banner row. Dropped with reason `banner_row` |
| Horizontal, inside one data row | Value kept in the first column, others left empty. Warning `horizontal_merge` |
| Block (several rows and columns) | Fill down in the first column only. Warning [J] |
| Over the header or preamble | Header grouping. Ignored |

`merged_cells: keep` in the mapping turns off fill-down.

### Hidden rows, columns and sheets

Hidden rows and columns are excluded by default and reported. Rows hidden by a filter are hidden rows in the file, so the report says "412 hidden rows excluded (possibly a filter)" [J]: someone who filtered a sheet to view a subset may not expect that. `include_hidden=True` keeps them. A column named explicitly in a template is used even if hidden, with a warning. When hidden rows cannot be checked (non-xlsx formats, or a workbook over the size limit), the report warns "hidden rows not checked".

### Dates, formulas, macros and encryption

- **Dates** follow the cell-to-text rules. Text that merely looks like a date is left alone.
- **Formulas with no cached value** appear as empty to calamine. A cell is `uncached_formula` when calamine returns empty and openpyxl returns text starting with `=`. The warning gives the count per column and advises opening and saving the file in Excel or LibreOffice. tailtune never evaluates formulas. If every value in a required column is uncached, those rows drop as empty and the stage fails blocking.
- **Macros** (`xl/vbaProject.bin` in xlsm or xlsb) are never executed. calamine and openpyxl are parsers and do not run VBA. The report says "workbook contains macros; they were not run, so values are those last saved".
- **Encrypted workbooks** fail blocking with "the workbook is password-protected: open it, remove the password, save a copy, and try again". Decryption is not supported at v0.1.
- **Text escapes.** OOXML `_xHHHH_` sequences such as `_x000D_` are decoded and counted, if calamine has not already decoded them [U].

## 7. Normalize

In order, for every table:

1. Select sheets and find the header (section 6).
2. Apply merges and hidden-row rules.
3. Drop fully empty rows (reason `empty_row`) and fully empty columns.
4. Trim spaces and tabs at both ends of each value. Keep inner whitespace and newlines [J]. `trim=False` turns this off.
5. Attach `source_row` to every row: the 1-based sheet row, the 1-based line for JSONL and CSV, or the index for in-memory data. Every later drop is reported against this number.
6. Keep nested records (`messages` lists) as they are, normalizing only key names and role values.

No silent Unicode changes. A mojibake heuristic gives a warning only. ftfy (Apache-2.0) [S](https://pypi.org/pypi/ftfy/json) is a candidate.

## 8. Detect

### Ordered rules

Detection samples up to 1,000 rows: the first 200, then evenly spaced rows [J]. For scale, Studio samples up to 100 rows [S](https://github.com/unslothai/unsloth/blob/5f1d55556fd82f889abfe17a446c3b3a97a2f42b/studio/backend/utils/datasets/format_detection.py) and `standardize_data_formats` samples 10 [S](https://github.com/unslothai/unsloth-zoo/blob/18dd2236add5a80dd79ccc8390dc9f49df37e5e5/unsloth_zoo/dataset_utils.py). Validation later checks every row.

The first rule that matches wins. Other matches are reported as alternatives.

| # | Format | Rule |
|---|---|---|
| R1 | ChatML `messages` | A column of lists of `{role, content}` objects. Column aliases: `messages`, `conversations`, `conversation`, `dialog` |
| R1b | ShareGPT | The same, with `{from, value}` keys |
| R2 | Preference | `chosen` and `rejected`, each a string or a message list, with an optional `prompt` |
| R3 | Prompt and completion | `prompt` and `completion`, as strings or message lists |
| R4 | Alpaca | `instruction` and `output`, with optional `input` and `system` |
| R5 | Long-format conversation table | A grouping column, a role column and a content column (see "Multi-turn conversations" below) |
| R6 | Wide-format conversation table | Paired columns such as `user_1, assistant_1, user_2, assistant_2` |
| R7 | Raw text | A single `text` column, or a table with one text column |
| R8 | Column aliases | One user column and one assistant column by the alias table |
| R9 | Positional fallback | Exactly two text columns: first is user, second is assistant. Always needs confirmation |

R7 has two cases. If most `text` values already contain the model's role markers, the data is pre-rendered chat text. It is used as is, and still gets the BOS and EOS checks. Otherwise it is a continued-pretraining corpus.

R2 at v0.1: preference training waits for DPO (D-16). tailtune reports "preference data detected" and stops, unless the user maps `assistant="{chosen}"` explicitly for SFT.

### Alias table

Aliases are matched case-insensitively, after treating spaces, hyphens and underscores alike. The English base lists follow ms-swift's Apache-2.0 preprocessor at commit `47b437a84e9ea481dbe24f516fc2318e7faf94da` [S](https://github.com/modelscope/ms-swift/blob/47b437a84e9ea481dbe24f516fc2318e7faf94da/swift/dataset/preprocessor/core.py). The additions, including the non-English headers, are untested [J].

| Role | From ms-swift | Added |
|---|---|---|
| user | `query, prompt, input, instruction, question, problem` | `user, human, request, q, user_message`, `pregunta, frage, domanda, pergunta, vraag, प्रश्न, 问题, 質問` |
| assistant | `response, answer, output, targets, target, answer_key, answers, solution, text, completion, content` | `reply, assistant, a, resolution, respuesta, antwort, réponse, risposta, resposta, antwoord, उत्तर, 答案, 回答` |
| system | `system, system_prompt` | `system_message, persona` |
| context (optional, appended to user) | none | `context, background, passage, document`, and `input` when `instruction` exists |
| conversation group | none | `conversation_id, conv_id, thread_id, session_id, dialog_id, chat_id` |
| turn order | none | `turn, turn_id, turn_index, step, seq, order` |
| role column | none | `role, speaker, from, author` |
| content column | none | `content, message, text, utterance, value` |
| role values | `user, human, input` → user; `gpt, assistant, output` → assistant (Unsloth's defaults) | `model, bot, agent, ai` → assistant; `customer, client` → user |

Collisions:

- `text` is treated as the assistant column only when a user alias is also present. Alone, it triggers R7.
- `input` is context when `instruction` exists, and user otherwise.

### Confidence

| Signal | Score [J] |
|---|---|
| Structural rules R1 to R7 | 0.95 × the share of sampled rows that parse |
| Exact alias match | 0.90 |
| Alias found as one word of a longer header, e.g. `customer_question` | 0.70 |
| Positional fallback R9 | 0.40 |
| Two columns tie for one role | −0.20 |
| Candidate column has 10 or fewer distinct values over 100 or more rows (a category) | Excluded from user and assistant |

Mapping confidence is the lowest score among the required roles.

| Confidence | Interactive | Non-interactive (`--yes`) |
|---|---|---|
| 0.85 or more | Preview, default Yes | Proceed |
| 0.60 to 0.85 | Preview, explicit confirmation | Proceed only with an explicit mapping, otherwise exit 2 |
| Below 0.60, or a tie | Ask the user to pick columns | Exit 2, listing candidates and the exact options to pass |

The thresholds are a judgement until fixtures and user reports calibrate them.

### Multi-turn conversations in tables

**Long format** (R5), one row per turn:

1. Group rows by the conversation column, keeping groups in order of first appearance.
2. Blank cells in the grouping column are filled down by default, because people group visually in spreadsheets. This is reported. `blank_group: drop` drops those rows instead.
3. Within a group, order rows by the turn column if one exists, sorting numbers numerically. Otherwise keep sheet order. Non-numeric turn values keep sheet order, with a warning.
4. Map role values through the role-value aliases. An unknown role drops the conversation with reason `invalid_role`.
5. Consecutive turns with the same role fail the role-alternation check, unless `merge_consecutive: true`, which joins them with a blank line [J].
6. A system message comes from a `system` column (the first non-empty value in the group) or from a constant in the mapping.

**Wide format** (R6): columns matching `^(user|question|q)[ _-]?(\d+)$` and `^(assistant|answer|a)[ _-]?(\d+)$` give turns ordered by number. Trailing empty pairs are skipped. A gap inside a conversation is a role-alternation error.

## 9. Map and the mapping file

The mapping says how columns become messages. It is model-independent, so it can be reused with another model. Model-dependent settings (template, sequence length, overflow policy, split) live in the `data:` section of `tailtune.yaml`, whose schema belongs to [04](04-api-and-ux.md).

```yaml
# mapping.yaml: design proposal. Written on every run, never required.
tailtune_mapping: 1
source:                     # informational except where noted
  path: faq.xlsx            # relative to this file
  sha256: 9b1e...           # of the source bytes when this mapping was made
  format: xlsx
  sheets: [Q1]              # used on reuse
  header_row: 3             # 1-based sheet row; used on reuse
  columns: [category, question, answer, context, internal_note]  # header fingerprint
  include_hidden: false
  merged_cells: fill_down   # fill_down | keep
format: columns             # columns | messages | sharegpt | alpaca | prompt_completion | preference | text | conversation_long | conversation_wide
messages:                   # for columns, alpaca, prompt_completion
  system: "You are the Acme support assistant."
  user: "{question}[[\n\nContext: {context}]]"
  assistant: "{answer}"
conversation:               # for conversation_long only
  group_by: conversation_id
  order_by: turn
  role_column: role
  content_column: content
  blank_group: fill_down    # fill_down | drop
  merge_consecutive: false
role_values:                # extra aliases for role values in the data
  agent: assistant
ignored: [category, internal_note]
detected:                   # written by tailtune; informational
  rule: R8
  confidence: 0.90
  reasons: ["question: exact user alias", "answer: exact assistant alias"]
```

### Templates

The `{col}` and `[[optional {col}]]` syntax follows Unsloth's documented `to_sharegpt` behaviour, where `{col}` is required and a `[[...]]` section is dropped when its column is empty [S](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide.md). The Apache-licensed `chat_templates.py` at commit `5f1d55556fd82f889abfe17a446c3b3a97a2f42b` could be reused with attribution after re-checking its header [S](https://github.com/unslothai/unsloth/blob/5f1d55556fd82f889abfe17a446c3b3a97a2f42b/unsloth/chat_templates.py). Writing it from this spec is simpler, so do that [J].

Grammar:

```text
template := ( text | field | optional )*
optional := "[[" ( text | field )+ "]]"     ; no nesting; must contain at least one field
field    := "{" column-name "}"            ; the normalized header, matched case-insensitively
text     := any characters; "{{" and "}}" are literal braces
```

Rules:

- A field outside an optional section is required. If it is empty for a row, the row is dropped with reason `empty_required:<column>`.
- An optional section is included only if every field inside it is non-empty.
- A template with no fields is a constant, such as a fixed system prompt.
- The result is trimmed at both ends.
- An unknown column is an error at mapping time, and the error lists the available columns.
- A literal `[[` cannot be written at v0.1. This is a known limitation [J].

```python
# Design proposal, untested.
_TOKEN = re.compile(r"\{\{|\}\}|\[\[|\]\]|\{[^{}]+\}")

def render_field_template(tpl: str, row: dict[str, str | None]) -> str | None:
    """Return the filled string, or None if a required field is empty."""
    out, section, in_opt, section_ok = [], [], False, True
    pos = 0
    for m in _TOKEN.finditer(tpl):
        buf = section if in_opt else out
        buf.append(tpl[pos:m.start()])
        tok, pos = m.group(), m.end()
        if tok == "{{" or tok == "}}":
            buf.append(tok[0])
        elif tok == "[[":
            if in_opt:
                raise TemplateError("nested [[ ]] sections are not supported")
            in_opt, section, section_ok = True, [], True
        elif tok == "]]":
            if not in_opt:
                raise TemplateError("]] without [[")
            if section_ok:
                out.extend(section)
            in_opt = False
        else:  # {column}
            value = row.get(normalize_header(tok[1:-1]))
            if not value:
                if in_opt:
                    section_ok = False
                else:
                    return None
            else:
                buf.append(value)
    if in_opt:
        raise TemplateError("[[ without ]]")
    out.append(tpl[pos:])
    return "".join(out).strip()
```

### Reuse

`tt.read(path, mapping="runs/run-001/mapping.yaml")` and the CLI equivalent apply a saved mapping.

- The data path given on the command line always wins. `source.path` is used only when no path is given, and it is then printed.
- `sheets` and `header_row` are reused unless overridden.
- A column referenced by a template that is missing from the new file is blocking.
- New columns not referenced anywhere produce a warning.
- A different source SHA-256 is expected (the file was updated) and is reported as information.

## 10. Render

1. **Template source.** In order of precedence:
   - a user Jinja file passed explicitly, recorded as `user:<sha256>`;
   - a tailtune override, applied only when the upstream template's hash exactly matches a known-broken entry, recorded as `override:<id>`;
   - the tokenizer's `chat_template` at the pinned model revision, recorded as `tokenizer`.

   Overrides exist because upstream templates "sometimes contain errors and may take time to be updated" [S](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide.md). Each override carries an ID, a reason and the upstream hash it replaces.
2. **Render** each conversation with `apply_chat_template(messages, tokenize=False, add_generation_prompt=False, **chat_template_kwargs)`.
3. **Tokenize** the rendered text with `add_special_tokens=False`. The Transformers docs warn that templates already include special tokens, and adding more "is often incorrect or duplicated" [S](https://github.com/huggingface/transformers/blob/0f53828e83f2212060233d2ea552cc76617b8712/docs/source/en/chat_templating.md). TRL hardcodes the same choice "to avoid adding the BOS twice" [S](https://github.com/huggingface/trl/blob/4ae4d34a86f385511c85fa14caedc2178095d0b8/trl/trainer/sft_trainer.py).
4. **Label masking**, by prefix difference. For each assistant turn *i*:
   - render `messages[:i]` with `add_generation_prompt=True` to get the prefix;
   - render `messages[:i+1]` to get the full text;
   - if the full text starts with the prefix, the assistant span is the remainder, including the end-of-turn token;
   - map character spans to token spans through the tokenizer's offsets.
5. **Templates that are not prefix-stable.** Some templates rewrite earlier turns, for example by stripping earlier reasoning blocks. If the prefix property fails on the sampled rows, fall back to training on the last assistant turn only, with warning `template_not_prefix_stable`. `multi_turn: expand` instead makes one example per assistant turn [J]. Transformers' own assistant-mask option for templates with generation markers is a candidate replacement, not verified in the research [U].
6. **`train_on`**: `assistant` (default, every assistant turn), `last_turn`, or `all`. Prompt and completion data trains on the completion. Raw text trains on everything.

### EOS

The research found the same lesson in Unsloth's Alpaca script, "Must add EOS_TOKEN, otherwise your generation will go on forever!" [S](https://github.com/unslothai/notebooks/blob/d85d113eacfc8a87a7003e73ffb2a86403a94344/python_scripts/Llama3.1_%288B%29-Alpaca.py), and in a popular issue on adding the EOS token [S](https://github.com/unslothai/unsloth/issues/419).

- The **stop set** is the tokenizer's `eos_token` plus every ID in the model's `generation_config.eos_token_id`.
- For chat data, the last supervised token of every assistant span must be in the stop set. If it is not, that is blocking `missing_eos`. `eos: append` adds the EOS token to the span instead.
- Raw text gets an EOS appended at the end of each document [J].

### System prompts

- Probe the template once with and without a system message.
- If it raises an error, or the output is identical (the system message was dropped silently), apply `system_policy`. The default, `prepend_to_user`, joins the system text to the first user turn with a blank line and warns. The alternative, `error`, blocks [J].
- If the template injects a default system prompt when none is given, the rendered samples show it, and the report notes it.

### Thinking-mode templates

- Templates with a thinking switch get explicit `chat_template_kwargs`, such as `{"enable_thinking": false}` for data without reasoning [J].
- The kwargs are part of the template hash, because they change the rendered text.
- A reasoning column can be mapped as `assistant_reasoning: "{thinking}"`. It is accepted only if the template has a known slot for reasoning content. Otherwise it is blocking [U]: which templates expose such a slot must be checked per launch model in [06](06-models.md).
- [09](09-evaluation.md) uses the same template, system prompt and thinking flag before and after training.

### Models with no chat template

- Raw text for continued pretraining needs no template.
- For chat data, it is blocking by default: "this model has no chat template; use its instruct variant, or pass a template".
- tailtune ships a small set of vetted generic templates. One that needs special tokens missing from the vocabulary is refused, because new tokens require training embeddings, and embeddings are frozen by default (D-15). In that case, offer a plain-text template that adds no tokens [J].

### Template hash

Two values are recorded, and both are shared with [10](10-export-and-deployment.md):

| Value | Input | Used for |
|---|---|---|
| `chat_template_sha256` | SHA-256 of the exact UTF-8 bytes of the template string | Comparing with templates embedded in exported files |
| `template_hash` | SHA-256 of canonical JSON (sorted keys, no whitespace) of `chat_template`, `chat_template_kwargs`, `bos_token`, `eos_token`, the stop set, `adds_bos` (whether the tokenizer adds BOS when asked for special tokens, probed once) and `system_policy` | The full render contract, stored in `lineage.json` |

## 11. Validate

### Check catalogue

Thresholds marked [J] are starting values to calibrate on fixtures. Row-level failures drop the row and record the reason. The run-level rule decides whether the whole run blocks.

| ID | Check | Level | Rule |
|---|---|---|---|
| `no_rows` | Rows remain after normalize | Blocking | 0 rows |
| `all_dropped` | Rows remain after map and render | Blocking | 0 rows |
| `drop_share` | Share of rows dropped | Warning above 10%, blocking above 50% [J] | Lists reasons with source row numbers |
| `sheet_ambiguous` | Sheet choice | Blocking when non-interactive | Section 6 |
| `mapping_ambiguous` | Mapping confidence | Blocking when non-interactive | Section 8 |
| `preference_data` | Preference pairs given to SFT | Blocking | Unless `assistant` is mapped explicitly |
| `uncached_formula` | Formula cells with no value in used columns | Warning | Count per column |
| `error_cells` | `#N/A` and similar in used columns | Warning | Count |
| `number_format` | Display differs from stored value | Warning | Column and format |
| `merged_cells` | Merges found and filled | Info | Count |
| `hidden_excluded` | Hidden rows, columns or sheets left out | Info; warning if rows were hidden | Count |
| `structure_unchecked` | Hidden-row or formula checks could not run | Warning | Format or size named |
| `macros_present` | VBA found | Info | Not run |
| `duplicates_exact` | Same canonical messages | Warning | Kept first, dropped the rest (`dedupe: exact`, default) [J] |
| `duplicates_conflicting` | Same user turns, different answers | Warning | Count and three examples |
| `role_order` | Optional system first, then user and assistant alternating, ending on assistant | Row dropped; blocking above 5% [J] | High rates mean a mapping error |
| `empty_assistant` | Assistant content empty | Row dropped | |
| `tools_unsupported` | Tool messages, but the template has no tool support | Blocking [J] | |
| `special_token_text` | User data contains a literal special-token string such as `<|im_end|>` | Warning | It can break role boundaries |
| `token_lengths` | min, median, p95 and max per example; total and supervised tokens | Info | |
| `over_length` | Examples longer than `max_seq_length` | `drop` (default): warning, blocking above 10% [J]; `truncate`: warning; `raise`: blocking | Policy names follow Axolotl's `drop \| truncate \| raise` [S](https://github.com/axolotl-ai-cloud/axolotl/blob/620c1efca4470c1bc2e4a5f46b4b106341e9dc41/src/axolotl/utils/schemas/config.py) |
| `double_bos` | More than one BOS at the start of any example | **Blocking** | Skipped if the tokenizer has no BOS |
| `missing_bos` | Tokenizer adds BOS by default, but the rendered text lacks it | Warning [J] | |
| `missing_eos` | Last supervised token of an assistant span is not in the stop set | **Blocking** | Truncated examples are exempt but counted |
| `all_masked` | An example has zero supervised tokens | **Blocking** | Catches Unsloth's documented "All labels in your dataset are -100" failure [S](https://unsloth.ai/docs/basics/troubleshooting-and-faqs.md) |
| `supervised_share` | Supervised tokens as a share of all tokens | Warning below 1% [J] | |
| `size_small` | Fewer than 100 kept rows | Warning | Unsloth's published floor, "bare minimum of at least 100 rows" [S](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide.md) |
| `size_modest` | Fewer than 1,000 kept rows | Info | Same source: over 1,000 preferable |
| `split_leakage` | Exact duplicates across train and eval | Blocking | Cannot happen after dedupe, so this checks the code |
| `split_near_dup` | Near-duplicates across train and eval | Warning | Section 13 |
| `contamination` | Overlap with evaluation items | Warning | Section 13 |
| `pii` | Possible personal data | Warning, optional [J] | See below |
| `estimate` | Steps per epoch | Info | Time estimates come from [07](07-hardware.md) |

**Personal data heuristics** [J]. On by default, never blocking, local only. Regular expressions find email addresses, phone numbers, card numbers that pass the Luhn checksum, and IBAN-like strings. The report gives counts per type and column, plus row numbers, and never prints the matched values. The point is a nudge before a model memorizes customer data, not a guarantee. `pii_scan=False` turns it off.

**Rendered samples.** Three samples are shown: the first kept row, the longest, and one chosen at random with the run seed. Special tokens are shown literally. Supervised spans are marked with `▶` and `◀` in plain text, and highlighted in a colour terminal.

### `data_report.json`

Abbreviated, illustrative values. Every check in the catalogue appears in `checks`.

```json
{
  "schema_version": 1,
  "tailtune_version": "0.1.0",
  "created_at": "2026-11-08T10:12:03Z",
  "status": "warn",
  "exit_code": 0,
  "source": {"kind": "file", "path": "faq.xlsx", "detected_format": "xlsx",
             "extension_mismatch": false, "sha256": "9b1e...", "bytes": 48213,
             "encoding": null, "hub": null},
  "readers": {"python-calamine": "0.9.1", "openpyxl": "3.1.5", "pyarrow": "26.0.1"},
  "sheets": [
    {"name": "Q1", "visible": "visible", "rows": 1207, "cols": 4, "selected": true},
    {"name": "Notes", "visible": "visible", "rows": 3, "cols": 2, "selected": false, "reason": "not data"},
    {"name": "Archive", "visible": "hidden", "rows": 880, "cols": 4, "selected": false, "reason": "hidden"}
  ],
  "normalize": {"header_row": 3, "preamble_rows": 2, "merged_ranges": 41, "cells_filled": 118,
                "banner_rows": 0, "hidden_rows_excluded": 0, "hidden_columns_excluded": 0,
                "structure_checks": {"hidden_rows": "checked", "formulas": "checked"}},
  "detection": {"rule": "R8", "confidence": 0.90,
                "roles": {"user": "question", "assistant": "answer"},
                "reasons": ["question: exact user alias", "answer: exact assistant alias"],
                "alternatives": []},
  "mapping": {"path": "mapping.yaml", "sha256": "41c0..."},
  "template": {"source": "tokenizer", "model": "Qwen/Qwen3-0.6B", "model_revision": "c1899de...",
               "chat_template_sha256": "e3b0...", "template_hash": "7d2a...",
               "chat_template_kwargs": {"enable_thinking": false},
               "system_policy": "as_is", "train_on": "assistant", "eos": "require"},
  "counts": {"rows_read": 1204, "rows_kept": 1201, "rows_dropped": 3,
             "dropped_by_reason": {
               "empty_required:answer": {"count": 2, "source_rows": [57, 911]},
               "duplicate": {"count": 1, "source_rows": [402]}}},
  "duplicates": {"exact": 1, "conflicting": 0, "near": null},
  "conversations": {"count": 1201, "multi_turn_share": 0.0, "turns": {"min": 2, "median": 2, "max": 2}},
  "tokens": {"min": 31, "median": 88, "p95": 212, "max": 640, "total": 117203,
             "supervised": 61904, "max_seq_length": 2048, "over_length": 0, "policy": "drop"},
  "token_checks": {"double_bos": 0, "missing_bos": 0, "missing_eos": 0, "all_masked": 0},
  "split": {"method": "random", "seed": 0, "train": 1141, "eval": 60,
            "group_by": null, "stratify_by": null, "near_dup_across": 0},
  "contamination": {"index": "exact+minhash", "suites": {}},
  "pii": {"email": {"count": 4, "columns": ["answer"], "source_rows": [12, 88, 301, 1150]}},
  "checks": [{"id": "pii", "level": "warning", "status": "warn", "count": 4,
              "message": "4 possible email addresses in column 'answer'"}],
  "samples": [{"source_row": 4, "rendered": "<|im_start|>user\nHow do I reset...", "supervised_spans": [[57, 141]]}],
  "frozen": {"path": "data.parquet", "data_hash": "c7f9...", "tokenized_hash": "18aa...", "rows": 1201}
}
```

`status` is `pass`, `warn` or `fail`. `exit_code` is 2 when any blocking check fails.

### Terminal layout

Illustrative output, not measured:

```text
Data  faq.xlsx  (xlsx, 48 KB, sha256 9b1e…)
  Sheets    Q1        1,204 rows   selected
            Notes         2 rows   ignored: not data
            Archive     880 rows   ignored: hidden
  Header    row 3 (2 title rows skipped)
  Merged    118 cells in "category" filled down
  Mapping   question → user        exact alias   0.90
            answer   → assistant   exact alias   0.90
            category, internal_note ignored          saved: mapping.yaml
  Template  Qwen/Qwen3-0.6B tokenizer, enable_thinking=false   hash 7d2a…

Rows      1,204 read · 1,201 kept · 3 dropped
            2  empty answer        sheet rows 57, 911
            1  duplicate           sheet row 402
Tokens    median 88 · p95 212 · max 640 · 0 over 2,048 (policy: drop)
Split     1,141 train · 60 eval (random, seed 0)

Check                 Result
BOS                   ok (no duplicate)
EOS                   ok (1,201 of 1,201)
Labels                ok (no fully masked example)
Duplicates            1 dropped
Personal data         warning: 4 possible email addresses (answer)
Size                  ok (≥ 100 rows)

Sample 1 of 3  (sheet row 4)
  <|im_start|>user
  How do I reset my password?<|im_end|>
  <|im_start|>assistant
  ▶Go to Settings, then Security, then Reset.<|im_end|>◀

Result  WARN · 0 blocking · 1 warning      details: runs/run-001/data_report.json
```

## 12. Freeze

`data.parquet` holds one row per kept example:

| Column | Type | Content |
|---|---|---|
| `tt_row_id` | int64 | Stable index, 0 to n−1, in training order before shuffling |
| `split` | string | `train` or `eval` |
| `source_file` | string, dictionary-encoded | Path relative to the input root |
| `source_sheet` | string or null | Sheet name |
| `source_row` | int32 | 1-based sheet row or line (the first row, for multi-row conversations) |
| `conversation_id` | string or null | From the data, or generated |
| `messages` | list of `{role, content}` | Canonical messages |
| `text` | string | Rendered text |
| `input_ids` | list of int32 | Token IDs |
| `loss_spans` | list of `{start, end}` int32 | Supervised token ranges, end exclusive |
| `num_tokens` | int32 | Length |
| `row_sha256` | string | Hash of the canonical messages |

Storage: zstd compression and row groups of about 10,000 rows [J]. Parquet key-value metadata carries `tailtune.schema_version`, `tailtune.data_hash`, `tailtune.tokenized_hash`, `tailtune.mapping_sha256` and `tailtune.template_hash`.

Hashes:

- `row_sha256` is the SHA-256 of canonical JSON (sorted keys, UTF-8, no whitespace) of the row's `messages`.
- `data_hash` is the SHA-256 over every row's `tt_row_id`, `split` and `row_sha256`, in `tt_row_id` order. It does not depend on Parquet encoding or on the tokenizer, so the same data gives the same hash for any model.
- `tokenized_hash` also covers `input_ids` and `loss_spans`, so it changes with the template or the tokenizer.

Freeze writes `data.parquet`, `mapping.yaml` and `data_report.json` into the run directory, with the names in the [plan index](README.md#shared-vocabulary). Each file is written to a temporary name and renamed, so a failed run never leaves a partial file. It passes the source SHA-256, `data_hash`, `mapping_sha256`, `template_hash` and reader versions to `tailtune.train`, which writes `lineage.json`.

Large spreadsheets are also cached once as a normalized Parquet table, keyed by source SHA-256, in the user cache directory, not the run directory. The same file is never parsed twice.

## 13. Train and eval split, leakage and contamination

### Defaults

| Situation | Behaviour [J] |
|---|---|
| The source has its own eval split (a Hub `validation` or `test` split, or a `split` column) | Use it, and report that |
| 200 or more kept rows | Hold out 5%, at least 20 rows and at most 1,000 |
| Fewer than 200 rows | No automatic eval split. Warn. `eval_fraction` forces one |

The split happens after deduplication. It is seeded, and the seed is recorded. It works by **group**: all turns of one conversation, or all rows sharing a `group_by` value (such as a source document), go to the same side.

**Stratification**, with `stratify_by=<column>`, applies when that column has 50 or fewer distinct values and every value has at least 2 rows. Otherwise it falls back to random, with a warning.

### Leakage

| Route | Control |
|---|---|
| Exact duplicates | Deduplicate before splitting. `split_leakage` asserts zero |
| Turns of one conversation on both sides | Group split |
| Paraphrases and near-copies | MinHash check across the split: warning with a count |
| Rows derived from one document | `group_by` |

### The contamination hook used by evaluation

[09](09-evaluation.md) owns the evaluation item lists, the known-adjacency warnings (such as MedMCQA against MMLU medical) and the reporting. This module owns the index and one shared normalization, so both sides match text the same way.

```python
# tailtune/data/contamination.py. Design proposal, untested.
def normalize_for_match(text: str) -> str:
    """NFKC, casefold, collapse whitespace, strip punctuation at token edges."""

def build_index(data: "FrozenData", *, shingle_words: int = 5,
                num_perm: int = 128) -> "ContaminationIndex":
    """Exact hashes and MinHash signatures of each user turn, each assistant turn and each whole example."""

def scan(index: "ContaminationIndex", items: Iterable["EvalItem"], *,
         jaccard: float = 0.8) -> "ContaminationResult":
    """Per suite: exact matches, near matches, item IDs and training row IDs."""
```

Parameters are a judgement [J]. Exact hashing ships at v0.1. MinHash ships by Gate 3, before the public benchmark. A MinHash implementation is small enough to write in-house, which avoids a dependency whose licence the research did not check. The index lives in the user cache, keyed by `data_hash`. Only the counts go into `data_report.json` and the forgetting report.

## 14. Performance

| Concern | Design |
|---|---|
| Memory | After the read stage, data moves as Arrow record batches of 10,000 rows. Peak memory should stay within about twice the normalized text size, plus the tokenizer [J]. To be measured at Gate 2 |
| Spreadsheet values | Read with calamine's `iter_rows` in chunks, written to a temporary Parquet file. Whether calamine holds a whole sheet in memory internally is unknown [U] |
| Structure pass | openpyxl read-only mode streams formulas and formats. openpyxl full mode, needed for hidden rows, runs only when sheet XML is under 50 MB uncompressed [J]. Above that, `structure_unchecked` is reported |
| Speed | calamine read 100,000 rows in 1.1 s against openpyxl's 12.2 s, in one run on small strings [M]. Progress is shown for anything over 5 seconds |
| Tokenization | Batched, using the fast tokenizer. Processes: 1 on Windows, where Unsloth's sample also uses `dataset_num_proc = 1` [S](https://unsloth.ai/docs/get-started/install/windows-installation.md); otherwise up to 8 [J] |
| Large cells | No 128 KB field limit. Python's `csv` module needs its field limit raised to a value that fits a C long on Windows [S](https://github.com/unslothai/unsloth/blob/5f1d55556fd82f889abfe17a446c3b3a97a2f42b/unsloth/dataprep/raw_text.py). The pyarrow reader avoids this. A fixture with a 1 MB cell proves it |
| Caching | Normalized tables are cached by source SHA-256 |
| Streaming (v0.2) | JSONL and Parquet as an `IterableDataset` [S](https://github.com/huggingface/datasets/blob/a1269ca9dbb6e0858a9c375e8982329a3fcc84af/docs/source/stream.mdx). Validation runs on the first 10,000 rows plus running counters, and the report says "sampled". Exact deduplication across the whole stream is not attempted |

## 15. Security

| Threat | Control |
|---|---|
| Zip bombs in xlsx, xlsm, xlsb, ods or docx | Before any parser runs, read the zip central directory: at most 10,000 entries, at most 2 GiB total uncompressed, and a compression ratio of at most 100 per entry [J]. All limits are configurable. Central-directory sizes can lie, so at Gate 4, move spreadsheet parsing into a child process with a timeout, and a memory limit where the platform allows one [J] |
| XML entity attacks (billion laughs, external entities) | tailtune's own XML reads use a hardened parser. Whether openpyxl and calamine are safe by default is unverified [U]. A fixture with an entity bomb must fail quickly or parse within limits |
| Path traversal | Files are never extracted to disk. For directory inputs, symlinks are resolved and anything outside the given root is refused. A reused mapping's `source.path` is resolved relative to the mapping and printed before use |
| Macros | Never executed (section 6) |
| Encrypted files | Refused with instructions (section 6) |
| Remote code in datasets | Dataset loading scripts are never run. `trust_remote_code` is never passed as true (D-08). A Hub dataset that needs a script is refused, with a request to use its Parquet export |
| Pickle | `.pkl` and pickled pandas files are refused, in line with safetensors-only (D-08) |
| Special-token injection | Warning `special_token_text` (section 11) |
| Credentials | The Hub token is read from the environment or the login cache, and never logged |
| Private data | Nothing leaves the machine. No cloud model reads the data. Personal-data counts are shown, values never are |

## 16. Licensing

| Dependency | Licence | Status |
|---|---|---|
| python-calamine | MIT [S](https://pypi.org/pypi/python-calamine/json) | Required |
| openpyxl | MIT [S](https://pypi.org/pypi/openpyxl/json). No release in over two years | Required. Watch its maintenance |
| pyarrow | Apache-2.0 [S](https://pypi.org/pypi/pyarrow/json) | Required |
| `datasets` | Apache-2.0 [S](https://pypi.org/pypi/datasets/json) | Required (D-05) |
| charset-normalizer | MIT [S](https://pypi.org/pypi/charset-normalizer/json) | Required |
| A YAML library | Confirm at pin time | Required |
| fsspec, s3fs, gcsfs | BSD [S](https://pypi.org/pypi/fsspec/json) | Optional, v0.2 |
| MarkItDown | MIT [S](https://pypi.org/pypi/markitdown/json). A third-party post reports a 251 MB install [U] | `[docs]` default |
| Docling | MIT [S](https://pypi.org/pypi/docling/json). The same post reports 1,032 MB [U](https://dev.to/nhirschfeld/i-benchmarked-4-python-text-extraction-libraries-2025-4e7j) | `[docs]`, if installed |
| pypdf, pdfplumber, python-docx | BSD-3-Clause, MIT, MIT [S](https://pypi.org/pypi/pypdf/json) | Optional fallbacks |
| **PyMuPDF** | "Dual Licensed - GNU AFFERO GPL 3.0 or Artifex Commercial License" [S](https://pypi.org/pypi/pymupdf/json) | **Never a required dependency.** CI checks that no extra pulls it in transitively |

**Do not copy:**

- Unsloth Studio's `format_detection.py` and `llm_assist.py` [S](https://github.com/unslothai/unsloth/blob/5f1d55556fd82f889abfe17a446c3b3a97a2f42b/studio/backend/utils/datasets/format_detection.py), `office_formats.py`, `ebooks.py` and `unsloth_cli/*`. All are AGPL. Implementers should work from this document, not from those files [J].
- `standardize_data_formats` in `unsloth_zoo`, which is LGPL [S](https://github.com/unslothai/unsloth-zoo/blob/18dd2236add5a80dd79ccc8390dc9f49df37e5e5/unsloth_zoo/dataset_utils.py). Reimplement it from the alias table.
- Unsloth's notebooks, which are LGPL [S](https://api.github.com/repos/unslothai/notebooks). Write tailtune's own.

**May reuse, with attribution in `NOTICE`:**

- The ms-swift alias lists at `47b437a84e9ea481dbe24f516fc2318e7faf94da` (Apache-2.0).
- Unsloth's `chat_templates.py` (`to_sharegpt`, `check_dataset`) at `5f1d55556fd82f889abfe17a446c3b3a97a2f42b`. Its file header is Apache-2.0, but re-verify the header on each copy (D-06).

## 17. Fixture corpus

Fixtures live in `tests/fixtures/data/`. Most are generated by one deterministic script that uses openpyxl. Those marked ★ must be saved by real Excel or LibreOffice and committed as binaries, with a note of the application and version that produced them.

| Fixture | Contents | Expected result |
|---|---|---|
| `qa_simple.xlsx` | One sheet: `question, answer`, 120 rows | R8 at 0.90. 120 kept. Pass |
| `qa_multi_sheet.xlsx` | Sheets Q1 and Q2 (Q&A) and Notes (`author, date`) | Non-interactive: exit 2, `sheet_ambiguous`, lists Q1 and Q2. With `sheet=["Q1","Q2"]`: rows combined, `source_sheet` set |
| `qa_notes_sheet.xlsx` | Q1 (Q&A) and Notes | Q1 auto-selected. Notes "ignored: not data" |
| `qa_mismatched_sheets.xlsx` | Q1 `question, answer`; Q2 `question, reply, extra` | Combining raises an error naming the differing columns |
| `banner_header.xlsx` | Merged title in row 1, blank row 2, header in row 3 | `header_row` 3, 2 preamble rows reported |
| `merged_category.xlsx` | Vertical merges in `category`, plus a full-width banner mid-sheet | Merges filled. Banner dropped as `banner_row` with its sheet row |
| `mixed_types.xlsx` | `answer` holds 42, `forty-two`, 3.14, TRUE, empty and a date | Text values `42`, `forty-two`, `3.14`, `TRUE`, (dropped as empty), ISO date. No Arrow error |
| `leading_zeros.xlsx` | IDs `00123` stored as text; an integer column with a blank | `00123` preserved. `1` not `1.0`. Blank → empty |
| `na_strings.xlsx` and `na_strings.csv` | `NA, None, null, N/A` as values | All preserved as text |
| `number_formats.xlsx` | Percent, currency and zero-padded formats | Stored values used. `number_format` warning per column |
| `formula_uncached.xlsx` | `=1+1` written by a script | `uncached_formula` warning, count 1 |
| `formula_cached.xlsx` ★ | The same, saved by Excel | Value `2`. No warning |
| `hidden.xlsx` | A hidden sheet, a very hidden sheet, a hidden row and a hidden column | All excluded and reported. `include_hidden=True` keeps the row |
| `filtered.xlsx` ★ | AutoFilter hiding 40% of rows | Rows excluded. Warning mentions a possible filter |
| `dates.xlsx` | Date, date-time, time and midnight date-time | `2026-01-02`, `2026-01-02T13:45:00`, `13:45:00`, `2026-01-02` |
| `macros.xlsm` ★ | VBA project | Values read. `macros_present`. Nothing executed |
| `encrypted.xlsx` ★ | Password-protected | Blocking, with the "remove the password" message. Exit 2 |
| `legacy.xls` ★, `binary.xlsb` ★, `calc.ods` ★ | The `qa_simple` content | Same rows as `qa_simple`. `structure_unchecked` reported where applicable |
| `xlsx_named.csv` | An xlsx file renamed `.csv` | Read as xlsx. Extension mismatch reported |
| `cp1252.csv` | `café?, naïve résumé` in cp1252 | Detected as cp1252. Text intact |
| `bom.csv` | UTF-8 with BOM | Clean headers |
| `type_flip.csv` | Integers for 12,000 rows, then text | Reads fully. No error |
| `big_cell.csv` | One 1 MB cell | Reads. No field-size error |
| `alpaca.json` | `instruction, input, output` | R4. `input` used as context |
| `sharegpt.jsonl` | `conversations` with `from, value` | R1b. Roles mapped |
| `messages.jsonl` | ChatML, including a system message | R1 at 0.95 |
| `messages_bad_roles.jsonl` | 10% of rows have user, user | `role_order` blocking (over 5%) |
| `prompt_completion.jsonl` | Strings | R3. Completion-only labels |
| `preference.jsonl` | `prompt, chosen, rejected` | `preference_data` blocking for SFT |
| `raw_text.jsonl`, `corpus/*.txt` | Plain documents | R7 continued pretraining. EOS per document |
| `prerendered.jsonl` | `text` already holding `<|im_start|>` markers | R7 pre-rendered. BOS and EOS checks run |
| `conversation_long.xlsx` | `conversation_id` (blanks after the first row of each group), `turn, role, content` | R5. Blank IDs filled down and reported. Order by `turn` |
| `conversation_wide.xlsx` | `user_1, assistant_1, user_2, assistant_2` | R6. Trailing empty pair skipped |
| `ambiguous_columns.xlsx` | `text_a, text_b, notes` | R9 at 0.40. Non-interactive: exit 2 with candidates |
| `non_english.xlsx` | `pregunta, respuesta`; second sheet `问题, 答案` | R8 matches both |
| `optional_context.xlsx` | `question, answer, context`, with `context` half empty | `[[...]]` included only where `context` is non-empty |
| `empty_required.xlsx` | 5 empty answers | 5 dropped, `empty_required:answer`, with sheet rows |
| `duplicates.xlsx` | 3 exact duplicates and 2 conflicting answers | 3 dropped. 2 reported as conflicting |
| `tiny.xlsx` | 40 rows | `size_small` warning. No eval split |
| `pii.xlsx` | Emails, phone numbers, a Luhn-valid card number | `pii` warning with counts. No values printed |
| `special_tokens.xlsx` | User text containing `<|im_end|>` | `special_token_text` warning |
| `zip_bomb.xlsx` | Highly compressed oversized member | Refused before parsing |
| `xml_entities.xlsx` | Entity expansion in a sheet part | Fails quickly, or parses with no expansion |
| `symlink_escape/` | A directory with a symlink pointing outside it | Linked file refused |
| `hub_script_dataset` (mocked) | A Hub dataset that needs a loading script | Refused. No code run |

Token-level fixtures run against the tokenizers of SmolLM2-135M and Qwen3-0.6B (D-20):

| Fixture | Setup | Expected |
|---|---|---|
| `tok_double_bos` | Rendering followed by tokenization with special tokens on, on a tokenizer with a BOS | `double_bos` blocking. Exit 2 |
| `tok_missing_eos` | A user template that drops the end-of-turn token | `missing_eos` blocking. Exit 2. With `eos: append`: passes |
| `tok_all_masked` | Masking forced to an absent marker | `all_masked` blocking. Exit 2 |
| `tok_no_system` | A template that rejects system messages | `prepend_to_user` applied, with a warning |
| `tok_no_template` | A tokenizer without a chat template, chat data | Blocking, with the "use the instruct variant" message |
| `tok_over_length` | 5% of rows over `max_seq_length` | `drop`: 5% dropped, warning; `raise`: exit 2; `truncate`: kept, EOS exemption counted |

## Implementation checklist

**Gate 0, Claim**

- [ ] Reserve the module layout `tailtune/data/{sniff,read,spreadsheet,normalize,detect,mapping,render,validate,freeze,split,contamination}.py`.

**Gate 1, Verify**

- [ ] Write the fixture generator script and the ★ fixtures, with the producing application and version noted.
- [ ] Re-run the section 6 probes on Excel-saved, `.xls`, `.xlsb` and `.ods` fixtures. Record which structure features calamine exposes for each format.

**Gate 2, v0.1 Door**

- [ ] Sniff: magic bytes, encoding detection, Hub ID resolution, directory walking, clear errors.
- [ ] Read: calamine values, openpyxl read-only and full structure passes, cell-to-text rules, CSV, JSON, Parquet, Arrow, `.txt`, `.md`, in-memory inputs.
- [ ] Normalize: sheet selection, header detection, merges, hidden rows and columns, `source_row`.
- [ ] Detect: rules R1 to R9, the alias table, confidence, multi-turn long and wide formats.
- [ ] Map: the template engine, `mapping.yaml` writing and reuse.
- [ ] Render: template source and overrides, prefix-difference masking, EOS, system and thinking policies, the template hash.
- [ ] Validate: every check in section 11, `data_report.json` and the terminal table, exit code 2.
- [ ] Freeze: `data.parquet`, hashes, atomic writes, the normalized-table cache.
- [ ] Split: defaults, group split, stratification, exact-duplicate leakage check.
- [ ] Contamination: `normalize_for_match`, the exact-hash index, `scan` (exact).
- [ ] Security: zip limits, hardened XML, symlink checks, script and pickle refusal.
- [ ] Licence check in CI: no AGPL or LGPL package in the required dependency tree, and no PyMuPDF.
- [ ] The T4 notebook accepts an uploaded `.xlsx`.

**Gate 3, v0.2 Proof**

- [ ] MinHash near-duplicate index, used by [09](09-evaluation.md) contamination scans and by `split_near_dup`.
- [ ] fsspec URLs (`hf://`, `s3://`, `gs://`).
- [ ] Streaming JSONL and Parquet with sampled validation.
- [ ] Calibrate detection thresholds against fixtures and reported issues.

**Gate 4, v0.3 Reach**

- [ ] `tailtune[docs]`: MarkItDown by default, Docling optional, for raw text.
- [ ] Optional ModelScope.
- [ ] Child-process isolation for spreadsheet parsing.

**Gate 5, v1.0**

- [ ] Preference data feeds DPO once it ships (D-16).
- [ ] Freeze the `mapping.yaml` schema as version 1.
- [ ] Vision and audio columns, only if multimodal training is planned.

## Acceptance tests

1. **Gate 2 exit, data:** the fixture corpus passes. Blocking checks catch a double BOS, a missing EOS and all-masked labels, with exit code 2 ([13](13-roadmap.md)).
2. `tailtune train qa_simple.xlsx` reaches the confirmation prompt with the correct mapping, and with `--yes` proceeds without a prompt.
3. `ambiguous_columns.xlsx` with `--yes` and no mapping exits with code 2 and prints the options that would resolve it.
4. Re-running with the written `mapping.yaml` and a different model gives the same `data_hash` and a different `tokenized_hash`.
5. Every dropped row in every fixture is listed in `data_report.json` with its original sheet row or line number.
6. No fixture produces a silent change: every merge, fill, exclusion, conversion with a format mismatch, and encoding choice appears in the report.
7. The `zip_bomb.xlsx` and `xml_entities.xlsx` fixtures finish within 5 seconds with a refusal or a bounded parse.
8. A 100,000-row xlsx reads, normalizes and validates (without tokenization) in under 30 seconds on a GitHub-hosted runner [J], with peak memory recorded.
9. The same inputs produce byte-identical `data.parquet` and identical hashes on Linux, macOS and Windows.

## Open questions

1. Does demand for Excel exist? Measure it after launch through issue reports and notebook use, as [01](01-product-strategy.md) proposes.
2. Which structure features does calamine expose for `.xls`, `.xlsb` and `.ods`? This decides whether hidden rows and merges are reported for those formats.
3. Should a "display values" mode apply Excel number formats, so that `12.50%` stays `12.50%`? Stored values are the safer default, but some users will expect what they see.
4. Are the detection thresholds and the 5%, 10% and 50% drop limits right? Calibrate at Gate 3.
5. How does freeze work for streaming corpora too large to copy into `data.parquet`? A manifest of source URIs and shard hashes may be needed. That would change the run-directory contract, so it needs an owner decision recorded in the [plan index](README.md) first.
6. Should data size relative to the preservation level trigger a stronger warning? The research suggested it, but gave no threshold.
7. Should literal special-token strings in user data be escaped automatically? That needs a tokenizer option the research did not verify.
8. Which templates expose a reasoning slot? Check this per launch model in [06](06-models.md).
