from cave_agent import AttributeRule, FunctionRule, ImportRule, RegexRule, SecurityChecker


# Files at all. The data are the runtime's variables, and `open` and the path
# modules are refused, but pandas and NumPy read and write files on their own.
# Runs share a machine, so a file one run writes (`/tmp/x.pkl`, `x.pkl` in the
# working directory) another can read. Saving or loading a file, a `/tmp` path,
# and `to_csv`/`to_json` given a destination are refused; `to_csv()`/`to_json()`
# returning text are not.
_FILE_IO = [
    (r"\.(?:to_(?:pickle|parquet|feather|hdf|excel|stata|orc|sql)"
     r"|read_(?:csv|pickle|parquet|feather|hdf|excel|table|fwf|stata|orc|json|html|xml|sas"
     r"|spss|clipboard))\s*\(", "reading or writing a file"),
    (r"\.to_(?:csv|json)\s*\(\s*(?![)\s]|(?:index|header|sep|orient|float_format|columns"
     r"|na_rep|date_format|lines|indent|double_precision|date_unit)\s*=)",
     "writing a file"),
    (r"\b(?:np|numpy)\.(?:save|savez|savez_compressed|load|loadtxt|savetxt|genfromtxt"
     r"|fromfile|memmap)\s*\(|\.tofile\s*\(", "reading or writing a file"),
    (r"[\'\"]/tmp\b", "a path in /tmp"),
]


SECURITY_CHECKER = SecurityChecker([
    # The repository's own top-level folders, which import as namespace packages
    # from the kernel's working directory. `datasets` would replace the injected
    # handle of the same name and expose the raw files through `__path__`; the
    # handle itself (`datasets.load(...)`) is not an import and stays usable.
    ImportRule({
        "os", "subprocess", "sys", "shutil", "pathlib", "socket", "urllib",
        "http", "ctypes", "inspect", "importlib", "evals", "cases", "core.data",
        "datasets", "config", "scripts",
        "pickle", "shelve", "tempfile", "joblib",
    }),
    FunctionRule({
        "exec", "compile", "open", "input", "exit", "quit", "__import__",
        "globals", "locals", "eval", "breakpoint",
    }),
    AttributeRule({
        "__globals__", "__code__", "__closure__", "__dict__", "__class__",
        "__bases__", "__mro__", "__subclasses__", "__builtins__",
    }),
    RegexRule(r"(?:evals|cases)[/\\.]", "benchmark answer-key access is forbidden"),
    RegexRule(r"dataset_new[/\\]", "the raw data files are not the runtime's data"),
    *(RegexRule(pattern, label) for pattern, label in _FILE_IO),
])
