import pytest

from src.ingestion.cleaner import normalize_text, tokenize

LOG_CAMEL = "PermissionDenied: User does not have SELECT on table main.crm.customers"
LOG_SNAKE = "PERMISSION_DENIED: principal lacks SELECT privilege on main.crm.customers"


def test_permission_denied_camel_case():
    assert tokenize(LOG_CAMEL) == [
        "permission", "denied", "user", "not", "select", "table", "main", "crm", "customers",
    ]


def test_permission_denied_upper_snake():
    assert tokenize(LOG_SNAKE) == [
        "permission", "denied", "principal", "lacks", "select", "privilege",
        "main", "crm", "customers",
    ]


def test_variants_share_tokens():
    shared = set(tokenize(LOG_CAMEL)) & set(tokenize(LOG_SNAKE))
    assert {"permission", "denied", "select", "main", "crm", "customers"} <= shared


def test_dotted_identifier_split():
    assert tokenize("main.crm.customers") == ["main", "crm", "customers"]


def test_brackets_backticks_and_duplicates_kept():
    text = "[UNRESOLVED_COLUMN.WITH_SUGGESTION] A column with name `order_total` cannot be resolved."
    assert tokenize(text) == [
        "unresolved", "column", "suggestion", "column", "name",
        "order", "total", "cannot", "resolved",
    ]


def test_out_of_memory_camel():
    assert tokenize("java.lang.OutOfMemoryError: Java heap space") == [
        "java", "lang", "out", "memory", "error", "java", "heap", "space",
    ]


def test_run_id_kept_whole():
    assert tokenize("Why did run r2002 fail?") == ["run", "r2002", "fail"]


def test_numbers_kept():
    assert tokenize("Run exceeded timeout of 1800 seconds.") == [
        "run", "exceeded", "timeout", "1800", "seconds",
    ]


def test_negation_is_not_removed():
    assert "not" in tokenize("User does not have SELECT")
    assert "cannot" in tokenize("table cannot be found")


def test_question_example():
    assert tokenize("Why did my latest pipeline run fail?") == [
        "latest", "pipeline", "run", "fail",
    ]


def test_keep_stopwords_option():
    assert tokenize("The run failed", remove_stopwords=False) == ["the", "run", "failed"]


def test_empty_and_whitespace():
    assert tokenize("") == []
    assert tokenize("   \n\t") == []


def test_punctuation_only():
    assert tokenize("!!! --- ...") == []


def test_non_string_raises():
    with pytest.raises(TypeError):
        tokenize(None)
    with pytest.raises(TypeError):
        tokenize(123)


def test_unicode_combining_accent_is_composed():
    assert tokenize("cafe\u0301 ERROR") == ["caf\u00e9", "error"]


def test_normalize_text_splits_and_folds():
    assert normalize_text("OutOfMemoryError") == "out of memory error"
    assert normalize_text("SQLException") == "sql exception"


def test_tokenize_is_idempotent_on_its_output():
    for s in (LOG_CAMEL, LOG_SNAKE, "Run exceeded timeout of 1800 seconds."):
        once = tokenize(s)
        assert tokenize(" ".join(once)) == once