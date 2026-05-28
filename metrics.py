"""
metrics.py
==========
Avalia predições Text-to-SQL e gera um relatório em arquivo de texto.

Métricas calculadas:
  - Syntax Validity
  - Execution Accuracy
  - Exact Match (AST)

Formatos de entrada aceitos:
  1. JSON — lista de objetos com chaves: db_id, question, gold, predicted
  2. Blocos de texto (chave<TAB>valor separados por linha em branco):
        db_id\t"concert_singer"
        question\t"How many singers?"
        gold\t"SELECT count(*) FROM singer"
        predicted\t"SELECT count(*) FROM singer"

Uso:
    python metrics.py <predictions_file> [--db_dir DIR] [--output FILE] [--format json|text|auto]

Exemplos:
    python metrics.py predictions.json
    python metrics.py predictions.json --db_dir spider/database --output report.txt
    python metrics.py predictions.txt  --format text
"""

import argparse
import json
import os
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import sqlglot


# ---------------------------------------------------------------------------
# Normalização
# ---------------------------------------------------------------------------

def normalize_value(v):
    if v is None:
        return None
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, str):
        return v.strip().lower()
    return v


def normalize_rows(rows):
    """Normaliza linhas brutas do cursor. Chamar uma única vez por resultado."""
    normalized = [tuple(normalize_value(v) for v in row) for row in rows]
    sorted_rows     = sorted(normalized, key=str)
    frozenset_rows  = sorted((frozenset(row) for row in normalized), key=str)
    return frozenset_rows, sorted_rows


# ---------------------------------------------------------------------------
# Execução SQL
# ---------------------------------------------------------------------------

def _db_path(db_dir: str, db_id: str) -> str:
    return os.path.join(db_dir, db_id, f"{db_id}.sqlite")


def _connect(db_dir: str, db_id: str):
    path = _db_path(db_dir, db_id)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Banco não encontrado: {path}")
    return sqlite3.connect(path)


def execute_query(db_dir: str, db_id: str, sql: str):
    """Retorna (frozenset_list, sorted_list) ou None em caso de erro."""
    conn = None
    try:
        conn = _connect(db_dir, db_id)
        cur  = conn.cursor()
        cur.execute(sql)
        return normalize_rows(cur.fetchall())
    except Exception:
        return None
    finally:
        if conn:
            conn.close()


def syntax_valid(db_dir: str, db_id: str, sql: str) -> bool:
    conn = None
    try:
        conn = _connect(db_dir, db_id)
        conn.cursor().execute(sql)
        return True
    except Exception:
        return False
    finally:
        if conn:
            conn.close()


# ---------------------------------------------------------------------------
# Métricas individuais
# ---------------------------------------------------------------------------

def execution_accuracy(db_dir: str, db_id: str, gold: str, pred: str) -> bool:
    gold_res = execute_query(db_dir, db_id, gold)
    pred_res = execute_query(db_dir, db_id, pred)
    if gold_res is None or pred_res is None:
        return False
    gold_fs, gold_s = gold_res
    pred_fs, pred_s = pred_res
    return gold_fs == pred_fs or gold_s == pred_s


def exact_match(gold: str, pred: str) -> bool:
    try:
        return sqlglot.parse_one(gold) == sqlglot.parse_one(pred)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Classificação de erros
# ---------------------------------------------------------------------------

def classify_error(db_dir: str, db_id: str, pred_sql: str, gold_sql: str) -> str:
    if not pred_sql or not pred_sql.strip():
        return "empty"

    if execute_query(db_dir, db_id, pred_sql) is None:
        return "invalid_sql"

    pred_lower = pred_sql.lower()
    gold_lower = gold_sql.lower()

    for keyword in ["group by", "having", "intersect", "except", "order by"]:
        if keyword in gold_lower and keyword not in pred_lower:
            return "missing_" + keyword.replace(" ", "_")

    if "join" in gold_lower and "join" not in pred_lower:
        where_idx   = pred_lower.find("where")
        has_subquery = "select" in pred_lower[where_idx:] if where_idx != -1 else False
        if not has_subquery:
            return "missing_join"

    return "wrong_columns_or_values"


# ---------------------------------------------------------------------------
# Parser de entrada
# ---------------------------------------------------------------------------

def _parse_text_blocks(text: str) -> list[dict]:
    records = []
    for block in re.split(r"\n{2,}", text.strip()):
        record = {}
        for line in block.strip().splitlines():
            if "\t" not in line:
                continue
            key, _, value = line.partition("\t")
            record[key.strip()] = value.strip().strip('"')
        if {"db_id", "gold", "predicted"}.issubset(record):
            records.append(record)
    return records


def load_predictions(path: str, fmt: str = "auto") -> list[dict]:
    text = Path(path).read_text(encoding="utf-8")
    if fmt == "auto":
        fmt = "json" if text.lstrip().startswith(("[", "{")) else "text"
    if fmt == "json":
        raw = json.loads(text)
        return raw if isinstance(raw, list) else [raw]
    return _parse_text_blocks(text)


# ---------------------------------------------------------------------------
# Avaliação
# ---------------------------------------------------------------------------

def evaluate(predictions: list[dict], db_dir: str) -> list[dict]:
    results = []
    for p in predictions:
        db_id = p["db_id"]
        gold  = p["gold"]
        pred  = p["predicted"]

        syn = syntax_valid(db_dir, db_id, pred)
        ex  = execution_accuracy(db_dir, db_id, gold, pred)
        em  = exact_match(gold, pred)

        results.append({
            **p,
            "syntax_valid":       syn,
            "execution_accuracy": ex,
            "exact_match":        em,
            "error_type":         None if ex else classify_error(db_dir, db_id, pred, gold),
        })
    return results


# ---------------------------------------------------------------------------
# Geração do relatório
# ---------------------------------------------------------------------------

SEP  = "=" * 80
DASH = "-" * 80


def build_report(evaluated: list[dict]) -> str:
    lines = []
    w = lines.append                       # alias para append

    total = len(evaluated)
    if total == 0:
        w("Nenhuma predição avaliada.")
        return "\n".join(lines)

    n_syn = sum(p["syntax_valid"]       for p in evaluated)
    n_ex  = sum(p["execution_accuracy"] for p in evaluated)
    n_em  = sum(p["exact_match"]        for p in evaluated)

    # ── Métricas globais ────────────────────────────────────────────────────
    w(SEP)
    w("GLOBAL METRICS")
    w(SEP)
    w(f"Total Examples       : {total}")
    w(f"Syntax Validity      : {n_syn/total:.4f}  ({n_syn}/{total})")
    w(f"Execution Accuracy   : {n_ex/total:.4f}  ({n_ex}/{total})")
    w(f"Exact Match (AST)    : {n_em/total:.4f}  ({n_em}/{total})")

    # ── Métricas por banco ──────────────────────────────────────────────────
    by_db = defaultdict(list)
    for p in evaluated:
        by_db[p["db_id"]].append(p)

    if len(by_db) > 1:
        w("")
        w(SEP)
        w("EXECUTION ACCURACY PER DATABASE")
        w(SEP)
        w(f"{'db_id':<35}  {'EA':>6}  {'n':>5}")
        w(DASH)
        for db_id, rows in sorted(by_db.items()):
            ea = sum(r["execution_accuracy"] for r in rows) / len(rows)
            w(f"{db_id:<35}  {ea:>6.4f}  {len(rows):>5}")

    # ── Queries inválidas ───────────────────────────────────────────────────
    invalid = [p for p in evaluated if not p["syntax_valid"]]
    w("")
    w(SEP)
    w(f"INVALID SQL  ({len(invalid)} of {total})")
    w(SEP)

    if not invalid:
        w("Nenhuma query inválida.")
    else:
        for i, p in enumerate(invalid, 1):
            w(f"\n[{i}] Question : {p.get('question', '-')}")
            w(f"    db_id     : {p['db_id']}")
            w(f"    Gold      : {p['gold']}")
            w(f"    Predicted : {p['predicted']}")
            w(DASH)

    # ── Distribuição de erros ───────────────────────────────────────────────
    errors = [p for p in evaluated if not p["execution_accuracy"]]

    w("")
    w(SEP)
    w(f"ERROR DISTRIBUTION  ({len(errors)} errors of {total} examples)")
    w(SEP)

    if not errors:
        w("Nenhum erro de execução.")
    else:
        error_counter = Counter(p["error_type"] for p in errors)
        w(f"{'error_type':<40}  {'count':>5}  {'%':>6}")
        w(DASH)
        for etype, count in error_counter.most_common():
            pct = 100 * count / len(errors)
            w(f"{etype:<40}  {count:>5}  {pct:>5.1f}%")

        # Exemplos agrupados por tipo
        grouped = defaultdict(list)
        for p in errors:
            grouped[p["error_type"]].append(p)

        for etype, examples in sorted(grouped.items()):
            w("")
            w(SEP)
            w(f"ERROR TYPE : {etype}  ({len(examples)} cases)")
            w(SEP)
            for i, p in enumerate(examples, 1):
                w(f"\n[{i}] Question : {p.get('question', '-')}")
                w(f"    db_id     : {p['db_id']}")
                w(f"    Gold      : {p['gold']}")
                w(f"    Predicted : {p['predicted']}")
                w(DASH)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Avalia predições Text-to-SQL e gera relatório em texto."
    )
    parser.add_argument(
        "predictions",
        help="Arquivo de predições (JSON ou blocos de texto)."
    )
    parser.add_argument(
        "--db_dir",
        default= "/mnt/storage_C1/igorzwirtes/poster_ic/spider/database",
        help="Diretório raiz com os bancos SQLite. "
             "Padrão: variável de ambiente SPIDER_DB_DIR ou './database'."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Arquivo de saída para o relatório. "
             "Padrão: <predictions_stem>_report.txt no mesmo diretório."
    )
    parser.add_argument(
        "--format",
        choices=["json", "text", "auto"],
        default="auto",
        help="Formato do arquivo de predições (padrão: auto)."
    )
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        raise SystemExit(f"Arquivo não encontrado: {pred_path}")

    if args.output is None:
        output_path = pred_path.with_name(pred_path.stem + "_report.txt")
    else:
        output_path = Path(args.output)

    print(f"Carregando: {pred_path}")
    predictions = load_predictions(str(pred_path), args.format)
    print(f"  {len(predictions)} exemplos carregados.")

    print(f"Avaliando com db_dir={args.db_dir} ...")
    evaluated = evaluate(predictions, args.db_dir)

    report = build_report(evaluated)

    output_path.write_text(report, encoding="utf-8")
    print(f"Relatório salvo em: {output_path}")

    # Imprime métricas globais no terminal também
    for line in report.splitlines():
        if any(k in line for k in ("GLOBAL", "Total", "Syntax", "Execution", "Exact")):
            print(line)


if __name__ == "__main__":
    main()