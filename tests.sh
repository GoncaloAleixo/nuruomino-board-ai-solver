#!/usr/bin/env bash

# (1) Diretório onde estão os testes
TESTDIR="sample-nuruominoboards1"

# (2) Pasta de saída gerada
TMPDIR="__test_outputs__"

# (3) Limpa/Cria a pasta de outputs gerados
if [ -d "$TMPDIR" ]; then
    rm -rf "$TMPDIR"
fi
mkdir "$TMPDIR"

any_failed=0

for infile in "$TESTDIR"/*.txt; do
    base=$(basename "$infile" .txt)

    expected="$TESTDIR/${base}.out"
    actual="$TMPDIR/actual_${base}.out"

    python3 nuruomino.py < "$infile" > "$actual"

    if cmp -s "$actual" "$expected"; then
        echo "$base: OK"
    else
        echo "$base: FAIL"
        …
    fi
done

if [ $any_failed -eq 0 ]; then
    echo
    echo "===== TODOS OS TESTES PASSARAM ====="
    exit 0
else
    echo
    echo "===== AO MENOS UM TESTE FALHOU ====="
    exit 1
fi