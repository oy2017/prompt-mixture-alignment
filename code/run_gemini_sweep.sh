#!/bin/bash
# Full MobLab sweep on a Gemini backbone: EM fit + eval per game, then GB.
# Quota-resilient: providers.api_call sleeps through daily-quota exhaustion,
# so this script can run unattended across quota resets.
#
# Usage: PMA_PROVIDER=gemini ./run_gemini_sweep.sh   (or gemini-pro)
set -uo pipefail
cd "$(dirname "$0")"
export PMA_PROVIDER=${PMA_PROVIDER:-gemini}
GEMINI_API_KEY=$(grep GEMINI_API_KEY ~/.bashrc | cut -d"'" -f2)
export GEMINI_API_KEY
PY=~/venvs/pma/bin/python
DEST=../reproduction

k_for() { [ "$1" = "Public_Goods" ] && echo 10 || echo 50; }

echo "=== sweep on $PMA_PROVIDER start: $(date) ==="

# --- EM: fit, evaluate with own weights, archive ---
for GAME in Dictator Proposer Responder Investor Banker Bomb Public_Goods; do
  OUT="$DEST/level3_${PMA_PROVIDER}_EM_${GAME}"
  # Public_Goods flash/pro EM already done under different dir names; skip if archived
  if [ -e "$OUT" ] || { [ "$GAME" = "Public_Goods" ] && ls $DEST/level3_gemini*_EM_Public_Goods >/dev/null 2>&1; }; then
    echo "skip EM $GAME (already archived)"; continue
  fi
  echo "--- EM $GAME (K=$(k_for $GAME)) $(date) ---"
  $PY EM_moblab.py --game "$GAME" --K "$(k_for $GAME)" --runs 1 || { echo "EM $GAME FAILED"; continue; }
  FINAL=$(ls "$GAME"/1_result/*_investor_EM_prompts_updated.csv 2>/dev/null | sort -V | tail -1)
  [ -z "$FINAL" ] && { echo "EM $GAME produced no prompts file"; continue; }
  $PY replay_eval.py --alg EM --game "$GAME" --workers 8 \
      --prompts-csv "$FINAL" --weights-pkl "$GAME/1_weights_lst.pkl" \
      --tag "level3_${PMA_PROVIDER}_EM_${GAME}" || echo "EM eval $GAME FAILED"
  mv "$GAME" "$OUT"
done

# --- GB: fit only (eval needs a weights parser; done host-side afterwards) ---
for GAME in Public_Goods Dictator Proposer Responder Investor Banker Bomb; do
  OUT="$DEST/level3_${PMA_PROVIDER}_GB_${GAME}"
  [ -e "$OUT" ] && { echo "skip GB $GAME"; continue; }
  echo "--- GB $GAME (maxIter=60) $(date) ---"
  $PY GB_moblab.py --game "$GAME" --runs 1 --maxIter 60 || { echo "GB $GAME FAILED"; continue; }
  mv "$GAME" "$OUT"
done

echo "=== sweep on $PMA_PROVIDER done: $(date) ==="
