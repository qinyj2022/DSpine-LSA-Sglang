#!/bin/bash
# Generate the four 200K-context launch variants from the standard scripts.
# Changed: --context-length 262144, per-step logging, and for the speculative
# paths a smaller mamba cache plus a higher static fraction so the KV pool can
# hold 200K tokens (0.82 -> 0.85, mamba 64 -> 16).
set -e
cd /mnt/sda/qyj1044

mk() {
  base=$1
  out=$2
  mamba=$3
  mem=$4
  sed -e 's/--context-length 131072/--context-length 262144/' \
      -e "s/--max-mamba-cache-size [0-9]*/--max-mamba-cache-size $mamba/" \
      -e "s/--mem-fraction-static [0-9.]*/--mem-fraction-static $mem/" \
      -e 's/^  --host 0.0.0.0 --port 30000/  --decode-log-interval 1 --host 0.0.0.0 --port 30000/' \
      "$base" > "$out"
  echo "--- $out (from $base) ---"
  grep -E 'context-length|mamba-cache-size|mem-fraction-static|decode-log-interval|attention-backend |page-size|max-running-requests|speculative-algorithm' "$out"
}

mk dense_base.sh dense_200k.sh 64 0.82
mk lsa_uds.sh lsa_200k.sh 64 0.85
mk bf16_src_uds.sh ds_200k.sh 16 0.85
mk graph_lsa.sh combo_200k.sh 16 0.85
