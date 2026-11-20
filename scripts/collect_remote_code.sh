#!/bin/bash
# Pack the DSpine + LSA adapted code from the remote host into two tarballs:
#   1. remote_code.tgz        -- our own trees (dspine plugin, lsa_code, patch scripts, launchers)
#   2. remote_site_packages.tgz -- only the site-packages files carrying our [PATCH markers
# plus MANIFEST.txt listing every path and its sha256.
set -e
SP=/mnt/nvme1n1/sglang522/lib/python3.12/site-packages
ROOT=/mnt/sda/qyj1044
PACK=$ROOT/pack
rm -rf "$PACK"
mkdir -p "$PACK"

# --- 1. our own trees -------------------------------------------------------
cd "$ROOT"
tar czf "$PACK/remote_code.tgz" \
  --exclude='__pycache__' --exclude='*.pyc' \
  dspine_code/DSpine-main lsa_code

# launch scripts + patch scripts (small, keep provenance)
shopt -s nullglob
scripts=( *.sh patch_*.py make_*.py *_probe.py step_stats.py mamba_math.py layer_map.py )
tar czf "$PACK/remote_scripts.tgz" --exclude='__pycache__' "${scripts[@]}"

# --- 2. patched site-packages files ----------------------------------------
grep -rlF '[PATCH' "$SP/sglang" > "$PACK/patched.txt" || true
grep -rlF '[PATCH' "$SP/lsa" >> "$PACK/patched.txt" || true
sort -u "$PACK/patched.txt" -o "$PACK/patched.txt"
echo "patched files: $(wc -l < "$PACK/patched.txt")"

cd "$SP"
sed "s|^$SP/||" "$PACK/patched.txt" > "$PACK/patched_rel.txt"
tar czf "$PACK/remote_site_packages.tgz" --exclude='__pycache__' --exclude='*.pyc' -T "$PACK/patched_rel.txt"

# whole lsa package too (its patches are spread across many files)
tar czf "$PACK/remote_lsa_pkg.tgz" --exclude='__pycache__' --exclude='*.pyc' lsa

# --- 3. manifest -----------------------------------------------------------
{
  echo "# packed from $(hostname) at $(date -Is)"
  echo "# site-packages: $SP"
  echo
  echo "## tarballs"
  ( cd "$PACK" && sha256sum remote_code.tgz remote_scripts.tgz remote_site_packages.tgz remote_lsa_pkg.tgz )
  echo
  echo "## patched files (relative to site-packages)"
  cat "$PACK/patched_rel.txt"
  echo
  echo "## patch marker lines per file"
  cd "$SP"
  while read -r rel; do
    printf '%s\n' "--- $rel"
    grep -oF '[PATCH' "$SP/$rel" | wc -l | sed 's/^/    markers: /'
  done < "$PACK/patched_rel.txt"
} > "$PACK/MANIFEST.txt"

echo
echo "== sizes =="
ls -lh "$PACK"/*.tgz
echo
echo "== totals =="
echo "patched sglang files : $(grep -c '^sglang/' "$PACK/patched_rel.txt" || true)"
echo "patched lsa files    : $(grep -c '^lsa/' "$PACK/patched_rel.txt" || true)"
