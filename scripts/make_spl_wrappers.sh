#!/bin/bash
# 为串行草稿（prompt lookup 续接草稿链）造带环境变量的启动包装脚本。
#   spl_mtp8_keep4.sh   SPL_KEEP=4（主草稿保留前 4 个，其余由 prompt lookup 续接）
#   spl_mtp8_keep0.sh   SPL_KEEP=0（整条链都由 prompt lookup 给，等于链内纯 NGRAM）
set -eu
cd /mnt/sda/qyj1044

for keep in 4 0; do
    cat > "spl_mtp8_keep${keep}.sh" <<EOF
#!/bin/bash
export SPL_ENABLE=1
export SPL_KEEP=${keep}
export SPL_MAXL=3
export SPL_WINDOW=4096
exec bash /mnt/sda/qyj1044/fp4_lsa_mtp8_kvbf16.sh
EOF
    chmod +x "spl_mtp8_keep${keep}.sh"
    echo "written spl_mtp8_keep${keep}.sh"
done

echo "=== spl_mtp8_keep4.sh ==="
cat spl_mtp8_keep4.sh
