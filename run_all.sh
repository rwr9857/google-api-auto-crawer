#!/bin/bash
# 실행 중 오류가 나면 바로 중단
set -e

echo "=== 1_update_docs.py 실행 ==="
python3 1_update_docs.py

echo "=== 2_remove_vs.py 실행 ==="
python3 2_remove_vs.py

echo "=== 3_insert_vs.py 실행 ==="
python3 3_insert_vs.py

echo "✅ 모든 작업 완료!"
