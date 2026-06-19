#!/bin/bash
cd ~/bio/AttrVR-main
echo '=== TIME:' $(date) '==='
echo '=== GPU ==='
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv | column -t
echo
echo '=== ACTIVE TRAINING PROC ==='
pgrep -af 'fs_BiomedVR_V8_rebuttal2' | head -3
echo
started=$(grep -c '^===== \[.*\] START:' rebuttal_campaign_out/campaign.log 2>/dev/null)
ended=$(grep -c '^===== END' rebuttal_campaign_out/campaign.log 2>/dev/null)
echo "=== CAMPAIGN PROGRESS:  STARTed=$started  completed=$ended  ==="
echo
echo '=== last 5 events ==='
tail -10 rebuttal_campaign_out/campaign.log
echo
echo '=== best-acc per finished dataset/seed ==='
for d in logs_rebuttal_5seed logs_rebuttal_negmode logs_rebuttal_mask logs_rebuttal_perclass logs_rebuttal_poscorrupt logs_rebuttal_post; do
    [ -d $d ] || continue
    echo "  -- $d --"
    for f in $d/*.log; do
        [ -f $f ] || continue
        bn=$(basename $f .log)
        [[ $bn == *_cs ]] && continue
        last=$(grep 'Best Acc=' $f 2>/dev/null | tail -1)
        epoch=$(echo "$last" | sed -E 's/Epoch=([0-9]+).*/\1/')
        best=$(echo "$last" | sed -E 's/.*Best Acc=([0-9.]+).*/\1/')
        printf '    %-30s ep=%-4s best=%s\n' "$bn" "$epoch" "$best"
    done
done
