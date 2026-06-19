#!/bin/bash
set -u
cd ~/bio/AttrVR-main
MAIN_PID=$1
echo "[chain] waiting for v2 PID=$MAIN_PID..."
while kill -0 $MAIN_PID 2>/dev/null; do sleep 60; done
echo "[chain] v2 done at $(date). Starting v3."
bash run_rebuttal_campaign_v3.sh > rebuttal_campaign_out/campaign_v3.log 2>&1
echo "[chain] v3 done at $(date). Starting post-campaign."
bash run_post_campaign.sh > rebuttal_campaign_out/post_campaign.log 2>&1
echo "[chain] all done at $(date)."
