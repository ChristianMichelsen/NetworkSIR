#!/bin/bash
# script to run the python program, save the output to a log file and save the log file.

pythonfile=$1
date_str="$(date '+%Y.%m.%d_%H.%M.%S')"
log_str="${date_str}_log.log"
# python_str="${date_str}_${pythonfile}"

export MPLBACKEND=agg

nohup python -u ${pythonfile} &> "logs/${log_str}" &

rm out.log
ln -s "logs/${log_str}" out.log

tail -f -s 1 "logs/${log_str}"
#less -rf +F "logs/${log_str}"

#nohup ./process &> "$TEMP_LOG_FILE" & tail -f "$TEMP_LOG_FILE" &

