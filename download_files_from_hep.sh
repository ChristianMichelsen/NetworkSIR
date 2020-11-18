rsync -e ssh mnv794@hep03.hpc.ku.dk:/groups/hep/mnv794/work/NetworkSIR/db.json .
rsync --update -hraztP -e ssh --info=progress2 --no-inc-recursive mnv794@hep03.hpc.ku.dk:/groups/hep/mnv794/work/NetworkSIR/Data/ABM ./Data/
rsync --update -hraztP -e ssh --info=progress2 --no-inc-recursive mnv794@hep03.hpc.ku.dk:/groups/hep/mnv794/work/NetworkSIR/Data/cfgs ./Data/


# rsync_param="-avr"
# folder_hep="mnv794@hep03.hpc.ku.dk:/groups/hep/mnv794/work/NetworkSIR/Data/ABM"
# folder_local="./Data/ABM2"
# rsync "$rsync_param" -e ssh $folder_hep $folder_local | pv -lep -s $(rsync $rsync_param -n $folder_hep $folder_local | awk 'NF' | wc -l)

