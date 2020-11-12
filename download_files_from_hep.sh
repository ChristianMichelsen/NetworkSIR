rsync -e ssh mnv794@hep03.hpc.ku.dk:/groups/hep/mnv794/work/NetworkSIR/db.json .
rsync --update -hraztP -e ssh --info=progress2 --no-inc-recursive mnv794@hep03.hpc.ku.dk:/groups/hep/mnv794/work/NetworkSIR/Data/ABM ./Data/
