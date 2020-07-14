install_dir="/groups/hep/mnv794/gifsicle/"

printf "\nInstalling gifsicle into ${install_dir}\n\n"
mkdir $install_dir
mkdir gifsicle_tmp
cd gifsicle_tmp
printf "\nDownload gifsicle\n\n"
git clone https://github.com/kohler/gifsicle.git
cd gifsicle
autoreconf -i
printf "\nconfigure and install\n\n"
./configure --disable-gifview --prefix=$install_dir
make install
cd ..
cd ..
rm -rf gifsicle_tmp
printf "\nFinished installing! \n\n"
printf "Remember to add the following line to your .bashrc/profile \n"
printf "export PATH=${install_dir}bin:\$PATH"