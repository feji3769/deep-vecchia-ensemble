apt remove --purge --auto-remove cmake
DEBIAN_FRONTEND="noninteractive" apt-get install wget -y
apt-get update
apt-get upgrade g++
apt-get upgrade -y
# need to install newest cmake because FAISS aggresively bumps version.
DEBIAN_FRONTEND="noninteractive" apt-get install software-properties-common -y
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
DEBIAN_FRONTEND="noninteractive" apt-get install $(cat /cfg/pkglist) -y