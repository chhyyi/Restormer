#!/bin/bash
#description : This is temporal shell script written by changhyun yi, 2023 OCT 6th. Hard coded! Should be executed outside of the docker container!
#2024-02-22 : renamed to 'init_container', as I required some script inside of the container...

: ${1?"Usage: $0 need argument. mount or umount or torch"} #this is parameter substitution. ${parameter?err_msg}

for i do
    if [ "$i" = "mount" ]; then
        echo "
        ######Examples to mount & share dir via LAN######

        # #---mount 2TB drive on local--- (used until 2024-01-11)
        #dataset_drive="/dev/sda2"
        #mount_directory="/media/twotb"
        #mount \$dataset_drive \$mount_directory

        #share it with samba. /etc/samba/smb.conf should be configured first
        #echo "try to restart smbd"
        #service smbd restart

        # #---Mount document folder on windows desktop
        # read -p "client ip?" client_ip

        # echo \$client_ip
        #client_ip="10.72.20.88"
        #mount_directory_shared="/home/chyi/shared/aberration_sbcho"
        # echo "try to mount mounted //\$client_ip/Users/ch/Documents/aberration_sbcho to \$mount_directory_shared"
        # sudo mount -t cifs -o username=ch_linux //\$client_ip/Users/ch/Documents/aberration_sbcho \$mount_directory_shared
        
        # # ---- cifs directory share from windows to linux server (20240912) ----
        # ## share directoryon windows (with administrator privillage):
        # net share chyi_windows=D:\dataset_shared /GRANT:ch_linux,FULL 
        # ## mount at ubuntu
        # sudo mount -t cifs -o username=ch_linux //10.72.20.88/chyi_windows /media/win_shared/
        "

    elif [ "$i" = "NGC_torch" ]; then
        echo "running pytorch:22.10:container"
        sudo docker run -d --gpus all -it --rm --ipc=host\
        --mount type=bind,source="/media/twotb/datasets",target=/app \
        --mount type=bind,source="/home/chyi/Restormer",target=/root/project \
        nvcr.io/nvidia/pytorch:22.10-py3

    elif [ "$i" = "init" ]; then
        pip install tifffile==2022.8.12 einops scikit-image==0.21 aotools
        
    fi
done