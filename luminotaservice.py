import os
import sys
import time
import requests

sys.path.append('/home/pi/projects/lumin/Lumin_FW_Src/audio_application/python/lumota')
from pixels import Pixels

def check_version():
    fd = os.open('/home/pi/projects/lumin/Lumin_FW_Src/audio_application/python/lumota/mversion.txt', os.O_RDONLY)
    readBytes = os.read(fd, 50)

    os.close(fd)

    return readBytes.decode('utf-8')


if __name__ == '__main__':
    pixels = Pixels()
    pixels.ota()

    # os.system("espeak --stdout 'Updating, please wait.' | aplay -Dsysdefault")

    timezone = None
    while True:
        r = requests.get("http://worldtimeapi.org/api/ip.txt")
        out = r.content.decode('utf-8')

        for line in out.splitlines():
            tokens = line.split(':')
            if tokens[0] == 'timezone':
                timezone = (tokens[-1]).strip()
                os.system("timedatectl set-timezone {0}".format(timezone))
                break

        if timezone is not None:
            break
        time.sleep(10)

    cmd = os.system('sh /home/pi/projects/lumin/Lumin_FW_Src/audio_application/python/lumingit.sh')
    exit_code = os.WEXITSTATUS(cmd)
    if exit_code == 0:
        model_name = check_version()
        print (model_name)
        if model_name.upper().strip() == 'NO_UPDATES':
           print ('no updates found.')
        else:
            os.system("espeak --stdout 'New updates are available, disabling the system to update.' | aplay -Dsysdefault")

            os.system('/usr/sbin/service luminled stop')
            os.system('sh /home/pi/projects/lumin/Lumin_FW_Src/audio_application/python/luminota.sh ' + model_name)
            os.system('cp ./LED_control.py /home/pi/projects/lumin/Lumin_FW_Src')
            os.system('/usr/sbin/service luminled start')
            time.sleep(2)
            os.system("espeak --stdout 'Update is done, system is online.' | aplay -Dsysdefault")

    # os.system('cd /home/pi/projects/lumin/Lumin_FW_Src/audio_application/rec;rm -rf .git;git init;git remote add origin https://luminota:lumin0$%^@github.com/luminota/lumindata.git;git add --all;git commit -am "first commit";git pull origin master --no-edit --allow-unrelated-histories;git commit -am "2nd commit";git push origin master')



    exit(0)
