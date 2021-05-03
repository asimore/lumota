import os
import sys
import time

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

        cmd = os.system('sh /home/pi/projects/lumin/Lumin_FW_Src/audio_application/python/lumingit.sh')
        exit_code = os.WEXITSTATUS(cmd)
        if exit_code == 0:
            model_name = check_version()
            print (model_name)
            if model_name.upper().strip() == 'NO_UPDATES':
               print ('no updates found.')
            else:
                # os.system("espeak --stdout 'New updates are available, disabling the system to update.' | aplay -Dsysdefault")

                os.system('/usr/sbin/service luminled stop')
                os.system('sh /home/pi/projects/lumin/Lumin_FW_Src/audio_application/python/luminota.sh ' + model_name)
                os.system('cp ./LED_control.py /home/pi/projects/lumin/Lumin_FW_Src')
                os.system('/usr/sbin/service luminled start')
                time.sleep(2)
                # os.system("espeak --stdout 'Update is done, system is online.' | aplay -Dsysdefault")

        print("testing")
        # os.system('cd /home/pi/projects/lumin/Lumin_FW_Src/audio_application/rec;git init;git remote add origin https://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@github.com/luminota/lumindata.git;git add --all;git commit -am "first commit";git pull origin master;git push -u origin master')
        os.system('cd /home/pi/projects/lumin/Lumin_FW_Src/audio_application/rec;git init;git remote add origin https://luminota:lumin0$%^@github.com/luminota/lumindata.git;git add --all;git commit -am "first commit";git pull origin master;git commit -am "first commit";git push -u origin master')



exit(0)
