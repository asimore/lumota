import os
import sys

def check_version():
    fd = os.open('/home/pi/projects/lumin/Lumin_FW_Src/audio_application/python/lumota/mversion.txt', os.O_RDONLY)
    readBytes = os.read(fd, 50)

    os.close(fd)

    return readBytes.decode('utf-8')

if __name__ == '__main__':

        os.system('sh /home/pi/projects/lumin/Lumin_FW_Src/audio_application/python/lumingit.sh')
        model_name = check_version()
        print (model_name)
        if model_name.upper().strip() == 'NO_UPDATES':
           print ('no updates found.')
        else:
           os.system('sh /home/pi/projects/lumin/Lumin_FW_Src/audio_application/python/luminota.sh ' + model_name)
           os.system('service luminled stop')
           os.system('cp ./LED_control.py /home/pi/projects/lumin/Lumin_FW_Src')
           os.system('service luminled start')
exit(0)
