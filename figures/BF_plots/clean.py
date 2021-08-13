import os

for folder in os.listdir():
    try:
        int(folder)
        for filename in os.listdir(folder+'/'):
            if '.png' in filename:
                os.remove(folder + '/' + filename)
    except ValueError:
        pass