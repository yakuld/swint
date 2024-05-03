import os
import fileinput
import sys

path = '/home/yakul/dataset/CelebDF/video_frames'

def replaceFrameCount(video):

    framesInDir = len(os.listdir(os.path.join(path, 'videos', video)))
    framesInTxt = -1
    replace = False
    # files = [ os.path.join(path, x) for x in ['train.txt', 'val.txt', 'test.txt']]
    for txt in ['train.txt', 'val.txt', 'test.txt']:
        for line in fileinput.input( os.path.join(path, txt), inplace=1):
            if video in line:
                framesInTxt = int(line.split()[2]) 
                if framesInTxt > framesInDir:
                    line = line.replace(str(framesInTxt), str(framesInDir))  
                    replace = True
                elif framesInTxt == framesInDir:
                    replace = False
            sys.stdout.write(line)
    
    return replace

for video in os.listdir(os.path.join(path, 'videos')):
    video_path = os.path.join(path, 'videos', video)
    isReplaced = replaceFrameCount(video)
    if isReplaced:
        frame_list = [int(x[:5]) for x in os.listdir(video_path)]
        frame_list.sort()
        for i, frameNo in enumerate(frame_list, 1):
            if(frameNo != i):
                os.rename(os.path.join(video_path, '{:05}.jpg'.format(frameNo)), os.path.join(video_path, '{:05}.jpg'.format(i)))


