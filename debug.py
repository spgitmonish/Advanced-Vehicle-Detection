# Import movie editor files
from moviepy.editor import VideoFileClip
from scipy import misc

def debugRun():
    project_clip = VideoFileClip("project_video.mp4")
    frameCount = 0
    for frame in project_clip.iter_frames():
        print("FC: " + str(frameCount))
        filename = "video_images/FC" + str(frameCount) + ".jpg"
        # Save file
        misc.imsave(filename, frame)
        frameCount = frameCount + 1

debugRun()
