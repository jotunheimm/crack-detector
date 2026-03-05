import cv2, os, glob, shutil

base = os.path.dirname(cv2.__file__)
dlls = glob.glob(os.path.join(base, "*ffmpeg*.dll"))
for dll in dlls:
    shutil.copy(dll, ".")
    print("Copied:", dll)

if not dlls:
    print("No ffmpeg DLLs found — skipping")
