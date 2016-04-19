# moving-object-detection
It's a tool which can find out the moving object based on OpenCV and VS2013.

The  main idea is optical flow:

         as for static background: Harris corners + SAD.
         as for dynamic background: Harris corners + LK sparse optical flow + Fundamental Matrix (epipolar constraint).
