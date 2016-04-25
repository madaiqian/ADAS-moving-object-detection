# moving-object-detection
It's a tool which can find out the moving object based on OpenCV and VS2013.

The  main idea is optical flow:

         as for static background: Harris corners + SAD.
         as for dynamic background: Harris corners + LK sparse optical flow + Fundamental Matrix (epipolar constraint).
         
         
![](https://github.com/madaiqian/moving-object-detection/blob/master/mod_result/frame_27.jpg)  
![](https://github.com/madaiqian/moving-object-detection/blob/master/mod_result/frame_120.jpg)  
![](https://github.com/madaiqian/moving-object-detection/blob/master/mod_result/frame_136.jpg)  
![](https://github.com/madaiqian/moving-object-detection/blob/master/mod_result/frame_243.jpg)  
![](https://github.com/madaiqian/moving-object-detection/blob/master/mod_result/frame_511.jpg)  
![](https://github.com/madaiqian/moving-object-detection/blob/master/mod_result/frame_1566.jpg)  
