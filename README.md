COMP 9517 Computer Vision S1, 2015
==================================

Group Project
-------------

This is a two-person project. Pick a project team member, and register your group 

ONLINE by MONDAY week 5.

Project Synopsis
----------------

The project consists of two parts, with checkpoints built in to enable you to manage the 

project and the deliverables. Read the project specification through carefully, and discuss 

with potential team member before registering your group!

Part 1
------

This part is common to all groups.

Image segmentation, feature description and object tracking form the foundation of many successful applications of computer vision. The objective of this task is for you to become familiar with these techniques and their implementation in OpenCV.

Given a video with several (upto 3) objects of interest, the tasks are to detect the objects and track them through the video.

1) You will need a video that contains objects that appear in most video frames, and 

in particular in the first frame. This first frame may be used as a model frame.

2) You can then use feature descriptors to match objects in the model frame to those 

in each frame (image) of the sequence.

3) You should be able to display the estimated object locations in each frame in the 

video.

4) You should also display the ongoing object trajectories, based on the estimated 

object locations in each frame in the video.

Here are examples of the output with a single tracked object:

http://www.youtube.com/watch?v=3dY4uvSwiwE 

http://www.youtube.com/watch?v=8q0h1VJLIpM

There will be two bonus marks for implementing additional features such as real-time tracking (at least 10+ fps), or the ability to automatically track occluded objects in some of the video frames using techniques such as the Kalman/Particle filter.

Testing of Part 1
-----------------

In the assignment section of the course website, you will see links to a video sequence that you may use for testing during development. On the day of evaluation of Part 1, you will demo your program on another similar video sequence.

You should also provide a TWO page report giving a high level overview of your approach and results.

Part 2
------

In the second part, your team should propose ONE direction of further work that builds on Part 1, do the necessary literature survey on techniques and algorithms, implement them and write a detailed report.

You should also find suitable datasets to work with. The week 9 presentation (see checkpoints below) will provide an opportunity to get feedback from lecturers on your chosen direction.

Your team is encouraged to do its own research and decide on a line of further development. Use the week 9 presentation or discuss with us to get feedback.

You will demo your program on your chosen datasets in week 13 and submit a report that explains your approach and evaluation results.

Possible directions
-------------------

A short list of top projects from the class in previous years is separately posted.

Checkpoints

Week 6 Demo on Part 1 in the lab (10 marks)

Week 9 Presentation + Report in the lecture class (10 marks)

Week 13 Final Demo in lab + Performance Evaluation + Report (10 + 10 + 10

marks)

Total marks: 50

Copyright: Arcot Sowmya, CSE, UNSW, with acknowledgments to COMP 9517 

teaching team past and present

25.03.2015
