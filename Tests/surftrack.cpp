//	surftrack.cpp
//	Tarje Sandvik, April 2015
//	2015-04-08 06:35 
//
// 	Opens up a capture from webcam, and lets the user crop out an object from the first frame. 
//	The program then streams the webcam and tracks the object.
//	Made from the tutorial "Features2D + Homography to find a known object":
// 	http://docs.opencv.org/doc/tutorials/features2d/feature_homography/feature_homography.html#feature-homography

#include <stdlib.h>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"

using namespace std;
using namespace cv;

// GLOBAL VARIABLES FOR MOUSEHANDLER
Point mouse_point1, mouse_point2; /* vertical points of the bounding box */
int mouse_drag = 0;
Rect mouse_rect; /* bounding box */
Mat mouse_img, mouse_roiImg; /* roiImg - the part of the image in the bounding box */
int mouse_select_flag = 0;

// mouseHandler function
void mouseHandler(int event, int x, int y, int flags, void* param) {
    if (event == CV_EVENT_LBUTTONDOWN && !mouse_drag) {
        /* left button clicked. ROI selection begins */
        mouse_point1 = Point(x, y);
        mouse_drag = 1;
    }
    if (event == CV_EVENT_MOUSEMOVE && mouse_drag) {
        /* mouse dragged. ROI being selected */
        Mat img1 = mouse_img.clone();
        mouse_point2 = Point(x, y);
        rectangle(img1, mouse_point1, mouse_point2, CV_RGB(255, 0, 0), 3, 8, 0);
        imshow("First frame", img1);
    }
    if (event == CV_EVENT_LBUTTONUP && mouse_drag) {
        mouse_point2 = Point(x, y);
        mouse_rect = Rect(mouse_point1.x,mouse_point1.y,x-mouse_point1.x,y-mouse_point1.y);
        mouse_drag = 0;
        mouse_roiImg = mouse_img(mouse_rect);
    }
    if (event == CV_EVENT_LBUTTONUP) {
       /* ROI selected */
        mouse_select_flag = 1;
        mouse_drag = 0;
    }
}

int main(){

	//some boolean variables for added functionality
	bool trackingEnabled = false; 	// Press 't' to toggle
	bool featuresEnabled = false;	// Press 'f' to toggle
	bool pause = false;				// Press 'p' to toggle
	
	// Timer variables
	time_t start = time(0); 
	time_t diff;
	int frames = 0;

	// minHessian used for detecting features
	int minHessian = 400;
	// Detector, extractor and matcher
	SurfFeatureDetector detector( minHessian );
	SurfDescriptorExtractor extractor;
	FlannBasedMatcher matcher;
	
	
	// image for reading capture
	Mat frame;
	//grayscale images
	Mat grayScene, grayObject;
	// Vectors containing keypoints
	vector<KeyPoint> keypoints_scene, keypoints_object;
	// Descriptor
	Mat descriptors_object, descriptors_scene;
	// image showing matching keypoints (toggle with 'f')
	Mat img_matches;
	// Homography
	Mat H;
	//video capture object.
	VideoCapture capture;

	cout << "SURFTRACK IS RUNNING" << endl;
	// Open video stream from webcam
	capture.open(0);
		if(!capture.isOpened()){
			cout<<"ERROR ACQUIRING VIDEO FEED\n";
			getchar();
			return -1;
		}

	// LET USER CROP OBJECT FROM FIRST FRAME
	// Get the first frame of capture
	capture.read(mouse_img);
	// Show first frame in window
	imshow("First frame", mouse_img);
	// Get points from mousecallback
	cout << "Use mouse to crop object (top-left -> bottom-right). Press 'enter' when satisfied. 'Esc' to quit." << endl;
	cvSetMouseCallback("First frame", mouseHandler, NULL);
	bool wait = true;
	while (wait == true){
		// wait untill enter is pressed
		switch (waitKey()){
		case 10: 	// 'enter' pressed: Finished cropping
			wait = false;
			cout<<"Object cropped from first frame" << endl;
			break;
		case 27:	// 'esc' pressed: Quit program
			return 0;
		}
	}
	// Close window of First frame
	destroyWindow("First frame");

	// Make mouse_roiImg
	mouse_rect = Rect(mouse_point1.x, mouse_point1.y, mouse_point2.x-mouse_point1.x, mouse_point2.y-mouse_point1.y);
	mouse_roiImg = mouse_img(mouse_rect);
	
	// Create mainImage for visualization and make 'object' 'and 'scene' point to positions in it
	Mat mainImage(max(mouse_img.rows, mouse_roiImg.rows), mouse_img.cols + mouse_roiImg.cols, CV_8UC3);
	Mat object(mainImage, Range(0, mouse_roiImg.rows), Range(0, mouse_roiImg.cols));
	Mat scene(mainImage, Range::all(), Range(mouse_roiImg.cols, mouse_roiImg.cols + mouse_img.cols));

	// Copy images into object and scene
	mouse_roiImg.copyTo(object);
	mouse_img.copyTo(scene);

	// DISPLAY MAIN IMAGE
	imshow("Main Image", mainImage);

	// IMAGE PROCESSING FOR OBJECT (outside while-loop, since we only need to do it once)
	//convert object to gray scale
	cvtColor(object, grayObject, COLOR_BGR2GRAY);
	detector.detect( grayObject, keypoints_object );
	extractor.compute( grayObject, keypoints_object, descriptors_object );
	//-- Get the corners from the object
	vector<Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( object.cols, 0 );
	obj_corners[2] = cvPoint( object.cols, object.rows ); obj_corners[3] = cvPoint( 0, object.rows );

		
	cout << "SETUP COMPLETED. PLAYING VIDEO..." << endl 
	<< "'t' - toggles tracking" << endl 
	<< "'f' - toggles feature mode" << endl
	<< "'p' - toggles pause" << endl  
	<< "'esc' - quit program" << endl
	<< endl;
	while(1){
		//read first frame
		capture.read(frame);			

		// IMAGE PROCESSING AND TRACKING


		// put frame in the main image
		frame.copyTo(scene);

		if (trackingEnabled) {
			
			//convert frame to gray scale
			cvtColor(frame, grayScene, COLOR_BGR2GRAY);
			// DETECT the keypoints using SURF Detector
			detector.detect( grayScene, keypoints_scene );
			// EXTRACT			
			extractor.compute( grayScene, keypoints_scene, descriptors_scene );
			// MATCH				
			vector< DMatch > matches;
			matcher.match( descriptors_object, descriptors_scene, matches );

			// FILTER OUT BAD MATCHES
			//-- Quick calculation of max and min distances between keypoints
			double max_dist = 0; double min_dist = 100;
			for( int i = 0; i < descriptors_object.rows; i++ )
			{ double dist = matches[i].distance;
			if( dist < min_dist ) min_dist = dist;
			if( dist > max_dist ) max_dist = dist;
			}

			//-- Keep only good matches (i.e. whose distance is less than 3*min_dist )
			std::vector< DMatch > good_matches;
			for( int i = 0; i < descriptors_object.rows; i++ )
			{ if( matches[i].distance < 3*min_dist )
			{ good_matches.push_back( matches[i]); }
			}

			//-- Localize the object
			vector<Point2f> obj_matches;
			vector<Point2f> scene_matches;

			//-- Get the keypoints from the good matches
			for( int i = 0; i < good_matches.size(); i++ ) {
				obj_matches.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
				scene_matches.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
			}
			

			if(obj_matches.size() >= 4) { // If we have 4 or more good matches
				H = findHomography( obj_matches, scene_matches, CV_RANSAC );

				// Use homography and obj_corners to find corners of object in the scene
				vector<Point2f> scene_corners(4);
				perspectiveTransform( obj_corners, scene_corners, H);

				// Draw matches
				if (featuresEnabled) {
	  				drawMatches( object, keypoints_object, scene, keypoints_scene,
	               	good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
	               	vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	               	imshow("Matches", img_matches);
	          	} else {
	          		destroyWindow("Matches");
	          	}

				// Frame the detected object
				line( scene, scene_corners[0], scene_corners[1], Scalar( 0, 255, 0), 4 );
				line( scene, scene_corners[1], scene_corners[2], Scalar( 0, 255, 0), 4 );
				line( scene, scene_corners[2], scene_corners[3], Scalar( 0, 255, 0), 4 );
				line( scene, scene_corners[3], scene_corners[0], Scalar( 0, 255, 0), 4 );
			}
			else {
				cout << "Less than 4 good matches: Failed to find homography" << endl;
			}

		}
		
		// DISPLAY
		imshow("Main Image",mainImage);

		// TIMER
		diff = time(0) - start;
	    frames++;
	    if (diff > 1) {
	    	// Print frame rate
	    	cout << "FPS: " << frames/diff << endl;
	      	start = time(0);
	      	frames = 0;
	   	}

		//CHECKS TO SEE IF A BUTTON HAS BEEN PRESSED AND TOGGLES FLAGS
		switch(waitKey(1)){
		case 27: //'esc' key has been pressed, exit program.
			return 0;
		case 116: //'t' has been pressed. this will toggle tracking
			trackingEnabled = !trackingEnabled;
			if(trackingEnabled == false) cout<<"Tracking disabled."<<endl;
			else cout<<"Tracking enabled."<<endl;
			break;
		case 102: //'f' has been pressed. this will toggle feature points
			featuresEnabled = !featuresEnabled;
			if(featuresEnabled == false) cout<<"Feature points disabled."<<endl;
			else cout<<"Feature points enabled."<<endl;
			break;
		case 112: //'p' has been pressed. this will pause/resume the code.
			pause = !pause;
			if(pause == true){ 
				cout<<"Code paused, press 'p' again to resume"<<endl;
				while (pause == true){ 
					switch (waitKey()){
					case 112: 
						pause = false;
						cout<<"Code resumed."<<endl;
						break;
					}
				}
			}
		}
	}
	return 0;
}