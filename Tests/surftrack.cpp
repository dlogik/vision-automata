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
#include <ctime>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <opencv2/video/tracking.hpp>

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

double diffclock(clock_t clock1,clock_t clock2) {
    double diffticks=clock1-clock2;
    double diffms=(diffticks)/(CLOCKS_PER_SEC/1000);
    return diffms;
}

void getCorners(Mat object, vector<Point2f> & obj_corners) {
	obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( object.cols, 0 );
	obj_corners[2] = cvPoint( object.cols, object.rows ); obj_corners[3] = cvPoint( 0, object.rows );
}

double compareCorners(vector<Point2f> old, vector<Point2f> fresh) {
	double diff = 0;
	for (int i; i< old.size(); i++) {
		diff += abs( norm(old[i])- norm(fresh[i]) ); 
	}
	return diff;
}

int selectFrame(VideoCapture* capture) {
	// LET USER CROP OBJECT FROM FIRST FRAME
	// Get the first frame of capture
	capture->read(mouse_img);
	// Show first frame in window
	imshow("First frame", mouse_img);
	// Get points from mousecallback
	cout << "Use mouse to crop object (top-left -> bottom-right). Press 'enter' when satisfied. 'Esc' to quit." << endl;

	cvSetMouseCallback("First frame", mouseHandler, NULL);
	bool wait = true;
	while (wait == true){
		// wait untill enter is pressed
		char waitKey = cvWaitKey(5);
		switch (waitKey) {
			case 'a':
			case 10: 	// 'enter' pressed: Finished cropping
				wait = false;
				cout<<"Object cropped from first frame" << endl;
				break;
			case 27:	// 'esc' pressed: Quit program
				return 0;
		}
	}

	// Make mouse_roiImg
	mouse_rect = Rect(mouse_point1.x, mouse_point1.y, mouse_point2.x-mouse_point1.x, mouse_point2.y-mouse_point1.y);
	mouse_roiImg = mouse_img(mouse_rect);

	// Close window of First frame
	destroyWindow("First frame");
	return 1;
}

void findGoodMatches(Mat descriptors_object, vector< DMatch > matches,vector< DMatch > & good_matches) {
	//-- Quick calculation of max and min distances between keypoints
	double max_dist = 0; double min_dist = 100;
	for( int i = 0; i < descriptors_object.rows; i++ )
	{ double dist = matches[i].distance;
	if( dist < min_dist ) min_dist = dist;
	if( dist > max_dist ) max_dist = dist;
	}
	//-- Keep only good matches (i.e. whose distance is less than 2*min_dist )
	for( int i = 0; i < descriptors_object.rows; i++ )
	{ if( matches[i].distance <= max(2*min_dist, 0.02) )
	{ good_matches.push_back( matches[i]); }
	}
}

void getKeyPoints(vector<KeyPoint> key1, vector<KeyPoint> key2, vector<DMatch> good_matches, vector<Point2f> & points1, vector<Point2f> & points2) {
	for( int i = 0; i < good_matches.size(); i++ ) {
		points1.push_back( key1[ good_matches[i].queryIdx ].pt );
		points2.push_back( key2[ good_matches[i].trainIdx ].pt );
	}
}

void drawObjectFrame(Mat & image, vector<Point2f> frame_corners) {
	line( image, frame_corners[0], frame_corners[1], Scalar( 0, 255, 0), 4 );
	line( image, frame_corners[1], frame_corners[2], Scalar( 0, 255, 0), 4 );
	line( image, frame_corners[2], frame_corners[3], Scalar( 0, 255, 0), 4 );
	line( image, frame_corners[3], frame_corners[0], Scalar( 0, 255, 0), 4 );
}

void matchAndTrack	(Mat object, Mat scene, vector<KeyPoint> keypoints_object, vector<KeyPoint> keypoints_scene,
					Mat descriptors_object, Mat descriptors_scene, vector<Point2f> obj_corners, Mat img_matches, bool featuresEnabled) 
{
	// MATCH
	clock_t start = clock();

	FlannBasedMatcher matcher;
	vector< DMatch > matches;	
	matcher.match( descriptors_object, descriptors_scene, matches );

	// FILTER OUT BAD MATCHES
	vector< DMatch > good_matches;
	findGoodMatches(descriptors_object, matches, good_matches);

	//-- Get the keypoints from the good matches
	vector<Point2f> obj_matches;
	vector<Point2f> scene_matches;
	getKeyPoints(keypoints_object, keypoints_scene, good_matches, obj_matches, scene_matches);

	clock_t end = clock();
	cout << "Find good matches " << diffclock(end, start) << "ms" << endl;
	

	if(obj_matches.size() >= 4) { // If we have 4 or more good matches
		
		start = clock();

		Mat H = findHomography( obj_matches, scene_matches, CV_RANSAC );
		end = clock();
		// Use homography and obj_corners to find corners of object in the scene
		vector<Point2f> scene_corners(4);
		perspectiveTransform( obj_corners, scene_corners, H);

		// Draw matches if featuresEnabled
		if (featuresEnabled) {
			drawMatches( object, keypoints_object, scene, keypoints_scene,
           	good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
           	vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
           	imshow("Matches", img_matches);
      	} else {
      		destroyWindow("Matches");
      	}

		drawObjectFrame(scene, scene_corners);

		cout << "Find homography and draw object frame: " << diffclock(end, start) << "ms" << endl;
		
	
	}
	else {
		cout << "Less than 4 good matches: Failed to find homography" << endl;
	}
}


class Timer {
	private:
		// Timer variables
		clock_t start = clock();
		int frames = 0;
		double diff;
	public:
		void display() {
			frames++;
			diff = diffclock(clock(), start);
			if (diff > 1000) {
				// Print frame rate
				cout << "FPS: " << frames*1000/diff << endl;
				start = clock();
				frames = 0;
			}
		}
};

class MainWindow {
	private:
		FlannBasedMatcher matcher;
		Mat img_matches;
		// Homography
		Mat H;

	public:
		Mat mainImage;
		Mat object;
		Mat scene;
		Mat grayScene, grayObject;
		// Vectors containing keypoints
		vector<KeyPoint> keypoints_scene, keypoints_object;
		Mat descriptors_object, descriptors_scene;
		vector<Point2f> obj_corners;
		// Detector, extractor and matcher
		SurfFeatureDetector detector;
		SurfDescriptorExtractor extractor;
		bool featuresEnabled;

		

		void init() {
		
			// SURF(double hessianThreshold, int nOctaves=4, int nOctaveLayers=2, bool extended=true, bool upright=false )
			featuresEnabled = false;
			detector = SurfFeatureDetector(400, 4, 2, true, false);
			extractor = SurfDescriptorExtractor(400, 4, 2, true, false);
			obj_corners = vector<Point2f>(4);

			// Create mainImage for visualization and make 'object' 'and 'scene' point to positions in it
			mainImage = Mat(max(mouse_img.rows, mouse_roiImg.rows), mouse_img.cols + mouse_roiImg.cols, CV_8UC3);
			object = Mat(mainImage, Range(0, mouse_roiImg.rows), Range(0, mouse_roiImg.cols));
			scene = Mat(mainImage, Range::all(), Range(mouse_roiImg.cols, mouse_roiImg.cols + mouse_img.cols));

			// Copy images into object and scene
			mouse_roiImg.copyTo(object);
			mouse_img.copyTo(scene);


			// IMAGE PROCESSING FOR OBJECT (outside while-loop, since we only need to do it once)
			//convert object to gray scale
			cvtColor(object, grayObject, COLOR_BGR2GRAY);
			detector.detect( grayObject, keypoints_object );
			extractor.compute( grayObject, keypoints_object, descriptors_object );
			
			getCorners(object, obj_corners);

			cout << "SETUP COMPLETED. PLAYING VIDEO..." << endl
			<< "'t' - toggles tracking" << endl
			<< "'f' - toggles feature mode" << endl
			<< "'r' - select another image region" << endl
			<< "'p' - toggles pause" << endl
			<< "'esc' - quit program" << endl
			<< endl;
		}

		void track(Mat* frame) {

			clock_t start = clock();
			//convert frame to gray scale
			cvtColor(*frame, grayScene, COLOR_BGR2GRAY);
			clock_t end = clock();
			cout << "Convert scene to grayscale: " << diffclock(end, start) << "ms" << endl;

			start = clock();
			// DETECT the keypoints using SURF Detector
			detector.detect( grayScene, keypoints_scene );
			end = clock();
			cout << "Detect scene: " << diffclock(end, start) << "ms" << endl;

			start = clock();
			// EXTRACT
			extractor.compute( grayScene, keypoints_scene, descriptors_scene );
			end = clock();
			cout << "Extract scene: " << diffclock(end, start) << "ms" << endl;

			// MATCH OBJECT WITH SCENE
			matchAndTrack(object, scene, keypoints_object, keypoints_scene, descriptors_object, descriptors_scene, 
							obj_corners, img_matches, featuresEnabled);
		}
};

bool toggleFunctionality(MainWindow & window, bool & trackingEnabled, bool & featuresEnabled, bool & redetect) {
	switch(waitKey(1)) {
		case 27: //'esc' key has been pressed, exit program.
			return false;
		case 116: //'t' has been pressed. this will toggle tracking
			trackingEnabled = !trackingEnabled;
			if(trackingEnabled == false) cout<<"Tracking disabled."<<endl;
			else cout<<"Tracking enabled."<<endl;
			break;
		case 102: //'f' has been pressed. this will toggle feature points
			featuresEnabled = !featuresEnabled;
			if(featuresEnabled == false) cout<<"Feature points disabled."<<endl;
			else cout<<"Feature points enabled."<<endl;
			window.featuresEnabled = featuresEnabled;
			break;
		case 112: //'p' has been pressed. this will pause/resume the code.
			cout<<"Code paused, press 'p' again to resume"<<endl;
			waitKey();
			cout<<"Code resumed."<<endl;
			break;
		case 'r':
			redetect = true;
	}
	return true;
}

int main(){

	//some boolean variables for added functionality
	bool trackingEnabled = true; 	// Press 't' to toggle
	bool featuresEnabled = false;	// Press 'f' to toggle
	bool redetect = false;			// Allows for new selection via mouse region.
	
	Timer timer = Timer();

	// image for reading capture
	Mat frame;

	//video capture object.
	VideoCapture capture;

	cout << "SURFTRACK IS RUNNING" << endl;
	// Open video stream from webcam
	//capture.open(0);
	capture.open(0);  // "clip_test.m4v"
	if(!capture.isOpened()) {
		cout<<"ERROR ACQUIRING VIDEO FEED\n";
		getchar();
		return -1;
	}

	capture.read(mouse_img);
	//mouse_roiImg = cvLoadImage("poster.png"); 

	// Skip first 5 frames, Macbook camera seems to take a while to initalise.
	for (int i = 0; i <= 5; i++) {
		capture.read(mouse_img);
	}

	if (selectFrame(&capture) == 0) {
		return 0;
	}

	MainWindow window = MainWindow();
	window.init();

	while(1){
		//read first frame
		capture.read(frame);			

		// put frame in the main image
		frame.copyTo(window.scene);

		if (redetect) {
			if (selectFrame(&capture) == 0) {
				return 0;
			} else {
				window.init();
				redetect = false;
			}
		}

		if (trackingEnabled) {
			clock_t start = clock();
			window.track(&frame);
			clock_t end = clock();
			cout << "Total tracking time: " << diffclock(end, start) << "ms" << endl << endl;
		}
		
		// DISPLAY
		imshow("Main Image",window.mainImage);

		// TIMER
		timer.display();

		//CHECKS TO SEE IF A BUTTON HAS BEEN PRESSED AND TOGGLES FLAGS.
		if (toggleFunctionality(window, trackingEnabled, featuresEnabled, redetect) == false) {
			return 0;
		}
		
	}
	return 0;
}
