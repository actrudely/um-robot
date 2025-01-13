#!/usr/bin/env python

"""
	To run this script, you need to run below lines in terminal:
	-[Robot] $ roslaunch rchomeedu_vision multi_astra.launch
	-[IO] $ roslaunch astra_camera astra.launch
	-[Juno] $ roslaunch usb_cam usb_cam-test.launch
	- roslaunch opencv_apps face_detection.launch image:=/camera/rgb/image_raw
	- roslaunch kids_module FoodCollectRobot.launch
"""

import rospy
from sound_play.libsoundplay import SoundClient
from opencv_apps.msg import FaceArrayStamped
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import time

greet_flag = 0
# greet_flag = 0 means haven't greet the guest
# greet_flag = 1 means greeted the guest

class FoodCollectRobot:
	def __init__(self):
		rospy.init_node('FoodCollectRobot')
		rospy.on_shutdown(self.cleanup)

		# Create the sound client object
		self.soundhandle = SoundClient()
		# self.soundhandle = SoundClient(blocking=True)

		self.validated = False
		self.block_validation = False
		self.wrong_keyword_times = 0
		self.start_time = time.time()
		self.face_lists = {
			"lyy": False
		}
		
		# Wait a moment to let the client connect to the sound_play server
		rospy.sleep(1)
		
		# Make sure any lingering sound_play processes are stopped.
		self.soundhandle.stopAll()
		rospy.loginfo("Ready, waiting for commands...")

		# Subscribe to the face detection output and set the callback function
		# rospy.Subscriber('/face_detection/facess', FaceArrayStamped, self.talkback)
		rospy.Subscriber('/face_recognition/output', FaceArrayStamped, self.callback)
		rospy.Subscriber("recognizer_whisper/output", String, self.parse_asr_result)
	
	def cleanup(self):
		self.soundhandle.stopAll()
		rospy.loginfo("Shutting down partybot node...")
	
	def callback(self, image: FaceArrayStamped):
		if len(image.faces) > 1:
			if not self.block_validation:
				self.block_validation = True
				self.soundhandle.stopAll()
				self.soundhandle.say("Too many faces, please try again")
		elif image.faces:
			cur_time = time.time()
			if cur_time - self.start_time > 5 and not self.validated:
				face_label = image.faces[0].label
				if self.face_lists.get(face_label)==False:
					self.face_lists[face_label] = True
					self.block_validation = False
					self.start_time = cur_time
					self.soundhandle.stopAll()
					self.soundhandle.say("Face recognized, please collect your food.")
	
	def parse_asr_result(self, msg): 
		if time.time() - self.start_time > 5:
			"""Function to perform action on detected word"""
			detected_words = msg.data.lower()
			if "thank you" in detected_words and not self.block_validation and not self.validated:
				self.soundhandle.say("Thank you. Please come again")
				self.validated = True
				self.wrong_keyword_times = 0
			elif self.wrong_keyword_times < 3:
				self.soundhandle.say("Validation error, please try again")
				self.wrong_keyword_times += 1
		

		

	def talkback(self, msg):
		faces = msg.faces
		# ======================== YOUR CODE HERE ========================
		# Instruction: Say Hello when face(s) is detected
		
		# Use global flag to let the robot will only greet people once
		global greet_flag
		
		# When the robot have not greet the guest
		if greet_flag == 0:
			# Check whether face(s) is detected
			if len(faces) > 0:
				# Since a robot may detect more than 1 face,
				# declare faces_data and eyes_data lists 
				# to access all face and eyes data
				faces_data = list()
				eyes_data = list()
				for i in faces:
					faces_data.append(i.face)
					
					# Eyes data will only listed if eyes are detected
					if i.eyes:
						rospy.loginfo(i.eyes)
						eyes_data.append(i.eyes)
					else:
						eyes_data.append("null")
				
				rospy.loginfo(eyes_data)
				# Will only greet the person if eyes are detected
				if eyes_data and [x for x in eyes_data if x != "null" ]:	
					# self.soundhandle.say("Good morning. How can I help you my friend?")
					greet_flag = 1
		
		else:
			# Reset the flag if face not detected
			if len(faces) < 1:
				greet_flag = 0
		# ================================================================

if __name__=="__main__":
	try:
		FoodCollectRobot()
		rospy.spin()
	except rospy.ROSInterruptException:
		rospy.loginfo("Partybot node terminated.")
