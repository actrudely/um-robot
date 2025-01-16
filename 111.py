    def on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        buf = sample.get_buffer()
        caps = sample.get_caps()
        # Extract audio data from the buffer
        array = self.buffer_to_array(buf, caps)
        if array is not None:
            # Silence detection logic
            if self.is_silent(array):
                rospy.logdebug("Silent audio detected. Skipping...")
                return Gst.FlowReturn.OK
            # Accumulate audio data
            self.audio_buffer = np.concatenate((self.audio_buffer, array))
            rospy.logdebug("Accumulated audio buffer size: %d", len(self.audio_buffer))
            # Check if enough audio data has been accumulated
            if len(self.audio_buffer) >= self.max_buffer_size:
                transcription = self.transcribe(self.audio_buffer)
                if transcription:
                    msg = String()
                    msg.data = transcription
                    rospy.loginfo("Transcription: %s", msg.data)
                    self.pub.publish(msg)
                # Clear the buffer
                self.audio_buffer = np.array([], dtype=np.float32)
        return Gst.FlowReturn.OK

    def is_silent(self, audio, threshold=0.01):
        """
        Check if the audio is silent.
        :param audio: Audio data as a NumPy array
        :param threshold: Amplitude threshold to determine silence
        :return: True if silent, False otherwise
        """
        return np.mean(np.abs(audio)) < threshold
