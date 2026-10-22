# We use realsense for PI0 data collection and rollouts instead of Orbbec

import cv2
import zmq
import time
import struct
import numpy as np
from collections import deque
from threading import Thread, Event, RLock

class CamerasSubscriber():
    def __init__(self, topics, server_addr='localhost', port=8082, obs_len=1, cam_freq=60, read_freq=20):

        self.topics = topics
        self.server_addr = server_addr
        self.port = port

        self._context = zmq.Context()
        self.stop_event = Event()
        self.record_event = Event()
        
        self.cam_freq = cam_freq
        self.read_freq = read_freq
        self.obs_len = obs_len
        self.buffer_len = int(np.ceil(obs_len * (cam_freq / read_freq))) + 3 # min buffer len + extra states
        
        self.buffer_lock = RLock()
        self.frames_buffer = {topic:{'frame':deque(maxlen=self.buffer_len), 'timestamp':deque(maxlen=self.buffer_len)} for topic in self.topics}
        self.recording_buffer = {topic: [] for topic in self.topics}

    def start_thread(self):
        self.topics_threads = {}
        for topic in self.topics:
            topic_thread = Thread(target=self.individual_receive_frame, args=((topic,)))
            topic_thread.daemon = True
            topic_thread.start()
            self.topics_threads[topic] = topic_thread
        time.sleep(1)
                    
        print("[*] Waiting to fill up each topics buffer")
        steps = 0
        got_all_topics = False
        while  not got_all_topics:
            frames = self.get_frames_buffer()
            
            full_topics = []
            for topic, data in frames.items():
                full_topic = False
                if len(data['frame']) == self.buffer_len:
                    full_topic = True
                full_topics.append(full_topic)

            if np.all(full_topic):
                got_all_topics = True

            steps += 1

            if (steps % 1000) == 0:
                print(f"[*] Expecting frames from: {self.topics}, but only got {frames.keys()}")
        print("[*] Subscriber initialized")

    def close_subscriber(self):
        self.stop_event.set()
        for topic in self.topics:
            self.topics_threads[topic].join()
        self._context.term()
        time.sleep(1)

    def start_recording(self):
        self.record_event.set()
    
    def stop_recording(self):
        self.record_event.clear()

    def clear_recording(self):
        for topic in self.topics:
            self.recording_buffer[topic].clear()

    def individual_receive_frame(self, topic: str):
        topic_socket = self._context.socket(zmq.SUB)
        topic_socket.connect(f"tcp://{self.server_addr}:{self.port}")
        topic_socket.setsockopt(zmq.SUBSCRIBE, topic.encode('utf-8'))
        topic_socket.setsockopt(zmq.RCVHWM, 1)
        topic_socket.setsockopt(zmq.LINGER, 0)

        while not self.stop_event.is_set():

            [topic, message] = topic_socket.recv_multipart()

            topic = topic.decode(encoding='utf-8')

            if 'img_rgb' in topic:
                h, w, c = struct.unpack(">III", message[:12])
                img = np.frombuffer(message[12:], dtype=np.uint8).reshape(h, w, c)

            elif 'img_depth' in topic:
                h, w = struct.unpack(">II", message[:8])
                img = np.frombuffer(message[8:], dtype=np.uint16).reshape(h, w)
            
            recv_time = time.time()
            self.frames_buffer[topic]['frame'].append(img)
            self.frames_buffer[topic]['timestamp'].append(recv_time)
            
            if self.record_event.is_set():
                self.recording_buffer[topic].append(img)
            
        topic_socket.close()
    
    def get_frames_buffer(self):
        with self.buffer_lock:
            return self.frames_buffer
    
    def get_last_obs(self):
        frames = self.get_frames_buffer()
        last_timestamp = np.max([buffer['timestamp'][-1] for buffer in frames.values()])
        obs_timestamps = last_timestamp - (np.arange(self.obs_len)[::-1]) * (1. / self.read_freq)
        cameras_frames = {}
        for topic, buffer in frames.items():
            timestamps = buffer['timestamp']
            indices = []
            for t in obs_timestamps:
                is_before_t = np.nonzero(timestamps < t)[0]
                idx = 0
                if len(is_before_t) > 0:
                    idx = is_before_t[-1]
                indices.append(idx)
            
            cameras_frames[topic] = np.asarray(buffer['frame'])[indices]

        return cameras_frames
    



if __name__ == '__main__':

    topics = ["gripper_img_rgb",
              "left_realsense_img_rgb",
              "right_realsense_img_rgb"
              ]
    
    cameras_subscriber = CamerasSubscriber(topics=topics,
                                           server_addr='localhost',
                                           port=8082,
                                           obs_len=1,
                                           cam_freq=60,
                                           read_freq=20)
    cameras_subscriber.start_thread()
    try:
        for topic in topics:
            cv2.namedWindow(topic, cv2.WINDOW_AUTOSIZE)

        steps = 0
        while True:
            frames = cameras_subscriber.get_last_obs()
            
            for topic, frame in frames.items():
                cv2.imshow(topic, cv2.cvtColor(frame[-1], cv2.COLOR_RGB2BGR))

            cv2.waitKey(1)
            steps += 1

    except KeyboardInterrupt:
        print("[*] Killing subscriber")
    finally:
        cv2.destroyAllWindows()
        cameras_subscriber.close_subscriber()