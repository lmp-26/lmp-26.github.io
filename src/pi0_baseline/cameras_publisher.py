from threading import Thread
import hydra
from omegaconf import OmegaConf, DictConfig
import time
import zmq
import struct


class CamerasPublisher():
    def __init__(self, cameras_conf, fps=30, port=8082):
        
        self.port = port
        self.cameras_obj = {}
        self.cameras_thread = {}

        self.freq = fps
        self.send_period = 1. / self.freq

        for cam_name, cam_conf in cameras_conf.items():
            print(f"[*] Initializing {cam_name} camera")
            camera = hydra.utils.instantiate(cam_conf)

            # Discard a few frames during initialization 
            for _ in range(30):
                frame = camera.get_frame()

            # Each camera will read frames on its own thread
            camera_thread = Thread(target=camera._run)
            camera_thread.deamon = True
            camera_thread.start()

            self.cameras_obj[cam_name] = camera
            self.cameras_thread[cam_name] = camera_thread
            print(f"[*] Camera thread started")

        time.sleep(2)

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.bind(f"tcp://*:{self.port}")


    def start_publisher(self):
        last_time = None
        try:
            while True:
                start_step = time.time_ns()

                for cam_name, camera in self.cameras_obj.items():
                    frame = camera.get_last_frame()

                    if frame is None:
                        continue
                    
                    if 'img_rgb' in frame:
                        h, w, c = frame['img_rgb'].shape
                        header = struct.pack(">III", h, w, c)
                        img_bytes = frame['img_rgb'].tobytes()

                        encoded_message = header + img_bytes
                        topic = f"{cam_name}_img_rgb".encode(encoding="utf-8")
                        self._socket.send_multipart([topic, encoded_message])

                    if 'img_depth' in frame:
                        h, w = frame['img_depth'].shape
                        header = struct.pack(">II", h, w)
                        img_bytes = frame['img_depth'].tobytes()

                        encoded_message = header + img_bytes
                        topic = f"{cam_name}_img_depth".encode(encoding="utf-8")
                        self._socket.send_multipart([topic, encoded_message])
                        
                if last_time == None:
                    last_time = time.time_ns()

                curr_time = time.time_ns()
                remaining_time = self.send_period - ((curr_time - last_time) / (10**9))
                if 0.0001 < remaining_time < 0.2:
                    time.sleep(remaining_time)

                last_time = time.time_ns()

                end_step = time.time_ns()
                print(f"[*] Time profile: {(end_step - start_step) / (10**9)}, Sleep Time: {remaining_time/(10**9)}, Send Period: {self.send_period}")
                
        except KeyboardInterrupt:
            print("[*] The program was interrupted by the user. Closing cameras and publisher")

        finally:
            self.close_cameras()
            self._socket.close()
            self._context.term()
        

    def close_cameras(self):
        self.done = True
        time.sleep(1)
        for camera in self.cameras_obj.values():
            camera.close_camera()

        for camera_thread in self.cameras_thread.values():
            camera_thread.join()

@hydra.main(version_base=None, config_path='config', config_name='cameras_publisher')
def main(cfg: DictConfig) -> None:

    print(OmegaConf.to_yaml(cfg, resolve=True))

    cameras_publisher = CamerasPublisher(cfg.cameras, cfg.fps, cfg.port)
    cameras_publisher.start_publisher()


if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
          
    main()