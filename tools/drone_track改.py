import cv2
import time
import torch
import argparse
from djitellopy import Tello
from yolox.utils import postprocess
from yolox.tracker.byte_tracker import BYTETracker
from yolox.exp import get_exp
from loguru import logger
from tkinter import Tk, Button, messagebox, Label
from threading import Thread
from yolox.data.data_augment import preproc
from yolox.utils.visualize import plot_tracking

#python tools/drone_track.py --demo webcam --camid 0 --exp_file exps/example/mot/yolox_x_mix_det.py --ckpt pretrained/bytetrack_x_mot17.pth.tar --fp16 --fuse --save_result --target_id 1 --mot20

class Predictor:
    def __init__(self, model, exp, device=torch.device("cpu"), fp16=False):
        self.model = model.to(device)
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16

        if self.fp16:
            self.model.half()
        else:
            self.model.float()

        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img):
        img_info = {"raw_img": img, "height": img.shape[0], "width": img.shape[1]}
        img, _ = preproc(img, self.test_size, self.rgb_means, self.std)
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)

        if self.fp16:
            img = img.half()
        else:
            img = img.float()

        with torch.no_grad():
            outputs = self.model(img)
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
        return outputs, img_info

class DroneTracker:
    def __init__(self, target_id, model, exp, args, device):
        self.tello = Tello()
        self.tello.connect()
        self.tello.streamon()
        logger.info(f"Tello Battery: {self.tello.get_battery()}%")
        
        self.target_id = target_id
        self.device = device
        self.tracking = False
        self.paused = False

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            model = model.to(self.device)
            logger.info("Using GPU for inference")
        else:
            self.device = device
            logger.info("Using CPU for inference")

        self.predictor = Predictor(model, exp, self.device, fp16=args.fp16)
        self.tracker = BYTETracker(args, frame_rate=args.fps)
        self.width, self.height = 960, 720
        self.center_x, self.center_y = self.width // 2, self.height // 2
        self.base_speed = 20
        self.max_speed = 100
        self.max_distance = 500  # Tello的最大移动距离为500cm
        self.rotation_speed = 30  # 旋转速度，用于水平调整

    def send_command_with_retry(self, command, max_retries=30):
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = self.tello.send_control_command(command)
                if response == "ok":
                    return
            except Exception as e:
                logger.warning(f"Command '{command}' failed: {e}")
            retry_count += 1
            time.sleep(0.2)  # 等待0.2秒再重试
        logger.error(f"Command '{command}' failed after {max_retries} retries")


    def adjust_drone_position(self, offset_x, offset_y):
        # 偏移量较小时保持高度
        if abs(offset_y) < self.height * 0.1:
            offset_y = 0

        kp_base = 0.2
        kp_dynamic = min(1.0, max(0.1, abs(offset_x) / self.width, abs(offset_y) / self.height)) * kp_base

        # 计算速度并限制在最大距离范围内
        speed_x = int(min(kp_dynamic * offset_x, self.max_distance))
        speed_y = int(min(kp_dynamic * offset_y, self.max_distance))

        # 限制移动速度在基础速度和最大速度之间
        speed_x = max(-self.max_speed, min(self.max_speed, speed_x))
        speed_y = max(-self.max_speed, min(self.max_speed, speed_y))

        # 检查是否需要旋转来调整水平偏移
        if abs(offset_x) > self.width * 0.5:
            rotate_direction = "cw" if offset_x > 0 else "ccw"
            self.send_command_with_retry(f"{rotate_direction} {self.rotation_speed}")
        else:
            # 横向调整
            if abs(speed_x) > self.base_speed:
                if speed_x > 0:
                    self.send_command_with_retry(f"right {abs(speed_x)}")
                else:
                    self.send_command_with_retry(f"left {abs(speed_x)}")

            # 垂直调整
            if abs(speed_y) > self.base_speed:
                if speed_y > 0:
                    self.send_command_with_retry(f"down {abs(speed_y)}")
                else:
                    self.send_command_with_retry(f"up {abs(speed_y)}")


    def track_and_adjust(self):
        frame_reader = self.tello.get_frame_read()
        retry_count = 0
        frame_skip = 3  # 进行抽帧设置

        while self.tracking:
            frame = frame_reader.frame
            if frame is None:
                retry_count += 1
                if retry_count > 5:
                    logger.warning("Video stream temporarily unavailable. Skipping frame.")
                    retry_count = 0
                time.sleep(0.1)
                continue
            retry_count = 0

            if frame_skip > 1:
                frame_skip -= 1  # 如果当前不需要处理帧，跳过
                continue

            try:
                outputs, img_info = self.predictor.inference(frame)
            except Exception as e:
                logger.error(f"Inference failed on this frame: {e}")
                continue

            frame_skip = 3  # 重置帧跳过计数

            target_detected = False


        closest_target = None
        closest_distance = float("inf")  # 初始值为无限大

        if outputs[0] is not None:
            online_targets = self.tracker.update(outputs[0], [img_info['height'], img_info['width']], self.predictor.test_size)
            
            for t in online_targets:
                x, y, w, h = map(int, t.tlwh)
                target_center_x = x + w // 2
                target_center_y = y + h // 2
                distance = ((target_center_x - self.center_x) ** 2 + (target_center_y - self.center_y) ** 2) ** 0.5

                if distance < closest_distance:
                    closest_distance = distance
                    closest_target = t

            if closest_target:
                x, y, w, h = map(int, closest_target.tlwh)
                target_center_x = x + w // 2
                target_center_y = y + h // 2
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {closest_target.track_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                target_detected = True

                # 偏移和控制逻辑
                offset_x = target_center_x - self.center_x
                offset_y = target_center_y - self.center_y
                # 调整无人机位置
                self.adjust_drone_position(offset_x, offset_y)

        if not target_detected:
            cv2.putText(frame, "No target detected", (10, self.height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


            cv2.imshow("Drone View", frame)
            if cv2.waitKey(1) & 0xFF == ord('p'):
                self.paused = not self.paused


        cv2.destroyAllWindows()

    def start_tracking(self):
        """启动目标追踪"""
        self.tracking = True
        tracking_thread = Thread(target=self.track_and_adjust)
        tracking_thread.start()

    def stop_tracking(self):
        """停止目标追踪"""
        self.tracking = False

    def takeoff(self):
        if self.tello.get_battery() < 10:
            logger.warning("Battery low, please charge to above 10%.")
            return
        logger.info("Calibrating IMU...")
        self.tello.takeoff()

    def land(self):
        for _ in range(3):
            try:
                self.tello.land()
                return
            except Exception as e:
                logger.warning(f"Landing failed: {e}")
        logger.error("Failed to land after multiple attempts.")

    def end(self):
        self.tello.end()




class DroneGUI:
    def __init__(self, tracker):
        self.tracker = tracker
        self.root = Tk()
        self.root.title("Tello Control Panel")
        
        Label(self.root, text="Tello Drone Control", font=("Helvetica", 16), fg="white", bg="#333").pack(pady=10)
        
        self.takeoff_button = Button(self.root, text="Take Off", command=self.tracker.takeoff, bg="#4CAF50", fg="white")
        self.takeoff_button.pack(pady=10, ipadx=10, ipady=5)

        self.land_button = Button(self.root, text="Land", command=self.tracker.land, bg="#F44336", fg="white")
        self.land_button.pack(pady=10, ipadx=10, ipady=5)

        self.start_track_button = Button(self.root, text="Start Tracking", command=self.tracker.start_tracking, bg="#2196F3", fg="white")
        self.start_track_button.pack(pady=10, ipadx=10, ipady=5)

        self.stop_track_button = Button(self.root, text="Stop Tracking", command=self.tracker.stop_tracking, bg="#FF9800", fg="white")
        self.stop_track_button.pack(pady=10, ipadx=10, ipady=5)

        self.exit_button = Button(self.root, text="Exit", command=self.exit_program, bg="#9E9E9E", fg="white")
        self.exit_button.pack(pady=10, ipadx=10, ipady=5)

    def exit_program(self):
        self.tracker.stop_tracking()
        self.tracker.end()
        self.root.quit()

    def run(self):
        self.root.mainloop()

def make_parser():
    parser = argparse.ArgumentParser("Drone Tracking with Tello")
    parser.add_argument("--demo", type=str, default="video", help="demo type, e.g., image, video, and webcam")
    parser.add_argument("--target_id", type=int, default=1, help="ID of the target person to track")
    parser.add_argument("--camid", type=int, default=0, help="Camera ID for webcam demo")
    parser.add_argument("-f", "--exp_file", default="exps/example/mot/yolox_x_mix_det.py", type=str, help="Experiment description file")
    parser.add_argument("-c", "--ckpt", default="pretrained/bytetrack_x_mot17.pth.tar", type=str, help="Path to checkpoint file")
    parser.add_argument("--device", default="cpu", type=str, help="Device to run model on, e.g., cpu or gpu")
    parser.add_argument("--track_thresh", type=float, default=0.5, help="Tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="Frames to keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="Matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="Aspect ratio threshold for bounding boxes")
    parser.add_argument('--min_box_area', type=float, default=10, help="Minimum area for tracking boxes")
    parser.add_argument("--fps", default=30, type=int, help="Frame rate (fps)")
    parser.add_argument("--fp16", action="store_true", help="Use half-precision for inference")
    parser.add_argument("--fuse", action="store_true", help="Fuse convolution and batch normalization layers")
    parser.add_argument("--save_result", action="store_true", help="Save tracking results")
    parser.add_argument("--mot20", action="store_true", help="Use MOT20 settings")  # 添加 mot20 参数
    return parser

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, None)
    model = exp.get_model().to(args.device)
    model.eval()
    
    checkpoint = torch.load(args.ckpt, map_location=args.device, weights_only=True)
    model.load_state_dict(checkpoint["model"])

    drone_tracker = DroneTracker(args.target_id, model, exp, args, args.device)
    gui = DroneGUI(drone_tracker)
    gui.run()
