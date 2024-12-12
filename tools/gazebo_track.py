import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import cv2
import torch
from threading import Thread
from yolox.tracker.byte_tracker import BYTETracker
from yolox.utils.visualize import plot_tracking
from yolox.exp import get_exp
from yolox.utils import fuse_model, postprocess

class DroneControlApp(Node):
    def __init__(self, args):
        super().__init__('drone_control_app')
        self.args = args
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)  # 发布控制指令到虚拟无人机
        self.is_tracking = False
        self.target_id = args.target_id

        # 初始化 YOLOX 模型和 BYTETracker
        self.model, self.exp = self.setup_yolox_model(args)
        self.tracker = BYTETracker(args, frame_rate=args.fps)
        
        # 初始化 ROS 2 订阅者，用于接收相机图像
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
        self.current_frame = None

    def setup_yolox_model(self, args):
        # 获取实验配置文件
        exp = get_exp(args.exp_file, args.name)
        model = exp.get_model()

        # 如果使用半精度推理
        if args.fp16:
            model.half()

        # 加载预训练权重
        ckpt = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model.eval()

        # 融合模型中的 batchnorm 和 conv
        if args.fuse:
            model = fuse_model(model)

        return model, exp

    def image_callback(self, msg):
        """从 Gazebo 相机接收图像，并将其存储到 current_frame 变量中"""
        # 将 ROS 2 图像消息转换为 OpenCV 格式
        self.current_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def start_tracking(self):
        """启动追踪目标的线程"""
        self.is_tracking = True
        tracking_thread = Thread(target=self.track_target)
        tracking_thread.start()

    def stop_tracking(self):
        self.is_tracking = False

    def track_target(self):
        while self.is_tracking:
            if self.current_frame is None:
                continue

            frame = self.current_frame
            # 使用 YOLOX 进行视觉追踪
            self.adjust_gimbal_with_visual_tracking(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def adjust_gimbal_with_visual_tracking(self, frame):
        img_info = {"id": 0}
        img, ratio = self.preproc(frame)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float()

        if self.args.fp16:
            img = img.half()

        outputs = self.model(img)
        outputs = postprocess(outputs, self.exp.num_classes, self.exp.test_conf, self.exp.nmsthre)

        if outputs[0] is not None:
            online_targets = self.tracker.update(outputs[0], [frame.shape[0], frame.shape[1]], self.exp.test_size)
            for t in online_targets:
                if t.track_id == self.target_id or self.target_id is None:
                    self.target_id = t.track_id
                    tlwh = t.tlwh
                    x1, y1, w, h = map(int, tlwh)
                    self.adjust_drone_to_target(x1 + w // 2, y1 + h // 2, frame.shape[1], frame.shape[0])

            online_im = plot_tracking(frame, [tlwh], [self.target_id], frame_id=0, fps=30)
            cv2.imshow("Tracking", online_im)

    def adjust_drone_to_target(self, target_x, target_y, frame_width, frame_height):
        """根据目标位置调整无人机的移动"""
        center_x, center_y = frame_width // 2, frame_height // 2
        delta_x = target_x - center_x
        delta_y = target_y - center_y

        # 基于目标偏移量生成速度控制指令
        twist = Twist()
        if abs(delta_x) > 50:
            yaw_speed = delta_x / frame_width * 2.0  # 横向控制速度
            twist.angular.z = yaw_speed

        if abs(delta_y) > 50:
            pitch_speed = delta_y / frame_height * 2.0  # 垂直控制速度
            twist.linear.z = -pitch_speed  # 无人机上下移动

        # 发布控制指令
        self.publisher.publish(twist)

    def preproc(self, img):
        from yolox.data.data_augment import preproc
        return preproc(img, self.exp.test_size, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def make_parser():
    parser = argparse.ArgumentParser("ROS 2 Gazebo Drone Control App with YOLOX Tracking")

    # YOLOX 模型相关参数
    parser.add_argument("--exp_file", type=str, help="Path to experiment description file")
    parser.add_argument("--ckpt", type=str, help="Path to model checkpoint")
    parser.add_argument("--name", type=str, default=None, help="Model name")
    parser.add_argument("--fp16", action="store_true", help="Use half precision evaluation")
    parser.add_argument("--fuse", action="store_true", help="Fuse conv and bn for evaluation")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate")
    parser.add_argument("--target_id", type=int, default=None, help="ID of the target person to track")

    return parser


def main(args):
    rclpy.init(args=None)
    app = DroneControlApp(args)
    rclpy.spin(app)
    rclpy.shutdown()


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    main(args)
