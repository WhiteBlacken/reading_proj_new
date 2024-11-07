from django.shortcuts import render
import zmq
import msgpack
from loguru import logger

# Create your views here.

class PupilRemoteManager:

    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.pupil_remote_instance = None
        self.subscriber_instance = None
        self.ctx = zmq.Context()

    def get_pupil_service_instance(self):
        try:
            if self.pupil_remote_instance is None:
                pupil_remote = zmq.Socket(self.ctx, zmq.REQ)
                pupil_remote.connect(f'tcp://{self.ip}:{self.port}')
                self.pupil_remote_instance = pupil_remote
        except Exception as e:
            logger.info("please restart pupil capture")
        return self.pupil_remote_instance
    

    def get_subscriber_instance(self):
        if self.pupil_remote_instance is None:
            self.get_pupil_service_instance()
        if self.subscriber_instance is None:
            self.pupil_remote_instance.send_string('SUB_PORT')
            sub_port = self.pupil_remote_instance.recv_string()
            subscriber = self.ctx.socket(zmq.SUB)
            subscriber.connect(f'tcp://{self.ip}:{sub_port}')
            subscriber.subscribe('gaze.')  # receive all gaze messages
            self.subscriber_instance = subscriber
        return self.subscriber_instance

    def track_gaze_sequence(self):
        if self.subscriber_instance is None:
            self.subscriber_instance = self.get_subscriber_instance()
        while True:
            _, payload = self.subscriber_instance.recv_multipart()
            message = msgpack.loads(payload)
            try:
                base_data = message['base_data'][0]
                x,y, timestamp, confidence = base_data['location'][0], base_data['location'][1], base_data['timestamp'], base_data['confidence']
                yield x,y, timestamp, confidence
            except Exception as e:
                print(e)


if __name__ == '__main__':
    # pupil_remote = PupilRemoteManager("127.0.0.1", 50020)
    # gaze_sequence = pupil_remote.track_gaze_sequence()
    # for msg in gaze_sequence:
    #     print(msg)
    #     break
    pass

# exmaple:
# {'eye_center_3d': [-53.83143631885662, -185.0000584339129, 113.11853543916071], 'gaze_normal_3d': [-0.549321666828619, 0.7776316240494721, 0.3058345363604451], 'gaze_point_3d': [-328.4922697331662, 203.8157535908231, 266.0358036193833], 'norm_pos': [-0.4646618321201702, -0.5640576632602725], 'topic': 'gaze.3d.0.', 'confidence': 0.5277440617894628, 'timestamp': 2971.9698189980004, 'base_data': [{'id': 0, 'topic': 'pupil.0.3d', 'method': 'pye3d 0.3.0 real-time', 'norm_pos': [0.32509474342285566, 0.630356204292179], 'diameter': 17.84542417225699, 'confidence': 0.5277440617894628, 'timestamp': 2971.9698189980004, 'sphere': {'center': [2.0138056520052707, -7.146129868664803, 101.95780274840983], 'radius': 10.392304845413264}, 'projected_sphere': {'center': [90.92118251478979, 17.514049295595065], 'axes': [165.67665468099682, 165.67665468099682], 'angle': 0.0}, 'circle_3d': {'center': [-3.0346881535852823, -4.6028261833392285, 93.23746349703187], 'normal': [-0.4857915429433107, 0.24472951122561462, -0.8391150357013211], 'radius': -8.493024749837403}, 'diameter_3d': -16.986049499674806, 'ellipse': {'center': [52.01515894765691, 44.35725548493852], 'axes': [14.645972954338225, 17.84542417225699], 'angle': 145.32211253080933}, 'location': [52.01515894765691, 44.35725548493852], 'model_confidence': 0.1, 'theta': 1.3235556160095097, 'phi': -2.0955814779756228}]}
# {'eye_center_3d': [-53.83143631885662, -185.0000584339129, 113.11853543916071], 'gaze_normal_3d': [-0.5050449173506334, 0.8069329525307454, 0.3062496393113072], 'gaze_point_3d': [-306.3538949941734, 218.46641783145986, 266.2433550948143], 'norm_pos': [-0.3989481836230796, -0.6396550355723105], 'topic': 'gaze.3d.0.', 'confidence': 0.5440139231159485, 'timestamp': 2971.934071085, 'base_data': [{'id': 0, 'topic': 'pupil.0.3d', 'method': 'pye3d 0.3.0 real-time', 'norm_pos': [0.3375023699927708, 0.649033484171011], 'diameter': 17.484464239071222, 'confidence': 0.5440139231159485, 'timestamp': 2971.934071085, 'sphere': {'center': [2.0138056520052707, -7.146129868664803, 101.95780274840983], 'radius': 10.392304845413264}, 'projected_sphere': {'center': [90.92118251478979, 17.514049295595065], 'axes': [165.67665468099682, 165.67665468099682], 'angle': 0.0}, 'circle_3d': {'center': [-2.6929391059385885, -4.935555339816153, 92.9600300220084], 'normal': [-0.45290672550095784, 0.21271263321574968, -0.8658110842824905], 'radius': -8.531249493496329}, 'diameter_3d': -17.062498986992658, 'ellipse': {'center': [54.000379198843326, 42.115981899478676], 'axes': [14.756050369457423, 17.484464239071222], 'angle': 146.2515677915804}, 'location': [54.000379198843326, 42.115981899478676], 'model_confidence': 0.1, 'theta': 1.3564460355331964, 'phi': -2.0527535569535336}]}