from flask import Flask, render_template, Response, request, url_for, redirect
import cv2

app = Flask(__name__)

camera = cv2.VideoCapture(0)  # use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)

def gen_frames(name):  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')
    
@app.route('/register', methods=["POST", "GET"])
def register():
    if request.method == "POST":
        user = request.form["nm"]
        return redirect(url_for("video_feed", name=user))
    return render_template('register.html')

@app.route('/video_feed/<name>')
def video_feed(name):
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(name), mimetype='multipart/x-mixed-replace; boundary=frame')





if __name__ == '__main__':
    app.run(debug=True)