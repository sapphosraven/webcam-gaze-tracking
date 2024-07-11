import cv2
import dlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
import win32gui
import win32con
import win32api

# Load pre-trained models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define the grid of calibration points
grid_cols = 3  # Number of columns in the grid
grid_rows = 3  # Number of rows in the grid
calibration_points = [(x, y) for y in range(grid_rows) for x in range(grid_cols)]
calibration_screen_points = [
    (int(1920 * (x + 0.5) / grid_cols), int(1080 * (y + 0.5) / grid_rows))  # Center of each grid cell
    for (x, y) in calibration_points
]
current_calibration_point = 0

# Webcam initialization
cap = cv2.VideoCapture(0)

# Function to extract eye images
def extract_eye(frame, landmarks, eye_indices):
    points = [landmarks.part(i) for i in eye_indices]
    region = np.array([(point.x, point.y) for point in points], dtype=np.int32)
    rect = cv2.boundingRect(region)
    x, y, w, h = rect
    eye_image = frame[y:y+h, x:x+w]
    eye_image = cv2.resize(eye_image, (64, 64))
    return eye_image

# Function to preprocess eye images
def preprocess_eye(eye_image):
    gray_eye = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
    normalized_eye = gray_eye / 255.0
    return normalized_eye.reshape(1, 64, 64, 1)

# Define a more advanced CNN model for gaze estimation
def create_gaze_model():
    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(64, 64, 3))
    x = base_model.output
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(2)(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='mse')
    return model

# Load or create the model
gaze_model = create_gaze_model()

# Function to estimate gaze direction
def estimate_gaze(left_eye, right_eye):
    left_eye = preprocess_eye(left_eye)
    right_eye = preprocess_eye(right_eye)
    left_gaze = gaze_model.predict(left_eye)
    right_gaze = gaze_model.predict(right_eye)
    gaze_x = (left_gaze[0][0] + right_gaze[0][0]) / 2
    gaze_y = (left_gaze[0][1] + right_gaze[0][1]) / 2
    return gaze_x, gaze_y

# Function to map gaze points to screen coordinates using homography
def map_gaze_to_screen(gaze_points, screen_points, gaze_x, gaze_y):
    gaze_points = np.array(gaze_points, dtype=np.float32)
    screen_points = np.array(screen_points, dtype=np.float32)
    homography_matrix, _ = cv2.findHomography(gaze_points, screen_points)
    gaze_point = np.array([[gaze_x, gaze_y]], dtype=np.float32).reshape(-1, 1, 2)
    screen_point = cv2.perspectiveTransform(gaze_point, homography_matrix)
    return screen_point[0][0]

class OverlayWindow:
    def __init__(self):
        self.hwnd = None
        self.create_window()
        self.dot_radius = 10
        self.dot_color_active = (0, 0, 255)  # Red color for active dot
        self.dot_color_inactive = (255, 255, 255)  # White color for inactive dots

    def create_window(self):
        wc = win32gui.WNDCLASS()
        wc.lpfnWndProc = self.wnd_proc
        wc.lpszClassName = 'OverlayWindowClass'
        wc.hInstance = win32api.GetModuleHandle(None)
        wc.hCursor = win32gui.LoadCursor(None, win32con.IDC_ARROW)
        wc.hbrBackground = win32con.COLOR_WINDOW + 1

        class_atom = win32gui.RegisterClass(wc)
        self.hwnd = win32gui.CreateWindowEx(
            win32con.WS_EX_TOPMOST | win32con.WS_EX_LAYERED | win32con.WS_EX_TOOLWINDOW,
            class_atom,
            'Gaze Pointer',
            win32con.WS_POPUP,
            0, 0, win32api.GetSystemMetrics(0), win32api.GetSystemMetrics(1),
            None, None, None, None
        )
        win32gui.SetLayeredWindowAttributes(self.hwnd, win32api.RGB(0, 0, 0), 0, win32con.LWA_COLORKEY)
        win32gui.ShowWindow(self.hwnd, win32con.SW_SHOW)

    def wnd_proc(self, hwnd, msg, wparam, lparam):
        if msg == win32con.WM_PAINT:
            hdc, paint_struct = win32gui.BeginPaint(hwnd)
            self.draw_calibration_dots(hdc)  # Draw calibration dots on paint event
            win32gui.EndPaint(hwnd, paint_struct)
        return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)

    def draw_calibration_dots(self, hdc):
        for idx, point in enumerate(calibration_screen_points):
            if idx == current_calibration_point:
                dot_color = self.dot_color_active
            else:
                dot_color = self.dot_color_inactive
            cv2.circle(hdc, point, self.dot_radius, dot_color, -1)

    def update(self, x, y):
        hdc = win32gui.GetDC(self.hwnd)
        rect = win32gui.GetClientRect(self.hwnd)
        win32gui.FillRect(hdc, rect, win32gui.GetStockObject(win32con.BLACK_BRUSH))  # Clear the previous gaze pointer
        win32gui.ReleaseDC(self.hwnd, hdc)

overlay = OverlayWindow()
gaze_history = []

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face detection
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)

        # Extract and preprocess left and right eye images
        left_eye_image = extract_eye(frame, landmarks, range(36, 42))
        right_eye_image = extract_eye(frame, landmarks, range(42, 48))

        # Estimate gaze direction
        gaze_x, gaze_y = estimate_gaze(left_eye_image, right_eye_image)

        # Calibration instructions
        if current_calibration_point < len(calibration_screen_points):
            overlay.update(0, 0)
            instruction = f"Look at: Dot {current_calibration_point + 1}/{len(calibration_screen_points)} and press 'c'"
            cv2.putText(frame, instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if cv2.waitKey(1) & 0xFF == ord('c'):  # Press 'c' to capture calibration point
                calibration_points.append((gaze_x, gaze_y))
                current_calibration_point += 1
        else:
            # Map gaze points to screen coordinates using calibration data
            screen_coords = map_gaze_to_screen(calibration_points, calibration_screen_points, gaze_x, gaze_y)
            gaze_history.append(screen_coords)
            smoothed_coords = screen_coords  # No smoothing
            overlay.update(int(smoothed_coords[0]), int(smoothed_coords[1]))

    # Display frame
    cv2.imshow("Gaze Tracker", frame)

    # Calibration cycle complete, reset if needed
    if current_calibration_point >= len(calibration_screen_points):
        current_calibration_point = 0
        calibration_points = []  # Reset calibration points

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Calibration data (you can use this for further analysis or mapping)
print("Calibration points:", calibration_points)
