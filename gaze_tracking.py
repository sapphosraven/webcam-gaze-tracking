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

# Smoothing function for gaze points
def smooth_gaze(gaze_points, alpha=0.2):
    if len(gaze_points) < 2:
        return gaze_points[-1]
    smoothed_point = alpha * gaze_points[-1] + (1 - alpha) * gaze_points[-2]
    return smoothed_point

class OverlayWindow:
    def __init__(self):
        self.hwnd = None
        self.create_window()

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
            win32gui.EndPaint(hwnd, paint_struct)
        return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)

    def update(self, x, y):
        hdc = win32gui.GetDC(self.hwnd)
        rect = win32gui.GetClientRect(self.hwnd)
        win32gui.FillRect(hdc, rect, win32gui.GetStockObject(win32con.BLACK_BRUSH))  # Clear the previous gaze pointer
        win32gui.FillRect(hdc, (x-5, y-5, x+5, y+5), win32gui.GetStockObject(win32con.WHITE_BRUSH))  # Draw the new gaze pointer
        win32gui.ReleaseDC(self.hwnd, hdc)
        
    def draw_calibration_grid(self, current_index):
        hdc = win32gui.GetDC(self.hwnd)
        rect = win32gui.GetClientRect(self.hwnd)

        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                x = j * grid_spacing + 100
                y = i * grid_spacing + 100
                color = (0, 0, 255) if (i, j) == current_index else (255, 255, 255)
                win32gui.FillRect(hdc, (x - 5, y - 5, x + 5, y + 5), win32gui.GetStockObject(win32con.BLACK_BRUSH))
                win32gui.FillRect(hdc, (x - 4, y - 4, x + 4, y + 4), win32gui.CreateSolidBrush(win32api.RGB(*color)))

        win32gui.ReleaseDC(self.hwnd, hdc)
        
# Calibration setup (Both original and grid)
calibration_instructions = ["Center", "Top Left", "Top Right", "Bottom Left", "Bottom Right"]
initial_screen_points = [
    (960, 540),  # Center
    (0, 0),      # Top Left
    (1920, 0),   # Top Right
    (0, 1080),   # Bottom Left
    (1920, 1080) # Bottom Right
]
grid_size = (3, 5)  # 3x5 grid of calibration points
grid_spacing = 300  # Adjust for your screen size/desired spacing
offset_x = 0  # Initial offset from the top-left corner
offset_y = 0

grid_screen_points = [(j * grid_spacing + offset_x, i * grid_spacing + offset_y)
                     for i in range(grid_size[0])
                     for j in range(grid_size[1])]

calibration_screen_points = initial_screen_points  # Combine both sets of points
calibration_points = []
current_calibration_index = 0

for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        x = j * grid_spacing + offset_x
        y = i * grid_spacing + offset_y
        calibration_screen_points.append((x, y))

overlay = OverlayWindow()
gaze_history = []

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
        if current_calibration_index < len(calibration_screen_points):
            overlay.update(0, 0)
            if current_calibration_index < 5:
                instruction_text = f"Look at the {calibration_instructions[current_calibration_index]}"
            else:
                instruction_text = f"Look at the dots row-wise, Point {current_calibration_index - 4}/{grid_size[0] * grid_size[1]}"

            cv2.putText(frame, instruction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Draw calibration grid on the overlay (only for grid points)
            if current_calibration_index >= 5:
                overlay.draw_calibration_grid(current_calibration_index - 5)  # Adjusted index

            # Capture calibration point
            if cv2.waitKey(1) & 0xFF == ord('c'):
                calibration_points.append((gaze_x, gaze_y))
                current_calibration_index += 1

                # Check if calibration is complete
                if current_calibration_index == len(calibration_screen_points):
                    break 
        else:
            # Map gaze points to screen coordinates using calibration data
            screen_coords = map_gaze_to_screen(calibration_points, calibration_screen_points, gaze_x, gaze_y)
            gaze_history.append(screen_coords)
            smoothed_coords = smooth_gaze(gaze_history)
            overlay.update(int(smoothed_coords[0]), int(smoothed_coords[1]))

    # Display frame
    cv2.imshow("Gaze Tracker", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Calibration data (you can use this for further analysis or mapping)
print("Calibration points:", calibration_points)

