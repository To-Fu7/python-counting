"""
Line Calibration Helper Script

Run this script to draw counting lines on full-resolution video.
Coordinates will be automatically scaled to your production resolution.

Usage:
    python calibrate_lines.py --source rtsp://camera_url --target-res 800x600
    python calibrate_lines.py --source ./1.mp4 --target-res 1280x720

Controls:
    - Left click: Add point (2 clicks = 1 line)
    - 'u': Undo last line
    - 's': Save and print scaled coordinates
    - 'q': Quit without saving
"""
import cv2
import argparse

lines = []
current_line = []


def mouse_callback(event, x, y, flags, param):
    global current_line, lines
    if event == cv2.EVENT_LBUTTONDOWN:
        current_line.append((x, y))
        if len(current_line) == 2:
            lines.append(current_line.copy())
            print(f"Line {len(lines)} added: {lines[-1]}")
            current_line = []


def main():
    global lines, current_line

    parser = argparse.ArgumentParser(description='Line calibration helper for person counting')
    parser.add_argument('--source', required=True, help='Video source (RTSP URL or file path)')
    parser.add_argument('--target-res', default='800x600', help='Target resolution (WxH)')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {args.source}")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from video source")
        cap.release()
        return

    orig_h, orig_w = frame.shape[:2]
    target_w, target_h = map(int, args.target_res.split('x'))

    print(f"Original resolution: {orig_w}x{orig_h}")
    print(f"Target resolution: {target_w}x{target_h}")
    print(f"Scale factors: X={target_w/orig_w:.4f}, Y={target_h/orig_h:.4f}")
    print("\nControls:")
    print("  - Left click: Add point (2 clicks = 1 line)")
    print("  - 'u': Undo last line")
    print("  - 's': Save and print scaled coordinates")
    print("  - 'q': Quit without saving")

    cv2.namedWindow('Draw Lines - Click to add points')
    cv2.setMouseCallback('Draw Lines - Click to add points', mouse_callback)

    while True:
        display = frame.copy()

        # Draw completed lines
        for i, line in enumerate(lines):
            color = (0, 255, 0) if i % 2 == 0 else (0, 255, 255)  # Alternate green/yellow
            cv2.line(display, line[0], line[1], color, 2)
            # Label the line
            mid_x = (line[0][0] + line[1][0]) // 2
            mid_y = (line[0][1] + line[1][1]) // 2
            label = chr(ord('A') + i)
            cv2.putText(display, f"line{label}", (mid_x, mid_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Draw current point if only one point selected
        if current_line:
            cv2.circle(display, current_line[0], 5, (0, 0, 255), -1)
            cv2.putText(display, "Click second point", (current_line[0][0] + 10, current_line[0][1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Show info
        cv2.putText(display, f"Lines: {len(lines)} | Press 's' to save, 'q' to quit",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Draw Lines - Click to add points', display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nQuitting without saving.")
            break
        elif key == ord('u'):
            # Undo last line
            if lines:
                removed = lines.pop()
                print(f"Removed line: {removed}")
            elif current_line:
                current_line = []
                print("Cleared current point")
        elif key == ord('s'):
            if not lines:
                print("\nNo lines to save!")
                continue

            # Scale and save
            scale_x = target_w / orig_w
            scale_y = target_h / orig_h

            print("\n" + "=" * 60)
            print("SCALED COORDINATES FOR .env FILE")
            print("=" * 60)
            print(f"# Target resolution: {target_w}x{target_h}")
            print(f"# Original resolution: {orig_w}x{orig_h}")
            print()

            for i, line in enumerate(lines):
                scaled = [(int(p[0] * scale_x), int(p[1] * scale_y)) for p in line]
                label = chr(ord('A') + i)
                print(f"line{label} = {scaled}")

            print()
            print("# Copy the lines above to your .env file")
            print("=" * 60)

            # Also print pairing info
            print("\nNOTE: Lines are paired as follows:")
            for i in range(0, len(lines), 2):
                if i + 1 < len(lines):
                    label1 = chr(ord('A') + i)
                    label2 = chr(ord('A') + i + 1)
                    print(f"  Gate {i//2 + 1}: line{label1} (IN) + line{label2} (OUT)")
                else:
                    label1 = chr(ord('A') + i)
                    print(f"  Gate {i//2 + 1}: line{label1} (IN) - OUT line will be auto-generated")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
