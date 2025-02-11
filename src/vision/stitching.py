import cv2


def find_keypoints_and_match(frame1, frame2):
    orb = cv2.ORB_create()

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    return kp1, kp2, matches


def stitch_frames(frame1, frame2):
    stitcher = cv2.Stitcher_create()
    status, stitched_frame = stitcher.stitch([frame1, frame2])

    if status != cv2.Stitcher_OK:
        print("Error during stitching:", status)
        return None

    return stitched_frame


def frame_stitcher(frame_queue):
    frames = {9000: None, 9001: None}

    while True:
        port, frame = frame_queue.get()
        frames[port] = frame
        if frames[9000] is not None and frames[9001] is not None:
            e1 = cv2.getTickCount()
            kp1, kp2, matches = find_keypoints_and_match(frames[9001], frames[9000])

            stitched = stitch_frames(frames[9001], frames[9000])

            e2 = cv2.getTickCount()
            print(f"Time elapsed: {(e2 - e1)/cv2.getTickFrequency()}")
            result_frame = cv2.drawMatches(frames[9001], kp1, frames[9000], kp2, matches[:10], None,
                                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow("Stitched Video", result_frame)

            if cv2.waitKey(1) == ord("q"):
                break

    cv2.destroyAllWindows()
