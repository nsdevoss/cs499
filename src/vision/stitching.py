import cv2

# DO NOT pay attention to anything here yet
# HUGE WIP, and it's something we don't really need, this was mainly for the coolness effect


def find_keypoints_and_match(frame1, frame2):
    orb = cv2.ORB_create()

    if frame1 is None or frame2 is None or frame1.size == 0 or frame2.size == 0:
        print("Invalid frames received, skipping keypoint matching.")
        return [], [], []

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        print("Not enough keypoints detected in one or both frames, skipping matching.")
        return [], [], []

    if des1.shape[1] != des2.shape[1]:
        print("Descriptor sizes do not match, skipping matching.")
        return [], [], []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    print(f"Keypoints found: {len(kp1)} in frame1, {len(kp2)} in frame2. Matches: {len(matches)}")

    return kp1, kp2, matches



def stitch_frames(frame1, frame2):
    if frame1 is None or frame2 is None:
        print("One of the frames is None, cannot stitch.")
        return None

    if frame1.size == 0 or frame2.size == 0:
        print("One of the frames is empty, cannot stitch.")
        return None

    print(f"Frame 1 shape: {frame1.shape}, Frame 2 shape: {frame2.shape}")

    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
    status, stitched_frame = stitcher.stitch([frame1, frame2])

    if status != cv2.Stitcher_OK:
        print(f"Error during stitching: {status} (Probably not enough features detected)")
        return None

    return stitched_frame


def frame_stitcher(frame_queue):
    frames = {9000: None, 9001: None}

    while True:
        port, frame = frame_queue.get()
        frames[port] = frame
        if frames[9000] is not None and frames[9001] is not None:
            kp1, kp2, matches = find_keypoints_and_match(frames[9000], frames[9001])
            if len(matches) < 10:
                print("Not enough matches found, skipping stitching for this frame.")
                continue

            e1 = cv2.getTickCount()
            stitched = stitch_frames(frames[9000], frames[9001])
            e2 = cv2.getTickCount()
            print(f"Stitching time elapsed: {(e2 - e1)/cv2.getTickFrequency()}s")

            if stitched is None:
                print("Stitching failed, skipping this frame.")
                continue

            if frames[9000] is None or frames[9001] is None or stitched is None:
                print("Invalid frame detected, skipping display.")
                continue

            if frames[9000].size == 0 or frames[9001].size == 0:
                print("One of the frames is empty, skipping display.")
                continue

            # This is the feature matching frame with the lines drawn
            result_frame = cv2.drawMatches(frames[9000], kp1, frames[9001], kp2, matches[:10], None,
                                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            if stitched is not None and stitched.size != 0:
                cv2.imshow("Stitched image", stitched)

            if result_frame is not None and result_frame.size != 0:
                cv2.imshow("Features detected", result_frame)

            if cv2.waitKey(1) == ord("q"):
                break

    cv2.destroyAllWindows()
