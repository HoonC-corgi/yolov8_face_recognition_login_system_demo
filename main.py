from roboflow import Roboflow
import cv2
import time

def main():

    rf = Roboflow(api_key="Lwa6RdWc3rV4qREe5uMP")
    project = rf.workspace().project("face-detection-mik1i")
    model = project.version(18).model

    # 웹캠 초기화
    cam = cv2.VideoCapture(0)
    max_fps = cam.get(cv2.CAP_PROP_FPS)
    cam.set(cv2.CAP_PROP_FPS, max_fps)

    verified = False  # 사용자 인증 상태 초기화
    verification_start_time = None
    verification_end_time = None  # 인증 조건 유지 시간

    while cam.isOpened():
        success, frame = cam.read()
        if not success:
            break  # 웹캠으로부터 프레임을 읽지 못하면 루프 중단

        if not verified:
            # 이미지 전처리 및 모델 예측
            predictions = model.predict(frame)

            # 예측 결과 중 가장 큰 박스 찾기 >> 화면과 가장 가까운 사람 한 명만 인식
            largest_area = 0
            largest_prediction = None

            for prediction in predictions:
                w = int(prediction['width'])
                h = int(prediction['height'])
                area = w * h

                if area > largest_area:
                    largest_area = area
                    largest_prediction = prediction

            # 가장 큰 박스 그리기
            if largest_prediction is not None:
                # 좌표와 크기 추출 및 변환
                center_x = int(largest_prediction['x'])
                center_y = int(largest_prediction['y'])
                w = int(largest_prediction['width'])
                h = int(largest_prediction['height'])

                # 중심 좌표에서 상자의 왼쪽 상단 모서리 좌표 계산
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # 직사각형 그리기
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # 라벨과 정확도 표시
                label = largest_prediction['class']
                accuracy = largest_prediction['confidence']
                cv2.putText(frame, f"{label}: {accuracy:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                            2)

                if label == "face" and accuracy >= 0.8:
                    if verification_start_time is None:
                        verification_start_time = time.time()
                    if verification_end_time is None:
                        verification_end_time = verification_start_time

                    # 현재 시간과 인증 시작 시간 간의 차이 계산
                    verification_end_time = time.time() - verification_start_time

                    # 5초 동안 인증 조건을 유지
                    if verification_end_time >= 5:
                        verified = True
                        countdown_start_time = time.time()
                        countdown_duration = 10  # 10초 동안 카운트 다운
                    else:
                        verified = False

        # 5초 동안 인증 조건을 유지한 경우
        if verified:
            countdown_remaining = countdown_duration - (time.time() - countdown_start_time)
            if countdown_remaining > 0:
                cv2.putText(frame, f"Verified ({int(countdown_remaining)} seconds left)",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                verified = False  # 인증 시간이 종료되면 인증 상태를 초기화
                break

        else:
            cv2.putText(frame, "Not Verified. Facial recognition should be maintained with at least 80% confidence for 5 seconds.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 결과 표시
        cv2.imshow("Face Recognition", frame)

        # 'q' 키를 누르면 루프 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 웹캠을 루프 종료 후 닫음
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
