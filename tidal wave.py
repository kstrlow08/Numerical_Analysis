import pygame
import sys
from collections import deque

# Pygame 초기화
pygame.init()

# 화면 크기 설정
screen = pygame.display.set_mode((1000, 1000))

# 색상 정의
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# 구면파의 최대 개수 및 유지 시간 설정
MAX_WAVES = 10  # 화면에 동시에 존재할 수 있는 구면파의 최대 개수
WAVE_LIFETIME = 5000  # 구면파가 화면에 유지되는 시간 (밀리초)
WAVE_SPEED = 0.4  # 구면파의 퍼져나가는 속도 (픽셀/밀리초)

# 구면파들을 저장할 큐
waves = deque()

# 메인 루프
while True:
    # 이벤트 처리
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # 마우스 클릭 위치와 현재 시간 가져오기
            pos = pygame.mouse.get_pos()
            time_clicked = pygame.time.get_ticks()

            # 구면파 추가 (위치, 생성 시간)
            waves.append((pos, time_clicked))

            # 구면파의 개수가 최대 개수를 넘으면 오래된 구면파 제거
            if len(waves) > MAX_WAVES:
                waves.popleft()

    # 배경 색상 설정
    screen.fill(BLACK)

    # 현재 시간 가져오기
    current_time = pygame.time.get_ticks()

    # 구면파 그리기 (유지 시간 내에 있는 구면파들만)
    for wave in list(waves):
        pos, time_clicked = wave
        time_elapsed = current_time - time_clicked
        if time_elapsed < WAVE_LIFETIME:
            radius = int(time_elapsed * WAVE_SPEED)
            pygame.draw.circle(screen, WHITE, pos, radius, 1)
        else:
            waves.popleft()  # 유지 시간이 지난 구면파 제거

    # 화면 업데이트
    pygame.display.flip()