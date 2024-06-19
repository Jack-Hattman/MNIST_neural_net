import pygame
import numpy as np

def get_user_image():
    pygame.init()
    pygame.font.init()

    WHITE = [255, 255, 255]
    BLACK = [0, 0, 0]

    FPS = 144

    WIDTH, HEIGHT = 280, 280

    ROWS = COLS = 28

    TOOLBAR_HEIGHT = HEIGHT - WIDTH

    PIXEL_SIZE = WIDTH // COLS

    BG_COLOR = BLACK

    DRAW_GRID_LINES = False

    def get_font(size):
        return pygame.font.SysFont("Arial", size)


    WIN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Drawing Program")

    def init_grid(rows, cols, color):
        grid = []

        for i in range(rows):
            grid.append([])
            for _ in range(cols):
                grid[i].append(color)

        return grid

    def draw_grid(win, grid):
        for i, row in enumerate(grid):
            for j, pixel in enumerate(row):
                pygame.draw.rect(win, pixel, (j * PIXEL_SIZE, i * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE))

    def draw(win, grid):
        win.fill(BG_COLOR)
        draw_grid(win, grid)
        pygame.display.update()

    def get_row_col_from_pos(pos):
        x, y = pos

        center = [y // PIXEL_SIZE, x // PIXEL_SIZE]

        pos_arr = [
            [[center[0] + 1, center[1] - 1],    [center[0] + 1, center[1]],      [center[0] + 1, center[1] + 1]],
            [[center[0], center[1] - 1],        [center[0], center[1]],          [center[0], center[1] + 1]],
            [[center[0] - 1, center[1] - 1],    [center[0] - 1, center[1]],      [center[0] - 1, center[1] + 1]]
        ]

        col_arr = [
            [False, False, False],
            [False, False, False],
            [False, False, False]
        ]

        for i in range(3):
            for j in range(3):

                x_dist = (x) - (pos_arr[i][j][1] * PIXEL_SIZE)
                y_dist = (y) - (pos_arr[i][j][0] * PIXEL_SIZE)
                dist = np.sqrt((x_dist**2)+(y_dist**2))

                if dist < (PIXEL_SIZE * 1.25):
                    col_arr[i][j] = True

        return pos_arr, col_arr

    run = True
    clock = pygame.time.Clock()
    grid = init_grid(ROWS, COLS, BG_COLOR)

    while run:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()

                try:
                    pos_arr, col_arr = get_row_col_from_pos(pos)

                    for i in range(3):
                        for j in range(3):

                            row = pos_arr[i][j][0]
                            col = pos_arr[i][j][1]

                            if col_arr[i][j]:
                                grid[row][col] = WHITE


                except IndexError:
                    pass


        draw(WIN, grid)

    pygame.image.save(WIN, 'num.png')

    pygame.quit()