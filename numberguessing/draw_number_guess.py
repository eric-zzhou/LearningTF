from tkinter import Tk
from tkinter import messagebox

from tensorflow import keras
import numpy as np
import pygame
from PIL import Image

# pygame.font.init()


WIN_WIDTH = WIN_HEIGHT = 560
# STAT_FONT = pygame.font.SysFont("comicsans", 36)

win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Number Guesser")

model = keras.models.load_model("mnist_model.h5")


class Pixel:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = (255, 255, 255)
        self.neighbors = []

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, (self.x, self.y, self.x + self.width, self.y + self.height))

    def getNeighbors(self, g):
        # Get the neighbours of each pixel in the grid, this is used for drawing thicker lines
        j = self.x // 20  # the var i is responsible for denoting the current col value in the grid
        i = self.y // 20  # the var j is responsible for denoting thr current row value in the grid
        rows = 28
        cols = 28

        # Horizontal and vertical neighbors
        if i < cols - 1:  # Right
            self.neighbors.append(g.pixels[i + 1][j])
        if i > 0:  # Left
            self.neighbors.append(g.pixels[i - 1][j])
        if j < rows - 1:  # Up
            self.neighbors.append(g.pixels[i][j + 1])
        if j > 0:  # Down
            self.neighbors.append(g.pixels[i][j - 1])

        # Diagonal neighbors
        if j > 0 and i > 0:  # Top Left
            self.neighbors.append(g.pixels[i - 1][j - 1])

        if j + 1 < rows and i > -1 and i - 1 > 0:  # Bottom Left
            self.neighbors.append(g.pixels[i - 1][j + 1])

        if j - 1 < rows and i < cols - 1 and j - 1 > 0:  # Top Right
            self.neighbors.append(g.pixels[i + 1][j - 1])

        if j < rows - 1 and i < cols - 1:  # Bottom Right
            self.neighbors.append(g.pixels[i + 1][j + 1])


class Grid:
    pixels = []

    def __init__(self, row, col, width, height):
        self.rows = row
        self.cols = col
        self.len = row * col
        self.width = width
        self.height = height
        self.generatePixels()
        pass

    def draw(self, surface):
        for row in self.pixels:
            for col in row:
                col.draw(surface)

    def generatePixels(self):
        x_gap = self.width // self.cols
        y_gap = self.height // self.rows
        self.pixels = []
        for r in range(self.rows):
            self.pixels.append([])
            for c in range(self.cols):
                self.pixels[r].append(Pixel(x_gap * c, y_gap * r, x_gap, y_gap))

        for r in range(self.rows):
            for c in range(self.cols):
                self.pixels[r][c].getNeighbors(self)

    def clicked(self, pos):  # Return the position in the grid that user clicked on
        try:
            t = pos[0]
            w = pos[1]
            g1 = int(t) // self.pixels[0][0].width
            g1 = int(t) // self.pixels[0][0].width
            g2 = int(w) // self.pixels[0][0].height
            return self.pixels[g2][g1]
        except:
            pass


def guess(li):
    newMatrix = [[[] for x in range(len(li))]]
    for i in range(len(li)):
        for j in range(len(li[i])):
            if li[i][j][0] == 255:
                newMatrix[0][i].append(0)
            else:
                newMatrix[0][i].append(1)
    prediction = model.predict(newMatrix)[0]
    t = np.argmax(prediction)
    print(f"I predict this number is a {t} with {prediction[t] * 100}% confidence")
    window = Tk()
    window.withdraw()
    messagebox.showinfo("Prediction", f"I predict this number is a {t} with {prediction[t] * 100}% confidence")
    window.destroy()


def main():
    global g
    g = Grid(28, 28, WIN_WIDTH, WIN_HEIGHT)
    thick = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if pygame.key.get_pressed()[pygame.K_RETURN]:
                pygame.image.save(win, "window.png")
                img = Image.open("window.png")
                img.thumbnail([28, 28])
                img.save("thumbnail.png")
                guess(np.array(img))
                g.generatePixels()
            if pygame.key.get_pressed()[pygame.K_SPACE]:
                thick = not thick
                Tk().withdraw()
                messagebox.showinfo("Thickness", f"Thick marker: {thick}")
            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                clicked = g.clicked(pos)
                clicked.color = (0, 0, 0)
                if thick:
                    for n in clicked.neighbors:
                        n.color = (0, 0, 0)

            if pygame.mouse.get_pressed()[2]:
                pos = pygame.mouse.get_pos()
                clicked = g.clicked(pos)
                clicked.color = (255, 255, 255)
                if thick:
                    for n in clicked.neighbors:
                        n.color = (255, 255, 255)

        g.draw(win)
        pygame.display.update()


if __name__ == '__main__':
    main()
