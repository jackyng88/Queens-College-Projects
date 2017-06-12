
""" Made by Nick Wayne, this is open source so edit it if you want, post any bug reports
	and any suggestions you have. I might work on text wrapping if people ask for it but
	currently i am not implementing it.

	thanks																				"""

#####INIT#####
import pygame
from pygame.locals import *




#####INIT#####
class input_page:  # the actual screen

    def __init__(self):
        self.lst = []  # list of text box's
        self.current = 0  # currently selected string

    def get_input(self, event, mouse_pos):
        if event.type == KEYDOWN:
            if event.key == K_RETURN or event.key == K_TAB:  # move to next box in the list
                if self.current < len(self.lst) - 1:
                    self.current += 1
        if event.type == MOUSEBUTTONDOWN:  # sees if you click in the box
            for i in range(len(self.lst)):
                if self.lst[i].rect.collidepoint(mouse_pos):
                    self.lst[i].current = True
                    self.current = i
                    for g in range(len(self.lst)):
                        if g != i:
                            self.lst[g].current = False

        for i in range(len(self.lst)):  # makes all other text boxes not the current one selected
            if i == self.current:
                self.lst[i].current = True
                self.lst[i].get_input(event)
                for g in range(len(self.lst)):
                    if g != i:
                        self.lst[g].current = False

    def render(self, screen):
        for i in range(len(self.lst)):  # renders the text boxes
            self.lst[i].render(screen)


class text_box:
    def __init__(self, location, width, height, question=None, text_color=(255, 255, 255), font=None, font_size=20):
        self.location = location
        self.text = ""
        self.question = question
        self.current = False
        self.rect = pygame.Rect((location), (width, max(height, 25)))
        self.font_size = font_size
        self.font = pygame.font.Font(font, font_size)
        self.text_color = text_color
        self.outline = (255, 255, 255)
        self.rect_color = (0, 0, 0)

    def render(self, screen):
        if self.current == True:
            temp = (self.rect[0] - 3, self.rect[1] - 3, self.rect[2] + 6, self.rect[
                3] + 6)  # if you change the numbers, the second two need to be multiplied by 2 and postive
            pygame.draw.rect(screen, (255, 105, 34), temp)
        pygame.draw.rect(screen, self.rect_color, self.rect)
        pygame.draw.rect(screen, self.outline, self.rect, 1)
        screen.blit(self.font.render(self.question, 1, self.text_color),
                    (self.location[0] - self.font.size(self.question)[0] - 100, self.location[1] + 4))
        screen.blit(self.font.render(self.text, 1, self.text_color), (self.location[0] + 2, self.location[1] + 4))

    def get_input(self, event):
        if event.type == KEYDOWN:
            if 31 < event.key < 127 and event.key != 8:  # making sure not backspace or error throwing key
                if event.mod & (KMOD_SHIFT | KMOD_CAPS):  # sees if caps or shift is on
                    if chr(event.key) in special.keys():  # Shifted keys
                        self.text += special[chr(event.key)]  # adds to the current string
                    else:
                        self.text += chr(event.key).upper()  # uppercase
                else:
                    self.text += chr(event.key)  # lowercase
            if event.key == 8:  # Backspace
                self.text = self.text[0:-1]
            if event.key == 127:  # delete entire string, comment out if you want
                self.text = ""
            if self.font.size(self.text)[0] > self.rect.size[
                0] - 5:  # makes sure it isn't making text outside of the rect
                self.text = self.text[0:-1]

def convert_to_float(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return 0.0



def makeButton(cur, rect):
    if rect.collidepoint(cur):
        print ("button pressed")
        return


