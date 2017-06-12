from pygame import *
import math


""" Made by Nick Wayne, this is open source so edit it if you want, post any bug reports
	and any suggestions you have. I might work on text wrapping if people ask for it but
	currently i am not implementing it.

	thanks																				"""

#####INIT#####
import pygame
from pygame.locals import *

pygame.init()
width = 800
height = 600
screen_size = width, height
screen = pygame.display.set_mode(screen_size)
done = False
pygame.font.init()
special = {"1": "!", "2": "@", "3": "#", "4": "$", "5": "%", "6": "^", "7": "&", "8": "*", "9": "(", "0": ")", "`": "~",
           "-": "_", "=": "+", ",": "<", ".": ">", "/": "?", ";": ":", "'": chr(34), "[": "{", "]": "}", chr(92): "|"}
pygame.display.set_caption('1-D Collision Simulation')


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

myfont = pygame.font.SysFont('Comic Sans MS', 30)

clock = pygame.time.Clock()
inp = input_page()  # make the page class
mass1box = text_box((int(width / 1.75), height / 2 - 25), 200, 25, "Please enter the mass of Object 1:")  # make the text box classes
velocity1box = text_box((int(width / 1.75), height / 2 + 25), 200, 25, "Please enter the velocity of Object 1:")
mass2box = text_box((int(width / 1.75), height / 2 + 50), 200, 25, "Please enter the mass of Object 2:")
velocity2box = text_box((int(width / 1.75), height / 2 + 75), 200, 25, "Please enter the velocity of Object 2")
elasticitybox = text_box((int(width / 1.75), height / 2 + 100), 200, 25, "Please enter the elasticity. (Between 0 and 1):")

def makeButton(cur, rect):
    if rect.collidepoint(cur):
        print ("button pressed")
        return



square = pygame.Rect((600,480), (108,32))
buttonLabel = myfont.render("Submit", 1, (255, 255, 255))


screen.fill((0, 0, 0))
screen.fill((255,0,0), square)
screen.blit(buttonLabel, (600,475))
inp.lst = [mass1box, velocity1box, mass2box, velocity2box, elasticitybox]  # add the boxes to a list
while done == False:


    pygame.display.update()


    pos = pygame.mouse.get_pos()
    for event in pygame.event.get():
        if event.type == QUIT:
            done = True
        inp.get_input(event, pos)  # give the boxes input

        if convert_to_float(elasticitybox.text) > 1 or convert_to_float(elasticitybox.text) < 0:
            label1 = myfont.render("Error in Elasticity Input!", 1, (255, 255, 0))
            screen.blit(label1, (100, 100))
            label2 = myfont.render ("Value must be between 0 and 1",1, (255,0,255))
            screen.blit(label2, (100, 130))

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # left mouse button?
                makeButton(event.pos, square)
                if event.pos > (640,480) and event.pos <= (672,588):
                    done = True
                #done = True


        '''
        if pygame.event.peek(inp.lst) == False:
            if event.key == K_RETURN:
                done = True
        '''

    inp.render(screen)  # render the boxes


    pygame.display.flip()

pygame.quit()



pygame.init()
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('1-D Collision Simulation')
pygame.font.init()

background_color = (255,255,255)
(width, height) = (800, 600)
screen.fill(background_color)


myfont = pygame.font.SysFont('Comic Sans MS', 30)

mass1 = mass1box.text
velocity1 = velocity1box.text
mass2 = mass2box.text
velocity2 = velocity2box.text
elasticity = elasticitybox.text

print ("Mass1:" + mass1)
print ("Velocity1:" + velocity1)
print ("Mass2:" + mass2)
print ("Velocity2:" + velocity2)
print ("Elasticity:" + elasticity)


drag = 0.999

def addVectors(angle1, length1, angle2, length2):
    x = math.sin(angle1) * length1 + math.sin(angle2) * length2
    y = math.cos(angle1) * length1 + math.cos(angle2) * length2

    angle = 0.5 * math.pi - math.atan2(y, x)
    length = math.hypot(x, y)

    return (angle, length)


def findParticle(particles, x, y):
    for p in particles:
        if math.hypot(p.x - x, p.y - y) <= p.size:
            return p
    return None


def collide(p1, p2):
    dx = p1.x - p2.x
    dy = p1.y - p2.y

    dist = math.hypot(dx, dy)
    if dist < p1.size + p2.size:
        angle = math.atan2(dy, dx) + 0.5 * math.pi
        total_mass = p1.mass + p2.mass

        p1.angle, p1.speed = addVectors(p1.angle, (p1.velocity * (p1.mass - p2.mass) / total_mass), angle, (2 * p2.velocity * p2.mass / total_mass))
        p2.angle, p2.speed = addVectors(p2.angle, (p2.velocity * (p2.mass - p1.mass) / total_mass), (angle + math.pi), (2 * p1.velocity * p1.mass / total_mass))
        p1.velocity *= elasticity
        p2.velocity *= elasticity

        overlap = 0.5 * (p1.size + p2.size - dist + 1)
        p1.x += math.sin(angle) * overlap
        p1.y -= math.cos(angle) * overlap
        p2.x -= math.sin(angle) * overlap
        p2.y += math.cos(angle) * overlap

class Particle():
    def __init__(self, x, y, size, color, angle, velocity, mass):
        self.x = x
        self.y = y
        self.size = size
        #self.colour = (0, 0, 255)
        self.color = color
        self.thickness = 0
        self.angle = angle
        self.velocity = velocity
        self.mass = mass

    def display(self):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.size, self.thickness)

    def move(self):
        #self.x += self.velocity
        self.x += math.sin(self.angle) * self.velocity
        self.y -= math.cos(self.angle) * self.velocity
        #self.velocity *= drag


    def bounce(self):
        if self.x > width - self.size:
            self.x = 2 * (width - self.size) - self.x
            self.angle = - self.angle
            self.velocity *= elasticity

        elif self.x < self.size:
            self.x = 2 * self.size - self.x
            self.angle = - self.angle
            self.velocity *= elasticity

        if self.y > height - self.size:
            self.y = 2 * (height - self.size) - self.y
            self.angle = math.pi - self.angle
            self.velocity *= elasticity

        elif self.y < self.size:
            self.y = 2 * self.size - self.y
            self.angle = math.pi - self.angle
            self.velocity *= elasticity

def convert_to_int(x):
    try:
        return int(x)
    except (ValueError, TypeError):
        return 0





#Object = (X,Y, Size, (Color,Color,Color), Movement Angle in Radians, Velocity, Mass)
#object1 = Particle (300,300, 30, (0,0,255), 1.5708, 1, 2)
#object2 = Particle (500,300, 30, (255,0,0), -1.5708, -2, 4)

mass1 = float(mass1)
mass2 = float(mass2)
velocity1 = float(velocity1)
velocity2 = float(velocity2)
elasticity = float(elasticity)

object1 = Particle (250,300, 30, (0,0,255), math.pi/2, velocity1, mass1)
object2 = Particle (550,300, 30, (255,0,0), -math.pi/2, velocity2 , mass2)


number_of_objects = 2
my_particles = []

particle = object1
my_particles.append(particle)
particle = object2
my_particles.append(particle)




running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(background_color)

    screen.fill(background_color)
    object1.move()
    object1.bounce()
    object2.move()
    object2.bounce()


    collide (object1, object2)
    object1.display()
    object2.display()

    object1label = myfont.render("Object 1 - Blue", 1, (0, 0, 255))
    screen.blit(object1label, (100, 100))

    object2label = myfont.render("Object 2 - Red", 1, (255, 0, 0))
    screen.blit(object2label, (500, 100))


    pygame.display.flip()