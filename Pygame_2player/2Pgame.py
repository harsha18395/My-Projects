import pygame
import os 
pygame.font.init()

WIDTH, HEIGHT = 900,500
WIN = pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption("First Game !")
BCOLOR = (0,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
YELLOW  = (255,255,0)
WHITE = (255,255,255)
BORDER = pygame.Rect(WIDTH//2 - 5,0,10,HEIGHT)
HEALTH_FONT = pygame.font.SysFont('comicsans',40)
FPS = 60
VEL = 5
BULLET_VEL =7
MAX_BULLETS = 3
SPACESHIP_WIDTH , SPACESHIP_HEIGHT = 40,40

YELLOW_HIT = pygame.USEREVENT + 1 # 1st hit 
RED_HIT = pygame.USEREVENT + 2 # 2nd event

dir = os.path.dirname(__file__)

BG = pygame.image.load(os.path.join(dir,'Assets','space.png'))
BG = pygame.transform.scale(BG,(WIDTH , HEIGHT))
YELLOW_SPACESHIP = pygame.image.load(os.path.join(dir,'Assets','spaceship_yellow.png'))
YELLOW_SPACESHIP = pygame.transform.rotate(pygame.transform.scale(YELLOW_SPACESHIP,(SPACESHIP_WIDTH , SPACESHIP_HEIGHT)),90)
RED_SPACESHIP = pygame.image.load(os.path.join(dir,'Assets','spaceship_red.png'))
RED_SPACESHIP = pygame.transform.rotate(pygame.transform.scale(RED_SPACESHIP,(SPACESHIP_WIDTH , SPACESHIP_HEIGHT)),-90)

def draw_window(red,yellow,yellow_bullets,red_bullets,red_health,yellow_health):
    WIN.blit(BG,(0,0))
    pygame.draw.rect(WIN,BLACK,BORDER)

    red_health_text = HEALTH_FONT.render('Health: ' + str(red_health), 1, WHITE)
    yellow_health_text = HEALTH_FONT.render('Health: ' + str(yellow_health), 1,WHITE)
    WIN.blit(red_health_text,(WIDTH - red_health_text.get_width()-10,10 ))
    WIN.blit(yellow_health_text,(10,10 ))

    WIN.blit(YELLOW_SPACESHIP,(yellow.x,yellow.y)) #draw surface
    WIN.blit(RED_SPACESHIP,(red.x,red.y))
    for bullet in yellow_bullets:
        pygame.draw.rect(WIN,YELLOW,bullet)
    for bullet in red_bullets:
        pygame.draw.rect(WIN,RED,bullet)    
    pygame.display.update()

def yellow_handle_movement(keys_pressed,yellow):
    if keys_pressed[pygame.K_a] and yellow.x - VEL > 0 : #left - yellow ship
        yellow.x -= VEL
    if keys_pressed[pygame.K_d] and yellow.x + VEL < WIDTH/2- 5 - yellow.width : #Right
        yellow.x += VEL
    if keys_pressed[pygame.K_w] and yellow.y - VEL > 0: #up
        yellow.y -= VEL
    if keys_pressed[pygame.K_s] and yellow.y + VEL < HEIGHT - yellow.height : #down
        yellow.y += VEL     

def red_handle_movement(keys_pressed,red):
    if keys_pressed[pygame.K_LEFT] and red.x - VEL > WIDTH/2+5: #left - red ship
        red.x -= VEL
    if keys_pressed[pygame.K_RIGHT] and red.x + VEL < WIDTH - red.width: #Right
        red.x += VEL
    if keys_pressed[pygame.K_UP] and red.y - VEL > 0: #up
        red.y -= VEL
    if keys_pressed[pygame.K_DOWN] and red.y + VEL < HEIGHT - red.height: #down
        red.y += VEL     

def handle_bullets(yellow_bullets,red_bullets,yellow,red):
    for bullet in yellow_bullets:
        bullet.x += BULLET_VEL
        if red.colliderect(bullet):
            pygame.event.post(pygame.event.Event(RED_HIT))
            yellow_bullets.remove(bullet) 
        elif  bullet.x > WIDTH:
            yellow_bullets.remove(bullet)         


    for bullet in red_bullets:
        bullet.x -= BULLET_VEL
        if yellow.colliderect(bullet):
            pygame.event.post(pygame.event.Event(YELLOW_HIT))
            red_bullets.remove(bullet)
        elif bullet.x+ bullet.width < 0:
            red_bullets.remove(bullet)     

def main():
    red = pygame.Rect(700,300,SPACESHIP_WIDTH,SPACESHIP_HEIGHT)
    yellow = pygame.Rect(100,300,SPACESHIP_WIDTH,SPACESHIP_HEIGHT)

    yellow_bullets = []
    red_bullets = []

    red_health = 10
    yellow_health = 10

    clock = pygame.time.Clock()
    run =True
    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run =False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LCTRL and len(yellow_bullets) < MAX_BULLETS:
                    bullet = pygame.Rect(yellow.x + yellow.width, yellow.y + yellow.height//2 - 2 , 5,2)
                    yellow_bullets.append(bullet)

                if event.key == pygame.K_RCTRL and len(red_bullets) < MAX_BULLETS:
                    bullet = pygame.Rect(red.x , red.y + red.height//2 - 2 , 5,2)
                    red_bullets.append(bullet)

            if event.type == RED_HIT:
                red_health -= 1
                
            if event.type == YELLOW_HIT:
                yellow_health -= 1    

        winner_text = ''
        if red_health <= 0 :
            winner_text = 'Yellow wins!'
        if yellow_health <= 0:
            winner_text = 'Red Wins!'            

        if winner_text != '':
            pass #won    

        keys_pressed = pygame.key.get_pressed()
        yellow_handle_movement(keys_pressed,yellow)
        red_handle_movement(keys_pressed,red)           

        handle_bullets(yellow_bullets,red_bullets,yellow,red)

        draw_window(red,yellow,yellow_bullets,red_bullets,red_health,yellow_health)
        
    pygame.quit()

main()    