# Calcul de la trajectoires d'une balle de tennis à partir d'une video stereo 

# Calculating the trajectory of a tennis ball from a stereo video.

------------------

Le principe de ce programme est de suivre une balle de tennis dans un court de tennis en 3D. 
Pour cela, on utilise deux cameras pour capturer la trajectoire de la balle. 
On commence par filtrer les couleurs de la balle pour detecter sa position dans chaque frame de la video. 
Ensuite, on calcule les coordonnees 3D de la balle dans le repere de chaque camera. 
Enfin, on transforme les coordonnees 3D de la balle dans un repere commun pour afficher la trajectoire de la balle dans un court de tennis en 3D.

Les videos sont issus d'une simulation de tennis sur unity que j'ai aussi realisé, pour capturer la même scene avec deux cameras.

-----------------

The principle of this program is to track a tennis ball in a tennis court in 3D.
To do this, we use two cameras to capture the trajectory of the ball.
We start by filtering the colors of the ball to detect its position in each frame of the video.
Then, we calculate the 3D coordinates of the ball in the reference of each camera.
Finally, we transform the 3D coordinates of the ball into a common reference to display the trajectory of the ball in a 3D tennis court.

The videos come from a tennis simulation on unity that I also made, to capture the same scene with two cameras.

https://github.com/rantaluca/ball_tracking/assets/102813576/69435848-0b3f-45fc-94bc-c70752da114d

https://github.com/rantaluca/ball_tracking/assets/102813576/669b6237-f1ab-45ca-865b-d2f8cfe60376

https://github.com/rantaluca/ball_tracking/assets/102813576/21a5eee0-e6c7-4fe9-8147-3ea805cee3f6

![Trajectoire_3D_repere](https://github.com/rantaluca/ball_tracking/assets/102813576/f1f78b13-3b47-49b2-bedc-623e04f927a1)
