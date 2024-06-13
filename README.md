# Computing the trajectory of a tennis ball using stereo video analysis.

------------------

The program uses stereo vision using two cameras, to track a tennis ball on a 3D court.

- First, it detects the ball and its coordinates in the two frames via color filtering.
- Then, using photogrammetry formulas and the coordinates from both the left and right image frames, it calculates the 3D coordinates of the ball within each camera's reference frame.
- Finally, these coordinates are transformed into a common reference system to visualize the ball's trajectory on the 3D court.

The footage comes from a tennis simulation I developed in Unity, designed to capture the scene simultaneously from two different camera perspectives.

https://github.com/rantaluca/ball_tracking/assets/102813576/69435848-0b3f-45fc-94bc-c70752da114d

https://github.com/rantaluca/ball_tracking/assets/102813576/bb6f24d6-d586-4a89-89e0-c222867ac8fa

https://github.com/rantaluca/ball_tracking/assets/102813576/669b6237-f1ab-45ca-865b-d2f8cfe60376

https://github.com/rantaluca/ball_tracking/assets/102813576/21a5eee0-e6c7-4fe9-8147-3ea805cee3f6

![Trajectoire_3D_repere](https://github.com/rantaluca/ball_tracking/assets/102813576/f1f78b13-3b47-49b2-bedc-623e04f927a1)

------------------

FR: 
Le principe de ce programme est de suivre une balle de tennis dans un court de tennis en 3D. 

- Pour cela, on utilise deux cameras pour capturer la trajectoire de la balle. 
- On commence par filtrer les couleurs de la balle pour detecter sa position dans chaque frame de la video. 
- Ensuite, on calcule les coordonnees 3D de la balle dans le repere de chaque camera. 
- Enfin, on transforme les coordonnees 3D de la balle dans un repere commun pour afficher la trajectoire de la balle dans un court de tennis en 3D.

Les videos sont issus d'une simulation de tennis sur unity que j'ai aussi realisé, pour capturer la même scene avec deux cameras.

