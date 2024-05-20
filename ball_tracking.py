#etape 1: importation des librairies:  

#Dans un terminal: 
#pip3 install opencv-python
#pip3 install numpy
#pip3 install matplotlib

import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
from scipy.signal import medfilt

def process_video(input_video, output_video_nb, output_video_contours, output_video_point, lower_bound, upper_bound,file_name):
    #Ouvrir la video
    cap = cv2.VideoCapture(input_video)
    
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la video")
        return

    #Recuperation des parametres de la video d'entree:
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    #Creation d'une video de sortie avec les memes parametres que la video d'entree:

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or 'XVID'
    out_nb = cv2.VideoWriter(output_video_nb, fourcc, fps, (frame_width, frame_height), isColor=False)
    out_cnt = cv2.VideoWriter(output_video_contours, fourcc, fps, (frame_width, frame_height), isColor=True)
    out_point = cv2.VideoWriter(output_video_point, fourcc, fps, (frame_width, frame_height), isColor=True)

    coords = np.empty((0, 2), int)

    #Analyse de chaque frame de la video d'entree:
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            
            # creation d'un masque pour filtrer la balle on filtre deux couleurs differentes pour gerer les ombres et les reflets:
            #couleur une:
            filtered_frame = cv2.inRange(frame, lower_bound, upper_bound)           
            
            cv2.imshow('Filtered Frame', filtered_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break               
            # Rajout de cette frame a la video de sortie:
            out_nb.write(filtered_frame)  


            contours, hierarchy = cv2.findContours(filtered_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                continue
            else:
                #frame_with_contours = cv2.drawContours(frame, contours, -1, (0,0,255), 3) #cette version dessine tous les contour
                max_contour=max(contours, key=cv2.contourArea)
                frame_with_contours = cv2.drawContours(frame, [max_contour], -1, (0, 0, 255), 2) #cette version dessine le contour de plus grande surface
                #cv2.imshow('Frame with Contours', frame_with_contours)
                out_cnt.write(frame_with_contours)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break     
                
                M = cv2.moments(max_contour)
                if M["m00"] != 0: 
                
                    mean_x= int(M["m10"]/M["m00"])
                    mean_y= int(M["m01"]/M["m00"])
                    coords = np.append(coords, [[mean_x, mean_y]], axis=0)  
                    print("Coordonnees de la balle: ", mean_x, mean_y) 
                    img_point=cv2.circle(frame,(mean_x,mean_y),10,(0,0,255),-1)                    
                    out_point.write(img_point)    
                    
                    cv2.imshow('Point', img_point)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break      
        
        else:
            break
    #Fermeture des fichiers:
    cap.release()
    out_nb.release()
    out_cnt.release()
    out_point.release()

    # Closes all the frames
    cv2.destroyAllWindows()     
    

    #on inverse les coordonnees y pour avoir le repere en bas a gauche (par defaut en haut a gauche sur openCV):
    #coords[:, 1] = frame_height - coords[:, 1]
    #filtarge des valeurs aberrantes:
    smoothed_x = medfilt(coords[:, 0]  , kernel_size=15)
    smoothed_y = medfilt(coords[:, 1], kernel_size=15)    
    
    coords = np.column_stack((smoothed_x, smoothed_y))
    #Affichage des coordonnees de la balle:
    print("Coordonnees de la balle:")
    print(coords)
    print("X: ", coords[:, 0]) 
    print("Y: ", coords[:, 1]) 


    #Affichage des coordonnees de la balle:
    plt.title("Trajectoire de la balle: " + file_name)
    plt.plot(coords[:, 0], coords[:, 1], marker='o')
    plt.savefig('Trajectoire_' + file_name + '.png')
    plt.show()

    #trajectoire en fonction du temps:
    time = np.arange(0, len(coords)/fps, 1/fps)
    plt.title("Trajectoire de la balle en fonction du temps: " + file_name)
    plt.plot(time, coords[:, 0], label='X')
    plt.plot(time, coords[:, 1], label='Y')
    plt.legend()
    plt.xlabel('Temps (s)')
    plt.ylabel('Position')
    plt.savefig('Trajectoire_temps_' + file_name + '.png')
    plt.show()

    #affichage trajectoire 3D en fonction du temps (on suppose vitesse constante, pas de calcul avec stereo vision):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(coords[:, 0], time, coords[:, 1]) #on a le temps en abscisse, la position en ordonnee et la vitesse en profondeur
    ax.set_xlabel('Horizontal(X)')
    ax.set_ylabel('Temps(Z)')
    ax.set_zlabel('Vertical(Y)')
    #set elevation and azimuth
    ax.view_init(elev=-165, azim=22)
    plt.savefig('Trajectoire_3D_' + file_name + '.png')
    plt.show()

    return coords

def compute_3D_position_cam(coords_cam_R, coords_cam_L):
    e = 8  #distance entre les deux cameras
    f = 20 #distance focal
    d = coords_cam_R[:, 0] - coords_cam_L[:, 0]
    #coordonnees 3D de la balle dans le repere de la camera droite et gauche:
    x_cam_R = coords_cam_R[:, 0] * e / d
    x_cam_L = coords_cam_L[:, 0] * e / d
   
    y_cam_R = coords_cam_R[:, 1] * e / d
    y_cam_L = coords_cam_L[:, 1] * e / d
   
    z_cam_R = e / d * f
    z_cam_L = e / d * f

    #affichage trajectoire 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_cam_R,  z_cam_R, y_cam_R,  label='Camera droite')
    ax.plot(x_cam_L, z_cam_L,  y_cam_L, label='Camera gauche')
    ax.set_xlabel('Horizontal(X)')
    ax.set_ylabel('Vertical(Y)')
    ax.set_zlabel('Profondeur(Z)')
    plt.savefig('Trajectoire_3D.png')
    plt.show()

    matrice_tranfo = np.array([
    [0.96592583, 0, 0.25881905,-9.3],
    [0, 1, 0, -3.5],
    [-0.25881905, 0, 0.96592583, -8.2 ],
    [0, 0, 0, 1]])


    coords_3D = np.vstack((x_cam_L, z_cam_L, y_cam_L, np.ones(len(x_cam_R))))

     
    coords_3D_transformed = np.dot(matrice_tranfo, coords_3D)
   
    #inversion de y et z pour avoir le repere en bas a gauche à cause de opencv:
    coords_3D_transformed[2] = -coords_3D_transformed[2]
    coords_3D_transformed[1] = -coords_3D_transformed[1]

    print("Plot des Coordonnees 3D de la balle:")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(coords_3D_transformed[0], coords_3D_transformed[1], coords_3D_transformed[2], label='Camera Gauche', color='red')
    ax.set_xlabel('Longueur')
    ax.set_ylabel('Largeur')
    ax.set_zlabel('Hauteur')    
    #afficher a partir de zero et dessiner un terrain de tennis en dessous  ( redefinir le slimites)
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 11)
    ax.set_zlim(0, 7)

    # Dessiner le terrain de tennis
    court_length = 23.77
    court_width = 10.97 - 1.37 #(10.97-8.23)/2 = 1.37 ( on affiche pas la ligne de double coté droit)
    single_court_width = 8.23
    net_height = 0.91  # assuming a net height for visualization

    # Les lignes de double
    ax.plot([0, 0, court_length, court_length, 0], [0, court_width, court_width, 0, 0], [0, 0, 0, 0, 0], color='slategray')

    # Les lignes de simple
    ax.plot([0, 0, court_length, court_length, 0],  [0, single_court_width, single_court_width, 0, 0], [0, 0, 0, 0, 0],color='slategray')

    # Ligne centrale de service
    ax.plot([court_length/2, court_length/2],  [0, single_court_width], [0, 0],color='slategray')

    # Filet
    ax.plot([court_length/2, court_length/2],  [0, court_width],[0, 0], color='slategray')

    # Lignes de service
    service_line_distance = 6.40  # Distance from net to service line
    ax.plot([court_length/2 - service_line_distance, court_length/2 - service_line_distance],  [0, single_court_width], [0, 0],color='slategray')
    ax.plot([court_length/2 + service_line_distance, court_length/2 + service_line_distance],  [0, single_court_width], [0, 0],color='slategray')

    # Filet (horizontal)
    ax.plot([court_length/2, court_length/2],  [0, court_width],[0, 0], color='slategray')

    # Afficher les limites du terrain
    ax.plot([0, 0, court_length, court_length, 0], [0, 0, 0, court_width, court_width],[0, 0, 0, 0, 0], color='slategray')
        
    plt.savefig('Trajectoire_3D_repere.png')
    plt.show()

def main(): 
    print("TIPE Tracking de balle:")
    
    #Importation de la video droite:
    input_video = 'videos/right_short.mp4'
    output_video_nb = 'videos/output_nb_r.mp4'
    output_video_contours = 'videos/output_cnt_r.mp4'
    output_video_point = 'videos/output_point_r.mp4'

    #parametres du filtre:
    color2 = np.array([41,200,43])

    #  tolerance
    tolerance = np.array([40, 100, 60])

    lower_bound = color2 - tolerance
    upper_bound = color2 + tolerance



    coords_cam_R = process_video(input_video, output_video_nb, output_video_contours,output_video_point, lower_bound, upper_bound, "droite")

    #Importation de la video gauche:
    input_video = 'videos/left_short.mp4'
    output_video_nb = 'videos/output_nb_l.mp4'
    output_video_contours = 'videos/output_cnt_l.mp4'
    output_video_point = 'videos/output_point_l.mp4'

    coords_cam_L = process_video(input_video, output_video_nb, output_video_contours,output_video_point, lower_bound, upper_bound , "gauche")

    compute_3D_position_cam(coords_cam_R, coords_cam_L)


if __name__ == "__main__":
    main()