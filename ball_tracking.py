#etape 1: importation des librairies:  

#Dans un terminal: 
#pip3 install opencv-python
#pip3 install numpy
#pip3 install matplotlib

import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt

def get_ball_color():
    # Couleur RGB pour la conversion
    color_rgb = np.uint8([[[49,154,90]]])  # BGR format in OpenCV
    color_hsv = cv2.cvtColor(color_rgb, cv2.COLOR_BGR2HSV)
    print("HSV Value:", color_hsv[0][0])

def process_video(input_video, output_video_nb, output_video_contours, output_video_point, lower_bound, upper_bound, file_name):
    
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

    x_cords = np.array([])
    y_cords = np.array([])

    #Analyse de chaque frame de la video d'entree:
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            
            # creation d'un masque pour filtrer la balle:
            filtered_frame = cv2.inRange(frame, lower_bound, upper_bound) 
            
            contours, hierarchy = cv2.findContours(filtered_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 0:
                continue
            else:
                #frame_with_contours = cv2.drawContours(frame, contours, -1, (0,0,255), 3) #cette version dessine tous les contour
                max_contour=max(contours, key=cv2.contourArea )
                frame_with_contours = cv2.drawContours(frame, max_contour, -1, (0,0,255), 2) #cette version dessine le contour de plus grande surface
                out_cnt.write(frame_with_contours)
                
                M = cv2.moments(max_contour)
        
                mean_x= int(M["m10"]/M["m00"])
                mean_y= int(M["m01"]/M["m00"])

                x_cords = np.append(x_cords,mean_x)
                y_cords = np.append(y_cords,mean_y)
                cv2.imshow('Frame with Contours', frame_with_contours)
            
            img_point=cv2.circle(frame,(mean_x,mean_y),10,(0,0,255),-1)
            # Rajout de cette frame a la video de sortie:
            out_nb.write(filtered_frame)  
            
            out_point.write(img_point)
            
            cv2.imshow('Filtered Frame', filtered_frame)         
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

    #Affichage des coordonnees de la balle:
    print("Coordonnees de la balle:")
    print("x: ", x_cords)
    print("y: ", y_cords)

    #Affichage des coordonnees de la balle:
    #donner le nom du fichier au plot
   
    plt.title("Trajectoire de la balle:" + input_video)
    plt.plot(x_cords, y_cords)
    #faire en sorte que le plot soit sauvegarde dans un fichier et que le programme continue sans l'affihcer
    plt.savefig('trajectoire_balle_'+file_name+'.png')



def main(): 
    print("TIPE Tracking de balle:")
    
    #Importation de la video droite:
    input_video = 'videos/right_short.mp4'
    output_video_nb = 'videos/output_nb_r.mp4'
    output_video_contours = 'videos/output_cnt_r.mp4'
    output_video_point = 'videos/output_point_r.mp4'

    #parametres du filtre:
    lower_bound = np.array([49,154,90]) #couleur bgr de la balle sombre
    upper_bound = np.array([119,255,226]) #couleur bgr de la balle claire
    
    process_video(input_video, output_video_nb, output_video_contours,output_video_point, lower_bound, upper_bound, "droite")

    #Importation de la video gauche:
    input_video = 'videos/left_short.mp4'
    output_video_nb = 'videos/output_nb_l.mp4'
    output_video_contours = 'videos/output_cnt_l.mp4'
    output_video_point = 'videos/output_point_l.mp4'

    process_video(input_video, output_video_nb, output_video_contours,output_video_point, lower_bound, upper_bound, "gauche")


if __name__ == "__main__":
    main()