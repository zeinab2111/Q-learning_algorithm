import numpy as np
import random
#creer un environnement
def create_matrix(nb):
    matrice = np.zeros([nb, nb], dtype = int)
    for i in range(nb):
        matrice[0,i]=1
        matrice[nb-1,i]=1
        matrice[i, 0] = 1
        matrice[i, nb-1] = 1
        matrice[i, nb//2] = 1
        matrice[nb//2, i] = 1
        matrice[1, nb//2] = 0
        matrice[(nb//2)+1, nb//2] = 0
        matrice[nb//2, 1] = 0
        matrice[nb//2, (nb//2)+1] = 0
    matrice[nb-1,1] = 10
    return matrice

#generer une position initiale aleatoire
def random_position(nb):
    x = random.uniform(1, (nb//2)-1)
    y = random.uniform((nb//2)+1, nb-2)
    row_index= round(random.choice([x, y]))
    column_index= round(random.choice([x, y]))
    position=[row_index, column_index]
    return position

#generer des actions (deplacemenet) d'une facon aleatoire
def random_movement(position):
    x= position[0]
    y= position[1]
    pos = []
    direction=""
    random_action = random.randint(1,4)
    #vers le haut
    if random_action == 1:
        x-=1
        pos=[x,y]
        direction="haut"
    #vers le bas
    elif random_action == 2:
        x+=1
        pos = [x, y]
        direction="bas"
    #a gauche:
    elif random_action == 3:
        y-=1
        pos = [x, y]
        direction="gauche"
    #a droite
    elif random_action == 4:
        y+=1
        pos = [x, y]
        direction="droite"
    return pos

#creer le matrice de recompenses a partice du matrice environnement
def recompenses(matrice,nb):
    recomp= matrice.copy()
    for i in range(nb):
        for j in range(nb):
            if matrice[i,j]==1:
                recomp[i,j]=-1
    recomp[nb-2,1]=10
    return recomp

#creer des matrices de connaissances initiales d'une facon aleatoire
def matrices_de_connaissances(nb):
    np.random.seed(1)
    matrice_haut = np.random.randint(10, size=(nb, nb))
    matrice_bas = np.random.randint(10, size=(nb, nb))
    matrice_gauche = np.random.randint(10, size=(nb, nb))
    matrice_droite = np.random.randint(10, size=(nb, nb))
    return matrice_gauche, matrice_droite, matrice_bas, matrice_haut

#creer un environnement, un matrice de recompense et des matrices de connaissances initiales fixent pour tout l'algorithme
# et une position initiale aleatoire pour effectuer le test (avant et apres apprentissage)
def env_recon(nb):
    environnement = create_matrix(nb)
    matrice_gauche, matrice_droite, matrice_bas, matrice_haut = matrices_de_connaissances(nb)
    recompense = recompenses(environnement, nb)
    position0 = random_position(nb)
    #position_copy = position
    return environnement, recompense, matrice_gauche, matrice_droite, matrice_bas, matrice_haut, position0
environnement, recompense, matrice_gauche, matrice_droite, matrice_bas, matrice_haut, position1 = env_recon(25)
print("environnement"+"\n")
print(environnement)
position0 =position1.copy()

#partir de la  position "position1" avant apprentissage et selon les connaissances initiales seulement
#lst_deplacement_1 est la liste des differentes positions occupes par l'agent pendant son parcours
lst_deplacement_1 =[]
lst_deplacement_1.append(position1)
#nombre de mouvement max
movement=100
while (recompense[position1[0], position1[1]] > -1) and (recompense[position1[0], position1[1]] != 10) and movement > 0:
    x_1 = position1[0]
    y_1 = position1[1]
    Qt_haut1 = matrice_haut[x_1, y_1]
    Qt_bas1 = matrice_bas[x_1, y_1]
    Qt_droite1 = matrice_droite[x_1, y_1]
    Qt_gauche1 = matrice_gauche[x_1, y_1]
    lst_gain1 = [Qt_haut1, Qt_bas1, Qt_gauche1, Qt_droite1]
    max_gain1 = max(lst_gain1)
    indice1 = lst_gain1.index(max_gain1)
    if indice1 == 0:
        x_1 -= 1
        position1 = [x_1, y_1]
    # vers le bas
    elif indice1 == 1:
        x_1 += 1
        position1 = [x_1, y_1]
    # a gauche:
    elif indice1 == 2:
        y_1 -= 1
        position1 = [x_1, y_1]
    # a droite
    elif indice1 == 3:
        y_1 += 1
        position1 = [x_1, y_1]
    lst_deplacement_1.append(position1)
    movement-=1
print("liste de deplacement avant apprentissage", lst_deplacement_1)



def apprentissage(nb, mov, episode):
    alpha = 0.2
    gamma = 0.9
    e = 0.1
    for i in range(episode):
        # a chaque episode on part d'une nouvelle position aleatoire
        position = random_position(nb)
        #reinitialiser le nb max des actions possibles a chaque episode
        movements = mov
        while (recompense[position[0], position[1]]>-1) and (recompense[position[0], position[1]]!=10)  and movements >0:
            r = recompense[position[0], position[1]]
            x =position[0]
            y =position[1]
            #Qt(st, at)
            Qt_haut = matrice_haut[x,y]
            Qt_bas = matrice_bas[x,y]
            Qt_droite = matrice_droite[x,y]
            Qt_gauche = matrice_gauche[x,y]
            lst_gain = [Qt_haut,Qt_bas, Qt_gauche, Qt_droite]
            random_choice = random.random()
            #se deplacer d'une facon aleatoire
            if random_choice < e:
                position=random_movement(position)
            else:
                #se deplacer en se basant sur les matrices de connaissances
                max_gain =  max(lst_gain)
                indice = lst_gain.index(max_gain)
                if indice == 0:
                    x -= 1
                    position = [x, y]
                # vers le bas
                elif indice == 1:
                    x += 1
                    position = [x, y]
                # a gauche:
                elif indice == 2:
                    y -= 1
                    position = [x, y]
                # a droite
                elif indice == 3:
                    y += 1
                    position = [x, y]
            #st+1
            x1 = position[0]
            y1 = position[1]
            #Qt(st+1, ai)
            Qt1_haut = matrice_haut[x1, y1]
            Qt1_bas = matrice_bas[x1, y1]
            Qt1_droite = matrice_droite[x1, y1]
            Qt1_gauche = matrice_gauche[x1, y1]
            lst_gain_1 = [Qt1_haut, Qt1_bas, Qt1_gauche, Qt1_droite]
            max_gaint_1 = max(lst_gain_1)
            #le nombre d'actions possible doit diminuer de 1
            movements-=1
            #mise a jour des matrices de connaissances
            matrice_haut[x, y] = ((1-alpha)*Qt_haut) + (alpha*(r + (gamma*max_gaint_1)))
            matrice_bas[x, y] = ((1-alpha)*Qt_bas) + (alpha*(r + (gamma*max_gaint_1)))
            matrice_droite[x, y] = ((1-alpha)*Qt_droite) + (alpha*(r + (gamma*max_gaint_1)))
            matrice_gauche[x, y] = ((1-alpha)*Qt_gauche) + (alpha*(r + (gamma*max_gaint_1)))
    return matrice_gauche, matrice_droite, matrice_bas, matrice_haut, recompense


#partir de la meme position "position1" mais apres apprentissage et selon les connaissances mises a jour
matrice_gauche2, matrice_droite2, matrice_bas2, matrice_haut2, recomponse2 = apprentissage(25, 100, 10000)
lst_deplacement_2=[]
position2= position0
lst_deplacement_2.append(position2)
while (recompense[position2[0], position2[1]]>-1) and (recompense[position2[0], position2[1]]!=10):
    r = recompense[position2[0], position2[1]]
    x2=position2[0]
    y2=position2[1]
    Qt_haut2 = matrice_haut2[x2,y2]
    Qt_bas2 = matrice_bas2[x2,y2]
    Qt_droite2 = matrice_droite2[x2,y2]
    Qt_gauche2 = matrice_gauche2[x2,y2]
    lst_gain2 = [Qt_haut2,Qt_bas2, Qt_gauche2, Qt_droite2]
    max_gain2 =  max(lst_gain2)
    indice2 = lst_gain2.index(max_gain2)
    if indice2 == 0:
        x2 -= 1
        position2 = [x2, y2]
    # vers le bas
    elif indice2 == 1:
        x2 += 1
        position2 = [x2, y2]
    # a gauche:
    elif indice2 == 2:
        y2 -= 1
        position2 = [x2, y2]
    # a droite
    elif indice2 == 3:
        y2 += 1
        position2 = [x2, y2]
    lst_deplacement_2.append(position2)
print("liste de deplacement apres apprentissage",lst_deplacement_2)
if [23,1] in lst_deplacement_2:
    print("But atteint")
else:
    print("But non atteint")
