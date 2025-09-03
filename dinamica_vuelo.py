# ============================================
# File: dinamica_vuelo.py
# Author: Ingrid Castillo, Juan Jesús Quesada and Maykol Echeverry
# Date: 28-Aug-2025
# Description: 
# Program to transform and visualize vector expressed in different coordinate vectors
# ============================================""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

def angular_rates():
    print("Enter the angular rates with respect to the aircraft")
    p_rate = float(input("Enter the angular rate p in deg/s: "))
    q_rate = float(input("Enter the angular rate q in deg/s: "))
    r_rate = float(input("Enter the angular rate r in deg/s: "))
    return p_rate, q_rate, r_rate

# Function for the selection of the flight cases
def menu_cases():
    print("Select the flight case of your preference")
    print("A. No wind, straight- and level flight")
    print("B. Crosswind vector present")
    print("C. Climb or descend with wind")
    print()
    
    while True:
        option2 = input("Select your option (A,B,C): ").upper()  # Works for both 'a' and 'A'
        if option2 == "A":
            print("You selected case A")
            return "A"
        elif option2 == "B":
            print("You selected case B")
            return "B"
        elif option2 == "C":
            print("You selected case C")
            return "C"
        else:
            print("Invalid option. Please enter A, B, or C.")

def inputs(flight_case):
    
    x = float(input("Enter vector x component in km/h: "))
    
    if flight_case == "A":  # straight & level
        y = 0.0
        z = 0.0
        print("y = 0, z = 0 (straight & level)")
    else:
        y = float(input("Enter vector y component in km/h: "))
    
        z = float(input("Enter vector z component in km/h: "))
    
    p = np.array([x, y, z])
    print()
    
    return x, y, z, p

def body_vehicle():
        # Euler angles to know the position of the aircraft
        yaw = float(input("Enter yaw (ψ) in degrees: "))
        pitch = float(input("Enter pitch (θ) in degrees:  "))
        roll = float(input("Enter roll (φ) in degrees: "))
        print()
    
        phi =  math.radians(roll)  
        theta = math.radians(pitch)
        psi = math.radians(yaw)    
      

        Rz = np.array([[ math.cos(psi), -math.sin(psi), 0 ],
                       [ math.sin(psi),  math.cos(psi), 0 ],
                       [    0,           0,               1 ]
                       ])          
    
        Ry = np.array([[ math.cos(theta), 0, math.sin(theta)],
                       [        0          , 1,      0          ],
                       [-math.sin(theta), 0, math.cos(theta)]
                       ])
    
        Rx = np.array([[  1            ,      0            ,      0          ],
                       [  0            , math.cos(phi)  , -math.sin(phi) ],
                       [  0            , math.sin(phi)  , math.cos(phi)  ]
                     ])
        return Rz, Ry, Rx, phi, theta, psi, yaw, pitch, roll
    
def wing_body(alpha,beta):

    alpha_rad = math.radians(alpha)
    beta_rad  = math.radians(beta)
    
    A = np.array([ [math.cos(alpha_rad), 0, -math.sin(alpha_rad)],
                   [0                   , 1,      0           ],
                   [math.sin(alpha_rad), 0,  math.cos(alpha_rad)]
                 ])
    B = np.array([ [ math.cos(-beta_rad), math.sin(-beta_rad), 0],
                   [-math.sin(-beta_rad), math.cos(-beta_rad), 0],
                   [0                   , 0                   , 1]
                 ])
    return A, B

def aircraft_angles(u,v,w,p,flight_case, pitch, Wind, Wd_x, Wd_z):
    V_nwind = math.sqrt(u**2 + v**2 + w**2) #No wind, still atmosphere
    V_inf = np.add(p, Wind) #np.add calculates the vectorial sum
    Vix, Viy, Viz =V_inf
    Magnitude_V_inf = math.sqrt(Vix**2 + Viy**2 + Viz**2)
    
    if flight_case == "A":  # Straight & level
        alpha = pitch      # angle of attack equals the pitch
        beta  = 0.0        # no lateral wind
        gamma = 0.0        # climb angle is zero

    elif flight_case == "B"  :      
        alpha = pitch
        beta  = math.degrees(math.asin(v/Magnitude_V_inf))
        gamma = pitch - alpha
        
    else:
        alpha = math.degrees(math.atan2((w + Wd_z),(u+ Wd_x)))
        beta  = math.degrees(math.asin(v / Magnitude_V_inf))
        gamma = pitch - alpha    
        
    return V_inf, alpha, beta, gamma, V_nwind, Magnitude_V_inf

# --- NUEVAS FUNCIONES PARA EL GRÁFICO 3D ---

def draw_aircraft(ax, R_mat):
    """
    Dibuja un modelo simplificado de avión en 3D.
    ax: Objeto de Axes3D.
    R_mat: Matriz de rotación para posicionar el avión.
    """
    # Coordenadas base del modelo del avión (en el sistema de referencia del cuerpo)
    fuselage_x = np.array([-1, 1, 1, -1, -1, 1, 1, -1])
    fuselage_y = np.array([0, 0, 0, 0, -0.1, -0.1, -0.1, -0.1])
    fuselage_z = np.array([0, 0, -0.2, -0.2, -0.2, -0.2, 0, 0])

    wing_x = np.array([0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0])
    wing_y = np.array([0, 0, 2, 2, 0, 0, -2, -2])
    wing_z = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    
    tail_x = np.array([-0.5, -1, -1, -0.5, -0.5, -1, -1, -0.5])
    tail_y = np.array([0.05, 0.05, 0.05, 0.05, -0.05, -0.05, -0.05, -0.05])  # grosor en Y
    tail_z = np.array([0, 0, -0.5, -0.5, 0, 0, -0.5, -0.5])  # altura de la cola

    # Transformar las coordenadas del avión con la matriz de rotación
    fuselage_rotated = R_mat @ np.vstack((fuselage_x, fuselage_y, fuselage_z))
    wing_rotated = R_mat @ np.vstack((wing_x, wing_y, wing_z))
    tail_rotated = R_mat @ np.vstack((tail_x, tail_y, tail_z))
    
    # Reformatear los arrays para plot_wireframe
    fuselage_X = fuselage_rotated[0].reshape(2, 4)
    fuselage_Y = fuselage_rotated[1].reshape(2, 4)
    fuselage_Z = fuselage_rotated[2].reshape(2, 4)

    wing_X = wing_rotated[0].reshape(2, 4)
    wing_Y = wing_rotated[1].reshape(2, 4)
    wing_Z = wing_rotated[2].reshape(2, 4)
    
    tail_X = tail_rotated[0].reshape(2, 4)
    tail_Y = tail_rotated[1].reshape(2, 4)
    tail_Z = tail_rotated[2].reshape(2, 4)
    
    # Dibujar las partes del avión como mallas
    ax.plot_wireframe(fuselage_X, fuselage_Y, fuselage_Z, color='gray', alpha=0.6)
    ax.plot_wireframe(wing_X, wing_Y, wing_Z, color='gray', alpha=0.6)
    ax.plot_wireframe(tail_X, tail_Y, tail_Z, color='gray', alpha=0.6)

def draw_vectors(ax, Vb, W, V_rel_I, R_mat, p_rate, q_rate, r_rate):
    """
    Dibuja los vectores de velocidad, viento y velocidad relativa,
    y los ejes del sistema de referencia del cuerpo (Body Frame).
    ax: Objeto de Axes3D.
    Vb: Vector de velocidad en el sistema del cuerpo (body frame).
    W: Vector de viento.
    V_rel_I: Vector de velocidad relativa al aire.
    R_mat: Matriz de rotación para posicionar los ejes del avión.
    p_rate, q_rate, r_rate: Tasas de velocidad angular.
    """
    origin = [0, 0, 0]
    
    # Vectores unitarios para los ejes del Body Frame
    x_body = np.array([1, 0, 0])
    y_body = np.array([0, -1, 0])
    z_body = np.array([0, 0, 1])
    
    # Rotar los vectores de los ejes para que se muevan con la aeronave
    x_body_rotated = R_mat @ x_body
    y_body_rotated = R_mat @ y_body
    z_body_rotated = R_mat @ z_body
    
    # Dibujar los ejes del Body Frame
    ax.quiver(origin[0], origin[1], origin[2], x_body_rotated[0], x_body_rotated[1], x_body_rotated[2], color='red', label='X_body')
    ax.quiver(origin[0], origin[1], origin[2], y_body_rotated[0], y_body_rotated[1], y_body_rotated[2], color='green', label='Y_body')
    ax.quiver(origin[0], origin[1], origin[2], z_body_rotated[0], z_body_rotated[1], z_body_rotated[2], color='blue', label='Z_body')
    ax.text(x_body_rotated[0], x_body_rotated[1], x_body_rotated[2], 'Xb', color='red')
    ax.text(y_body_rotated[0], y_body_rotated[1], y_body_rotated[2], 'Yb', color='green')
    ax.text(z_body_rotated[0], z_body_rotated[1], z_body_rotated[2], 'Zb', color='blue')
    
    # --- NUEVA SECCIÓN PARA EL EJE DEL VEHÍCULO FIJO ---
    # Dibujar los ejes del sistema de referencia fijo (Vehicle/Inertial Frame)
    ax.quiver(origin[0], origin[1], origin[2], 1, 0, 0, color='darkred', linestyle='--', label='X_vehicule')
    ax.quiver(origin[0], origin[1], origin[2], 0, -1, 0, color='darkgreen', linestyle='--', label='Y_vehicule')
    ax.quiver(origin[0], origin[1], origin[2], 0, 0, 1, color='darkblue', linestyle='--', label='Z_vehicule')
    ax.text(1.1, 0, 0, 'X_v', color='darkred')
    ax.text(0, -1.1, 0, 'Y_v', color='darkgreen')
    ax.text(0, 0, 1.1, 'Z_v', color='darkblue')
    
    # --- SECCIÓN PARA LOS VECTORES DE VELOCIDAD ANGULAR ---
    # Convertir las tasas angulares en vectores en el sistema del cuerpo
    p_vector_body = np.array([p_rate, 0, 0])
    q_vector_body = np.array([0, -q_rate, 0])
    r_vector_body = np.array([0, 0, r_rate])
    
    # Normalizar y rotar los vectores de velocidad angular
    scale_angular = 0.5 
    
    if np.linalg.norm(p_vector_body) > 0:
        p_rotated = R_mat @ (p_vector_body / np.linalg.norm(p_vector_body)) * scale_angular
        ax.quiver(origin[0], origin[1], origin[2], p_rotated[0], p_rotated[1], p_rotated[2], color='gold', label='p')
        ax.text(p_rotated[0], p_rotated[1], p_rotated[2], 'p', color='gold')
    
    if np.linalg.norm(q_vector_body) > 0:
        q_rotated = R_mat @ (q_vector_body / np.linalg.norm(q_vector_body)) * scale_angular
        ax.quiver(origin[0], origin[1], origin[2], q_rotated[0], q_rotated[1], q_rotated[2], color='orange', label='q')
        ax.text(q_rotated[0], q_rotated[1], q_rotated[2], 'q', color='orange')
    
    if np.linalg.norm(r_vector_body) > 0:
        r_rotated = R_mat @ (r_vector_body / np.linalg.norm(r_vector_body)) * scale_angular
        ax.quiver(origin[0], origin[1], origin[2], r_rotated[0], r_rotated[1], r_rotated[2], color='purple', label='r')
        ax.text(r_rotated[0], r_rotated[1], r_rotated[2], 'r', color='purple')
    
    # Calcular las magnitudes de los vectores de velocidad
    mag_Vb = np.linalg.norm(Vb)
    mag_W = np.linalg.norm(W)
    mag_V_rel_I = np.linalg.norm(V_rel_I)
    
    # Encontrar la magnitud máxima para escalar
    max_magnitude = max(mag_Vb, mag_W, mag_V_rel_I)
    
    if max_magnitude > 0:
        scale_factor = 2.0 / max_magnitude
    else:
        scale_factor = 1.0
    
    # Dibujar el vector Vb (velocidad del cuerpo)
    if mag_Vb > 0:
        Vb_scaled = (R_mat @ (Vb / mag_Vb)) * scale_factor
        ax.quiver(origin[0], origin[1], origin[2], Vb_scaled[0], Vb_scaled[1], Vb_scaled[2], color='blue', label='Vb')
        ax.text(Vb_scaled[0], Vb_scaled[1], Vb_scaled[2], 'Vb', color='blue')
    
    # Dibujar el vector W (viento)
    if mag_W > 0:
        W_scaled = (W / mag_W) * scale_factor
        ax.quiver(origin[0], origin[1], origin[2], W_scaled[0], W_scaled[1], W_scaled[2], color='green', label='W')
        ax.text(W_scaled[0], W_scaled[1], W_scaled[2], 'W', color='green')
    
    # --- Modificaciones para el vector V_inf (Impacto) ---
    
    # El vector V_inf en el sistema del vehículo es V_rel_I.
    # Queremos un vector de impacto que apunte hacia el morro.
    
    # Escalar el vector de impacto para que sea visible, independientemente de los demás.
    # Puedes usar un factor de escala fijo, por ejemplo, 1.5, para que se destaque.
    impacto_scale_factor = 1.5  # Puedes ajustar este valor
    
    if mag_V_rel_I > 0:
        # El vector de impacto será -V_rel_I, que apunta hacia el avión
        V_impacto = -V_rel_I
        mag_V_impacto = np.linalg.norm(V_impacto)
    
        if mag_V_impacto > 0:
            V_impacto_scaled = (V_impacto / mag_V_impacto) * impacto_scale_factor
    
            # El punto de origen es el final del vector de impacto (el morro del avión)
            start_point = V_impacto_scaled
    
            ax.quiver(start_point[0], start_point[1], start_point[2],
                      -V_impacto_scaled[0], -V_impacto_scaled[1], -V_impacto_scaled[2],
                      color='magenta', label='V_inf (Impact)')
    
            # Añadir una etiqueta al final del vector (en el morro)
            ax.text(start_point[0], start_point[1], start_point[2], 'V_inf', color='magenta')
    
    # El vector original V_rel_I se mantiene igual, aunque ahora el de "impacto" es más grande
    if mag_V_rel_I > 0:
        V_rel_I_scaled = (V_rel_I / mag_V_rel_I) * scale_factor
        ax.quiver(origin[0], origin[1], origin[2], V_rel_I_scaled[0], V_rel_I_scaled[1], V_rel_I_scaled[2], color='darkmagenta', label='V_rel_I')
        ax.text(V_rel_I_scaled[0], V_rel_I_scaled[1], V_rel_I_scaled[2], 'V_rel_I', color='darkmagenta')
        
def draw_wind_axes(ax, V_inf):
    """
    Dibuja los ejes del sistema de referencia del viento (wing-fixed).
    El eje X_w se alinea con V_inf.
    ax: Objeto de Axes3D.
    V_inf: Vector de velocidad relativa al aire (en el sistema de referencia del vehículo).
    """
    origin = np.array([0, 0, 0])
    
    # Si no hay velocidad, no se pueden dibujar los ejes
    if np.linalg.norm(V_inf) == 0:
        return

    # Normalizar el vector de velocidad del aire (V_inf) para definir el eje X_w
    x_wind_vector = V_inf / np.linalg.norm(V_inf)
    
    # Para Y_w, tomamos el producto cruz del eje Z del vehículo y X_w
    # para asegurar que Y_w esté en el plano horizontal (o lo más cerca posible)
    z_vehicle = np.array([0, 0, 1])
    y_wind_vector = np.cross(z_vehicle, x_wind_vector)
    
    if np.linalg.norm(y_wind_vector) != 0:
        y_wind_vector = y_wind_vector / np.linalg.norm(y_wind_vector)
    else: # Si x_wind es paralelo a z_vehicle (vuelo vertical), usa x_vehicle para el producto cruz
        x_vehicle = np.array([1, 0, 0])
        y_wind_vector = np.cross(x_vehicle, x_wind_vector)
        y_wind_vector = y_wind_vector / np.linalg.norm(y_wind_vector)
        
    # Para Z_w, tomamos el producto cruz de X_w y Y_w para mantener la regla de la mano derecha
    z_wind_vector = np.cross(x_wind_vector, y_wind_vector)

    # Dibujar los ejes del Viento
    ax.quiver(origin[0], origin[1], origin[2], x_wind_vector[0], x_wind_vector[1], x_wind_vector[2], color='cyan', linestyle='-', label='X_wind')
    ax.quiver(origin[0], origin[1], origin[2], y_wind_vector[0], y_wind_vector[1], y_wind_vector[2], color='magenta', linestyle='-', label='Y_wind')
    ax.quiver(origin[0], origin[1], origin[2], z_wind_vector[0], z_wind_vector[1], z_wind_vector[2], color='orange', linestyle='-', label='Z_wind')
    
    ax.text(x_wind_vector[0] * 1.2, x_wind_vector[1] * 1.2, x_wind_vector[2] * 1.2, 'Xw', color='cyan')
    ax.text(y_wind_vector[0] * 1.2, y_wind_vector[1] * 1.2, y_wind_vector[2] * 1.2, 'Yw', color='magenta')
    ax.text(z_wind_vector[0] * 1.2, z_wind_vector[1] * 1.2, z_wind_vector[2] * 1.2, 'Zw', color='orange')

def plot_3d_model(Vb, W, V_rel_I, R_mat, title, p_rate, q_rate, r_rate):
    """
    Configura y muestra el gráfico 3D en la ventana de Spyder.
    Vb, W, V_rel_I: Vectores a visualizar.
    R_mat: Matriz de rotación del avión.
    title: Título del gráfico.
    p_rate, q_rate, r_rate: Tasas de velocidad angular para la visualización.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    
    # Dibujar el avión y los vectores
    draw_aircraft(ax, R_mat)
    # Se pasan los nuevos parámetros para la velocidad angular y el viento
    draw_vectors(ax, Vb, W, V_rel_I, R_mat, p_rate, q_rate, r_rate)
    
    # Ajustar límites y etiquetas de los ejes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])
    
    # Invertir el eje Z para que apunte hacia abajo (convención aeronáutica)
    ax.invert_zaxis()
    
    ax.legend()
    plt.show()

# --- CÓDIGO PRINCIPAL MODIFICADO CON LLAMADAS A LA FUNCIÓN DE GRÁFICO ---

#Menu1
print("Select the target reference frame")
print("1. Body Fixed Reference Frame")
print("2. Vehicle Carried Coordinate Frame")
print("3. Air trajectory Reference Frame, wing-fixed coordinate system")
print()


while True:
    option = input("Select your option (1,2,3):")
    print()
    
    if (option == "1"):
        
        print("Turning vector into Body Frame")
        print()
        flight_case=menu_cases()
        print()
        
        print("Select the origin reference frame of this vector:")
        print("1. Vehicle-carried Reference Frame")
        print("2. Air/Wing Trajectory Reference Frame")
        origin = input("Select origin frame (1 or 2): ")
        print()
        
        origin_name = "Vehicle Frame" if origin == "1" else "Air/Wing Frame"
        
        p_rate, q_rate, r_rate = angular_rates()
        
        if flight_case == "A": #Still atmosphere
            
            x, y, z, p = inputs(flight_case)
            
            print("Vector entered (km/h) from", origin_name)
            print(f"[ {x} ")
            print(f"  {y} ")
            print(f"  {z} ]")
            print()
                        
            Wd_x = 0
            Wd_y = 0
            Wd_z = 0
            Wind = np.array([Wd_x, Wd_y, Wd_z])
            
            if origin == "1": #vehicle al body (RzRyRxp)Transpuesta
                
                Rz, Ry, Rx, phi, theta, psi, yaw, pitch, roll = body_vehicle()
                
                # Complete rotation matrix. Python 3.5 allows @ to multiply matrixes without using np.dot
                Rot_mat = Rz @ Ry @ Rx
                p_transformed = Rot_mat.T @ p
                u, v, w = p_transformed
                
                print("Vector velocity in Body Frame (km/h):")
                print(f"[ {u:.2f} ")
                print(f"  {v:.2f} ")
                print(f"  {w:.2f} ]")
                print()
                
                V_inf, alpha, beta, gamma, V_nwind, Magnitude_V_inf = aircraft_angles(u,v,w,p,flight_case, pitch, Wind, Wd_x, Wd_z)
                print(f"V_nwind = {V_nwind:.2f} km/h")
                print()
                
                print("Aircraft angles")
                print(f"Angle of attack, α = {alpha:.2f}°")
                print(f"Sideslip angle, β = {beta:.2f}°")
                print(f"Climb angle, γ = {gamma:.2f}°")
                print()
                
                plot_3d_model(p_transformed, Wind, p, Rot_mat, "Body Frame Transformation", p_rate, q_rate, r_rate)
                break
                
          
            elif origin == "2": # Air/Wing Trajectory Reference Frame to Body Fixed
                print("Note: Enter the desired angle of attack (α) to set the aircraft's orientation in level flight")
                alpha = float(input("Enter AoA in degrees: "))
                beta = 0
                pitch = alpha
                gamma = pitch - alpha
    
                A,B = wing_body(alpha, beta)
                LBW = A @ B
                
                p_transformed = LBW @ p
                
                u, v, w = p_transformed
                print("Vector velocity in Body Frame (km/h):")
                print(f"[ {u:.2f} ")
                print(f"  {v:.2f} ")
                print(f"  {w:.2f} ]")
                print()
                
                V_nwind = math.sqrt(u**2 + v**2 + w**2)
                print(f"V_nwind = {V_nwind:.2f} km/h")
                print()
                print(f"Pitch={pitch:.2f}°")
                
                # Para este caso, el avión está alineado con la velocidad del aire, por lo que la matriz de rotación es la identidad.
                # V_inf es lo mismo que p_transformed
                plot_3d_model(p_transformed, Wind, p_transformed, np.identity(3), "Body Frame Transformation (Air/Wing)", p_rate, q_rate, r_rate)
                break
                
        elif flight_case == "B": #Crosswind
            
            x, y, z, p = inputs(flight_case)
            
            print("Vector entered (km/h) from", origin_name)
            print(f"[ {x} ")
            print(f"  {y} ")
            print(f"  {z} ]")
            print()
            
            print("Enter the Wind components:")
            Wd_x = float(input("Enter vector x component in km/h: "))
            Wd_y = float(input("Enter vector y component in km/h: "))
            Wd_z = float(input("Enter vector z component in km/h: "))
            Wind = np.array([Wd_x, Wd_y, Wd_z])
            
            if origin == "1": #vehicle al body (RzRyRxp)Transpuesta
            
                Rz, Ry, Rx, phi, theta, psi, yaw, pitch, roll = body_vehicle()
            
                Rot_mat = Rz @ Ry @ Rx
                p_transformed = Rot_mat.T @ p
                u, v, w = p_transformed
                
                print("Vector velocity in Body Frame (km/h):")
                print(f"[ {u:.2f} ")
                print(f"  {v:.2f} ")
                print(f"  {w:.2f} ]")
                print()
                
                V_inf, alpha, beta, gamma, V_nwind, Magnitude_V_inf = aircraft_angles(u,v,w,p, flight_case, pitch, Wind,Wd_x, Wd_z)
                print(f"Magnitude_V_inf = {Magnitude_V_inf:.2f} km/h")
                print()
                
                print("Aircraft angles")
                print(f"Angle of attack, α = {alpha:.2f}°")
                print(f"Sideslip angle, β = {beta:.2f}°")
                print(f"Climb angle, γ = {gamma:.2f}°")
                print()
                
                plot_3d_model(p_transformed, Wind, V_inf, Rot_mat, "Body Frame with Crosswind", p_rate, q_rate, r_rate)
                break
            
            if origin == "2": #Air to body
                
                print("Note: Enter the desired angle of attack (α) to set the aircraft's orientation in level flight")
                alpha = float(input("Enter AoA in degrees: "))
                beta = float(input("Enter sideslip in degrees: "))
                pitch = alpha
                gamma = pitch - alpha
    
                A,B = wing_body(alpha, beta)
                LBW = A @ B
                
                p_transformed = LBW @ p
                
                u, v, w = p_transformed
                print("Vector velocity in Body Frame (km/h):")
                print(f"[ {u:.2f} ")
                print(f"  {v:.2f} ")
                print(f"  {w:.2f} ]")
                print()
                
                V_nwind = math.sqrt(u**2 + v**2 + w**2)
                print(f"V_nwind = {V_nwind:.2f} km/h")
                print()
                print(f"Pitch={pitch:.2f}°")
                
                V_inf = np.add(p_transformed, Wind)
                
                # Para este caso, el avión está alineado con la velocidad del aire, por lo que la matriz de rotación es la identidad.
                # V_inf es lo mismo que p_transformed
                plot_3d_model(p_transformed, Wind, V_inf, np.identity(3), "Body Frame (Air/Wing) with Crosswind", p_rate, q_rate, r_rate)
                break
                
        elif flight_case == "C": # Crosswind and Climb
        
            x, y, z, p = inputs(flight_case)
            
            print("Vector entered (km/h) from", origin_name)
            print(f"[ {x} ")
            print(f"  {y} ")
            print(f"  {z} ]")
            print()
            
            print("Enter the Wind components:")
            Wd_x = float(input("Enter vector x component in km/h: "))
            Wd_y = float(input("Enter vector y component in km/h: "))
            Wd_z = float(input("Enter vector z component in km/h: "))
            Wind = np.array([Wd_x, Wd_y, Wd_z])
            
            if origin == "1": #vehicle al body (RzRyRxp)Transpuesta
            
                Rz, Ry, Rx, phi, theta, psi, yaw, pitch, roll = body_vehicle()
            
                Rot_mat = Rz @ Ry @ Rx
                p_transformed = Rot_mat.T @ p
                u, v, w = p_transformed
                
                print("Vector velocity in Body Frame (km/h):")
                print(f"[ {u:.2f} ")
                print(f"  {v:.2f} ")
                print(f"  {w:.2f} ]")
                print()
                
                V_inf, alpha, beta, gamma, V_nwind, Magnitude_V_inf = aircraft_angles(u,v,w,p, flight_case, pitch, Wind,Wd_x, Wd_z)
                print(f"Magnitude_V_inf = {Magnitude_V_inf:.2f} km/h")
                print()
                
                print("Aircraft angles")
                print(f"Angle of attack, α = {alpha:.2f}°")
                print(f"Sideslip angle, β = {beta:.2f}°")
                print(f"Climb angle, γ = {gamma:.2f}°")
                print()
                
                plot_3d_model(p_transformed, Wind, V_inf, Rot_mat, "Body Frame with Crosswind & Climb/Descent", p_rate, q_rate, r_rate)
                break
            
            elif origin == "2": #Air to body case C
                
                print("Note: Enter the desired angle of attack (α) to set the aircraft's orientation in level flight")
                alpha = float(input("Enter AoA in degrees: "))
                beta  = float(input("Enter sideslip in degrees: "))
                pitch = float(input("Enter pitch in degrees: "))
                gamma = pitch - alpha
                
                # Determine climb or descent
                if gamma > 0:
                    gamma_state = "Climb"
                else:  # gamma < 0 means descent
                    gamma_state = "Descent"
    
                A,B = wing_body(alpha, beta)
                LBW = A @ B
                
                p_transformed = LBW @ p
                
                u, v, w = p_transformed
                print("Vector velocity in Body Frame (km/h):")
                print(f"[ {u:.2f} ")
                print(f"  {v:.2f} ")
                print(f"  {w:.2f} ]")
                print()
                
                V_nwind = math.sqrt(u**2 + v**2 + w**2) #No wind, still atmosphere
                V_inf = np.add(p_transformed, Wind) #np.add calculates the vectorial sum
                Vix, Viy, Viz = V_inf
                Magnitude_V_inf = math.sqrt(Vix**2 + Viy**2 + Viz**2)
                
                print(f"V_ = {Magnitude_V_inf:.2f} km/h")
                print()
                print(f"Pitch={pitch:.2f}°")
                
                print(f"{gamma_state}, γ = {gamma:.2f}°")
                print()
                
                # Aquí también asumimos una matriz de rotación de identidad ya que el avión se alinea con el vector de velocidad del aire
                plot_3d_model(p_transformed, Wind, V_inf, np.identity(3), "Body Frame (Air/Wing) with Crosswind & Climb/Descent", p_rate, q_rate, r_rate)
                break
                
    elif (option =="2"):
        
        print("Note: The vector will be automatically transformed from the Body Frame to the Vehicle Frame.")
        print()
        flight_case=menu_cases()
        print()
        
        x, y, z, p = inputs(flight_case)
        
        u, v, w = p
        print("Vector velocity in Body Frame (km/h):")
        print(f"[ {u:.2f} ")
        print(f"  {v:.2f} ")
        print(f"  {w:.2f} ]")
        print()
        
        p_rate, q_rate, r_rate = angular_rates()
        
        
        if flight_case == "A": #Still atmosphere
                Wd_x = 0
                Wd_y = 0
                Wd_z = 0
                Wind = np.array([Wd_x, Wd_y, Wd_z])
                
                Rz, Ry, Rx, phi, theta, psi, yaw, pitch, roll = body_vehicle()
                V_inf, alpha, beta, gamma, V_nwind, Magnitude_V_inf = aircraft_angles(u, v, w,p,flight_case, pitch, Wind, Wd_x, Wd_z)
                print(f"V_nwind = {V_nwind:.2f} km/h")
                print()
                
                print("Aircraft angles")
                print(f"Angle of attack, α = {alpha:.2f}°")
                print(f"Sideslip angle, β = {beta:.2f}°")
                print(f"Climb angle, γ = {gamma:.2f}°")
                print()
                
                Rot_mat = Rz @ Ry @ Rx
                p_vehicle = Rot_mat @ p # Body to Vehicle. Rotation matrix, no transpose
                
                Vx, Vy, Vz = p_vehicle
                print("Vector velocity in Vehicle Frame (km/h):")
                print(f"[ {Vx:.2f} ")
                print(f"  {Vy:.2f} ")
                print(f"  {Vz:.2f} ]")
                print()

                # Como estamos en el sistema del vehículo, el avión se dibuja en la posición de los ejes principales (identidad)
                plot_3d_model(p, Wind, p_vehicle, np.identity(3), "Vehicle Frame Transformation", p_rate, q_rate, r_rate)
                break
            
        elif flight_case == "B":

            print("Enter the Wind components:")
            Wd_x = float(input("Enter vector x component in km/h: "))
            Wd_y = float(input("Enter vector y component in km/h: "))
            Wd_z = float(input("Enter vector z component in km/h: "))
            Wind = np.array([Wd_x, Wd_y, Wd_z])
            
            Rz, Ry, Rx, phi, theta, psi, yaw, pitch, roll = body_vehicle()
            
            V_inf, alpha, beta, gamma, V_nwind, Magnitude_V_inf = aircraft_angles(u,v,w,p,flight_case, pitch, Wind, Wd_x, Wd_z)
            print("Aircraft angles")
            print(f"Angle of attack, α = {alpha:.2f}°")
            print(f"Sideslip angle, β = {beta:.2f}°")
            print(f"Climb angle, γ = {gamma:.2f}°")
            print()
            
            Rot_mat = Rz @ Ry @ Rx
            p_transformed = Rot_mat @ p
            Vx, Vy, Vz = p_transformed
            
            print("Vector velocity in Vehicle Frame (km/h):")
            print(f"[ {Vx:.2f} ")
            print(f"  {Vy:.2f} ")
            print(f"  {Vz:.2f} ]")
            print()
            
            plot_3d_model(p, Wind, V_inf, np.identity(3), "Vehicle Frame with Crosswind", p_rate, q_rate, r_rate)
            break
            
        elif flight_case == "C": #Crosswind and climb/descend. Body to vehicle
            
            print("Enter the Wind components:")
            Wd_x = float(input("Enter vector x component in km/h: "))
            Wd_y = float(input("Enter vector y component in km/h: "))
            Wd_z = float(input("Enter vector z component in km/h: "))
            Wind = np.array([Wd_x, Wd_y, Wd_z])
            
            Rz, Ry, Rx, phi, theta, psi, yaw, pitch, roll = body_vehicle()
            
            V_inf, alpha, beta, gamma, V_nwind, Magnitude_V_inf = aircraft_angles(u,v,w,p,flight_case, pitch, Wind, Wd_x, Wd_z)
            print("Aircraft angles")
            print(f"Angle of attack, α = {alpha:.2f}°")
            print(f"Sideslip angle, β = {beta:.2f}°")
            print(f"Climb angle, γ = {gamma:.2f}°")
            print()
            
            Rot_mat = Rz @ Ry @ Rx
            p_transformed = Rot_mat @ p
            Vx, Vy, Vz = p_transformed
            
            print("Vector velocity in Vehicle Frame (km/h):")
            print(f"[ {Vx:.2f} ")
            print(f"  {Vy:.2f} ")
            print(f"  {Vz:.2f} ]")
            print()
            
            plot_3d_model(p, Wind, V_inf, np.identity(3), "Vehicle Frame with Crosswind & Climb/Descent", p_rate, q_rate, r_rate)
            break
            
    elif (option =="3"):
        
        print("Note: The vector will be transformed from the Body Frame to the Air/Wing Frame only.")
        print()
        flight_case=menu_cases()
        print()
        
        x, y, z, p = inputs(flight_case)
        
        u, v, w = p
        print("Vector velocity in Body Frame (km/h):")
        print(f"[ {u:.3f} ")
        print(f"  {v:.3f} ")
        print(f"  {w:.3f} ]")
        print()
        
        p_rate, q_rate, r_rate = angular_rates()
        
        if flight_case == "A":
            
            Wd_x = 0
            Wd_y = 0
            Wd_z = 0
            Wind = np.array([Wd_x, Wd_y, Wd_z])
            
            Rz, Ry, Rx, phi, theta, psi, yaw, pitch, roll = body_vehicle()
            
            V_inf, alpha, beta, gamma, V_nwind, Magnitude_V_inf = aircraft_angles(u,v,w,p, flight_case, pitch, Wind, Wd_x, Wd_z)
            print(f"V_nwind = {V_nwind:.2f} km/h")
            print()
            
            print("Aircraft angles")
            print(f"Angle of attack, α = {alpha:.2f}°")
            print(f"Sideslip angle, β = {beta:.2f}°")
            print(f"Climb angle, γ = {gamma:.2f}°")
            print()
                
            A,B = wing_body(alpha, beta)
            LBW = A @ B
            
            p_transformed = LBW.T @ p
            
            Vx, Vy, Vz = p_transformed
            print("Vector velocity in Air/Wing Fixed Frame (km/h):")
            print(f"[ {Vx:.2f} ")
            print(f"  {Vy:.2f} ")
            print(f"  {Vz:.2f} ]")
            print()
            
            # Dibujar en el sistema del avión (identidad)
            plot_3d_model(p, Wind, p_transformed, np.identity(3), "Air/Wing Fixed Frame", p_rate, q_rate, r_rate)
            break
            
        elif flight_case == "B": #Body al Air/Wing
        
            print("Enter the Wind components:")
            Wd_x = float(input("Enter vector x component in km/h: "))
            Wd_y = float(input("Enter vector y component in km/h: "))
            Wd_z = float(input("Enter vector z component in km/h: "))
            Wind = np.array([Wd_x, Wd_y, Wd_z])
            
            
            Rz, Ry, Rx, phi, theta, psi, yaw, pitch, roll = body_vehicle()
            V_inf, alpha, beta, gamma, V_nwind, Magnitude_V_inf = aircraft_angles(u,v,w,p, flight_case, pitch, Wind, Wd_x, Wd_z)
            
            print("Aircraft angles")
            print(f"Angle of attack, α = {alpha:.2f}°")
            print(f"Sideslip angle, β = {beta:.2f}°")
            print(f"Climb angle, γ = {gamma:.2f}°")
            print()
            
            A,B = wing_body(alpha, beta)
            LBW = A @ B
            
            p_transformed = LBW.T @ p
            
            Xw, Yw, Zw = p_transformed
            print("Vector velocity in Body Frame (km/h):")
            print(f"[ {Xw:.2f} ")
            print(f"  {Yw:.2f} ")
            print(f"  {Zw:.2f} ]")
            print()
            
            plot_3d_model(p, Wind, V_inf, np.identity(3), "Air/Wing Fixed Frame with Crosswind", p_rate, q_rate, r_rate)
            break
            
        elif flight_case == "C":
            print("Enter the Wind components:")
            Wd_x = float(input("Enter vector x component in km/h: "))
            Wd_y = float(input("Enter vector y component in km/h: "))
            Wd_z = float(input("Enter vector z component in km/h: "))
            Wind = np.array([Wd_x, Wd_y, Wd_z])    
            
            Rz, Ry, Rx, phi, theta, psi, yaw, pitch, roll = body_vehicle()
                                
            V_inf, alpha, beta, gamma, V_nwind, Magnitude_V_inf = aircraft_angles(u,v,w,p,flight_case, pitch, Wind, Wd_x, Wd_z)
            
            print("Aircraft angles")
            print(f"Angle of attack, α = {alpha:.2f}°")
            print(f"Sideslip angle, β = {beta:.2f}°")
            print(f"Climb angle, γ = {gamma:.2f}°")
            print()
            
            A,B = wing_body(alpha, beta)
            LBW = A @ B
            
            p_transformed = LBW.T @ p
            
            Xw, Yw, Zw = p_transformed
            print("Vector velocity in Body Frame (km/h):")
            print(f"[ {Xw:.2f} ")
            print(f"  {Yw:.2f} ")
            print(f"  {Zw:.2f} ]")
            print()

            plot_3d_model(p, Wind, V_inf, np.identity(3), "Air/Wing Fixed Frame with Crosswind & Climb/Descent", p_rate, q_rate, r_rate)
            break
            
    else:
        print("Invalid option. Please enter 1, 2, or 3.")