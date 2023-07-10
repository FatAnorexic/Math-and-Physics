import math

G=6.6743e-11 #Gravitational constant

def Capture_Calculation(
        stellar_radius: float, obj_velocity: float, stellar_mass: float
        )->float:
    #escape velocity squared| short hand for legibility
    evs=(2*G*stellar_mass)/stellar_radius

    capture_radius=stellar_radius*math.sqrt( 1+ evs/math.pow(obj_velocity,2))
    return capture_radius

def Effective_Capture_Area(capture_radius: float)->float:
    sigma=math.pi*(capture_radius*capture_radius)
    return sigma

def main():
    print('This will calculate the radius of impact between a star and an object\n'+
          'with initial velocity v')
    
    star_mass=float(input('What is the mass of the star in kg? IE 6e24: '))
    star_radius=float(input('What is the radius of the target star in meters| IE 6.957e8: '))
    object_vel=float(input('What is the objects initial velocity in m/s| IE 7000: '))

    b_capture=Capture_Calculation(star_radius,object_vel,star_mass)
    sigma=Effective_Capture_Area(b_capture)

    print(
        f'\nThe effective radii of impact is {b_capture} meters\n'+ 
        f'and its effective area {sigma} meters**2')

if __name__=='__main__':
    import doctest
    doctest.testmod()
    main()