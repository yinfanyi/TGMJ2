import keyboard  # Make sure to install this library if you haven't already  


while True:  
    # Check if the 'Esc' key is pressed  
    if keyboard.is_pressed('esc'):  
        print("Exiting simulation...")  
        break  # Exit the loop if 'Esc' is pressed  

    print("Simulation is running...") 