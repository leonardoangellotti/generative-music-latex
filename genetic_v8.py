
"""Genetic Algorithm for music generation. Takes a user-defined key and tempo and evolves a melody accordingly."""

import random
from midiutil import MIDIFile
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt

# This constant defines the number of genomes (musical sequences) in each generation. 
# the population size determines how many solutions are being evaluated and evolved in each generation. 
# A larger population size can increase genetic diversity but also requires more computational resources.
POPULATION_SIZE = 10

# This constant sets the maximum number of generations the genetic algorithm will run. 
# Each generation involves evaluating, selecting, and breeding genomes to create the next generation. 
# The algorithm stops after reaching this number, even if the optimal solution (maximum fitness) has not been found.
MAX_GENERATIONS = 100

# This constant represents the probability of mutating each note in a genome during the mutation phase. 
# A mutation rate of 0.05 means there is a 5% chance that any given note will be altered.
# Mutation introduces genetic diversity and helps the algorithm explore new potential solutions.
MUTATION_RATE = 0.05

# This constant defines the target fitness score the algorithm aims to achieve. 
# The fitness score is a measure of how well a genome (musical sequence) meets the desired criteria 
# (e.g., harmony, rhythm, smoothness). The algorithm may stop early if a genome reaches this fitness score, 
# indicating an optimal or near-optimal solution has been found.
MAX_FITNESS = 30

# The scaleStructures dictionary provides interval patterns for different musical scales, defining how to construct each scale starting from any root note.
# The patterns are repeated twice to ensure they cover a full two-octave range.
# This dictionary allows the program to build various scales dynamically based on user input for key and scale type, facilitating the creation of diverse musical sequences.
scaleStructures = {
    "major": [2, 2, 1, 2, 2, 2, 1] * 2,
    "minor": [2, 1, 2, 2, 1, 2, 2] * 2,
}

# list of notes duration for each type of rythm
rythms = {
    "rock" : [1, 0.5, 1, 0.5, 1, 0.5, 0.5, 0.5],
    "jazz" : [0.75, 0.75, 0.5, 0.25, 0.25, 0.5, 0.5, 1],
    "dance" : [0.5, 0.5, 0.5, 1, 0.5, 0.5, 0.5, 1],
    "bossa nova" : [1.5, 0.5, 0.75, 0.75, 1, 0.5, 0.5, 0.25],
}

# where to store the fitness weight for each generation
fitness_weight = []

# MIDI codes for notes starting from A below middle C
scales = {}
a3 = 45 # MIDI code for A3 (A below middle C)
currentCode = a3

# Generate MIDI note codes for all notes including sharps and flats
for note in "abcdefg":

    scales[note] = currentCode # Assign the current code to the note

    # Add the sharp note (if applicable)
    if note not in ["b", "e"]:
        scales[f"{note}#"] = currentCode + 1 # Sharp notes are one semitone above the current note
        currentCode += 2 # Move to the next note

    # Add the flat note (if applicable)
    if note not in ["c", "f"]:
        scales[f"{note}b"] = currentCode - 1 # Flat notes are one semitone below the current note
        currentCode += 2 # Move to the next note
    else:
        currentCode += 1 # Move to the next note without adding a flat

### ------------------------------- MAIN ------------------------------- ###

def genetic_algorithm(key, root, tempo, rythm):

    """Main function for running all helper functions and handling user input."""

    # List available scale options for the user to choose from
    scaleOptions = list(scaleStructures.keys())
    
    # Get the user to select a scale from the available options
    key = getUserInput(key, scaleOptions)
    
    # Get the user to enter the root note for the chosen scale
    root = getUserInput(root, scales.keys())
    
    # Get and validate the tempo input from the user
    # tempo = getValidTempo()

    # Build the scale based on the root note and chosen scale type
    scale = buildScale(root, key)

    # Run the genetic algorithm to evolve a melody using the specified mutation rate and scale
    evolvedMelody = runEvolution(MUTATION_RATE, scale)

    # print(evolvedMelody[0])

    # NORMALIZED MIDI
    # every note is transpose to a single octave C3-C4 in order to avoid high gap between notes
    # and having more omogenuos melody.
    normalized_melody = normalizedMidi(evolvedMelody)

    # write on disk the best melody
    writeMidiToDisk(normalized_melody[0], rythm, f"best_genome.mid", tempo)

    # Save the other evolved melody to a separate MIDI files
    for i, melody in enumerate(normalized_melody[1:len(normalized_melody)]):
        
        writeMidiToDisk(melody, rythm, f"genome_{i}.mid", tempo)

    # REPEAT MELODY
    repeated_melody = repeatMelody(normalized_melody[0])

    # write on disk the repeated melody 
    writeMidiToDisk(repeated_melody, rythm, f"repeated_melody.mid", tempo)

    # display message
    status_var.set("Generation Completed")

### ------------------------------- FUNCTIONS ------------------------------- ###

# If the user input is not valid (i.e., it doesn't match any of the valid options), 
# an error message is printed, and the loop continues to prompt the user again.
def getUserInput(userinput, validOptions):

    """Gets user input and validates it."""

    # Infinite loop to continuously prompt the user until a valid input is received
    while True:

        # Read user input, converting it to lowercase and stripping any leading/trailing whitespace
        choice = userinput.lower().strip()

        # Check if the user input matches any of the valid options
        if choice in validOptions:

            # If a valid option is chosen, return it and exit the loop
            return choice
        
        # If the input is not valid, inform the user and prompt again
        print("Invalid choice. Please try again.")

# By following this process, the getValidTempo function ensures that the user 
# provides a valid tempo value within the specified range before proceeding.
def getValidTempo():

    """Gets and validates the tempo input from the user."""
    
    # Infinite loop to continuously prompt the user until a valid tempo is provided
    while True:

        # Prompt the user to input a tempo value and strip any leading/trailing whitespace
        tempo = input("Pick a tempo (integer) between 30 and 300 bpm: ").strip()
    
        # Check if the provided tempo value is valid by calling the isValidTempo function
        if isValidTempo(tempo):

            # If valid, convert the tempo value to an integer and return it
            return int(tempo)
        
        # If the tempo value is not valid, inform the user and prompt again
        print("Invalid tempo. Please try again.")

# By following this process, the buildScale function constructs 
# a complete musical scale by iterating through the pattern of intervals for the specified scale type,
# starting from the root note
def buildScale(root, key):

    """Builds scale based on the root and key."""
    
    # Retrieve the MIDI note value for the root note from the scales dictionary
    # ex: a3: 45
    rootCode = scales[root]
    
    # Retrieve the pattern of intervals (in semitones) for the specified scale type from the scaleStructures dictionary
    # major/ minor
    pattern = scaleStructures[key]
    
    # Initialize the scale list with the root note's MIDI note value
    scale = [rootCode]

    # Iterate over the interval pattern to construct the full scale
    for step in pattern:

        # Add the interval step to the rootCode to get the next note in the scale
        rootCode += step
        
        # Append the new note's MIDI note value to the scale list
        scale.append(rootCode)

    # print(scale)

    return scale

# The function returns this 2D list, representing a musical genome as a sequence of notes organized into 8 bars, 
# each containing 16 notes. This random selection of notes from the scale forms the basis of the initial population
# for the genetic algorithm used in music generation.
def generateGenome(scale):

    """generate genome containing 8 bars, each containing 16 notes"""
    
    # Create a list of 8 bars, where each bar is a list of 16 random notes selected from the scale
    return [[random.choice(scale) for _ in range(8)] for _ in range(8)]

# The function returns this list, representing the entire population of genomes. 
# Each genome in the population is a 2D list of note sequences organized into 8 bars, 
# each containing 16 notes, all randomly chosen from the specified scale.
def generatePopulation(size, scale):

    """Generates a population of genomes."""

    # Create a list of 'size' genomes, where each genome is generated using the generateGenome function with the given scale
    return [generateGenome(scale) for _ in range(size)]

# Calculates the final fitness score by weighting and summing
# the component scores for smoothness, rhythm, and harmony, and returns the result.
def fitnessFunction(genome):

    """Calculates the fitness of a sequence based on smoothness, rhythm, and harmony."""
    
    # Weights for the different fitness components
    smoothnessWeight = 15
    restWeight = 5
    harmonyWeight = 20

    # Initialize fitness scores
    smoothnessScore = 0
    restScore = 0
    harmonyScore = 0

    # Harmony intervals table for scoring harmony
    harmonyIntervalsTable = {
        0: 50, 1: 5, 2: 5, 3: 50, 4: 50, 5: 30, 6: -10, 7: 50, 8: 10, 9: 40, 10: -2, 11: -2, 12: 10,
        13: -5, 14: 5, 15: 5, 16: 50, 17: 50, 18: 30, 19: -10, 20: 50, 21: 10, 22: 40, 23: -2, 24: -2, 25: 10
    }

    # Calculate the number of rest notes in the genome
    numRests = sum(1 for bar in genome for note in bar if note is None)
    # number of consecutive rests
    consecutiveRests = 0

    # Iterate over each bar in the genome
    for bar in genome:

        # Iterate over each note in the bar, starting from the second note
        for i in range(1, len(bar)):
        
            note = bar[i]
            prevNote = bar[i - 1]

            # Calculate scores for smoothness and harmony if both current and previous notes are not None
            if note is not None and prevNote is not None:
                noteDifference = abs(note - prevNote)
                harmonyScore += harmonyIntervalsTable.get(noteDifference, 0)

                # Adjust smoothness score based on the difference between consecutive notes
                if noteDifference == 0:
                    smoothnessScore /= 10
                elif noteDifference <= 2:
                    smoothnessScore += 1
                elif noteDifference == 11:
                    smoothnessScore /= 2
                else:
                    smoothnessScore += 1 / noteDifference if noteDifference != 0 else 0

                # Further adjust smoothness score if notes are an octave apart
                if abs(note - (prevNote + 12)) in [1, 2] or abs((note + 12) - prevNote) in [1, 2]:
                    smoothnessScore += 0.5

            # Count consecutive rests
            if note is None and prevNote is None:
                consecutiveRests += 1

    # Adjust rhythm score based on the number of rest notes
    if numRests * 10 <= len(flatten(genome)):
        restScore += 10

    # Penalize the rhythm score for consecutive rests
    if consecutiveRests:
        restScore -= (consecutiveRests * 10)

    # Calculate the final fitness score by weighting and summing the component scores 
    # for a single genome in the population
    fitness_weight_genome = (smoothnessScore * smoothnessWeight) + (restScore * restWeight) + (harmonyScore * harmonyWeight)

    # append the score to the list
    fitness_weight.append(fitness_weight_genome)

    return fitness_weight_genome

# By following this process, the selectParents function ensures that genomes
# with higher fitness scores have a higher probability of being selected as 
# parents for the next generation, thereby promoting the evolution of more optimal solutions.
def selectParents(population):

    """Selects two sequences from the population based on their fitness."""
    
    # Calculate the fitness score for each genome in the population
    weights = [fitnessFunction(genome) for genome in population]

    # Use the calculated fitness scores as weights to randomly select two genomes from the population
    return random.choices(population, weights=weights, k=2)

# By following this process, the crossoverFunction ensures that genetic information
# from both parents is mixed to produce new offspring, promoting genetic diversity 
# and potentially combining beneficial traits from both parents in the resulting child genomes.
def multipointCrossover(parentA, parentB, num_points=5):

    """Performs multipoint crossover on two sequences."""
    
    # Flatten the 2D arrays of the parent genomes into 1D arrays
    noteStringA = flatten(parentA)
    noteStringB = flatten(parentB)

    # Ensure both parent sequences are of the same length
    if len(noteStringA) != len(noteStringB):

        raise ValueError("Parent sequences are not the same length")

    # Select multiple unique crossover points and sort them
    # random.sample ensures that the crossover points are unique
    # sorted ensures the points are in ascending order
    crossover_points = sorted(random.sample(range(1, len(noteStringA)), num_points))

    # Initialize empty lists to hold the flattened child genomes
    childAFlat = []
    childBFlat = []

    # Alternate segments between parents based on crossover points
    last_point = 0  # Start from the beginning of the genome
    swap = False    # Start with not swapping, so the first segment is from parentA to childA

    # Loop through each crossover point and the end of noteStringA
    for point in crossover_points + [len(noteStringA)]:

        # If swap is True, swap the segments from parent strings
        if swap:
        
            # Swap segments: childA gets segment from parentB, and childB gets segment from parentA

            # childAFlat.extend(noteStringB[last_point:point]):
            # This line adds the segment from noteStringB (from last_point to point) to the end of childAFlat.
            childAFlat.extend(noteStringB[last_point:point])

            # childBFlat.extend(noteStringA[last_point:point]):
            # This line adds the segment from noteStringA (from last_point to point) to the end of childBFlat.
            childBFlat.extend(noteStringA[last_point:point])

        else:
        
            # No swap: childA gets segment from parentA, and childB gets segment from parentB
            childAFlat.extend(noteStringA[last_point:point])
            childBFlat.extend(noteStringB[last_point:point])
        
        # Toggle the swap flag for the next segment
        swap = not swap

        # Update the last crossover point
        last_point = point

    # Unflatten the child genomes back into 2D arrays with the original bar length
    # (the lenght of parentA and parentB are the same)
    return unflatten(childAFlat, len(parentA[0])), unflatten(childBFlat, len(parentA[0]))

# By following this process, the mutateGenome function introduces random changes
# to the sequence of notes based on the mutation rate, promoting genetic diversity 
# in the population and potentially exploring new musical ideas that may improve the 
# fitness of the genome.
def mutateGenome(genome, mutationRate, scale):

    """Mutates a sequence according to a mutation probability."""
    
    # Iterate over each bar in the genome
    for i, bar in enumerate(genome):
    
        # Iterate over each note in the bar
        for j, note in enumerate(bar):
    
            # Generate a random number between 0 and 1 and compare it to the mutation rate
            if random.uniform(0, 1) <= mutationRate:
    
                # whit a probability of 50% the note become rest
                if random.uniform(0, 1) < 0.7:
                    genome[i][j] = None

                # or is changed by another note from the scale
                else:
                    genome[i][j] = random.choice(scale)
                
    # Return the mutated genome
    return genome

# Function to display notes on a piano roll
def displayPianoRoll(population):

    fig, axs = plt.subplots(len(population), 1, figsize=(12, len(population) * 3), sharex=True)
    
    note_height = 0.6  # Height of the note rectangles

    for idx, genome in enumerate(population):

        ax = axs[idx]
        time = 0

        for bar in genome: # for every bar in the genome (8 bars)

            for note in bar: # for every note in the genome (16 notes)
                
                if note is not None:  # Check if the note is not a rest

                    ax.add_patch(plt.Rectangle((time, note - note_height / 2), 1, note_height, edgecolor='black', facecolor='blue'))
                
                time += 1  # Increment time by 1 unit for each note

        ax.set_xlabel('Time')
        ax.set_ylabel(idx)
        ax.set_ylim(45, 90)  # Adjust based on the range of notes
        ax.set_xlim(0, time)
        ax.grid(True)
    
    plt.show()

# Function to display weightd on a scatter plot
def displayWeight(weights):

    plt.figure(figsize=(20, 20))  # width: 10 inches, height: 6 inches
    plt.scatter(range(len(weights)), weights, s = 5)
    plt.xlabel('epoch')
    plt.ylabel('weights')
    plt.title('weights of genomes')
    plt.show()
    plt.show()

# By following this process, the runEvolution function implements a genetic algorithm 
# that iteratively improves the population of musical genomes through selection, 
# crossover, and mutation, aiming to maximize the fitness of the genomes based 
# on the defined fitness function.
def runEvolution(mutationRate, scale):

    """Runs the genetic algorithm until a genome with the specified MAX_FITNESS score is reached."""
    
    # Generate the initial population of genomes using the specified scale
    population = generatePopulation(POPULATION_SIZE, scale)

    # Iterate for a maximum number of generations
    for _ in range(MAX_GENERATIONS):

        # Sort the population based on fitness scores in descending order
        population = sorted(population, key=fitnessFunction, reverse=True)
        
        # Select the top 2 genomes (the fittest) to carry over to the next generation
        nextGeneration = population[:2]

        # Generate the rest of the next generation through crossover and mutation
        for _ in range(len(population) // 2 - 1):

            # Select two parent genomes based on their fitness
            parentA, parentB = selectParents(population)
            # Perform crossover to produce two child genomes
            childA, childB = multipointCrossover(parentA, parentB)
            # Mutate the child genomes and add them to the next generation
            nextGeneration += [mutateGenome(childA, mutationRate, scale), mutateGenome(childB, mutationRate, scale)]

        # Update the population with the new generation
        population = nextGeneration

    # print(population)

    displayPianoRoll(population)

    displayWeight(fitness_weight)

    # Return the final sorted population based on fitness scores in descending order
    return sorted(population, key=fitnessFunction, reverse=True)

# normalized every midi genome in the population bringing every note in one octave range
def normalizedMidi(population):

    """normalized every midi genome in the population bringing every note in one octave range"""

    for genome in population:    

        # Iterate over each bar in the genome
        for i, bar in enumerate(genome):
        
            # Iterate over each note in the bar
            for j, note in enumerate(bar):
        
                # Generate a random number between 0 and 1 and compare it to the mutation rate
                if genome[i][j] is None:
                    pass

                # if the note is lower than C3 (60 midi code)
                elif genome[i][j] < 60:
                    # encrease by one octave (12 semitones)
                    genome[i][j] += 12

                # if the note is higther than C4 (72 midi code)
                elif genome[i][j] > 72:
                    # dencrease by one octave (12 semitones)
                    genome[i][j] -= 12
        
    # Return the normalized population
    return population

# repeat some bars from one genome in order to increase precived regularity 
def repeatMelody(genome):

    # number of bar to take
    K = 3

    # number of repetition
    repeat = 4 

    repeated_genome = []

    # Iterate over each bar in the genome
    for i, bar in enumerate(genome):
    
        for _ in range(repeat - 1):
        
            repeated_genome.append(genome[i])

        if i == K - 1: break

    return repeated_genome


# By following this process, the writeMidiToDisk function converts 
# the generated sequence of musical notes into a MIDI file, with appropriate track name,
# tempo, note durations, and volumes, and writes it to disk with the specified filename.
def writeMidiToDisk(sequence, rythm, filename="out", userTempo=60):

    """Writes the generated sequence to a MIDI file."""
    
    # Create a new MIDI file with one track
    midiFile = MIDIFile(1)
    track = 0 # Track number
    time = 0 # Start time for the track

    # Add track name and tempo to the MIDI file
    midiFile.addTrackName(track, time, filename)
    midiFile.addTempo(track, time, userTempo)

    # Flatten the 2D sequence array into a 1D array of notes
    fSequence = flatten(sequence)
    channel = 0 # MIDI channel (0-15)
    volume = 100 # Volume level (0-127)

    # select the rythm and the list of notes duration
    # each rythm contain 8 note duration
    # multiplying by 2 we obtain the lenght for a single bar
    # multiply by 8 we obtain the duration for 8 bars (one genome)
    duration = rythms[rythm] * 2 * 8

    # Iterate over each note in the flattened sequence
    for i, pitch in enumerate(fSequence):

        # If the pitch is not None (not a rest), add the note to the MIDI file
        if pitch is not None:

            midiFile.addNote(track, channel, pitch, time, duration[i], volume)

        # Increment the time 
        time += duration[i]
        # if you would use random choice duration
        # time += random.choice([0.25, 0.5, 1])

    # Open a file in binary write mode and write the MIDI data to it
    with open(filename, 'wb') as outf:
        midiFile.writeFile(outf)

# By following this process, the flatten function effectively combines
# all the nested sublists in a 2D array into a single 1D list, 
# which is useful for simplifying operations that need to process all
# elements in the array sequentially.
def flatten(arr):

    """Flattens a 2D array into a 1D array."""
    
    # List comprehension that iterates through each sublist (bar) in the 2D array (arr)
    # and then iterates through each element (note) in those sublists to create a 1D list
    return [note for bar in arr for note in bar]

# By following this process, the unflatten function takes a 1D list and converts
# it back into a 2D list with a specified number of elements per sublist, 
# which is useful for organizing data into a structured format, such as musical bars.
def unflatten(arr, barLength):

    """Converts a 1D array back into a 2D array with specified bar length."""
    
    # List comprehension that slices the 1D array (arr) into sublists of length barLength
    # It starts at index 0 and increments by barLength to create each sublist
    return [arr[i:i + barLength] for i in range(0, len(arr), barLength)]

# By following this process, the isValidTempo function ensures that the input value is 
# a valid integer within the specified tempo range (30 to 300 BPM). 
# If the input is invalid (either not an integer or outside the valid range), 
# the function returns False. This validation is crucial for ensuring that
# the user provides a reasonable tempo for the musical sequence.
def isValidTempo(val):

    """Validates the tempo input."""
    
    # Attempt to convert the input value to an integer
    try:
        val = int(val)

        # Check if the integer value is within the valid tempo range (30 to 300 BPM)
        return 30 <= val <= 300
    
    # If a ValueError occurs during the conversion, return False indicating invalid input
    except ValueError:
        return False

### ------------------------------- EXECUTE ------------------------------- ###

def generate_music():

    key = key_var.get()
    root = root_var.get()
    tempo = tempo_var.get()
    rythm = rythm_var.get()
    
    genetic_algorithm(key, root, tempo, rythm)

app = tk.Tk()
app.title("Music Generator")

ttk.Label(app, text="Key:").grid(column=0, row=0)
key_var = tk.StringVar(value="major")
ttk.Combobox(app, textvariable=key_var, values=["major", "minor"]).grid(column=1, row=0)

ttk.Label(app, text="Root Note:").grid(column=0, row=1)
root_var = tk.StringVar(value="C")
ttk.Combobox(app, textvariable=root_var, values=["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]).grid(column=1, row=1)

ttk.Label(app, text="Tempo:").grid(column=0, row=2)
tempo_var = tk.IntVar(value=120)
ttk.Entry(app, textvariable=tempo_var).grid(column=1, row=2)

ttk.Label(app, text="Rythm:").grid(column=0, row=3)
rythm_var = tk.StringVar(value="rock")
ttk.Combobox(app, textvariable=rythm_var, values=["rock", "jazz", "dance", "bossa nova"]).grid(column=1, row=3)

ttk.Button(app, text="Generate", command=generate_music).grid(column=0, row=4, columnspan=2)

# Status label to display completion message
status_var = tk.StringVar(value="")
ttk.Label(app, textvariable=status_var).grid(column=0, row=5, columnspan=2)

app.mainloop()
