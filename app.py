import numpy as np
import random
import json

subjects = np.array(['Preliminary Activities', 'Edukasyon sa Pagpapakatao', 'Mother Tongue', 'Mathematics',
                     'RECESS', 'Filipino', 'Araling Panlipunan', 'English', 'MAPEH', 'Remedial Teaching', 'Anciliary'])
teachers = np.array(['teach1', 'teach2', 'teach3', 'teach4', 'teach5', 'teach6', 'teach7', 'teach8', 'teach9', 'teach10'])
sections = np.array(["Daffodil", "Everlasting", "Sampaguita", "Daisy", "Camia"])
homeroom_teacher = np.array(["Fevie Grace F. Habin", "Julie Ann R. Abingona", "Eglicelda A. Molo", "Arlene A. Montes", "Nery B. Sarion"])

subject_time = {
    'Preliminary Activities': 10,
    'Edukasyon sa Pagpapakatao': 30,
    'Mother Tongue': 50,
    'Mathematics': 50,
    'RECESS': 10,
    'Filipino': 30,
    'Araling Panlipunan': 40,
    'English': 40,
    'MAPEH': 40,
    'Remedial Teaching': 30,
    'Anciliary': 40
}

subject_teacher = {
    'Preliminary Activities': 'teach1',
    'Edukasyon sa Pagpapakatao': 'teach2',
    'Mother Tongue': 'teach3',
    'Mathematics': 'teach4',
    'RECESS': None,
    'Filipino': 'teach5',
    'Araling Panlipunan': 'teach6',
    'English': 'teach7',
    'MAPEH': 'teach8',
    'Remedial Teaching': 'teach9',
    'Anciliary': 'teach10'
}

timeslots = 11
recess_timeslot = 4

class Particle:
    def __init__(self, num_sections):
        self.position = np.zeros((num_sections, timeslots), dtype=np.int64)
        self.best_position = np.zeros((num_sections, timeslots), dtype=np.int64)
        self.velocity = np.zeros((num_sections, timeslots), dtype=np.int64)
        self.fitness = float('inf')
        self.best_fitness = float('inf')

def initialize_swarm(num_particles, num_sections):
    return [Particle(num_sections) for _ in range(num_particles)]

def generate_schedule(num_sections):
    schedule = np.zeros((num_sections, timeslots), dtype=np.int64)
    for section_idx in range(num_sections):
        subjects_copy = np.copy(subjects)
        schedule[section_idx, 0] = np.where(subjects == 'Preliminary Activities')[0][0]
        subjects_copy = np.delete(subjects_copy, np.where(subjects_copy == 'Preliminary Activities'))
        for timeslot in range(1, timeslots):
            if subjects_copy.size == 0:
                schedule[section_idx, timeslot] = np.where(subjects == 'RECESS')[0][0]
                continue
            subject_idx = random.choice(np.arange(subjects_copy.size))
            subject = subjects_copy[subject_idx]
            schedule[section_idx, timeslot] = np.where(subjects == subject)[0][0]
            subjects_copy = np.delete(subjects_copy, subject_idx)
    return schedule

def evaluate_fitness(schedule):
    fitness = 0
    num_sections = schedule.shape[0]
    for section_idx in range(num_sections):
        section_schedule = schedule[section_idx]
        for timeslot in range(timeslots):
            subject_idx = section_schedule[timeslot]
            subject = subjects[subject_idx]
            if subject == 'RECESS':
                if timeslot != recess_timeslot:
                    fitness += 100
                if timeslot >= 7 and timeslot <= 10:
                    fitness += 100
                if timeslot >= 0 and timeslot <= 2:
                    fitness += 100
                continue
            if subject == 'Preliminary Activities' and timeslot != 0:
                fitness += 100
                continue
            teacher = subject_teacher[subject]
            for other_timeslot in range(timeslots):
                if other_timeslot == timeslot:
                    continue
                other_subject_idx = section_schedule[other_timeslot]
                other_subject = subjects[other_subject_idx]
                if other_subject in ['RECESS', 'Preliminary Activities']:
                    continue
                other_teacher = subject_teacher[other_subject]
                if teacher == other_teacher:
                    fitness += 1000
                if subject == other_subject:
                    fitness += 1000
                if other_timeslot == timeslot + subject_time[subject] - 1:
                    fitness += 1000
    return fitness

def pso(num_particles, num_iterations, c1, c2, w, mutation_prob):
    swarm = initialize_swarm(num_particles, len(sections))
    global_best_fitness = float('inf')
    global_best_position = None
    for iteration in range(num_iterations):
        for particle in swarm:
            schedule = generate_schedule(len(sections))
            particle.position = schedule.copy()
            particle.fitness = evaluate_fitness(schedule)
            if particle.fitness < particle.best_fitness:
                particle.best_position = particle.position.copy()
                particle.best_fitness = particle.fitness
            if particle.fitness < global_best_fitness:
                global_best_fitness = particle.fitness
                global_best_position = particle.position.copy()
        for particle in swarm:
            r1 = np.random.random(particle.velocity.shape)
            r2 = np.random.random(particle.velocity.shape)
            particle.velocity = (w * particle.velocity +
                                 c1 * r1 * (particle.best_position - particle.position) +
                                 c2 * r2 * (global_best_position - particle.position))
            particle.velocity = np.clip(particle.velocity, -1, 1)
            particle.position = np.round(particle.position + particle.velocity).astype(np.int64)
            particle.position = np.clip(particle.position, 0, len(subjects) - 1)
            if np.random.random() < mutation_prob:
                mutated_position = particle.position.copy()
                for section_idx in range(len(sections)):
                    if mutated_position[section_idx, 0] != 0:
                        rand_subject_idx = random.choice(np.arange(1, len(subjects)))
                        mutated_position[section_idx, 0], mutated_position[section_idx, rand_subject_idx] = \
                            mutated_position[section_idx, rand_subject_idx], mutated_position[section_idx, 0]
                mutated_fitness = evaluate_fitness(mutated_position)
                if mutated_fitness < particle.best_fitness:
                    particle.best_position = mutated_position.copy()
                    particle.best_fitness = mutated_fitness
    return global_best_position

best_schedule = pso(num_particles=200, num_iterations=100, c1=3.50, c2=0.50, w=0.85, mutation_prob=0.1)

schedule_data = []
for section_idx in range(best_schedule.shape[0]):
    section_name = sections[section_idx]
    homeroom = homeroom_teacher[section_idx]
    section_schedule = []
    current_time = 6 * 60
    for timeslot in range(timeslots):
        subject_idx = best_schedule[section_idx, timeslot]
        subject = subjects[subject_idx]
        subject_duration = subject_time[subject]
        time_start = f"{current_time // 60:02d}:{current_time % 60:02d}"
        current_time += subject_duration
        time_end = f"{current_time // 60:02d}:{current_time % 60:02d}"
        minutes = subject_duration
        teacher = subject_teacher[subject]
        section_schedule.append({
            "time_start": time_start,
            "time_end": time_end,
            "minutes": minutes,
            "subject": subject,
            "teacher": teacher,
        })
    schedule_data.append({
        "section": section_name,
        "homeroom_teacher": homeroom,
        "schedule": section_schedule,
    })

with open('schedule.json', 'w') as f:
    json.dump(schedule_data, f)

print("Schedule data saved to 'schedule.json'.")
