import re
import pandas as pd
import numpy as np
import matplotlib

def find_pairs(file):
    # Hardcoded per file
    with open(file) as f:

        # Get rid of header file
        _ = f.readline()

        executives = list()
        analysts = list()


        count = 0
        while True:
            # Extract name from line
            line = f.readline()

            if "Executives" in line or "Participants" in line:
                break


            count += 1
            if count > 100:
                return None, None, None

        count = 0
        while True:

            # Extract name from line
            line = f.readline()
            name = line.split("-")[0].strip()

            if "Analysts" in line or "Participants" in line:
                break

            executives.append(name)

            count += 1
            if count > 100:
                return None, None, None

        count = 0
        while True:

            # Extract name from line
            line = f.readline()
            name = line.split("-")[0].strip()

            if "Operator" in line:
                break

            analysts.append(name)

            count += 1
            if count > 100:
                return None, None, None

        # Extract body of text now
        body = list()

        count = 0
        while True:

            # Extract name from line
            line = f.readline().strip()

            flag = False

            count += 1
            if count > 100:
                return None, None, None

            for name in executives:
                if name in line:
                    flag = True
                    break

            if flag:
                continue

            if ("Question-and-Answer" in line or "Questions-and-Answer" in line) and len(line) < 100:
                break

            body.append(line + " ")




        questions = list()
        while True:

            line = f.readline().strip()

            if not line:
                break

            for name in analysts:
                if name in line and len(line) < 100:
                    question = f.readline().strip()

                    if '?' in question:
                        questions.append(question)
                    break


        body = "".join(body)

        # Return body and one random question
        return body.lower(), questions, True if body and questions else False


def readfiles():
    print("Reading lines...")

    pairs = list()


    import os

    count = 0
    unusable_count = 0

    with os.scandir('cleaned_scrape/') as entries:


        for entry in entries:
            if entry.path.endswith(".txt"):
                count += 1

                body, questions, flag = find_pairs(entry)
                if flag:
                    pairs.append( (body, questions))
                else:
                    unusable_count += 1

    print("Total number of documents: " + str(count))
    print("Number of usuable documents: "+  str(len(pairs)))
    print("Number of unusuable documents: " + str(unusable_count))
    return pairs


pairs = readfiles()
def overall_statistics(pairs):
    exposition_length = np.empty([len(pairs)])
    num_questions = np.empty([len(pairs)])

    questions_list = list()

    for i in range(len(pairs)):
        body = pairs[i][0]
        questions = pairs[i][1]

        exposition_length[i] = len(body.split(" "))

        # if exposition_length[i] < 14:
        #     print(body)

        num_questions[i] = len(questions)

        for question in questions:
            questions_list.append(len(question))

    questions_list = np.array(questions_list)


    print()
    print("Overall Statistics For Exposition Length")


    print("Exposition Max Length: {}".format(np.amax(exposition_length)))
    print("Exposition Min Length: {}".format(np.amin(exposition_length)))

    print("Exposition Mean Length: {}".format(exposition_length.mean()))
    print("Exposition Standard Deviation: {}".format(exposition_length.std()))

    import matplotlib.pyplot as plt
    plt.hist(exposition_length)
    plt.ylabel('exposition length')
    plt.savefig("Exposition_Overall.png")
    plt.clf()

    print()
    print("Overall Statistics For Number of Questions")

    print("Num Questions Max Length: {}".format(np.amax(num_questions)))
    print("Num Questions Min Length: {}".format(np.amin(num_questions)))

    print("Num Questions Mean Length: {}".format(num_questions.mean()))
    print("Num Questions Standard Deviation: {}".format(num_questions.std()))

    import matplotlib.pyplot as plt
    plt.hist(num_questions)
    plt.ylabel('num_questions')
    plt.savefig("Num_Questions.png")
    plt.clf()

    print()
    print("Overall Statistics For Length of Questions")

    print("Length of Questions Max Length: {}".format(np.amax(questions_list)))
    print("Length of QuestionsMin Length: {}".format(np.amin(questions_list)))

    print("Length of Questions Mean Length: {}".format(questions_list.mean()))
    print("Length of Questions Standard Deviation: {}".format(questions_list.std()))

    import matplotlib.pyplot as plt
    plt.hist(questions_list)
    plt.ylabel('questions_length')
    plt.savefig("questions_lengt.png")
    plt.clf()

if __name__ == '__main__':

    overall_statistics(pairs)