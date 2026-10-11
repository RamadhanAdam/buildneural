#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "elmann_rnn.c"

/*
 * GENETIC ALGORITHM TRAINER FOR ELMAN RNN
 *
 * Instead of using backpropagation (which requires computing gradients),
 * we treat the RNN weights like DNA and evolve them over generations.
 *
 * The idea:
 *   1. Start with 50 random brains (each brain = 168 weights)
 *   2. Test each brain on a task (predict the next number in a sequence)
 *   3. Better brains are more likely to reproduce
 *   4. Children inherit mixed weights from two parents, with small random mutations
 *   5. Repeat 100 times, the population gets smarter each generation
 *
 * The task we are training on:
 *   The sequence 0 1 2 0 1 2 0 1 2
 *   Given the current number, predict the next one.
 *   This loops, so the RNN must remember context to do it well.
 */

#define POP_SIZE 50       // How many RNNs we evolve in parallel
#define GENERATIONS 100   // How many rounds of evolution
#define MUTATION_RATE 0.05 // 5% chance any single weight gets nudged

/*
 * A Chromosome represents one candidate RNN.
 * gene[] holds all 168 weights as a flat array.
 * fitness measures how well it predicts the sequence.
 * Higher fitness = smaller prediction error.
 */
typedef struct {
    double gene[TOTAL_WEIGHTS];
    double fitness;
} Chromosome;

Chromosome population[POP_SIZE];     // current generation
Chromosome new_population[POP_SIZE]; // next generation (built during reproduction)

/*
 * init_population
 * Give every RNN in the population random weights between -1 and 1.
 * This is generation zero, pure randomness, no skill yet.
 */
void init_population()
{
    for (int i = 0; i < POP_SIZE; i++)
    {
        for (int j = 0; j < TOTAL_WEIGHTS; j++)
        {
            population[i].gene[j] =
                ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        }
        population[i].fitness = 0.0;
    }
}

/*
 * load_weights
 * The GA stores all weights as a flat array (gene[]).
 * The RNN needs them as 2D matrices.
 * This function unpacks the flat array into the weight matrices.
 *
 * Order: input->hidden weights, then hidden->hidden, then hidden->output.
 */
void load_weights(double *gene)
{
    int index = 0;

    for (int i = 0; i < HIDDEN_NEURONS; i++)
        for (int j = 0; j < INPUT_NEURONS + 1; j++)
            w_input_hidden[i][j] = gene[index++];

    for (int i = 0; i < HIDDEN_NEURONS; i++)
        for (int j = 0; j < HIDDEN_NEURONS; j++)
            w_hidden_hidden[i][j] = gene[index++];

    for (int i = 0; i < OUTPUT_NEURONS; i++)
        for (int j = 0; j < HIDDEN_NEURONS; j++)
            w_hidden_output[i][j] = gene[index++];
}

/*
 * select_parent
 * Tournament selection: pick two random candidates, return the better one.
 * This gives fitter individuals a higher chance to reproduce,
 * but does not completely exclude weaker ones (keeps diversity).
 */
int select_parent()
{
    int a = rand() % POP_SIZE;
    int b = rand() % POP_SIZE;

    return (population[a].fitness > population[b].fitness) ? a : b;
}

/*
 * reproduce
 * Build the next generation from the current one.
 *
 * For each new child:
 *   - Pick two parents via tournament selection
 *   - For each weight, randomly inherit from parent 1 or parent 2 (crossover)
 *   - With 5% probability, nudge that weight slightly (mutation)
 *
 * Mutation adds a small random value between -0.1 and +0.1.
 * This prevents the population from getting stuck.
 *
 * After building new_population, replace the old one.
 */
void reproduce()
{
    for (int i = 0; i < POP_SIZE; i++)
    {
        int p1 = select_parent();
        int p2 = select_parent();

        for (int j = 0; j < TOTAL_WEIGHTS; j++)
        {
            // crossover: flip a coin to pick which parent this weight comes from
            if (rand() % 2)
                new_population[i].gene[j] = population[p1].gene[j];
            else
                new_population[i].gene[j] = population[p2].gene[j];

            // mutation: occasionally nudge the weight slightly
            if (((double)rand() / RAND_MAX) < MUTATION_RATE)
                new_population[i].gene[j] +=
                    ((double)rand() / RAND_MAX) * 0.2 - 0.1;
        }
    }

    for (int i = 0; i < POP_SIZE; i++)
        population[i] = new_population[i];
}

/*
 * evaluate_population
 * Score every RNN in the population.
 *
 * We feed each RNN the sequence 0 1 2 0 1 2 0 1 2 one step at a time.
 * At each step we give it the current number and ask it to predict the next.
 * We measure how wrong it is (squared error).
 *
 * Fitness = 1 / (1 + total_error)
 * So fitness is always between 0 and 1.
 * Perfect prediction = fitness close to 1.
 * Terrible prediction = fitness close to 0.
 *
 * The input is one-hot encoded:
 *   input[0] = 1.0 always (bias neuron)
 *   input[1] = 1 if current letter is 0, else 0
 *   input[2] = 1 if current letter is 1, else 0
 *   input[3] = 1 if current letter is 2, else 0
 *   (inputs 4 and 5 unused in this task, available for extension)
 */
void evaluate_population()
{
    for (int i = 0; i < POP_SIZE; i++)
    {
        load_weights(population[i].gene);
        reset_context(); // clear RNN memory before each evaluation

        double total_error = 0.0;

        int sequence[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
        int length = 9;

        for (int t = 0; t < length - 1; t++)
        {
            for (int j = 0; j < INPUT_NEURONS + 1; j++)
                input[j] = 0.0;

            input[0] = 1.0;                  // bias always on
            input[sequence[t] + 1] = 1.0;   // one-hot encode current step

            RNN_feed_forward();

            int target = sequence[t + 1]; // what we expect the RNN to predict

            for (int k = 0; k < OUTPUT_NEURONS; k++)
            {
                double expected = (k == target) ? 1.0 : 0.0;
                double diff = expected - outputs[k];
                total_error += diff * diff;
            }
        }

        population[i].fitness = 1.0 / (1.0 + total_error);
    }
}

/*
 * main
 * The full evolution loop:
 *   1. Seed random number generator
 *   2. Create random initial population
 *   3. For each generation: evaluate fitness, then reproduce
 *   4. After all generations, find the best RNN and demo its predictions
 */
int main()
{
    srand(time(NULL));

    init_population();

    for (int gen = 0; gen < GENERATIONS; gen++)
    {
        evaluate_population();
        reproduce();
        printf("Generation %d complete\n", gen);
    }

    // Find the best individual in the final population
    int best = 0;
    for (int i = 1; i < POP_SIZE; i++)
        if (population[i].fitness > population[best].fitness)
            best = i;

    printf("\nBest fitness: %f\n", population[best].fitness);

    // Load the best weights into the RNN and run the sequence
    load_weights(population[best].gene);
    reset_context();

    int demo[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    printf("\nPredictions from best individual:\n");

    for (int t = 0; t < 8; t++)
    {
        for (int j = 0; j < INPUT_NEURONS + 1; j++) input[j] = 0.0;
        input[0] = 1.0;
        input[demo[t] + 1] = 1.0;
        RNN_feed_forward();

        // The predicted next number is whichever output neuron fired strongest
        int predicted = 0;
        for (int k = 1; k < OUTPUT_NEURONS; k++)
            if (outputs[k] > outputs[predicted])
                predicted = k;

        printf("Input: %d -> Predicted: %d (expected %d)\n",
               demo[t], predicted, demo[t + 1]);
    }

    return 0;
}