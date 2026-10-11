#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// NETWORK CONFIGURATION
#define INPUT_NEURONS 6
#define HIDDEN_NEURONS 8
#define OUTPUT_NEURONS 6
// #define TOTAL_WEIGHTS \
//     (HIDDEN_NEURONS * (INPUT_NEURONS + 1) + \
//      HIDDEN_NEURONS * HIDDEN_NEURONS + \
//      OUTPUT_NEURONS * HIDDEN_NEURONS) // This is not a variable, it is a compile time number, compiler replaces it with an actual number
#define TOTAL_WEIGHTS 168
// Input (+1 for bias)
double input[INPUT_NEURONS + 1];

// Current hidden state (memory at this step)
double hidden[HIDDEN_NEURONS];

// Output prediction
double outputs[OUTPUT_NEURONS];

// Previous hidden state (memory from last step)
double context[HIDDEN_NEURONS];

// Weights
double w_input_hidden[HIDDEN_NEURONS][INPUT_NEURONS + 1];
double w_hidden_hidden[HIDDEN_NEURONS][HIDDEN_NEURONS];
double w_hidden_output[OUTPUT_NEURONS][HIDDEN_NEURONS];

// ACTIVATION FUNCTION
double sigmoid(double x){
    return 1.0 / (1.0 + exp(-x));
}

// RNN FEED FORWARD
void RNN_feed_forward(void)
{
    int i, j;

    // Update hidden state
    for (i = 0; i < HIDDEN_NEURONS; i++)
    {
        double sum = 0.0;

        // Input contribution
        for (j = 0; j < INPUT_NEURONS + 1; j++)
            sum += w_input_hidden[i][j] * input[j];

        // Recurrent contribution
        for (j = 0; j < HIDDEN_NEURONS; j++)
            sum += w_hidden_hidden[i][j] * context[j];

        hidden[i] = tanh(sum);
    }

    // Compute output
    for (i = 0; i < OUTPUT_NEURONS; i++)
    {
        double sum = 0.0;

        for (j = 0; j < HIDDEN_NEURONS; j++)
            sum += w_hidden_output[i][j] * hidden[j];

        outputs[i] = sigmoid(sum);
    }

    // Update memory
    for (i = 0; i < HIDDEN_NEURONS; i++)
        context[i] = hidden[i];
}

// resetting memory
void reset_context()
{
    for (int i =0 ; i < HIDDEN_NEURONS; i++)
        context[i] = 0.0;
}