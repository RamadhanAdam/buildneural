# Elman RNN trained with a Genetic Algorithm

A recurrent neural network written in C, trained without backpropagation.
Instead of computing gradients like a normal person, we treat the weights
like DNA and let evolution figure it out. Surprisingly, it works.

This is not how real production RNNs are trained. Nobody at Google is doing this.
But it is a clean way to understand what weights actually do and how a network
can learn sequential patterns without anyone doing any calculus.


## What it learns

The network is given a repeating sequence of numbers, for example:

    0  1  2  0  1  2  0  1  2

Given the current number, predict the next one.
Simple on paper. But because the sequence loops, the network has to remember
where it is in the pattern. That is what the recurrent connections are for.
You can change the sequence to anything you like in the visualizer.


## How it trains

1. Start with 50 random networks. They are all terrible.
2. Test each one. Measure how wrong its predictions are.
3. The better ones are more likely to have children.
4. Children inherit weights from two parents, with small random mutations.
5. Repeat 100 times. The population gets smarter each generation.
6. By the end, at least one of them usually has it figured out.

This is a genetic algorithm. The fitness of each network is scored as:

    fitness = 1 / (1 + total_error)

Perfect prediction scores close to 1. Completely wrong scores close to 0.


## Files

elmann_rnn.c — the network itself. Forward pass, activation functions, memory reset.
Read this first if you want to understand what is actually happening.
It is short and every line is commented.

ga.c — the genetic algorithm that trains it. Reads like plain English.
Includes comments explaining every decision. Does not require understanding
of calculus or gradients. Just selection, crossover, and mutation.

visualizer.c — optional, standalone. Opens a window where you can watch
everything happen live, interact with the network, and change parameters.
Does not require ga.c or elmann_rnn.c to be compiled separately.
All logic is embedded inside it.


## How to run

You have two options. Pick one. You do not need both.

Option 1 — terminal only, no extra dependencies:

    gcc ga.c -lm -o rnn_ga
    ./rnn_ga

Prints each generation and shows final predictions in the terminal.
ga.c includes elmann_rnn.c automatically so you only compile one file.

Option 2 — interactive visual window:

    gcc visualizer.c -I/opt/homebrew/include -L/opt/homebrew/lib -lraylib -lm -o visualizer
    ./visualizer

This is completely standalone. It contains all the GA and RNN logic internally.
You do not need to run ga.c first or at all.


## Setting up on Mac

If you get an error about missing developer tools run this first:

    xcode-select --install

For the visualizer you need raylib. Install it with Homebrew:

    brew install raylib

If you do not have Homebrew:

    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"


## Setting up on Windows

Option 1 works on Windows with MinGW installed:

    gcc ga.c -lm -o rnn_ga.exe
    rnn_ga.exe

For the visualizer on Windows, download raylib from raylib.com and follow
the Windows setup instructions there.


## Visualizer controls

    SPACE          start or pause evolution
    R              reset everything and start over
    H              show or hide the hidden state memory panel
    [ and ]        fewer or more hidden neurons (resets on change)
    + and -        slower or faster evolution speed
    E              edit the training sequence (type digits 0-2, press ENTER)
    click a node   highlight all its connections and weights
    hover a node   see what that neuron does and its current value
    type 0, 1, 2   while paused: feed a number in manually and watch the signal flow


## What the visualizer shows

Network panel on the left shows all nodes and connections live.
Input neurons are blue, hidden neurons are orange, output neurons are green.
Line color shows whether each weight is positive (red) or negative (blue).
Line thickness shows how strong the weight is.
The fill arc inside each node shows how active it is right now.

Hidden state panel next to the network shows the memory from the previous step.
These are the values that get fed back into the hidden layer, giving the network
its sense of time. Press H to hide this panel if you want a cleaner view.

Fitness graph top right shows the best fitness in the population climbing
over generations. A flat line means the population is stuck. A climbing line
means evolution is working.

Predictions panel shows all steps of the sequence, whether the best network
got each one right or wrong, and the score out of total steps.

Info panel at the bottom describes whatever node you are hovering over,
including its current value and what role it plays in the network.


## Architecture

    Input layer:   6 neurons plus 1 bias
    Hidden layer:  8 neurons with recurrent connections (adjustable in visualizer)
    Output layer:  6 neurons
    Total weights: 168

The recurrent connections are what make this an Elman RNN specifically.
The hidden layer saves its state and feeds it back in on the next step,
giving the network a form of short term memory.


## Why not just use backpropagation

You can. BPTT, backpropagation through time, is the standard way to train RNNs
and it is faster and more accurate for serious tasks.

The genetic algorithm is here because it requires no gradient computation,
no learning rate tuning, and no understanding of calculus to read the code.
It is slower and less efficient, but the logic is human readable from top to bottom.
That was the point.


## Building on this

Change the number of hidden neurons in the visualizer and watch fitness change.
Press E and type a different sequence to train on something other than 0 1 2.
Replace the GA with actual BPTT and compare how fast each one learns.
Add a second hidden layer and see if more memory helps.