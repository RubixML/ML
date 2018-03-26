<?php

namespace Rubix\Engine;

use Rubix\Engine\NeuralNetwork\Bias;
use Rubix\Engine\NeuralNetwork\Input;
use Rubix\Engine\NeuralNetwork\Neuron;
use Rubix\Engine\NeuralNetwork\Hidden;
use Rubix\Engine\NeuralNetwork\Output;
use Rubix\Engine\NeuralNetwork\ActivationFunctions\Sigmoid;
use Rubix\Engine\NeuralNetwork\ActivationFunctions\ActivationFunction;
use InvalidArgumentException;
use SplObjectStorage;

class MultiLayerPerceptron implements Estimator
{
    /**
     * The layers of the neural network.
     *
     * @var array
     */
    protected $layers = [
        //
    ];

    /**
     * The maximum number of training epochs. i.e. training rounds.
     *
     * @var int
     */
    protected $epochs;

    /**
     * The number of samples to consider per training round.
     *
     * @var int
     */
    protected $batchSize;

    /**
     * The learning rate. i.e. the size of each step towards the minimum during
     * backpropagation.
     *
     * @var float
     */
    protected $rate;

    /**
     * @param  int  $inputs
     * @param  array  $hidden
     * @param  array  $outcomes
     * @param  int  $epochs
     * @param  int  $batchSize
     * @param  float  $rate
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $inputs, array $hidden, array $outcomes, int $epochs = 1000, int $batchSize = 3, float $rate = 0.3)
    {
        if ($inputs < 1) {
            throw new InvalidArgumentException('The number of inputs must be greater than 1.');
        }

        foreach ($hidden as &$layer) {
            if (!is_array($layer)) {
                $layer = [$layer];
            }

            if (!is_int($layer[0]) || $layer[0] < 1) {
                throw new InvalidArgumentException('The size parameter of a hidden layer must be an integer greater than 0.');
            }

            if (isset($layer[1])) {
                if (!$layer[1] instanceof ActivationFunction) {
                    throw new InvalidArgumentException('The second hidden layer parameter must be an instance of an ActivationFunction.');
                }
            }
        }

        $outcomes = array_unique($outcomes);

        if (count($outcomes) < 1) {
            throw new InvalidArgumentException('The number of unique outcomes must be greater than 1.');
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException('Epoch parameter must be an integer greater than 0.');
        }

        if ($batchSize < 1) {
            throw new InvalidArgumentException('Batch size cannot be less than 1.');
        }

        $this->epochs = $epochs;
        $this->batchSize = $batchSize;
        $this->rate = $rate;

        $this->addInputLayer($inputs);

        foreach ($hidden as $layer) {
            $this->addHiddenLayer($layer[0], $layer[1] ?? new Sigmoid());
        }

        $this->addOutputLayer($outcomes);

        foreach (range(count($this->layers) - 1, 1, -1) as $i) {
            $this->connectLayers($this->layers[$i], $this->layers[$i - 1]);
        }
    }

    /**
     * Return the input layer.
     *
     * @return array
     */
    public function inputs() : array
    {
        return $this->layers[0];
    }

    /**
     * Return the output layer.
     *
     * @return array
     */
    public function outputs() : array
    {
        return $this->layers[count($this->layers) - 1];
    }

    /**
     * Train the model using backpropagation.
     *
     * @param  array  $samples
     * @param  array  $outcomes
     * @return void
     */
    public function train(array $samples, array $outcomes) : void
    {
        foreach (range(1, $this->epochs) as $epoch) {
            $order = range(0, count($samples) - 1);

            shuffle($order);

            array_multisort($order, $samples, $outcomes);

            $queue = [
                array_slice($samples, 0),
                array_slice($outcomes, 0),
            ];

            while (!empty($queue[0])) {
                $batch = [
                    array_splice($queue[0], 0, $this->batchSize),
                    array_splice($queue[1], 0, $this->batchSize),
                ];

                $deltas = new SplObjectStorage();

                foreach ($batch[0] as $i => $sample) {
                    $this->feed($sample);

                    $this->backpropagate($batch[1][$i], $deltas);
                }

                foreach ($deltas as $synapse) {
                    $synapse->adjustWeight($deltas[$synapse]);
                }
            }
        }
    }

    /**
     * Predict a sample.
     *
     * @param  array  $sample
     * @return array
     */
    public function predict(array $sample) : array
    {
        $this->feed($sample);

        $best = ['activation' => -INF, 'neuron' => null];

        foreach ($this->outputs() as $neuron) {
            if ($neuron->output() > $best['activation']) {
                $best = ['activation' => $neuron->output(), 'neuron' => $neuron];
            }
        }

        return [
            'outcome' => $best['neuron']->outcome(),
            'activation' => $best['neuron']->output(),
        ];
    }

    /**
     * Feed forward.
     *
     * @param  array  $sample
     * @throws \InvalidArgumentException
     * @return void
     */
    protected function feed(array $sample) : void
    {
        if (count($sample) !== count($this->layers[0]) - 1) {
            throw new InvalidArgumentException('Feature columns must equal the number of input neurons.');
        }

        $this->reset();

        foreach ($this->inputs() as $input) {
            if ($input instanceof Input) {
                $input->prime(next($sample));
            }
        }

        foreach ($this->outputs() as $output) {
            $output->fire();
        }
    }

    /**
     * Backpropgate the errors and save the sum of the computed deltas to be used
     * when adjusting the synapse weights.
     *
     * @param  mixed  $outcome
     * @param  \SplObjectStorage  $deltas
     * @return void
     */
    protected function backpropagate($outcome, SplObjectStorage $deltas) : void
    {
        foreach (range(count($this->layers) - 1, 1, -1) as $i) {
            $sigmas = new SplObjectStorage();

            foreach ($this->layers[$i] as $j => $neuron) {
                if ($neuron instanceof Hidden) {
                    $output = $neuron->output();
                    $sigma = $neuron->error();

                    if ($i === count($this->layers) - 1) {
                        $value = $neuron->outcome() === $outcome ? 1.0 : 0.0;

                        $sigma *= ($value - $output);
                    } else {
                        $prev = 0.0;

                        foreach ($prevSigmas as $prevNode) {
                            foreach ($prevNode->synapses() as $synapse) {
                                $prev += $synapse->weight() * $prevSigmas[$prevNode];
                            }
                        }

                        $sigma *= $prev;
                    }

                    $sigmas->attach($neuron, $sigma);

                    foreach ($neuron->synapses() as $synapse) {
                        $delta = $this->rate * $sigma * $synapse->neuron()->output();

                        if ($deltas->contains($synapse)) {
                            $deltas[$synapse] += $delta;
                        } else {
                            $deltas->attach($synapse, $delta);
                        }
                    }
                }
            }

            $prevSigmas = $sigmas;
        }
    }

    /**
     * Add the input layer of neurons.
     *
     * @param  int  $inputs
     * @return array
     */
    protected function addInputLayer(int $inputs) : array
    {
        $layer = [];

        foreach (range(1, $inputs) as $i) {
            $layer[] = new Input();
        }

        array_push($layer, new Bias());

        $this->layers[] = $layer;

        return $layer;
    }

    /**
     * Add a hidden layer of n neurons using given activation function.
     *
     * @param  int  $n
     * @param  \Rubix\Engine\NeuralNetwork\ActivationsFunctions\ActivationFunction
     * @return array
     */
    protected function addHiddenLayer(int $n, ActivationFunction $activationFunction) : array
    {
        $layer = [];

        foreach (range(1, $n) as $i) {
            $layer[] = new Hidden($activationFunction);
        }

        array_push($layer, new Bias());

        $this->layers[] = $layer;

        return $layer;
    }

    /**
     * Add an output layer of neurons.
     *
     * @param  array  $outcomes
     * @return array
     */
    protected function addOutputLayer(array $outcomes) : array
    {
        $outcomes = array_unique($outcomes);
        $layer = [];

        foreach ($outcomes as $outcome) {
            $layer[] = new Output($outcome);
        }

        $this->layers[] = $layer;

        return $layer;
    }

    /**
     * Fully connect layer a to layer b.
     *
     * @param  array  $a
     * @param  array  $b
     * @return self
     */
    protected function connectLayers(array $a, array $b) : self
    {
        foreach ($a as $next) {
            if ($next instanceof Neuron) {
                foreach ($b as $current) {
                    $next->connect($current);
                }
            }
        }

        return $this;
    }

    /**
     * Reset the z values for all neurons in the network.
     *
     * @return void
     */
    protected function reset() : void
    {
        foreach (range(1, count($this->layers) - 1) as $i) {
            foreach ($this->layers[$i] as $neuron) {
                if ($neuron instanceof Hidden) {
                    $neuron->reset();
                }
            }
        }
    }
}
