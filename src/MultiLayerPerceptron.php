<?php

namespace Rubix\Engine;

use Rubix\Engine\NeuralNetwork\Hidden;
use Rubix\Engine\NeuralNetwork\Network;
use Rubix\Engine\NeuralNetwork\ActivationFunctions\Sigmoid;
use InvalidArgumentException;
use RuntimeException;
use SplObjectStorage;

class MultiLayerPerceptron extends Network implements Classifier
{
    const CATEGORICAL = 1;
    const CONTINUOUS = 2;

    /**
     * The fixed number of training epochs. i.e. the number of times to iterate
     * over the entire training set.
     *
     * @var int
     */
    protected $epochs;

    /**
     * The number of training samples to consider per iteration of gradient descent.
     *
     * @var int
     */
    protected $batchSize;

    /**
     * The learning rate. i.e. the size of each step towards the minimum during
     * gradient descent.
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
    public function __construct(int $inputs, array $hidden, array $outcomes, int $epochs = 100, int $batchSize = 10, float $rate = 0.1)
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

        $this->addOutputLayer($outcomes, new Sigmoid());

        foreach (range(count($this->layers) - 1, 1, -1) as $i) {
            $this->connectLayers($this->layers[$i], $this->layers[$i - 1]);
        }
    }

    /**
     * Train the model using backpropagation.
     *
     * @param  \Rubix\Engine\SupervisedDataset  $data
     * @return void
     */
    public function train(SupervisedDataset $data) : void
    {
        foreach ($data->types() as $type) {
            if ($type !== self::CONTINUOUS) {
                throw new InvalidArgumentException('This estimator only works with continuous input data.');
            }
        }

        list($samples, $outcomes) = $data->toArray();

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
                    $synapse->adjustWeight($this->rate * $deltas[$synapse]);
                }

                unset($deltas);
            }
        }
    }

    /**
     * Feed a sample through the network and make a prediction based on the highest
     * activated output neuron.
     *
     * @param  array  $sample
     * @return array
     */
    public function predict(array $sample) : array
    {
        $best = ['activation' => -INF, 'neuron' => null];

        $this->feed($sample);

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
     * Feed forward and calculate the output of each neuron in the network.
     *
     * @param  array  $sample
     * @throws \RuntimeException
     * @return void
     */
    protected function feed(array $sample) : void
    {
        if (count($sample) !== count($this->layers[0]) - 1) {
            throw new RuntimeException('Feature columns must equal the number of input neurons.');
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
        foreach (range(count($this->layers) - 1, 1, -1) as $layer) {
            $sigmas = new SplObjectStorage();

            foreach ($this->layers[$layer] as $neuron) {
                if ($neuron instanceof Hidden) {
                    $sigma = $neuron->error();

                    if ($layer === count($this->layers) - 1) {
                        $value = $neuron->outcome() === $outcome ? 1.0 : 0.0;

                        $sigma *= ($value - $neuron->output());
                    } else {
                        $prev = 0.0;

                        foreach ($prevSigmas as $prevNode) {
                            foreach ($prevNode->synapses() as $synapse) {
                                if ($synapse->neuron() === $neuron) {
                                    $prev += $synapse->weight() * $prevSigmas[$prevNode];
                                }
                            }
                        }

                        $sigma *= $prev;
                    }

                    $sigmas->attach($neuron, $sigma);

                    foreach ($neuron->synapses() as $synapse) {
                        $delta = $sigma * $synapse->neuron()->output();

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

        unset($sigmas);
        unset($prevSigmas);
    }
}
