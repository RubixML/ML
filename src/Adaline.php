<?php

namespace Rubix\Engine;

use Rubix\Engine\NeuralNetwork\Bias;
use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\NeuralNetwork\Input;
use Rubix\Engine\NeuralNetwork\Neuron;
use Rubix\Engine\NeuralNetwork\Synapse;
use Rubix\Engine\Persisters\Persistable;
use Rubix\Engine\NeuralNetwork\Optimizers\Adam;
use Rubix\Engine\NeuralNetwork\Optimizers\Optimizer;
use InvalidArgumentException;

class Adaline extends Neuron implements Estimator, Classifier, Persistable
{
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
     * The gradient descent optimizer.
     *
     * @var \Rubix\Engine\NeuralNetwork\Optimizers\Optimizer
     */
    protected $optimizer;

    /**
     * The actual labels of the binary class outcomes.
     *
     * @var array
     */
    protected $labels = [
        //
    ];

    /**
     * @param  int  $inputs
     * @param  int  $epochs
     * @param  int  $batchSize
     * @param  \Rubix\Engine\NeuralNetwork\Optimizers\Optimizer  $optimizer
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $inputs, int $epochs = 10, int $batchSize = 10, Optimizer $optimizer = null)
    {
        if ($inputs < 1) {
            throw new InvalidArgumentException('The number of inputs must be greater than 0.');
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException('Epoch parameter must be an integer greater than 0.');
        }

        if ($batchSize < 1) {
            throw new InvalidArgumentException('Batch size cannot be less than 1.');
        }

        if (!isset($optimizer)) {
            $optimizer = new Adam();
        }

        $this->epochs = $epochs;
        $this->batchSize = $batchSize;
        $this->optimizer = $optimizer;

        for ($i = 0; $i < $inputs; $i++) {
            $this->connect(new Synapse(new Input()));
        }

        $this->connect(new Synapse(new Bias()));
    }

    /**
     * Return the weight paramters of the neuron.
     *
     * @return array
     */
    public function weights() : array
    {
        return array_map(function ($synapse) {
            return $synapse->weight();
        }, $this->synapses);
    }

    /**
     * Perform mini-batch gradient descent with given optimizer over the training
     * set and update the input weights accordingly.
     *
     * @param  \Rubix\Engine\Datasets\Supervised  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function train(Supervised $dataset) : void
    {
        $labels = $dataset->labels();

        if (count($labels) !== 2) {
            throw new InvalidArgumentException('The number of unique outcomes must be exactly 2, ' . (string) count($labels) . ' found.');
        }

        if (in_array(self::CATEGORICAL, $dataset->columnTypes())) {
            throw new InvalidArgumentException('This estimator only works with continuous samples.');
        }

        $this->labels = [1 => $labels[0], -1 => $labels[1]];

        $this->zap();

        for ($epoch = 0; $epoch < $this->epochs; $epoch++) {
            foreach ($this->generateMiniBatches(clone $dataset) as $batch) {
                $outcomes = $batch->outcomes();
                $sigmas = array_fill(0, count($this->synapses), 0.0);
                $delta = 0.0;

                foreach ($batch as $row => $sample) {
                    $activation = $this->feed($sample);

                    $output = $activation > 0 ? 1 : -1;

                    $expected = $this->labels[$output] === $outcomes[$row] ? $output : -$output;

                    $delta += ($expected - $output);

                    foreach ($this->synapses as $i => $synapse) {
                        $sigmas[$i] += $delta * $synapse->neuron()->output();
                    }
                }

                foreach ($this->synapses as $i => $synapse) {
                    $this->optimizer->step($synapse, $sigmas[$i]);
                }
            }
        }
    }

    /**
     * Read the activation of the neuron and make a prediction.
     *
     * @param  array  $sample
     * @return \Rubix\Engine\Prediction
     */
    public function predict(array $sample) : Prediction
    {
        $activation = $this->feed($sample);

        return new Prediction($this->labels[$activation > 0 ? 1 : -1], [
            'activation' => $activation,
        ]);
    }

    /**
     * Feed a sample into the network and return the output of the neuron.
     *
     * @param  array  $sample
     * @return float
     */
    public function feed(array $sample) : float
    {
        $column = 0;
        $z = 0.0;

        foreach ($this->synapses as $synapse) {
            $neuron = $synapse->neuron();

            if ($neuron instanceof Input) {
                $neuron->prime($sample[$column++]);
            }
        }

        foreach ($this->synapses as $synapse) {
            $z += $synapse->impulse();
        }

        return $z;
    }

    /**
     * Generate a collection of mini batches from the training data.
     *
     * @param  \Rubix\Engine\Datasets\Supervised  $dataset
     * @return array
     */
    protected function generateMiniBatches(Supervised $dataset) : array
    {
        $dataset->randomize();

        $batches = [];

        while (!$dataset->isEmpty()) {
            $batches[] = $dataset->take($this->batchSize);
        }

        return $batches;
    }
}
