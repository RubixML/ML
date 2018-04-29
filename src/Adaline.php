<?php

namespace Rubix\Engine;

use Rubix\Engine\NeuralNet\Bias;
use Rubix\Engine\NeuralNet\Input;
use Rubix\Engine\NeuralNet\Neuron;
use Rubix\Engine\NeuralNet\Synapse;
use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Persisters\Persistable;
use Rubix\Engine\NeuralNet\LearningRates\Adam;
use Rubix\Engine\NeuralNet\LearningRates\LearningRate;
use InvalidArgumentException;
use RuntimeException;

class Adaline extends Neuron implements Estimator, Classifier, Persistable
{
    /**
     * The number of training samples to consider per iteration of gradient descent.
     *
     * @var int
     */
    protected $batchSize;

    /**
     * The gradient descent optimizer.
     *
     * @var \Rubix\Engine\NeuralNet\LearningRates\LearningRate
     */
    protected $rate;

    /**
     * The minimum gradient descent step before the algorithm terminates early.
     *
     * @var float
     */
    protected $threshold;

    /**
     * The maximum number of training epochs. i.e. the number of times to iterate
     * over the entire training set.
     *
     * @var int
     */
    protected $epochs;

    /**
     * The actual labels of the binary class outcomes.
     *
     * @var array
     */
    protected $labels = [
        //
    ];

    /**
     * A history of the magnitude of each step of gradient descent.
     *
     * @var array
     */
    protected $steps = [
        //
    ];

    /**
     * @param  int  $inputs
     * @param  int  $epochs
     * @param  int  $batchSize
     * @param  \Rubix\Engine\NeuralNet\LearningRates\LearningRate  $rate
     * @param  float  $threshold
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $inputs, int $batchSize = 10, LearningRate $rate = null,
                                float $threshold = 1e-8, int $epochs = PHP_INT_MAX)
    {
        if ($inputs < 1) {
            throw new InvalidArgumentException('The number of inputs must be greater than 0.');
        }

        if ($batchSize < 1) {
            throw new InvalidArgumentException('Batch size cannot be less than 1.');
        }

        if ($threshold < 0) {
            throw new InvalidArgumentException('Early stopping threshold parameter must be 0 or greater.');
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException('Epoch parameter must be an integer greater than 0.');
        }

        if (!isset($rate)) {
            $rate = new Adam();
        }

        $this->batchSize = $batchSize;
        $this->rate = $rate;
        $this->threshold = $threshold;
        $this->epochs = $epochs;

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
     * @return array
     */
    public function steps() : array
    {
        return $this->steps;
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
        $this->steps = [];

        $this->zap();

        for ($epoch = 1; $epoch <= $this->epochs; $epoch++) {
            $magnitude = 0.0;

            foreach ($this->generateMiniBatches(clone $dataset) as $batch) {
                $outcomes = $batch->outcomes();
                $sigmas = array_fill(0, count($this->synapses), 0.0);
                $gradient = 0.0;

                foreach ($batch as $row => $sample) {
                    $activation = $this->feed($sample);

                    $output = $activation > 0 ? 1 : -1;

                    $expected = $this->labels[$output] === $outcomes[$row] ? $output : -$output;

                    $gradient += $expected - $output;

                    foreach ($this->synapses as $i => $synapse) {
                        $sigmas[$i] += $gradient * $synapse->neuron()->output();
                    }
                }

                foreach ($this->synapses as $i => $synapse) {
                    $step = $this->rate->step($synapse, $sigmas[$i]);

                    $synapse->adjustWeight($step);

                    $magnitude += abs($step);
                }
            }

            $this->steps[] = $magnitude;

            if ($magnitude < $this->threshold && $epoch > 1) {
                break 1;
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
            'activation' => abs($activation),
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
        if (count($sample) !== count($this->synapses) - 1) {
            throw new RuntimeException('The ratio of feature columns to inputs is unequal, '
                . (string) count($sample) . ' found, ' . (string) (count($this->synapses) - 1) . ' needed.');
        }

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
