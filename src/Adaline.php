<?php

namespace Rubix\Engine;

use Rubix\Engine\NeuralNet\Bias;
use Rubix\Engine\NeuralNet\Input;
use Rubix\Engine\NeuralNet\Neuron;
use Rubix\Engine\NeuralNet\Synapse;
use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Persisters\Persistable;
use Rubix\Engine\NeuralNet\Optimizers\Adam;
use Rubix\Engine\NeuralNet\Optimizers\Optimizer;
use Rubix\Engine\NeuralNet\ActivationFunctions\Identity;
use InvalidArgumentException;
use RuntimeException;

class Adaline extends Neuron implements Estimator, Classifier, Persistable
{
    /**
     * The maximum number of training epochs. i.e. the number of times to iterate
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
     * @var \Rubix\Engine\NeuralNet\Optimizers\Optimizer
     */
    protected $optimizer;

    /**
     * The minimum absolute difference in magnitude of a gradient descent step
     * before the algorithm terminates early.
     *
     * @var float
     */
    protected $threshold;

    /**
     * The function that quantizes the input values into a binary output value.
     *
     * @var \Rubix\Engine\NeuralNet\ActivationFunctions\ActivationFunction
     */
    protected $quantizer;

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
     * @param  float  $threshold
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $inputs, int $epochs = 100, int $batchSize = 5,
                                Optimizer $optimizer = null, float $threshold = 1e-8)
    {
        if ($inputs < 1) {
            throw new InvalidArgumentException('The number of inputs must be greater than 0.');
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException('Estimator must train for at least 1 epoch.');
        }

        if ($batchSize < 1) {
            throw new InvalidArgumentException('Batch size cannot be less than 1.');
        }

        if ($threshold < 0) {
            throw new InvalidArgumentException('Early stopping threshold parameter must be 0 or greater.');
        }

        if (!isset($optimizer)) {
            $optimizer = new Adam();
        }

        parent::__construct(new Identity());

        for ($i = 0; $i < $inputs; $i++) {
            $this->connect(new Synapse(new Input()));
        }

        $this->connect(new Synapse(new Bias()));

        $this->batchSize = $batchSize;
        $this->epochs = $epochs;
        $this->optimizer = $optimizer;
        $this->threshold = $threshold;
    }

    /**
     * Return the weight parameters of the neuron.
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
        $last = 0.0;

        $this->zap();

        for ($epoch = 1; $epoch <= $this->epochs; $epoch++) {
            $magnitude = 0.0;

            foreach ($this->generateMiniBatches(clone $dataset) as $batch) {
                $sigmas = array_fill(0, count($this->synapses), 0.0);
                $outcomes = $batch->outcomes();

                foreach ($batch as $row => $sample) {
                    list($z, $activation, $output) = $this->feed($sample);

                    $expected = $this->labels[$output] === $outcomes[$row] ? $output : -$output;

                    $gradient = $this->activationFunction->differentiate($z, $activation)
                        * ($expected - $activation);

                    foreach ($this->synapses as $i => $synapse) {
                        $sigmas[$i] += $gradient * $synapse->node()->output();
                    }
                }

                foreach ($this->synapses as $i => $synapse) {
                    $step = $this->optimizer->step($synapse, $sigmas[$i]);

                    $synapse->adjustWeight($step);

                    $magnitude += abs($step);
                }
            }

            $this->steps[] = $magnitude;

            $change = abs($last - $magnitude);

            if ($change < $this->threshold && $epoch > 1) {
                break 1;
            }

            $last = $magnitude;
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
        list($z, $activation, $output) = $this->feed($sample);

        return new Prediction($this->labels[$output], [
            'activation' => $activation,
        ]);
    }

    /**
     * Feed a sample into the network and return the activation and output of the
     * neuron.
     *
     * @param  array  $sample
     * @return array
     */
    public function feed(array $sample) : array
    {
        if (count($sample) !== count($this->synapses) - 1) {
            throw new RuntimeException('The ratio of feature columns to inputs is unequal, '
                . (string) count($sample) . ' found, ' . (string) (count($this->synapses) - 1) . ' needed.');
        }

        $column = 0;
        $z = 0.0;

        foreach ($this->synapses as $synapse) {
            $neuron = $synapse->node();

            if ($neuron instanceof Input) {
                $neuron->prime($sample[$column++]);
            }
        }

        foreach ($this->synapses as $synapse) {
            $z += $synapse->impulse();
        }

        $activation = $this->activationFunction->compute($z);

        $output = $activation >= 0 ? 1 : -1;

        return [$z, $activation, $output];
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
