<?php

namespace Rubix\Engine;

use Rubix\Engine\NeuralNetwork\Input;
use Rubix\Engine\NeuralNetwork\Hidden;
use Rubix\Engine\NeuralNetwork\Network;
use Rubix\Engine\NeuralNetwork\Optimizers\Adam;
use Rubix\Engine\NeuralNetwork\Optimizers\Optimizer;
use InvalidArgumentException;
use RuntimeException;
use SplObjectStorage;

class MultiLayerPerceptron extends Network implements Classifier
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
     * @param  int  $inputs
     * @param  array  $hidden
     * @param  array  $outcomes
     * @param  int  $epochs
     * @param  int  $batchSize
     * @param  \Rubix\Engine\NeuralNetwork\Optimizers\Optimizer  $optimizer
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $inputs, array $hidden, array $outcomes, int $epochs = 100, int $batchSize = 10, Optimizer $optimizer = null)
    {
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

        parent::__construct($inputs, $hidden, $outcomes);
    }

    /**
     * Train the network using mini-batch gradient descent with backpropagation.
     *
     * @param  \Rubix\Engine\Dataset  $data
     * @throws \InvalidArgumentException
     * @return void
     */
    public function train(Dataset $data) : void
    {
        if (!$data instanceof SupervisedDataset) {
            throw new InvalidArgumentException('This estimator requires a supervised dataset.');
        }

        if (in_array(self::CATEGORICAL, $data->columnTypes())) {
            throw new InvalidArgumentException('This estimator only works with continuous samples.');
        }

        $this->randomizeWeights();

        for ($epoch = 0; $epoch < $this->epochs; $epoch++) {
            foreach ($this->generateMiniBatches(clone $data) as $batch) {
                $sigmas = new SplObjectStorage();

                foreach ($batch as $row => $sample) {
                    $this->feed($sample);

                    $this->backpropagate($batch->getOutcome($row), $sigmas);
                }

                foreach ($sigmas as $synapse) {
                    $this->optimizer->step($synapse, $sigmas[$synapse]);
                }
            }
        }
    }

    /**
     * Feed a sample through the network and make a prediction based on the highest
     * activated output neuron.
     *
     * @param  array  $sample
     * @return \Rubix\Engine\Prediction
     */
    public function predict(array $sample) : Prediction
    {
        $this->feed($sample);

        $best = ['activation' => -INF, 'outcome' => null];

        foreach ($this->outputs() as $neuron) {
            $activation = $neuron->output();

            if ($activation > $best['activation']) {
                $best = [
                    'activation' => $activation,
                    'outcome' => $neuron->outcome(),
                ];
            }
        }

        return new Prediction($best['outcome'], [
            'activation' => $best['activation'],
        ]);
    }

    /**
     * Backpropgate the errors and save the sum of the computed deltas to be used
     * during gradient descent.
     *
     * @param  mixed  $outcome
     * @param  \SplObjectStorage  $sigmas
     * @return void
     */
    protected function backpropagate($outcome, SplObjectStorage $sigmas) : void
    {
        for ($layer = count($this->layers) - 1; $layer > 0; $layer--) {
            $deltas = new SplObjectStorage();

            foreach ($this->layers[$layer] as $neuron) {
                if ($neuron instanceof Hidden) {
                    $delta = $neuron->derivative();

                    if ($layer === count($this->layers) - 1) {
                        $expected = $neuron->outcome() === $outcome ? 1.0 : 0.0;

                        $delta *= ($expected - $neuron->output());
                    } else {
                        $previous = 0.0;

                        foreach ($previousDeltas as $previousNeuron) {
                            foreach ($previousNeuron->synapses() as $synapse) {
                                if ($synapse->neuron() === $neuron) {
                                    $previous += $synapse->weight() * $previousDeltas[$previousNeuron];
                                }
                            }
                        }

                        $delta *= $previous;
                    }

                    $deltas->attach($neuron, $delta);

                    foreach ($neuron->synapses() as $synapse) {
                        $sigma = $delta * $synapse->neuron()->output();

                        if ($sigmas->contains($synapse)) {
                            $sigmas[$synapse] += $sigma;
                        } else {
                            $sigmas->attach($synapse, $sigma);
                        }
                    }
                }
            }

            $previousDeltas = $deltas;
        }

        unset($deltas);
        unset($previousDeltas);
    }

    /**
     * Generate a collection of mini batches from the training data.
     *
     * @param  \Rubix\Engine\SupervisedDataset  $data
     * @return array
     */
    protected function generateMiniBatches(SupervisedDataset $data) : array
    {
        $data->randomize();

        $batches = [];

        while (!$data->isEmpty()) {
            $batches[] = $data->take($this->batchSize);
        }

        return $batches;
    }
}
