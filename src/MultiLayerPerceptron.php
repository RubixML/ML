<?php

namespace Rubix\Engine;

use Rubix\Engine\NeuralNet\Input;
use Rubix\Engine\NeuralNet\Hidden;
use Rubix\Engine\NeuralNet\Network;
use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Persisters\Persistable;
use Rubix\Engine\NeuralNet\Optimizers\Adam;
use Rubix\Engine\NeuralNet\Optimizers\Optimizer;
use InvalidArgumentException;
use RuntimeException;
use SplObjectStorage;

class MultiLayerPerceptron extends Network implements Estimator, Classifier, Persistable
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
     * @var \Rubix\Engine\NeuralNet\Optimizers\Optimizer
     */
    protected $optimizer;

    /**
     * @param  int  $inputs
     * @param  array  $hidden
     * @param  array  $outcomes
     * @param  int  $epochs
     * @param  int  $batchSize
     * @param  \Rubix\Engine\NeuralNet\Optimizers\Optimizer  $optimizer
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
     * @param  \Rubix\Engine\Datasets\Supervised  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function train(Supervised $dataset) : void
    {
        if (in_array(self::CATEGORICAL, $dataset->columnTypes())) {
            throw new InvalidArgumentException('This estimator only works with continuous samples.');
        }

        $this->randomizeWeights();

        for ($epoch = 0; $epoch < $this->epochs; $epoch++) {
            foreach ($this->generateMiniBatches(clone $dataset) as $batch) {
                $sigmas = new SplObjectStorage();

                foreach ($batch as $row => $sample) {
                    $this->feed($sample);

                    $this->backpropagate($sigmas, $batch->getOutcome($row));
                }

                foreach ($sigmas as $synapse) {
                    $step = $this->optimizer->step($synapse, $sigmas[$synapse]);

                    $synapse->adjustWeight($step);
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
     * Feed a sample through the network and calculate the output of each neuron.
     *
     * @param  array  $sample
     * @throws \RuntimeException
     * @return void
     */
    public function feed(array $sample) : void
    {
        if (count($sample) !== count($this->layers[0]) - 1) {
            throw new RuntimeException('The number of feature columns must equal the number of input neurons.');
        }

        $this->reset();

        $column = 0;

        foreach ($this->inputs() as $input) {
            if ($input instanceof Input) {
                $input->prime($sample[$column++]);
            }
        }

        foreach ($this->outputs() as $output) {
            $output->fire();
        }
    }

    /**
     * Backpropgate the error through the network and return the sums of the partial
     * derivatives for each parameter.
     *
     * @param  \SplObjectStorage  $sigmas
     * @param  mixed  $outcome
     * @return void
     */
    protected function backpropagate(SplObjectStorage $sigmas, $outcome) : void
    {
        for ($layer = count($this->layers) - 1; $layer > 0; $layer--) {
            $gradients = new SplObjectStorage();

            foreach ($this->layers[$layer] as $neuron) {
                if ($neuron instanceof Hidden) {
                    if ($layer === count($this->layers) - 1) {
                        $expected = $neuron->outcome() === $outcome ? 1.0 : 0.0;

                        $gradient = $neuron->derivative() * ($expected - $neuron->output());
                    } else {
                        $previousGradient = 0.0;

                        foreach ($previousGradients as $previousNeuron) {
                            foreach ($previousNeuron->synapses() as $synapse) {
                                if ($synapse->neuron() === $neuron) {
                                    $previousGradient += $synapse->weight() * $previousGradients[$previousNeuron];
                                }
                            }
                        }

                        $gradient = $neuron->derivative() * $previousGradient;
                    }

                    $gradients->attach($neuron, $gradient);

                    foreach ($neuron->synapses() as $synapse) {
                        $sigma = $gradient * $synapse->neuron()->output();

                        if ($sigmas->contains($synapse)) {
                            $sigmas[$synapse] += $sigma;
                        } else {
                            $sigmas->attach($synapse, $sigma);
                        }
                    }
                }
            }

            $previousGradients = $gradients;
        }
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
